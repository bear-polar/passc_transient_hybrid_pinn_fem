#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1D Parabolic PDE Solver: Stabilized FEM (SUPG) + PINN Correction

   u_t - eps*u_xx + b(x)*u_x = f(t,x),   (t,x) in (0,1] x (0,1)
   u(0,x) = u_0(x),                       0 < x < 1
   u(t,0) = 0,  u(t,1) = 0,              0 <= t <= 1

   b(x) = 1 + x(1-x)

   Exact solution:
   u(t,x) = exp(-t) * (C1 + C2*x - exp((x-1)/eps))
   C1 = exp(-1/eps),  C2 = 1 - exp(-1/eps)

   Reference: Gowrisankar & Natesan (2014)

Workflow:
  Step 1 → Stabilized FEM (SUPG) solves from t=0 to t=1
           Snapshots saved every snapshot_interval steps.
  Step 2 → PINN corrects the FEM solution at the *terminal time* only.

KEY DESIGN (matching solver_three_hybrid_lwc.py architecture):
  1) Source term f defined in UFL for FEM, numpy/torch for PINN
  2) Time dependence via Constant() objects updated each step
  3) Solution readout via point evaluation U(Point(x))
  4) Error computation via UFL assemble((U - u_exact)^2 * dx)
  5) PINN uses autograd for ALL derivatives (du/dt, du/dx, d2u/dx2)
     No FEM data in PDE loss — 100% clean physics
"""

from __future__ import print_function
import numpy as np
if not hasattr(np, 'product'):
    np.product = np.prod

from fenics import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import time as timer_module

import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if str(device) == "cuda":
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version (PyTorch): {torch.version.cuda}")

parameters["allow_extrapolation"] = True
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["no-evaluate_basis_derivatives"] = False
set_log_level(LogLevel.INFO)


# =============================================================================
# PROBLEM PARAMETERS
# =============================================================================
epsilon = 1e-4
t_final = 1.0
N_t     = 400
dt_val  = t_final / N_t
snapshot_interval = 10

# Derived constants for the exact solution
C1_val = np.exp(-1.0 / epsilon)
C2_val = 1.0 - np.exp(-1.0 / epsilon)

print("=" * 70)
print("1D PARABOLIC — SUPG + YZβ (Gowrisankar & Natesan, 2014)")
print("=" * 70)
print("  epsilon = %e" % epsilon)
print("  b(x) = 1 + x(1-x)")
print("  t_f = %g, dt = %g, N_t = %d" % (t_final, dt_val, N_t))
print("  C1 = %.6e, C2 = %.6e" % (C1_val, C2_val))


# =============================================================================
# MESH & FUNCTION SPACE
# =============================================================================
nx_mesh = 200
mesh = UnitIntervalMesh(nx_mesh)
V = FunctionSpace(mesh, 'CG', 1)
coords = V.tabulate_dof_coordinates().flatten()  # (n_dof,) — 1D
n_dof = V.dim()

print("  Number of Cells:", mesh.num_cells())
print("  Number of DOFs: ", n_dof)
print("  h_max = %.6f, h_min = %.6f" % (mesh.hmax(), mesh.hmin()))

# Peclet number estimate (b_max ≈ 1.25 at x=0.5)
b_max_est = 1.25
Pe_h = b_max_est * mesh.hmax() / (2.0 * epsilon)
print("  Pe_h ≈ %.0f" % Pe_h)


# =============================================================================
# ANALYTICAL SOLUTION (numpy, for post-processing)
# =============================================================================
def analytical_solution_np(t, x):
    """Exact solution at time t, spatial points x (numpy array)."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-t) * (C1_val + C2_val * x - np.exp((x - 1.0) / epsilon))


# =============================================================================
# SOURCE TERM f(t,x) — numpy (for verification and PINN)
# =============================================================================
class ComputeRHS:
    """Compute f(t,x) for the 1D parabolic problem — numpy version."""
    def __init__(self, eps):
        self.eps = eps
        self.C1 = np.exp(-1.0 / eps)
        self.C2 = 1.0 - self.C1

    def compute_f(self, t, x):
        """
        f(t,x) = e^{-t} [ -C1 + C2(1 - x²) + e^{(x-1)/ε}(1 - x(1-x)/ε) ]
        """
        t = np.asarray(t, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        exp_term = np.exp((x - 1.0) / self.eps)
        return np.exp(-t) * (
            -self.C1 + self.C2 * (1.0 - x**2)
            + exp_term * (1.0 - x * (1.0 - x) / self.eps)
        )


rhs_computer = ComputeRHS(epsilon)


# =============================================================================
# SOURCE TERM & EXACT SOLUTION — PURE UFL (for FEM)
#
# u_exact = exp(-t) * (C1 + C2*x - exp((x-1)/eps))
#
# f = u_t - eps*u_xx + b(x)*u_x
#   = e^{-t} [ -C1 + C2(1-x²) + e^{(x-1)/ε}(1 - x(1-x)/ε) ]
# =============================================================================

# Spatial coordinate
xx = SpatialCoordinate(mesh)
x_ufl = xx[0]

# UFL Constants
eps_c = Constant(epsilon)
dt_c  = Constant(dt_val)
C1_c  = Constant(C1_val)
C2_c  = Constant(C2_val)

# Velocity field b(x) = 1 + x(1-x)
b_ufl = 1.0 + x_ufl * (1.0 - x_ufl)

# Time-dependent multiplier (updated each step)
exp_neg_t = Constant(1.0)  # exp(-t), updated in loop

# Source term f(t,x) in UFL
exp_bl = exp((x_ufl - 1.0) / eps_c)   # boundary layer exponential
f_ufl = exp_neg_t * (
    -C1_c + C2_c * (1.0 - x_ufl**2)
    + exp_bl * (1.0 - x_ufl * (1.0 - x_ufl) / eps_c)
)

# Exact solution in UFL (for error computation)
u_exact_ufl = exp_neg_t * (C1_c + C2_c * x_ufl - exp_bl)

# Verification
print("")
print("  u_exact(0.0, 0.5) = %.6e" % analytical_solution_np(0.0, 0.5))
print("  u_exact(0.5, 0.5) = %.6e" % analytical_solution_np(0.5, 0.5))
print("  u_exact(1.0, 0.5) = %.6e" % analytical_solution_np(1.0, 0.5))


# =============================================================================
# FEM SETUP — Two solvers: (A) SUPG-only, (B) SUPG + YZβ
# =============================================================================

# Functions (shared — reset between runs)
U   = Function(V, name="u")
U_n = Function(V, name="u_n")
phi1 = TestFunction(V)

# Boundary conditions: u(t,0) = 0, u(t,1) = 0
bc_left  = DirichletBC(V, Constant(0.0), 'near(x[0], 0.0)')
bc_right = DirichletBC(V, Constant(0.0), 'near(x[0], 1.0)')
bcs = [bc_left, bc_right]


# -----------------------------------------------------------------
# SUPG Stabilization (Tezduyar/Shakib style) — shared by both
# -----------------------------------------------------------------
h = CellDiameter(mesh)
b_mag_ufl = abs(b_ufl)

tau_t = pow((2.0 / dt_c)**2
            + (2.0 * b_mag_ufl / h)**2
            + 9.0 * (4.0 * eps_c / (h * h))**2,
            Constant(-0.5))
tau_u = sqrt(tau_t * tau_t)

tau_est = 1.0 / np.sqrt((2.0 / dt_val)**2 + (2.0 * b_max_est / mesh.hmax())**2
                         + 9.0 * (4.0 * epsilon / mesh.hmax()**2)**2)
print("  tau_SUPG (est) = %.6e" % tau_est)


# -----------------------------------------------------------------
# Strong-form residual (CG1: u_xx = 0 element-wise)
# -----------------------------------------------------------------
residual_u = (U - U_n) / dt_c + b_ufl * U.dx(0) - f_ufl


# -----------------------------------------------------------------
# YZbeta Shock Capturing
# -----------------------------------------------------------------
Y = 0.1
shock_raw = (1.0 / Y) * abs(residual_u) * (h / 2.0)**2
shock_visc = shock_raw


# -----------------------------------------------------------------
# Galerkin + SUPG (shared base)
# -----------------------------------------------------------------
F_galerkin = (((U - U_n) / dt_c) * phi1 * dx
              + eps_c * inner(grad(U), grad(phi1)) * dx
              + b_ufl * U.dx(0) * phi1 * dx
              - f_ufl * phi1 * dx)

SUPG = tau_u * b_ufl * phi1.dx(0) * residual_u * dx

SHOCK = shock_visc * inner(grad(U), grad(phi1)) * dx


# -----------------------------------------------------------------
# Form A: SUPG only
# -----------------------------------------------------------------
F_A = F_galerkin + SUPG
J_A = derivative(F_A, U)
problem_A = NonlinearVariationalProblem(F_A, U, bcs, J_A)
solver_A  = NonlinearVariationalSolver(problem_A)

prm_A = solver_A.parameters["newton_solver"]
prm_A["absolute_tolerance"]  = 1E-9
prm_A['relative_tolerance']  = 1E-9
prm_A['maximum_iterations']  = 50
prm_A['convergence_criterion'] = 'residual'
prm_A['krylov_solver']['absolute_tolerance']   = 1E-10
prm_A['krylov_solver']['relative_tolerance']   = 1E-10
prm_A['krylov_solver']['maximum_iterations']   = 1000
prm_A['krylov_solver']['nonzero_initial_guess'] = True
prm_A['krylov_solver']['error_on_nonconvergence'] = True
prm_A['krylov_solver']['report'] = True


# -----------------------------------------------------------------
# Form B: SUPG + YZβ
# -----------------------------------------------------------------
F_B = F_galerkin + SUPG + SHOCK
J_B = derivative(F_B, U)
problem_B = NonlinearVariationalProblem(F_B, U, bcs, J_B)
solver_B  = NonlinearVariationalSolver(problem_B)

prm_B = solver_B.parameters["newton_solver"]
prm_B["absolute_tolerance"]  = 1E-9
prm_B['relative_tolerance']  = 1E-9
prm_B['maximum_iterations']  = 50
prm_B['convergence_criterion'] = 'residual'
prm_B['krylov_solver']['absolute_tolerance']   = 1E-10
prm_B['krylov_solver']['relative_tolerance']   = 1E-10
prm_B['krylov_solver']['maximum_iterations']   = 1000
prm_B['krylov_solver']['nonzero_initial_guess'] = True
prm_B['krylov_solver']['error_on_nonconvergence'] = True
prm_B['krylov_solver']['report'] = True


# =============================================================================
# INITIAL CONDITION
# =============================================================================
u_init_expr = Expression('C1 + C2*x[0] - exp((x[0]-1.0)/eps)',
                         degree=4, C1=C1_val, C2=C2_val, eps=epsilon)

u_ic_center = analytical_solution_np(0.0, 0.5)
print("  IC at x=0.5: Exact=%.6e" % u_ic_center)


# =============================================================================
# GENERIC TIME-STEPPING FUNCTION
# =============================================================================
def run_fem_timestepping(solver_obj, label):
    """Run time stepping with given solver, return snapshots."""
    print(f"\n  --- Running {label} ---")

    # Reset IC
    U_n.assign(interpolate(u_init_expr, V))
    U.assign(U_n)

    snap_times = []
    snap_solutions = []
    u_prev_snap = None
    t_loc = 0.0
    step_loc = 0

    t_wall = timer_module.time()

    while t_loc < t_final - 1e-14:
        t_loc += dt_val
        step_loc += 1

        # Update time-dependent UFL constant
        exp_neg_t.assign(np.exp(-t_loc))

        # Solve
        solver_obj.solve()

        # Capture u_prev at last step
        if step_loc == N_t:
            u_prev_snap = np.array([U_n(Point(coords[i]))
                                     for i in range(n_dof)])

        # Update previous solution
        U_n.assign(U)

        # Save snapshot
        if step_loc % snapshot_interval == 0 or step_loc == N_t:
            snap_times.append(t_loc)
            u_snap = np.array([U(Point(coords[i])) for i in range(n_dof)])
            snap_solutions.append(u_snap)

        # Diagnostics
        if step_loc % 50 == 0 or step_loc == N_t:
            u_fem_mid = U(Point(0.5))
            u_exact_mid = analytical_solution_np(t_loc, 0.5)
            l2_err = sqrt(assemble((U - u_exact_ufl)**2 * dx))
            print("  t = %.4f | %s(0.5)=%.4e | Exact=%.4e | L2=%.4e"
                  % (t_loc, label[:8], u_fem_mid, u_exact_mid, l2_err))

    elapsed_loc = timer_module.time() - t_wall
    print(f"  {label} completed in {elapsed_loc:.1f} s, {len(snap_times)} snapshots")
    return snap_times, snap_solutions, u_prev_snap, elapsed_loc


# =============================================================================
# RUN BOTH FEM SOLVERS
# =============================================================================
print("\n" + "=" * 70)
print("TIME STEPPING — TWO FEM SOLVERS")
print("=" * 70)

snap_times_A, snap_sols_A, u_prev_A, elapsed_A = run_fem_timestepping(solver_A, "SUPG-only")
snap_times_B, snap_sols_B, u_prev_B, elapsed_B = run_fem_timestepping(solver_B, "SUPG+YZβ")

# Use SUPG+YZβ as the primary for PINN training
snapshot_times     = snap_times_B
snapshot_solutions = snap_sols_B
u_prev_snapshot    = u_prev_B
elapsed            = elapsed_B


# =============================================================================
# ERROR SUMMARY — both solvers
# =============================================================================
print("\n" + "=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

h_est = 1.0 / nx_mesh
sort_idx = np.argsort(coords)

def compute_l2_errors(snap_times_list, snap_sols_list):
    errors = []
    for t_s, u_s in zip(snap_times_list, snap_sols_list):
        u_exact = analytical_solution_np(t_s, coords)
        diff = u_exact - u_s
        errors.append(np.sqrt(np.sum(diff**2) * h_est))
    return errors

l2_errors_A = compute_l2_errors(snap_times_A, snap_sols_A)
l2_errors_B = compute_l2_errors(snap_times_B, snap_sols_B)
l2_errors_all = l2_errors_B  # alias for PINN section

print("\n  SUPG-only:")
for i in range(0, len(snap_times_A), max(1, len(snap_times_A) // 6)):
    print("    t = %.4f | L2 ~ %.6e" % (snap_times_A[i], l2_errors_A[i]))
print("    t = %.4f | L2 ~ %.6e" % (snap_times_A[-1], l2_errors_A[-1]))

print("\n  SUPG+YZβ:")
for i in range(0, len(snap_times_B), max(1, len(snap_times_B) // 6)):
    print("    t = %.4f | L2 ~ %.6e" % (snap_times_B[i], l2_errors_B[i]))
print("    t = %.4f | L2 ~ %.6e" % (snap_times_B[-1], l2_errors_B[-1]))


# =============================================================================
# VISUALIZATION — always compare: Analytical vs SUPG vs SUPG+YZβ
# =============================================================================
print("\n" + "=" * 70)
print("VISUALIZATION")
print("=" * 70)

x_plot = np.linspace(0, 1, 500)
t_end = snapshot_times[-1]

n_snaps_A = len(snap_times_A)
n_snaps_B = len(snap_times_B)


def interp_snap(u_snap_vals, x_eval):
    """Interpolate 1D DOF values onto evaluation points."""
    return np.interp(x_eval, coords[sort_idx], u_snap_vals[sort_idx])


# ----- PLOT: FEM Snapshot Lines (both solvers + analytical) -----
print("  Plotting FEM snapshots (both solvers) ...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (a) Analytical
colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_snaps_A))
for idx in range(0, n_snaps_A, max(1, n_snaps_A // 8)):
    t_s = snap_times_A[idx]
    u_anal = analytical_solution_np(t_s, x_plot)
    axes[0].plot(x_plot, u_anal, color=colors[idx], linewidth=1.5,
                 label=f't={t_s:.2f}')
axes[0].set_xlabel('$x$', fontsize=13)
axes[0].set_ylabel('$u$', fontsize=13)
axes[0].set_title('Analytical', fontsize=13)
axes[0].legend(fontsize=7, ncol=2)
axes[0].grid(True, alpha=0.3)

# (b) SUPG-only
for idx in range(0, n_snaps_A, max(1, n_snaps_A // 8)):
    t_s = snap_times_A[idx]
    u_interp = interp_snap(snap_sols_A[idx], x_plot)
    axes[1].plot(x_plot, u_interp, color=colors[idx], linewidth=1.5,
                 label=f't={t_s:.2f}')
axes[1].set_xlabel('$x$', fontsize=13)
axes[1].set_title('SUPG', fontsize=13)
axes[1].legend(fontsize=7, ncol=2)
axes[1].grid(True, alpha=0.3)

# (c) SUPG+YZβ
for idx in range(0, n_snaps_B, max(1, n_snaps_B // 8)):
    t_s = snap_times_B[idx]
    u_interp = interp_snap(snap_sols_B[idx], x_plot)
    axes[2].plot(x_plot, u_interp, color=colors[idx], linewidth=1.5,
                 label=f't={t_s:.2f}')
axes[2].set_xlabel('$x$', fontsize=13)
axes[2].set_title('SUPG-YZ$β$', fontsize=13)
axes[2].legend(fontsize=7, ncol=2)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fem_snapshots_comparison.png', dpi=400, bbox_inches='tight')
plt.close()
print("  Saved: fem_snapshots_comparison.png")


# ----- PLOT: Terminal cross-section (Analytical vs SUPG vs SUPG+YZβ) -----
print("  Plotting terminal cross-section ...")
u_anal_end = analytical_solution_np(t_end, x_plot)
u_supg_end = interp_snap(snap_sols_A[-1], x_plot)
u_yzb_end  = interp_snap(snap_sols_B[-1], x_plot)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_plot, u_anal_end, 'k-', linewidth=2.5, label='Analytical')
ax.plot(x_plot, u_supg_end, 'b--', linewidth=2, label='SUPG')
ax.plot(x_plot, u_yzb_end, 'g-.', linewidth=2, label='SUPG-YZ$β$')
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$u(1, x)$', fontsize=14)
#ax.set_title('FEM Solutions at Terminal Time', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cross_section_terminal.png', dpi=400, bbox_inches='tight')
plt.close()
print("  Saved: cross_section_terminal.png")


# ----- PLOT: Boundary layer zoom at x≈1 -----
print("  Plotting boundary layer zoom ...")
bl_mask = x_plot > 0.9
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_plot[bl_mask], u_anal_end[bl_mask], 'k-', linewidth=2.5, label='Analytical')
ax.plot(x_plot[bl_mask], u_supg_end[bl_mask], 'b--', linewidth=2, label='SUPG')
ax.plot(x_plot[bl_mask], u_yzb_end[bl_mask], 'g-.', linewidth=2, label='SUPG-YZ$β$')
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$u$', fontsize=14)
#ax.set_title('Boundary Layer Zoom ($x > 0.95$) at $t = %.2f$' % t_end, fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('boundary_layer_zoom_fem.png', dpi=400, bbox_inches='tight')
plt.close()
print("  Saved: boundary_layer_zoom_fem.png")


# ----- PLOT: L2 Error Over Time (both solvers) -----
print("  Plotting L2 error evolution ...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(snap_times_A, l2_errors_A, 'b-o', linewidth=1.5, markersize=3,
            label='SUPG', alpha=0.8)
ax.semilogy(snap_times_B, l2_errors_B, 'g-s', linewidth=1.5, markersize=3,
            label='SUPG-YZ$β$', alpha=0.8)
ax.set_xlabel('Time t', fontsize=13)
ax.set_ylabel('$L^2$ Error', fontsize=13)
#ax.set_title('$L^2$ Error Evolution: SUPG vs SUPG+YZβ', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('l2_error_evolution.png', dpi=400, bbox_inches='tight')
plt.close()
print("  Saved: l2_error_evolution.png")


# ----- PLOT: Pointwise error at terminal time -----
print("  Plotting pointwise error at terminal ...")
err_supg = np.abs(u_supg_end - u_anal_end)
err_yzb  = np.abs(u_yzb_end - u_anal_end)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(x_plot, err_supg + 1e-16, 'b--', linewidth=1.5, label='|SUPG − Analytical|')
ax.semilogy(x_plot, err_yzb + 1e-16, 'g-.', linewidth=1.5, label='|SUPG-YZ$β$ − Analytical|')
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('Pointwise Error', fontsize=14)
#ax.set_title('Pointwise Error at $t = %.2f$' % t_end, fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pointwise_error_fem.png', dpi=400, bbox_inches='tight')
plt.close()
print("  Saved: pointwise_error_fem.png")


# ===== FINAL REPORT =====
print("\n" + "=" * 70)
print("FINAL REPORT — FEM")
print("=" * 70)
print("  Terminal time:   t = %.4f" % t_end)
print("  SUPG     — L2: %.6e | Time: %.1f s" % (l2_errors_A[-1], elapsed_A))
print("  SUPG+YZβ — L2: %.6e | Time: %.1f s" % (l2_errors_B[-1], elapsed_B))
print("  Snapshots: %d per solver" % len(snapshot_times))
print("=" * 70)

# Export (SUPG+YZβ is primary)
np.savez('fem_results.npz',
         snapshot_times=np.array(snapshot_times),
         snapshot_solutions=np.array(snapshot_solutions),
         snapshot_solutions_supg=np.array(snap_sols_A),
         coordinates=coords,
         l2_errors=np.array(l2_errors_all),
         l2_errors_supg=np.array(l2_errors_A),
         terminal_solution=snapshot_solutions[-1],
         u_prev_solution=u_prev_snapshot,
         terminal_time=t_end,
         dt=dt_val)
print("  Saved: fem_results.npz")
print("=" * 70)


# #############################################################################
#                                                                             #
#  STEP 2:  PINN CORRECTION AT TERMINAL TIME                                 #
#           Network input: (t, x) — 2D                                       #
#           ∂u/∂t via autograd (no FEM u_prev dependency!)                    #
#                                                                             #
# #############################################################################

print("\n" + "=" * 70)
print("STEP 2: PINN CORRECTION AT TERMINAL TIME")
print("  ∂u/∂t via autograd — no FEM u_prev dependency")
print("=" * 70)


# =============================================================================
# GPU-native source term f(t,x) — avoids CPU↔GPU transfer
# =============================================================================
class ComputeRHS_Torch:
    """Compute f(t,x) for the 1D parabolic problem — PyTorch GPU version."""
    def __init__(self, eps):
        self.eps = eps
        self.C1 = np.exp(-1.0 / eps)
        self.C2 = 1.0 - self.C1

    def compute_f(self, t, x):
        """
        f(t,x) = e^{-t} [ -C1 + C2(1 - x²) + e^{(x-1)/ε}(1 - x(1-x)/ε) ]
        All inputs/outputs are torch tensors on same device.
        """
        exp_term = torch.exp((x - 1.0) / self.eps)
        return torch.exp(-t) * (
            -self.C1 + self.C2 * (1.0 - x**2)
            + exp_term * (1.0 - x * (1.0 - x) / self.eps)
        )


rhs_computer_gpu = ComputeRHS_Torch(epsilon)


# =============================================================================
# PINN ARCHITECTURE — input: (t, x), output: u_θ(t, x)
#                     ∂u/∂t via autograd 

# =============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x
        out = self.act(self.lin1(x))
        out = self.lin2(out)
        out = self.norm(out + identity)
        return self.act(out)


class HybridSUPGPINN1D(nn.Module):

    def __init__(self, hidden_dim=128, num_blocks=8,
                 num_fourier_features=32, sigma=4.0):
        super().__init__()
        self.num_fourier_features = num_fourier_features
        # B: (2, nf) for input (t, x)
        self.register_buffer('B', torch.randn((2, num_fourier_features)) * sigma)
        input_dim = 2 + 2 * num_fourier_features  # raw (t,x) + sin/cos
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)  # from working solver
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def fourier_features(self, x):
        projection = torch.matmul(x, self.B)  # (N, 2) @ (2, nf) = (N, nf)
        return torch.cat([x, torch.sin(projection), torch.cos(projection)], dim=-1)

    def forward(self, x):
        """
        x: (N, 2) = (t, x_coord)
        Returns: u_θ(t, x) with Dirichlet BCs enforced via distance function.
        """
        features = self.fourier_features(x)
        h = self.input_layer(features)
        for block in self.blocks:
            h = block(h)
        raw = self.output_layer(h)
        # Distance function d(x) = x*(1-x) — vanishes at x=0 and x=1
        x_coord = x[:, 1:2]   # spatial coordinate (column 1)
        d = x_coord * (1.0 - x_coord)
        return raw * d

    def compute_pde_residual(self, x_points, eps, rhs_fn_gpu):
        """
        Compute strong-form PDE residual via autograd.

        R = ∂u_θ/∂t − ε*u_xx + b(x)*u_x − f(t,x)

        b(x) = 1 + x(1-x)  — spatially varying!

        x_points: (N, 2) = (t, x) — requires_grad=True
        ALL derivatives via autograd. No FEM u_prev involved.
        """
        if not x_points.requires_grad:
            x_points = x_points.clone().detach().requires_grad_(True)

        u = self.forward(x_points).squeeze()

        # First derivatives: (∂u/∂t, ∂u/∂x)
        grads = torch.autograd.grad(
            u, x_points, torch.ones_like(u),
            create_graph=True, retain_graph=True)[0]

        u_t = grads[:, 0]   # ∂u/∂t  ← autograd, CLEAN!
        u_x = grads[:, 1]   # ∂u/∂x

        # Second spatial derivative: ∂²u/∂x²
        u_xx = torch.autograd.grad(
            u_x, x_points, torch.ones_like(u_x),
            create_graph=True, retain_graph=True)[0][:, 1]

        # Spatially varying velocity: b(x) = 1 + x(1-x)
        x_coord = x_points[:, 1]
        b_x = 1.0 + x_coord * (1.0 - x_coord)

        # Source term f(t, x) — GPU-native
        f_t = rhs_fn_gpu.compute_f(x_points[:, 0], x_points[:, 1])

        # PDE residual — pure autograd, no finite differences
        residual = (u_t
                    - eps * u_xx
                    + b_x * u_x
                    - f_t)
        return residual


# =============================================================================
# HYPERPARAMETERS — from working stationary solver (solver_one.py)
# =============================================================================
HIDDEN_DIM = 128
NUM_BLOCKS = 8
NUM_FOURIER = 24
FOURIER_SCALE = 4.0
NUM_EPOCHS = 5000
BATCH_SIZE = 256    # ← KEY: from working solver! tiny batch = stable convergence
LR_INIT = 1e-4             # ← KEY: from working solver! conservative
N_SNAPSHOTS_PINN = 10

print(f"\nPINN Hyperparameters (adapted from working solver_one.py):")
print(f"  Hidden dim: {HIDDEN_DIM}, Blocks: {NUM_BLOCKS}")
print(f"  Fourier features: {NUM_FOURIER}, scale: {FOURIER_SCALE}")
print(f"  Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}")
print(f"  Initial LR: {LR_INIT}")
print(f"  Training snapshots: last {N_SNAPSHOTS_PINN}")


# =============================================================================
# DATA PREPARATION — Multi-snapshot (t, x) → u_FEM
# =============================================================================
print("\nPreparing multi-snapshot training data...")

t_terminal = t_end
n_total_snaps = len(snapshot_times)
snap_start = max(0, n_total_snaps - N_SNAPSHOTS_PINN)

# Collect (t, x, u_FEM) from last N snapshots
t_data_list = []
x_data_list = []
u_data_list = []

for k in range(snap_start, n_total_snaps):
    t_k = snapshot_times[k]
    u_k = snapshot_solutions[k]
    n_pts = len(coords)
    t_data_list.append(np.full(n_pts, t_k))
    x_data_list.append(coords.copy())
    u_data_list.append(u_k)

t_train = np.concatenate(t_data_list)
x_train = np.concatenate(x_data_list)
u_train = np.concatenate(u_data_list)

# Build (t, x) tensor
txc = np.column_stack([t_train, x_train])
coords_tensor = torch.tensor(txc, dtype=torch.float32)
u_fem_tensor = torch.tensor(u_train, dtype=torch.float32)

# Time window for PDE collocation
t_pinn_start = snapshot_times[snap_start]
t_pinn_end = snapshot_times[-1]

print(f"  Time window: [{t_pinn_start:.4f}, {t_pinn_end:.4f}]")
print(f"  Snapshots used: {n_total_snaps - snap_start}")
print(f"  Total training points: {len(u_train)}")
print(f"  FEM data range: [{u_train.min():.4e}, {u_train.max():.4e}]")

# Analytical at terminal for reference
u_analytical_terminal = analytical_solution_np(t_terminal, coords)

# Identify boundary DOFs
tol_bdy = 1e-8
is_boundary = (np.abs(coords) < tol_bdy) | (np.abs(coords - 1.0) < tol_bdy)
print(f"  Interior DOFs: {np.sum(~is_boundary)}, Boundary DOFs: {np.sum(is_boundary)}")

# BC points — NOT used (distance function handles BCs)
# Kept minimal for logging only
n_bc_times = 5
bc_t_vals = np.linspace(t_pinn_start, t_pinn_end, n_bc_times)
bc_points_list = []
for t_bc in bc_t_vals:
    bc_points_list.append(torch.tensor([[t_bc, 0.0]], dtype=torch.float32, device=device))
    bc_points_list.append(torch.tensor([[t_bc, 1.0]], dtype=torch.float32, device=device))
bc_points = torch.cat(bc_points_list, dim=0)
bc_points.requires_grad_(True)
print(f"  BC collocation points: {len(bc_points)}")


# =============================================================================
# TRAINING — PERFORMANCE-OPTIMIZED
# =============================================================================
def train_hybrid_pinn():
    print("\n" + "=" * 70)
    print("HYBRID PINN TRAINING — autograd ∂u/∂t (OPTIMIZED)")
    print(f"  Time window: [{t_pinn_start:.4f}, {t_pinn_end:.4f}]")
    print(f"  Target: terminal time t = {t_terminal:.4f}")
    print("  Performance optimizations:")
    print("    • PDE loss every PDE_EVERY batches (not every batch)")
    print("    • Full-batch GPU preload (no DataLoader overhead)")
    print("    • Vectorized NaN check (single pass)")
    print("    • Model checkpoint every SAVE_EVERY epochs")
    print("    • AMP mixed precision for forward pass")
    print("=" * 70)

    model = HybridSUPGPINN1D(
        hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS,
        num_fourier_features=NUM_FOURIER, sigma=FOURIER_SCALE).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")


    all_x = coords_tensor.to(device)       # (N, 2)
    all_y = u_fem_tensor.unsqueeze(1).to(device)  # (N, 1)
    N_data = all_x.shape[0]

    # ── PERF tuning knobs ──
    PDE_EVERY = 3     # compute PDE loss every N batches (autograd is ~5x costlier)
    SAVE_EVERY = 50   # checkpoint to disk every N epochs (not every improvement)
    USE_AMP = False
    if device.type == 'cuda':
        try:
            # Test AMP availability
            with torch.amp.autocast('cuda', enabled=True):
                pass
            USE_AMP = True
        except (RuntimeError, AttributeError):
            try:
                with torch.cuda.amp.autocast(enabled=True):
                    pass
                USE_AMP = True
            except Exception:
                USE_AMP = False

    # AMP scaler
    if USE_AMP:
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=True)
        except (TypeError, AttributeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=False)
        except (TypeError, AttributeError):
            scaler = torch.cuda.amp.GradScaler(enabled=False)

    # AMP context manager
    def amp_autocast():
        if USE_AMP:
            try:
                return torch.amp.autocast('cuda', enabled=True)
            except (TypeError, AttributeError):
                return torch.cuda.amp.autocast(enabled=True)
        else:
            import contextlib
            return contextlib.nullcontext()

    def hybrid_loss(model, data_x, data_y, pde_pts, epoch, compute_pde=True):
        """Hybrid loss with optional PDE computation."""

        # 1. DATA LOSS — under AMP for speed
        with amp_autocast():
            data_pred = model(data_x)
            data_loss = F_torch.mse_loss(data_pred, data_y)
        if torch.isnan(data_loss) or torch.isinf(data_loss):
            data_loss = torch.tensor(1.0, device=data_x.device)

        # 2. PDE LOSS — only when requested (PERF 1)
        # autograd needs float32, so no AMP here
        pde_loss = torch.tensor(0.0, device=data_x.device)
        if compute_pde and pde_pts is not None and len(pde_pts) > 0:
            try:
                x_c = pde_pts[:, 1]
                dist = torch.min(torch.stack([x_c, 1.0 - x_c]), dim=0)[0]
                interior_mask = dist > 0.00000001
                interior = pde_pts[interior_mask]

                if len(interior) > 5:
                    res = model.compute_pde_residual(
                        interior, epsilon, rhs_computer_gpu)
                    res = torch.clamp(res, -5.0, 5.0)
                    if not (torch.isnan(res).any() or torch.isinf(res).any()):
                        pde_loss = torch.mean(res ** 2)
                        if torch.isnan(pde_loss) or torch.isinf(pde_loss):
                            pde_loss = torch.tensor(0.0, device=data_x.device)
            except Exception as e:
                if epoch % 500 == 0:
                    print(f"  PDE loss failed at epoch {epoch}: {e}")

        bc_loss = torch.tensor(0.0, device=data_x.device)

        if epoch < 1500:
            w_data, w_pde, w_bc = 1.0, 0.5, 0.1
        elif epoch < 3000:
            w_data, w_pde, w_bc = 0.8, 0.95, 0.1
        else:
            w_data, w_pde, w_bc = 0.35, 5.0, 0.1

        total = w_data * data_loss + w_pde * pde_loss + w_bc * bc_loss
        if torch.isnan(total) or torch.isinf(total):
            total = data_loss
        total = torch.clamp(total, 0.0, 10.0)
        return total, {
            'data': data_loss.item(), 'pde': pde_loss.item(),
            'bc': bc_loss.item(), 'total': total.item(),
            'weights': {'data': w_data, 'pde': w_pde, 'bc': w_bc}
        }

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT,
                                   weight_decay=1e-7, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9,
                                  patience=100, min_lr=1e-7)

    history = {'epoch': [], 'total_loss': [], 'data_loss': [], 'pde_loss': [],
               'bc_loss': [], 'lr': [], 'data_weight': [], 'pde_weight': [], 'bc_weight': []}
    best_loss = float('inf')
    best_epoch = 0
    best_state = None  
    start_time_pinn = timer_module.time()
    nan_count = 0
    patience_counter = 0
    max_patience = 5000

    
    n_batches = max(1, (N_data + BATCH_SIZE - 1) // BATCH_SIZE)
    print(f"Training for {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}")
    print(f"  Batches/epoch: {n_batches}, Data points: {N_data}")
    print(f"  PDE every {PDE_EVERY} batches, Save every {SAVE_EVERY} epochs")
    print(f"  AMP: {USE_AMP}")

    global_batch = 0  # tracks total batch count for PDE scheduling

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_losses = {'total': 0.0, 'data': 0.0, 'pde': 0.0, 'bc': 0.0}
        batch_count = 0

       
        perm = torch.randperm(N_data, device=device)

        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE : min((b + 1) * BATCH_SIZE, N_data)]
            batch_x = all_x[idx]
            batch_y = all_y[idx]

            # ── PERF 1: PDE only every PDE_EVERY batches ──
            do_pde = (global_batch % PDE_EVERY == 0)
            pde_pts = None
            if do_pde:
                n_pde = 512#256---------------------------------------------------------------------------------------------------------------------
                pde_pts = torch.rand(n_pde, 2, device=device)
                pde_pts[:, 0] = pde_pts[:, 0] * (t_pinn_end - t_pinn_start) + t_pinn_start
                pde_pts.requires_grad_(True)

            optimizer.zero_grad(set_to_none=True)

            total_loss, lc = hybrid_loss(
                model, batch_x, batch_y, pde_pts, epoch, compute_pde=do_pde)

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                nan_count += 1
                global_batch += 1
                if nan_count > 3:
                    print("Too many NaN occurrences, stopping")
                    break
                continue

            # ── PERF 5: AMP-aware backward ──
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)

            # ── PERF 3: Vectorized NaN check — single pass ──
            grad_flat = torch.cat([p.grad.view(-1) for p in model.parameters()
                                   if p.grad is not None])
            valid = torch.isfinite(grad_flat).all().item()

            if valid:
                scaler.step(optimizer)
                scaler.update()
                batch_count += 1
                for k in epoch_losses:
                    epoch_losses[k] += lc.get(k, lc.get('total', 0.0))
            else:
                scaler.update()  # must still call update
                nan_count += 1

            global_batch += 1

        if batch_count == 0:
            print(f"No valid batches at epoch {epoch}, stopping.")
            break

        avg = {k: v / batch_count for k, v in epoch_losses.items()}
        scheduler.step(avg['total'])

        # Best loss tracking
        if avg['total'] < best_loss:
            best_loss = avg['total']
            best_epoch = epoch
            patience_counter = 0
            # ── PERF 4: Keep best state in RAM ──
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

       
        if best_state is not None and (epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS - 1):
            torch.save(best_state, 'hybrid_pinn_terminal.pt')

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch} (patience {max_patience} exceeded)")
            break

        if epoch % 50 == 0 or epoch == NUM_EPOCHS - 1:
            w = lc['weights']
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:5d} | Total: {avg['total']:.4e} | "
                  f"Data: {avg['data']:.4e} | PDE: {avg['pde']:.4e} | "
                  f"BC: {avg['bc']:.4e} | LR: {lr:.1e} | "
                  f"Best: {best_loss:.4e} (ep {best_epoch})")
            history['epoch'].append(epoch)
            history['total_loss'].append(avg['total'])
            history['data_loss'].append(avg['data'])
            history['pde_loss'].append(avg['pde'])
            history['bc_loss'].append(avg['bc'])
            history['lr'].append(lr)
            history['data_weight'].append(w['data'])
            history['pde_weight'].append(w['pde'])
            history['bc_weight'].append(w['bc'])

    # Final save
    if best_state is not None:
        torch.save(best_state, 'hybrid_pinn_terminal.pt')

    elapsed_pinn = timer_module.time() - start_time_pinn
    print(f"\nHybrid PINN training completed in {elapsed_pinn:.2f} s")
    print(f"Best loss: {best_loss:.8f} at epoch {best_epoch}")
    print(f"NaN count: {nan_count}")

    torch.save({'state_dict': best_state, 'history': history,
                'model_config': {'hidden_dim': HIDDEN_DIM, 'num_blocks': NUM_BLOCKS,
                                 'num_fourier_features': NUM_FOURIER, 'sigma': FOURIER_SCALE}
                }, 'hybrid_pinn_terminal_full.pt')
    return model, history


# ===== RUN PINN TRAINING =====
model_hybrid, history_hybrid = train_hybrid_pinn()
del model_hybrid
torch.cuda.empty_cache()
gc.collect()


##############################################################################
#                                                                             #
#  STEP 3:  EVALUATION & COMPARISON                                           #
#                                                                             #
# #############################################################################

print("\n" + "=" * 70)
print("STEP 3: EVALUATION & VISUALIZATION")
print("=" * 70)

# Load best model
model_eval = HybridSUPGPINN1D(
    hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS,
    num_fourier_features=NUM_FOURIER, sigma=FOURIER_SCALE).to(device)
model_eval.load_state_dict(
    torch.load('hybrid_pinn_terminal.pt', map_location=device, weights_only=True))
model_eval.eval()


def evaluate_pinn(model, x_pts, t_val, dev):
    """Evaluate PINN at fixed time t_val on 1D spatial points."""
    n = len(x_pts)
    t_col = np.full(n, t_val)
    pts_2d = np.column_stack([t_col, x_pts])
    tensor_in = torch.tensor(pts_2d, dtype=torch.float32).to(dev)
    with torch.no_grad():
        u_pred = model(tensor_in).cpu().numpy().flatten()
    return u_pred


# Evaluation grid
x_ev = np.linspace(0, 1, 500)

# Compute solutions at terminal time
u_anal_ev = analytical_solution_np(t_terminal, x_ev)
u_supg_ev = interp_snap(snap_sols_A[-1], x_ev)    # SUPG-only
u_yzb_ev  = interp_snap(snap_sols_B[-1], x_ev)    # SUPG+YZβ
u_pinn_ev = evaluate_pinn(model_eval, x_ev, t_terminal, device)


# ===== PLOT 1: Solution comparison at terminal time =====
print(f"\nCreating solution comparison at t = {t_terminal:.2f} ...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_ev, u_anal_ev, 'k-', linewidth=2.5, label='Analytical')
ax.plot(x_ev, u_supg_ev, 'b--', linewidth=2, label='SUPG', alpha=0.8)
ax.plot(x_ev, u_yzb_ev, 'g-.', linewidth=2, label='SUPG-YZ$β$', alpha=0.8)
ax.plot(x_ev, u_pinn_ev, 'r:', linewidth=2, label='Hybrid PINN')
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel(f'$u(1,x)$', fontsize=14)
#ax.set_title(f'Solution Comparison at $t = {t_terminal:.2f}$', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('terminal_solution_comparison.png', dpi=400, bbox_inches='tight')
plt.close()
print("  ✅ Saved: terminal_solution_comparison.png")


# ===== PLOT 2: Error comparison =====
print("Creating error comparison ...")
err_supg = np.abs(u_supg_ev - u_anal_ev)
err_yzb  = np.abs(u_yzb_ev - u_anal_ev)
err_pinn = np.abs(u_pinn_ev - u_anal_ev)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(x_ev, err_supg + 1e-16, 'b--', linewidth=2, label='|SUPG − Analytical|')
ax.semilogy(x_ev, err_yzb + 1e-16, 'g-.', linewidth=2, label='|SUPG-YZ$β$ − Analytical|')
ax.semilogy(x_ev, err_pinn + 1e-16, 'r-', linewidth=2, label='|PINN − Analytical|')
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('Pointwise Error', fontsize=14)
#ax.set_title(f'Error Comparison at $t = {t_terminal:.2f}$', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('terminal_error_comparison.png', dpi=400, bbox_inches='tight')
plt.close()
print("  ✅ Saved: terminal_error_comparison.png")


# ===== PLOT 3: Zoom into boundary layer at x=1 =====
print("Creating boundary layer zoom ...")
bl_mask = x_ev > 0.9
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_ev[bl_mask], u_anal_ev[bl_mask], 'k-', linewidth=2.5, label='Analytical')
ax.plot(x_ev[bl_mask], u_supg_ev[bl_mask], 'b--', linewidth=2, label='SUPG')
ax.plot(x_ev[bl_mask], u_yzb_ev[bl_mask], 'g-.', linewidth=2, label='SUPG-YZ$β$')
ax.plot(x_ev[bl_mask], u_pinn_ev[bl_mask], 'r:', linewidth=2, label='Hybrid PINN')
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$u$', fontsize=14)
#ax.set_title('Boundary Layer Zoom ($x > 0.95$)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('boundary_layer_zoom.png', dpi=400, bbox_inches='tight')
plt.close()
print("  ✅ Saved: boundary_layer_zoom.png")


# ===== PLOT 4: L2 error over time + PINN point =====
print("Creating L2 error plot ...")
u_pinn_dof = evaluate_pinn(model_eval, coords, t_terminal, device)
u_analytical_terminal = analytical_solution_np(t_terminal, coords)
diff_pinn = u_analytical_terminal - u_pinn_dof
l2_pinn_terminal = np.sqrt(np.sum(diff_pinn**2) * h_est)

diff_supg_term = u_analytical_terminal - snap_sols_A[-1]
l2_supg_terminal = np.sqrt(np.sum(diff_supg_term**2) * h_est)

diff_yzb_term = u_analytical_terminal - snap_sols_B[-1]
l2_yzb_terminal = np.sqrt(np.sum(diff_yzb_term**2) * h_est)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(snap_times_A, l2_errors_A, 'b-', linewidth=1.5, label='SUPG', alpha=0.8)
ax.semilogy(snap_times_A[::5], l2_errors_A[::5], 'bo', markersize=3)
ax.semilogy(snap_times_B, l2_errors_B, 'g-', linewidth=1.5, label='SUPG-YZ$β$', alpha=0.8)
ax.semilogy(snap_times_B[::5], l2_errors_B[::5], 'gs', markersize=3)
ax.semilogy(t_terminal, l2_pinn_terminal, 'r*', markersize=15,
            label=f'Hybrid PINN ($t={t_terminal:.2f}$)', zorder=5)
ax.set_xlabel('Time ($t$)', fontsize=14)
ax.set_ylabel('$L^2$ Error', fontsize=14)
#ax.set_title('$L^2$ Error: SUPG vs SUPG+YZβ vs PINN', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('l2_error_over_time_with_pinn.png', dpi=400, bbox_inches='tight')
plt.close()
print("  ✅ Saved: l2_error_over_time_with_pinn.png")


# ===== PLOT 5: Training history =====
print("Creating training history ...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
epochs_hist = history_hybrid['epoch']
axes[0].semilogy(epochs_hist, history_hybrid['total_loss'], 'b-', linewidth=1.5, label='Total')
axes[0].semilogy(epochs_hist, history_hybrid['data_loss'], 'g--', linewidth=1.5, label='Data')
axes[0].semilogy(epochs_hist, history_hybrid['pde_loss'], 'r-.', linewidth=1.5, label='PDE')
axes[0].semilogy(epochs_hist, history_hybrid['bc_loss'], 'm:', linewidth=1.5, label='BC')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Losses'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(epochs_hist, history_hybrid['lr'], 'k-', linewidth=1.5)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('LR')
axes[1].set_title('Learning Rate'); axes[1].grid(True, alpha=0.3)
axes[2].plot(epochs_hist, history_hybrid['data_weight'], 'g-', linewidth=1.5, label='Data')
axes[2].plot(epochs_hist, history_hybrid['pde_weight'], 'r--', linewidth=1.5, label='PDE')
axes[2].plot(epochs_hist, history_hybrid['bc_weight'], 'm:', linewidth=1.5, label='BC')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Weight')
axes[2].set_title('Adaptive Weights'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_history_terminal.png', dpi=400, bbox_inches='tight')
plt.close()
print("  ✅ Saved: training_history_terminal.png")


# =============================================================================
# FINAL ERROR TABLE
# =============================================================================
print("\n" + "=" * 70)
print("FINAL ERROR TABLE")
print("=" * 70)
print(f"\nTerminal-time comparison (t = {t_terminal:.4f}):")
print(f"  L2(SUPG)     = {l2_supg_terminal:.6e}")
print(f"  L2(SUPG+YZβ) = {l2_yzb_terminal:.6e}")
print(f"  L2(PINN)     = {l2_pinn_terminal:.6e}")
if l2_pinn_terminal < l2_yzb_terminal:
    ratio = l2_yzb_terminal / max(l2_pinn_terminal, 1e-16)
    print(f"  ✓ PINN improved over SUPG+YZβ by {ratio:.2f}x")
else:
    ratio = l2_pinn_terminal / max(l2_yzb_terminal, 1e-16)
    print(f"  ✗ PINN worse than SUPG+YZβ by {ratio:.2f}x")

# Pointwise comparison at x = 0.5
u_anal_center = analytical_solution_np(t_terminal, 0.5)
u_supg_center = np.interp(0.5, coords[sort_idx], snap_sols_A[-1][sort_idx])
u_yzb_center  = np.interp(0.5, coords[sort_idx], snap_sols_B[-1][sort_idx])
u_pinn_center = evaluate_pinn(model_eval, np.array([0.5]), t_terminal, device)[0]
print(f"\nPointwise at x = 0.5:")
print(f"  Analytical: {u_anal_center:.6e}")
print(f"  SUPG:       {u_supg_center:.6e}  (err: {abs(u_anal_center - u_supg_center):.4e})")
print(f"  SUPG+YZβ:   {u_yzb_center:.6e}  (err: {abs(u_anal_center - u_yzb_center):.4e})")
print(f"  PINN:       {u_pinn_center:.6e}  (err: {abs(u_anal_center - u_pinn_center):.4e})")

# Pointwise comparison at x = 0.99 (boundary layer)
u_anal_bl = analytical_solution_np(t_terminal, 0.99)
u_supg_bl = np.interp(0.99, coords[sort_idx], snap_sols_A[-1][sort_idx])
u_yzb_bl  = np.interp(0.99, coords[sort_idx], snap_sols_B[-1][sort_idx])
u_pinn_bl = evaluate_pinn(model_eval, np.array([0.99]), t_terminal, device)[0]
print(f"\nPointwise at x = 0.99 (boundary layer):")
print(f"  Analytical: {u_anal_bl:.6e}")
print(f"  SUPG:       {u_supg_bl:.6e}  (err: {abs(u_anal_bl - u_supg_bl):.4e})")
print(f"  SUPG+YZβ:   {u_yzb_bl:.6e}  (err: {abs(u_anal_bl - u_yzb_bl):.4e})")
print(f"  PINN:       {u_pinn_bl:.6e}  (err: {abs(u_anal_bl - u_pinn_bl):.4e})")

print("\n" + "=" * 70)
print("✅ ALL COMPUTATIONS COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\n📊 Generated plots:")
print("  • fem_snapshots.png               (FEM solution over time)")
print("  • analytical_snapshots.png         (Analytical solution over time)")
print("  • l2_error_evolution.png           (L2 error curve)")
print("  • cross_section_terminal.png       (FEM vs Analytical at t_end)")
print("  • terminal_solution_comparison.png (FEM vs PINN contour)")
print("  • terminal_error_comparison.png    (Error comparison)")
print("  • boundary_layer_zoom.png          (BL zoom at x≈1)")
print("  • l2_error_over_time_with_pinn.png (L2 + PINN point)")
print("  • training_history_terminal.png    (Loss curves)")
print("=" * 70)


# ── Load saved history ──
data = torch.load('hybrid_pinn_terminal_full.pt', map_location='cpu', weights_only=False)
h = data['history']
epochs = h['epoch']

FIGSIZE = (10, 6)
DPI = 400

# ── Figure 1: Training Losses ──
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.semilogy(epochs, h['total_loss'], 'b-',  linewidth=1.5, label='Total')
ax.semilogy(epochs, h['data_loss'],  'g--', linewidth=1.5, label='Data')
ax.semilogy(epochs, h['pde_loss'],   'r-.', linewidth=1.5, label='PDE')
ax.semilogy(epochs, h['bc_loss'],    'm:',  linewidth=1.5, label='BC')
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Loss',  fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_losses.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: training_losses.png")

# ── Figure 2: Learning Rate ──
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(epochs, h['lr'], 'k-', linewidth=1.5)
ax.set_xlabel('Epoch',         fontsize=14)
ax.set_ylabel('Learning Rate', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_lr.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: training_lr.png")

# ── Figure 3: Adaptive Weights ──
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(epochs, h['data_weight'], 'g-',  linewidth=1.5, label='$w_{\\mathrm{data}}$')
ax.plot(epochs, h['pde_weight'],  'r--', linewidth=1.5, label='$w_{\\mathrm{pde}}$')
ax.plot(epochs, h['bc_weight'],   'm:',  linewidth=1.5, label='$w_{\\mathrm{bc}}$')
ax.set_xlabel('Epoch',  fontsize=14)
ax.set_ylabel('Weight', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_weights.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved: training_weights.png")

print("\n  All 3 figures generated (10x6, 400 dpi).")






























