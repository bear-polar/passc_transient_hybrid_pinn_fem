#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stabilized FEM (SUPG + YZbeta) + PINN Correction
for the Traveling Wave Problem (Example 4, Giere 2015)

   du/dt - eps*Lap(u) + b.grad(u) + c*u = f,   in (0,1)^2
   u = 0 on boundary,   u(0,.) = u_0(.)

eps = 1e-8, b = (cos(pi/3), sin(pi/3)), c = 1, t_f = 1

Exact solution:
   u(t,x1,x2) = 0.5 sin(pi x1) sin(pi x2) [tanh((x1+x2-t-0.5)/sqrt(eps)) + 1]

The solution possesses a moving internal layer of width O(sqrt(eps)).

Workflow:
  Step 1 → Stabilized FEM (SUPG + YZβ) solves from t=0 to t_f
           Snapshots saved every 5 time steps for visualization.
  Step 2 → PINN corrects the FEM solution at the *terminal time* only.

"""

from __future__ import print_function
import numpy as np
if not hasattr(np, 'product'):
    np.product = np.prod

from fenics import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, LinearNDInterpolator
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
parameters['form_compiler']['quadrature_degree'] = 8
set_log_level(LogLevel.INFO)


# =============================================================================
# PROBLEM PARAMETERS
# =============================================================================
epsilon = 1e-8
b1_val  = np.cos(np.pi / 3.0)   # = 0.5
b2_val  = np.sin(np.pi / 3.0)   # = sqrt(3)/2 ≈ 0.8660
c_val   = 1.0
t_final = 1.0
N_t     = 1000
dt_val  = t_final / N_t
snapshot_interval = 5
sqrt_eps_val = np.sqrt(epsilon)  # = 1e-4

print("=" * 70)
print("TRAVELING WAVE PROBLEM — SUPG + YZbeta (Example 4, Giere 2015)")
print("=" * 70)
print("  epsilon = %e" % epsilon)
print("  sqrt(epsilon) = %e" % sqrt_eps_val)
print("  b = (cos(pi/3), sin(pi/3)) = (%.6f, %.6f)" % (b1_val, b2_val))
print("  c = %g" % c_val)
print("  t_f = %g, dt = %g, N_t = %d" % (t_final, dt_val, N_t))


# =============================================================================
# MESH & FUNCTION SPACE
# =============================================================================
nx_mesh = 64
mesh = UnitSquareMesh(nx_mesh, nx_mesh, "right")
V = FunctionSpace(mesh, 'CG', 1)
coords = V.tabulate_dof_coordinates()
n_dof = V.dim()

print("  Number of Cells:", mesh.num_cells())
print("  Number of Nodes:", mesh.num_vertices())
print("  Number of DOFs: ", n_dof)
print("  h_max = %.4f, h_min = %.4f" % (mesh.hmax(), mesh.hmin()))

b_mag = np.sqrt(b1_val**2 + b2_val**2)
Pe_h = b_mag * mesh.hmax() / (2.0 * epsilon)
print("  |b| = %.6f" % b_mag)
print("  Pe_h = %.0f" % Pe_h)


# =============================================================================
# ANALYTICAL SOLUTION (numpy, for post-processing only)
# =============================================================================
def analytical_solution_np(t, x1, x2):
    """
    u(t,x1,x2) = 0.5 sin(pi x1) sin(pi x2) [tanh((x1+x2-t-0.5)/sqrt(eps)) + 1]
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    t  = np.asarray(t, dtype=np.float64)
    eta = (x1 + x2 - t - 0.5) / sqrt_eps_val
    return 0.5 * np.sin(np.pi * x1) * np.sin(np.pi * x2) * (np.tanh(eta) + 1.0)


# =============================================================================
# SOURCE TERM & EXACT SOLUTION
#
# u_exact = 0.5 sin(pi x1) sin(pi x2) [tanh(eta) + 1]
# eta = (x1 + x2 - t - 0.5) / sqrt(eps)
#
# f = du/dt - eps * Lap(u) + b.grad(u) + c * u
# =============================================================================

# Spatial coordinates (for A, A_x1, A_x2 which are smooth — safe in UFL)
xx = SpatialCoordinate(mesh)
x1_ufl = xx[0]
x2_ufl = xx[1]

# UFL Constants
eps_c      = Constant(epsilon)
sqrt_eps_c = Constant(sqrt_eps_val)
b1_c       = Constant(b1_val)
b2_c       = Constant(b2_val)
c_c        = Constant(c_val)
dt_c       = Constant(dt_val)
b_ufl      = as_vector([b1_c, b2_c])
pi_c       = Constant(np.pi)

# Time constant (updated each time step — used only by UFL parts)
t_c = Constant(0.0)

# ---- C++ Expressions ----
tanh_expr = Expression(
    'tanh((x[0] + x[1] - t - 0.5) / se)',
    degree=8, t=0.0, se=sqrt_eps_val)

sech2_expr = Expression(
    '1.0 - pow(tanh((x[0] + x[1] - t - 0.5) / se), 2)',
    degree=8, t=0.0, se=sqrt_eps_val)

u_exact_expr = Expression(
    '0.5 * sin(pi*x[0]) * sin(pi*x[1]) * (tanh((x[0]+x[1]-t-0.5)/se) + 1.0)',
    degree=8, pi=np.pi, t=0.0, se=sqrt_eps_val)

# Use u_exact_expr for L2 error computation
u_exact_ufl = u_exact_expr

# ---- Smooth spatial parts ----
# A(x1,x2) = 0.5 sin(pi x1) sin(pi x2)
A_ufl = Constant(0.5) * sin(pi_c * x1_ufl) * sin(pi_c * x2_ufl)

# Spatial derivatives of A
A_x1_ufl = Constant(0.5) * pi_c * cos(pi_c * x1_ufl) * sin(pi_c * x2_ufl)
A_x2_ufl = Constant(0.5) * sin(pi_c * x1_ufl) * pi_c * cos(pi_c * x2_ufl)

# ---- Build source term f ----
# W = tanh(eta) + 1
W_ufl = tanh_expr + Constant(1.0)

# Useful constants
inv_se  = Constant(1.0 / sqrt_eps_val)    # 1 / sqrt(eps)
inv_eps = Constant(1.0 / epsilon)          # 1 / eps

# du/dt = A * sech^2(eta) * (-1/sqrt(eps))
du_dt_ufl = A_ufl * sech2_expr * (- inv_se)

# grad(u) components
u_x1_ufl = A_x1_ufl * W_ufl + A_ufl * sech2_expr * inv_se
u_x2_ufl = A_x2_ufl * W_ufl + A_ufl * sech2_expr * inv_se

# Laplacian(u) — fully explicit
lap_u_ufl = (Constant(-2.0) * pi_c * pi_c * A_ufl * W_ufl
             + Constant(2.0) * (A_x1_ufl + A_x2_ufl) * sech2_expr * inv_se
             + Constant(-4.0) * A_ufl * tanh_expr * sech2_expr * inv_eps)

# Source term: f = du/dt - eps * Lap(u) + b · grad(u) + c * u
f_ufl = (du_dt_ufl
         - eps_c * lap_u_ufl
         + b1_c * u_x1_ufl + b2_c * u_x2_ufl
         + c_c * A_ufl * W_ufl)

# Verification
print("")
print("  u_exact(0.0, 0.5, 0.5) = %.6e" % analytical_solution_np(0.0, 0.5, 0.5))
print("  u_exact(0.5, 0.5, 0.5) = %.6e" % analytical_solution_np(0.5, 0.5, 0.5))
print("  u_exact(1.0, 0.5, 0.5) = %.6e" % analytical_solution_np(1.0, 0.5, 0.5))


# =============================================================================
# FEM SETUP
# =============================================================================

# Functions
U   = Function(V, name="u")
U_n = Function(V, name="u_n")
phi1 = TestFunction(V)

# ---- Initial condition ----
# u(0,x1,x2) = 0.5 sin(pi x1) sin(pi x2) [tanh((x1+x2-0.5)/sqrt(eps)) + 1]
u0_expr = Expression(
    '0.5 * sin(pi*x[0]) * sin(pi*x[1]) * (tanh((x[0]+x[1]-0.5)/se) + 1.0)',
    degree=5, pi=np.pi, se=sqrt_eps_val)
U_n.assign(interpolate(u0_expr, V))
print("  Initial condition set: ||u_0||_inf = %.6e" % U_n.vector().norm('linf'))

# Boundary condition
bc = DirichletBC(V, Constant(0.0), 'on_boundary')


# -----------------------------------------------------------------
# SUPG Stabilization (Tezduyar/Shakib)
# -----------------------------------------------------------------
h = CellDiameter(mesh)
velocity_u = b_ufl

tau_t = pow((2.0/dt_c)**2
            + (2.0*sqrt(dot(velocity_u, velocity_u))/h)**2
            + 9.0*(4.0*eps_c/(h*h))**2,
            Constant(-0.5))
tau_u = sqrt(tau_t * tau_t)

tau_est = 1.0 / np.sqrt((2.0/dt_val)**2 + (2.0*b_mag/mesh.hmax())**2
                         + 9.0*(4.0*epsilon/mesh.hmax()**2)**2)
print("  tau_SUPG (est) = %.6e" % tau_est)


# -----------------------------------------------------------------
# Strong-form residual (CG1: Laplacian(U) = 0 element-wise)
# -----------------------------------------------------------------
residual_u = (U - U_n)/dt_c + dot(b_ufl, grad(U)) + c_c*U - f_ufl


# -----------------------------------------------------------------
# YZbeta Shock Capturing
# -----------------------------------------------------------------

Y = 0.25
shock_raw = (1.0/Y) * abs(residual_u) * (h / 2.0)**2
shock_cap = tau_u * dot(b_ufl, b_ufl)  # maximum allowed viscosity
shock_visc = shock_raw


# -----------------------------------------------------------------
# Variational Form: F = 0
# -----------------------------------------------------------------
# Galerkin
F_galerkin = (((U - U_n)/dt_c) * phi1 * dx
              + eps_c * inner(grad(U), grad(phi1)) * dx
              + dot(b_ufl, grad(U)) * phi1 * dx
              + c_c * U * phi1 * dx
              - f_ufl * phi1 * dx)

# SUPG
SUPG = tau_u * dot(velocity_u, grad(phi1)) * residual_u * dx

# Shock capturing
SHOCK = shock_visc * inner(grad(U), grad(phi1)) * dx

F = F_galerkin + SUPG + SHOCK

# Jacobian
J_form = derivative(F, U)


# -----------------------------------------------------------------
# Solver
# -----------------------------------------------------------------
problem = NonlinearVariationalProblem(F, U, bc, J_form)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters["newton_solver"]
prm["absolute_tolerance"]  = 1E-9
prm['relative_tolerance']  = 1E-9
prm['maximum_iterations']  = 50
prm['convergence_criterion'] = 'residual'
prm['krylov_solver']['absolute_tolerance']   = 1E-10
prm['krylov_solver']['relative_tolerance']   = 1E-10
prm['krylov_solver']['maximum_iterations']   = 1000
prm['krylov_solver']['nonzero_initial_guess'] = True
prm['krylov_solver']['error_on_nonconvergence'] = True
prm['krylov_solver']['report'] = True


# =============================================================================
# TIME-STEPPING LOOP
# =============================================================================
print("\n" + "=" * 70)
print("TIME STEPPING")
print("=" * 70)

snapshot_times     = []
snapshot_solutions = []  # each entry: numpy array from point evaluation
u_prev_snapshot    = None
t = 0.0
step = 0

start_wall = timer_module.time()

while t < t_final - 1e-14:
    t += dt_val
    step += 1

    # ---- Update time constant ----
    t_c.assign(t)
    tanh_expr.t = t
    sech2_expr.t = t
    u_exact_expr.t = t

    # ---- Solve ----
    solver.solve()

    # ---- Capture u_prev before overwriting (for PINN PDE residual) ----
    if step == N_t:
        print("  ★ Capturing u_prev (t=%.4f) for PINN..." % (t - dt_val))
        u_prev_snapshot = np.array([U_n(Point(coords[i, 0], coords[i, 1]))
                                     for i in range(n_dof)])

    # ---- Update previous solution ----
    U_n.assign(U)

    # ---- Save snapshot ----
    if step % snapshot_interval == 0 or step == N_t:
        snapshot_times.append(t)
        u_snap = np.array([U(Point(coords[i, 0], coords[i, 1]))
                           for i in range(n_dof)])
        snapshot_solutions.append(u_snap)

    # ---- Diagnostics (point eval + UFL L2 error) ----
    if step % 10 == 0 or step <= 3 or step == N_t:
        u_fem_center = U(Point(0.5, 0.5))
        u_exact_center = analytical_solution_np(t, 0.5, 0.5)

        l2_err = sqrt(assemble((U - u_exact_ufl)**2 * dx))

        # Solution norm
        u_norm = U.vector().norm('linf')

        print("  t = %.4f | FEM(0.5,0.5)=%.4e | Exact=%.4e | ||u||_inf=%.4e | L2=%.4e"
              % (t, u_fem_center, u_exact_center, u_norm, l2_err))

elapsed = timer_module.time() - start_wall
print("\nFEM completed in %.1f s" % elapsed)
print("Snapshots: %d" % len(snapshot_times))


# =============================================================================
# ERROR SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

l2_errors_all = []
for i_snap, (t_s, u_s) in enumerate(zip(snapshot_times, snapshot_solutions)):
    u_exact = analytical_solution_np(t_s, coords[:, 0], coords[:, 1])
    diff = u_exact - u_s
    h_est = 1.0 / nx_mesh
    l2_approx = np.sqrt(np.sum(diff**2) * h_est**2)
    l2_errors_all.append(l2_approx)

for i in range(0, len(snapshot_times), 5):
    print("  t = %.4f | L2 ~ %.6e" % (snapshot_times[i], l2_errors_all[i]))
print("  t = %.4f | L2 ~ %.6e" % (snapshot_times[-1], l2_errors_all[-1]))


# =============================================================================
# SUPG-ONLY SOLVE (no YZbeta shock capturing)
# =============================================================================
print("\n" + "=" * 70)
print("SUPG-ONLY SOLVE (no shock capturing)")
print("=" * 70)

U_so   = Function(V, name="u_supg_only")
U_n_so = Function(V, name="u_n_supg_only")
phi1_so = TestFunction(V)

# Reset initial condition
U_n_so.assign(interpolate(u0_expr, V))

# Strong-form residual (SUPG-only)
residual_so = (U_so - U_n_so)/dt_c + dot(b_ufl, grad(U_so)) + c_c*U_so - f_ufl

# SUPG stabilization parameter (same formula)
tau_t_so = pow((2.0/dt_c)**2
               + (2.0*sqrt(dot(velocity_u, velocity_u))/h)**2
               + 9.0*(4.0*eps_c/(h*h))**2,
               Constant(-0.5))
tau_u_so = sqrt(tau_t_so * tau_t_so)

# Variational form: Galerkin + SUPG only (NO shock capturing)
F_so = (((U_so - U_n_so)/dt_c) * phi1_so * dx
        + eps_c * inner(grad(U_so), grad(phi1_so)) * dx
        + dot(b_ufl, grad(U_so)) * phi1_so * dx
        + c_c * U_so * phi1_so * dx
        - f_ufl * phi1_so * dx
        + tau_u_so * dot(velocity_u, grad(phi1_so)) * residual_so * dx)

J_so = derivative(F_so, U_so)
bc_so = DirichletBC(V, Constant(0.0), 'on_boundary')

problem_so = NonlinearVariationalProblem(F_so, U_so, bc_so, J_so)
solver_so  = NonlinearVariationalSolver(problem_so)

prm_so = solver_so.parameters["newton_solver"]
prm_so["absolute_tolerance"]  = 1E-9
prm_so['relative_tolerance']  = 1E-9
prm_so['maximum_iterations']  = 50
prm_so['convergence_criterion'] = 'residual'
prm_so['krylov_solver']['absolute_tolerance']   = 1E-10
prm_so['krylov_solver']['relative_tolerance']   = 1E-10
prm_so['krylov_solver']['maximum_iterations']   = 1000
prm_so['krylov_solver']['nonzero_initial_guess'] = True
prm_so['krylov_solver']['error_on_nonconvergence'] = True
prm_so['krylov_solver']['report'] = True

t_so = 0.0
step_so = 0
supg_only_terminal = None   # will hold DOF array at t_final
start_wall_so = timer_module.time()

while t_so < t_final - 1e-14:
    t_so += dt_val
    step_so += 1

    # Update time constant (shared with main solve)
    t_c.assign(t_so)
    tanh_expr.t = t_so
    sech2_expr.t = t_so
    u_exact_expr.t = t_so

    solver_so.solve()
    U_n_so.assign(U_so)

    if step_so % 50 == 0 or step_so == N_t:
        u_so_center = U_so(Point(0.5, 0.5))
        u_ex_center = analytical_solution_np(t_so, 0.5, 0.5)
        l2_so = sqrt(assemble((U_so - u_exact_ufl)**2 * dx))
        print("  [SUPG-only] t = %.4f | FEM(0.5,0.5)=%.4e | Exact=%.4e | L2=%.4e"
              % (t_so, u_so_center, u_ex_center, l2_so))

# Capture terminal solution
supg_only_terminal = np.array([U_so(Point(coords[i, 0], coords[i, 1]))
                                for i in range(n_dof)])

elapsed_so = timer_module.time() - start_wall_so
print("SUPG-only completed in %.1f s" % elapsed_so)

# L2 error
u_exact_term = analytical_solution_np(t_final, coords[:, 0], coords[:, 1])
diff_so = u_exact_term - supg_only_terminal
l2_so_terminal = np.sqrt(np.sum(diff_so**2) * (1.0/nx_mesh)**2)
print("  SUPG-only terminal L2 = %.6e" % l2_so_terminal)


# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("VISUALIZATION")
print("=" * 70)

nx_plot, ny_plot = 80, 80
x_plot = np.linspace(0, 1, nx_plot)
y_plot = np.linspace(0, 1, ny_plot)
X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

# Global color scale — use initial time peak
u_anal_peak = analytical_solution_np(0.0, X_plot, Y_plot)
global_vmax = max(u_anal_peak.max(), 1e-6)
u_anal_mid = analytical_solution_np(0.5, X_plot, Y_plot)
global_vmax = max(global_vmax, u_anal_mid.max())
global_levels = np.linspace(0.0, global_vmax * 1.05, 20)

n_snaps = len(snapshot_times)
n_cols = 10
n_rows = int(np.ceil(n_snaps / n_cols))


def snap_to_grid(u_snap_vals, coords, X_g, Y_g):
    """Interpolate DOF values onto plot grid via scipy griddata."""
    return griddata(coords, u_snap_vals, (X_g, Y_g),
                    method='linear', fill_value=0.0)


# ----- FEM Snapshot Grid -----
print("  Plotting FEM snapshots ...")
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(n_cols * 2.2, n_rows * 2.2),
                          squeeze=False)
for idx in range(n_rows * n_cols):
    row, col = divmod(idx, n_cols)
    ax = axes[row][col]
    if idx < n_snaps:
        t_s = snapshot_times[idx]
        u_plot = snap_to_grid(snapshot_solutions[idx], coords, X_plot, Y_plot)
        ax.contourf(X_plot, Y_plot, u_plot, levels=global_levels,
                    cmap='viridis', extend='both')
        ax.set_title('t=%.3f\nmax=%.3f' % (t_s, u_plot.max()), fontsize=6, pad=2)
    else:
        ax.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
fig.suptitle('SUPG+YZbeta FEM Solution (Traveling Wave)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('fem_snapshots_every5.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: fem_snapshots_every5.png")

# ----- Analytical Snapshot Grid -----
print("  Plotting analytical snapshots ...")
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(n_cols * 2.2, n_rows * 2.2),
                          squeeze=False)
for idx in range(n_rows * n_cols):
    row, col = divmod(idx, n_cols)
    ax = axes[row][col]
    if idx < n_snaps:
        t_s = snapshot_times[idx]
        u_anal = analytical_solution_np(t_s, X_plot, Y_plot)
        ax.contourf(X_plot, Y_plot, u_anal, levels=global_levels,
                    cmap='viridis', extend='both')
        ax.set_title('t=%.3f\nmax=%.3f' % (t_s, u_anal.max()), fontsize=6, pad=2)
    else:
        ax.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
fig.suptitle('Analytical Solution (Traveling Wave)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('analytical_snapshots_every5.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: analytical_snapshots_every5.png")

# ----- Error Snapshot Grid -----
print("  Plotting error snapshots ...")
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(n_cols * 2.2, n_rows * 2.2),
                          squeeze=False)
for idx in range(n_rows * n_cols):
    row, col = divmod(idx, n_cols)
    ax = axes[row][col]
    if idx < n_snaps:
        t_s = snapshot_times[idx]
        u_anal = analytical_solution_np(t_s, X_plot, Y_plot)
        u_plot = snap_to_grid(snapshot_solutions[idx], coords, X_plot, Y_plot)
        err = np.abs(u_plot - u_anal)
        ax.contourf(X_plot, Y_plot, err, levels=15, cmap='hot')
        ax.set_title('t=%.3f' % t_s, fontsize=7, pad=2)
    else:
        ax.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
fig.suptitle('|FEM - Analytical| Error (Traveling Wave)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('error_snapshots_every5.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: error_snapshots_every5.png")

# ----- L2 Error Over Time -----
print("  Plotting L2 error ...")
fig, ax = plt.subplots(figsize=(12, 5))
ax.semilogy(snapshot_times, l2_errors_all, 'g-o', linewidth=1.5, markersize=3)
ax.set_xlabel('Time t', fontsize=13)
ax.set_ylabel('$L^2$ Error', fontsize=13)
ax.set_title('$L^2$ Error (SUPG + YZbeta) — Traveling Wave', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('l2_error_evolution.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: l2_error_evolution.png")

# ----- Cross-section at x2 = 0.5 -----
print("  Plotting cross-section ...")
t_end = snapshot_times[-1]
u_anal_line = analytical_solution_np(t_end, x_plot, 0.5)
u_snap_end = snapshot_solutions[-1]
pts_line = np.column_stack([x_plot, 0.5 * np.ones_like(x_plot)])
u_fem_line = griddata(coords, u_snap_end, pts_line, method='linear', fill_value=0.0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_plot, u_anal_line, 'k-', linewidth=2.5, label='Analytical')
u_so_line = griddata(coords, supg_only_terminal, pts_line, method='linear', fill_value=0.0)
ax.plot(x_plot, u_so_line, 'b:', linewidth=2, label='SUPG-only')
ax.plot(x_plot, u_fem_line, 'g--', linewidth=2, label='SUPG+YZbeta FEM')
ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$u(x_1, 0.5)$ at $t = %.2f$' % t_end, fontsize=14)
ax.set_title('Cross-Section at $x_2 = 0.5$ — Traveling Wave', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cross_section_x2_05.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: cross_section_x2_05.png")

# ----- 3D Surface -----
print("  Plotting 3D surface ...")
fig = plt.figure(figsize=(21, 6))
u_anal_end = analytical_solution_np(t_end, X_plot, Y_plot)
u_fem_end = snap_to_grid(u_snap_end, coords, X_plot, Y_plot)
u_supg_only_end = snap_to_grid(supg_only_terminal, coords, X_plot, Y_plot)

for idx_p, (data, title) in enumerate([
        (u_anal_end, 'Analytical (t=%.2f)' % t_end),
        (u_supg_only_end, 'SUPG-only (t=%.2f)' % t_end),
        (u_fem_end, 'SUPG+YZbeta (t=%.2f)' % t_end)]):
    ax = fig.add_subplot(1, 3, idx_p + 1, projection='3d')
    ax.plot_surface(X_plot, Y_plot, data, cmap='viridis', alpha=0.85,
                    rstride=2, cstride=2)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_zlabel('$u$')
    ax.set_title(title, fontsize=12)
    ax.view_init(elev=25, azim=135)
plt.tight_layout()
plt.savefig('3d_surface.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: 3d_surface.png")


# ===== FINAL REPORT =====
print("\n" + "=" * 70)
print("FINAL REPORT")
print("=" * 70)
print("  Terminal time:   t = %.4f" % t_end)
print("  SUPG+YZβ L2:    %.6e" % l2_errors_all[-1])
print("  SUPG-only L2:   %.6e" % l2_so_terminal)
print("  Max L2 error:    %.6e" % max(l2_errors_all))
print("  Solver time:     %.1f s" % elapsed)
print("  Snapshots:       %d" % len(snapshot_times))
print("=" * 70)

# Export for PINN step
np.savez('fem_results.npz',
         snapshot_times=np.array(snapshot_times),
         snapshot_solutions=np.array(snapshot_solutions),
         coordinates=coords,
         l2_errors=np.array(l2_errors_all),
         terminal_solution=snapshot_solutions[-1],
         u_prev_solution=u_prev_snapshot,
         terminal_time=t_end,
         dt=dt_val)
print("  Saved: fem_results.npz")
print("=" * 70)


# #############################################################################
#                                                                             #
#  STEP 2:  PINN CORRECTION AT TERMINAL TIME                                 #
#           Network input: (t, x₁, x₂)                                       #
#                                                                             #
# #############################################################################

print("\n" + "=" * 70)
print("STEP 2: PINN CORRECTION AT TERMINAL TIME")
print("=" * 70)

# =============================================================================
# RHS COMPUTATION (numpy, for PINN PDE residual)
# =============================================================================
class ComputeRHS:
    """Compute source term f for the traveling wave PDE in numpy (for PINN training).

    u(t,x1,x2) = 0.5 sin(pi x1) sin(pi x2) [tanh(eta) + 1]
    eta = (x1 + x2 - t - 0.5) / sqrt(eps)

    f = du/dt - eps * Lap(u) + b.grad(u) + c * u

    All inputs can be arrays (vectorized).
    """
    def __init__(self, eps, b, c):
        self.eps = eps
        self.sqrt_eps = np.sqrt(eps)
        self.b1, self.b2 = b
        self.c = c

    def compute_f(self, t_val, x1, x2):
        """Compute f(t, x1, x2). All inputs can be arrays."""
        x1 = np.asarray(x1, dtype=np.float64)
        x2 = np.asarray(x2, dtype=np.float64)
        t_val = np.asarray(t_val, dtype=np.float64)
        eps = self.eps
        se = self.sqrt_eps
        b1, b2, c = self.b1, self.b2, self.c

        eta = (x1 + x2 - t_val - 0.5) / se
        T = np.tanh(eta)     # tanh(eta)
        S = 1.0 - T**2       # sech^2(eta)
        W = T + 1.0           # tanh(eta) + 1

        # A(x1,x2) = 0.5 sin(pi x1) sin(pi x2)
        sin1 = np.sin(np.pi * x1)
        sin2 = np.sin(np.pi * x2)
        cos1 = np.cos(np.pi * x1)
        cos2 = np.cos(np.pi * x2)

        A     = 0.5 * sin1 * sin2
        A_x1  = 0.5 * np.pi * cos1 * sin2
        A_x2  = 0.5 * sin1 * np.pi * cos2

        # du/dt = A * sech^2(eta) * (-1/sqrt(eps))
        du_dt = A * S * (-1.0 / se)

        # grad u = (A_x1 * W + A * S / se,  A_x2 * W + A * S / se)
        u_x1 = A_x1 * W + A * S / se
        u_x2 = A_x2 * W + A * S / se

        # Laplacian of u:
        # Lap(u) = -2*pi^2*A*W + 2*(A_x1+A_x2)*S/se - 4*A*T*S/eps
        lap_u = (-2.0 * np.pi**2 * A * W
                 + 2.0 * (A_x1 + A_x2) * S / se
                 - 4.0 * A * T * S / eps)

        # f = du/dt - eps * Lap(u) + b.grad(u) + c * u
        f = (du_dt
             - eps * lap_u
             + b1 * u_x1 + b2 * u_x2
             + c * A * W)
        return f


rhs_computer = ComputeRHS(epsilon, [b1_val, b2_val], c_val)


# =============================================================================
# PINN ARCHITECTURE — input: (t, x₁, x₂), output: u_θ(t, x₁, x₂)
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


class HybridSUPGPINN2D(nn.Module):
    """
    Physics-Informed Neural Network for correcting SUPG+YZβ FEM solution.

    Input:  (t, x₁, x₂) — 3D
    Output: u_θ(t, x₁, x₂)
    """
    def __init__(self, hidden_dim=96, num_blocks=6,
                 num_fourier_features=16, sigma=4.0):
        super().__init__()
        self.num_fourier_features = num_fourier_features
        # B: (3, nf) for 3D input (t, x1, x2)
        self.register_buffer('B', torch.randn((3, num_fourier_features)) * sigma)
        input_dim = 3 + 2 * num_fourier_features  # raw + sin/cos
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def fourier_features(self, x):
        projection = torch.matmul(x, self.B)  # (N, 3) @ (3, nf) = (N, nf)
        return torch.cat([x, torch.sin(projection), torch.cos(projection)], dim=-1)

    def forward(self, x):
        """
        x: (N, 3) = (t, x₁, x₂)
        Returns: u_θ(t, x₁, x₂) with Dirichlet BCs enforced via distance function.
        """
        features = self.fourier_features(x)
        h = self.input_layer(features)
        for block in self.blocks:
            h = block(h)
        raw = self.output_layer(h)
        # Distance function d(x) = x₁(1-x₁)x₂(1-x₂) — vanishes on ∂Ω
        x1 = x[:, 1:2]
        x2 = x[:, 2:3]
        d = x1 * (1.0 - x1) * x2 * (1.0 - x2)
        return raw * d

    def compute_pde_residual(self, x_points, eps, b_vec, c_react, rhs_fn):
        """
        Compute strong-form PDE residual via autograd.

        R = ∂u_θ/∂t − ε∆u_θ + b·∇u_θ + cu_θ − f

        x_points: (N, 3) = (t, x₁, x₂) — requires_grad=True
        """
        if not x_points.requires_grad:
            x_points = x_points.clone().detach().requires_grad_(True)

        u = self.forward(x_points).squeeze()

        # First derivatives: (∂u/∂t, ∂u/∂x₁, ∂u/∂x₂)
        grads = torch.autograd.grad(
            u, x_points, torch.ones_like(u),
            create_graph=True, retain_graph=True)[0]

        u_t  = grads[:, 0]
        u_x1 = grads[:, 1]
        u_x2 = grads[:, 2]

        # Second spatial derivatives
        u_x1x1 = torch.autograd.grad(
            u_x1, x_points, torch.ones_like(u_x1),
            create_graph=True, retain_graph=True)[0][:, 1]

        u_x2x2 = torch.autograd.grad(
            u_x2, x_points, torch.ones_like(u_x2),
            create_graph=True, retain_graph=True)[0][:, 2]

        laplacian_u = u_x1x1 + u_x2x2

        # Source term f(t, x₁, x₂) via numpy
        x_np = x_points.detach().cpu().numpy()
        f_vals = rhs_fn.compute_f(x_np[:, 0], x_np[:, 1], x_np[:, 2])
        f_t = torch.tensor(f_vals, dtype=torch.float32, device=x_points.device)

        residual = (u_t
                    - eps * laplacian_u
                    + b_vec[0] * u_x1 + b_vec[1] * u_x2
                    + c_react * u
                    - f_t)
        return residual


# =============================================================================
# HYPERPARAMETERS
# =============================================================================
HIDDEN_DIM = 96
NUM_BLOCKS = 6
NUM_FOURIER = 16
FOURIER_SCALE = 4.0
NUM_EPOCHS = 5000
BATCH_SIZE = 16000
LR_INIT = 8e-5 * (BATCH_SIZE / 256)**0.5
N_SNAPSHOTS_PINN = 5   # how many final snapshots to use as training data

print(f"\nPINN Hyperparameters:")
print(f"  Hidden dim: {HIDDEN_DIM}, Blocks: {NUM_BLOCKS}")
print(f"  Fourier features: {NUM_FOURIER}, scale: {FOURIER_SCALE}")
print(f"  Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}")
print(f"  Initial LR: {LR_INIT:.2e}")
print(f"  Training snapshots: last {N_SNAPSHOTS_PINN}")


# =============================================================================
# DATA PREPARATION — Multi-snapshot (t, x₁, x₂) → u_FEM
# =============================================================================
print("\nPreparing multi-snapshot training data...")

t_terminal = t_end
n_total_snaps = len(snapshot_times)
snap_start = max(0, n_total_snaps - N_SNAPSHOTS_PINN)

# Collect (t, x₁, x₂, u_FEM) from last N snapshots
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

# Build (t, x1, x2) tensor
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
u_analytical_terminal = analytical_solution_np(t_terminal, coords[:, 0], coords[:, 1])

# Identify boundary DOFs (for diagnostics)
tol_bdy = 1e-8
is_boundary = (
    (np.abs(coords[:, 0]) < tol_bdy) |
    (np.abs(coords[:, 0] - 1.0) < tol_bdy) |
    (np.abs(coords[:, 1]) < tol_bdy) |
    (np.abs(coords[:, 1] - 1.0) < tol_bdy))
print(f"  Interior DOFs: {np.sum(~is_boundary)}, Boundary DOFs: {np.sum(is_boundary)}")


# =============================================================================
# TRAINING
# =============================================================================
def train_hybrid_pinn():
    print("\n" + "=" * 70)
    print("HYBRID PINN TRAINING")
    print(f"  Time window: [{t_pinn_start:.4f}, {t_pinn_end:.4f}]")
    print(f"  Target: terminal time t = {t_terminal:.4f}")
    print("=" * 70)

    model = HybridSUPGPINN2D(
        hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS,
        num_fourier_features=NUM_FOURIER, sigma=FOURIER_SCALE).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    all_x = coords_tensor.to(device)
    all_y = u_fem_tensor.unsqueeze(1).to(device)
    N_data = all_x.shape[0]

    PDE_EVERY = 3

    USE_AMP = (device.type == 'cuda')
    if USE_AMP:
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=True)
            def amp_ctx():
                return torch.amp.autocast('cuda', enabled=True)
        except (TypeError, AttributeError):
            try:
                scaler = torch.cuda.amp.GradScaler(enabled=True)
                def amp_ctx():
                    return torch.cuda.amp.autocast(enabled=True)
            except Exception:
                USE_AMP = False
    if not USE_AMP:
        import contextlib
        scaler = None
        def amp_ctx():
            return contextlib.nullcontext()

    def hybrid_loss(model, data_x, data_y, pde_pts, epoch, compute_pde=True):
        # --- Data fidelity loss ---
        with amp_ctx():
            pred = model(data_x)
            data_loss = F_torch.mse_loss(pred, data_y)
        if torch.isnan(data_loss) or torch.isinf(data_loss):
            data_loss = torch.tensor(1.0, device=data_x.device)

        # --- PDE residual loss ---
        pde_loss = torch.tensor(0.0, device=data_x.device)
        if compute_pde and pde_pts is not None and len(pde_pts) > 0:
            try:
                # Filter boundary-adjacent points
                x1_c, x2_c = pde_pts[:, 1], pde_pts[:, 2]
                dist = torch.min(torch.stack([x1_c, 1.0-x1_c, x2_c, 1.0-x2_c]), dim=0)[0]
                mask = dist > 0.02
                interior = pde_pts[mask]
                if len(interior) > 30:
                    res = model.compute_pde_residual(
                        interior, epsilon, [b1_val, b2_val], c_val, rhs_computer)
                    res = torch.clamp(res, -10.0, 10.0)
                    if not (torch.isnan(res).any() or torch.isinf(res).any()):
                        pde_loss = torch.mean(res ** 2)
                        if torch.isnan(pde_loss) or torch.isinf(pde_loss):
                            pde_loss = torch.tensor(0.0, device=data_x.device)
            except Exception as e:
                if epoch % 500 == 0:
                    print(f"  PDE loss failed at epoch {epoch}: {e}")

        bc_loss = torch.tensor(0.0, device=data_x.device)

        # --- Adaptive weight scheduling ---
        if epoch < 1000:
            w_data, w_pde, w_bc = 1.0, 0.05, 0.0
        elif epoch < 1500:
            w_data, w_pde, w_bc = 0.5, 0.1, 0.0
        else:
            w_data, w_pde, w_bc = 0.1, 0.5, 0.0

        total = w_data * data_loss + w_pde * pde_loss
        if torch.isnan(total) or torch.isinf(total):
            total = data_loss
        total = torch.clamp(total, 0.0, 100.0)
        return total, {
            'data': data_loss.item(), 'pde': pde_loss.item(),
            'bc': bc_loss.item(), 'total': total.item(),
            'weights': {'data': w_data, 'pde': w_pde, 'bc': w_bc}
        }

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT,
                                   weight_decay=1e-7, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9,
                                  patience=150, min_lr=1e-6)

    history = {'epoch': [], 'total_loss': [], 'data_loss': [], 'pde_loss': [],
               'bc_loss': [], 'lr': [], 'data_weight': [], 'pde_weight': [], 'bc_weight': []}
    best_loss = float('inf')
    best_state = None
    start_time_pinn = timer_module.time()
    nan_count = 0
    SAVE_EVERY = 100

    n_batches = max(1, (N_data + BATCH_SIZE - 1) // BATCH_SIZE)
    print(f"Training for {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}")
    print(f"  Batches/epoch: {n_batches}, Data points: {N_data}")
    print(f"  PDE every {PDE_EVERY} batches, AMP: {USE_AMP}")
    print(f"  Model: {NUM_BLOCKS} blocks × {HIDDEN_DIM} dim, {NUM_FOURIER} Fourier")

    global_batch = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_losses = {'total': 0.0, 'data': 0.0, 'pde': 0.0, 'bc': 0.0}
        batch_count = 0

        perm = torch.randperm(N_data, device=device)

        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE : min((b + 1) * BATCH_SIZE, N_data)]
            batch_x = all_x[idx]
            batch_y = all_y[idx]

            do_pde = (global_batch % PDE_EVERY == 0)
            pde_pts = None
            if do_pde:
                n_pde = 384
                pde_pts = torch.rand(n_pde, 3, device=device)
                pde_pts[:, 0] = pde_pts[:, 0] * (t_pinn_end - t_pinn_start) + t_pinn_start
                pde_pts.requires_grad_(True)

            optimizer.zero_grad(set_to_none=True)
            total_loss, lc = hybrid_loss(model, batch_x, batch_y, pde_pts,
                                          epoch, compute_pde=do_pde)

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                nan_count += 1
                global_batch += 1
                if nan_count > 10:
                    break
                continue

            if USE_AMP:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
            else:
                total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            grad_tensors = [p.grad for p in model.parameters() if p.grad is not None]
            if grad_tensors:
                grad_flat = torch.cat([g.view(-1) for g in grad_tensors])
                valid = torch.isfinite(grad_flat).all().item()
            else:
                valid = False

            if valid:
                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                batch_count += 1
                for k in epoch_losses:
                    epoch_losses[k] += lc.get(k, lc.get('total', 0.0))
            else:
                if USE_AMP:
                    scaler.update()
                nan_count += 1

            global_batch += 1

        if batch_count == 0:
            print(f"No valid batches at epoch {epoch}, stopping.")
            break

        avg = {k: v / batch_count for k, v in epoch_losses.items()}
        scheduler.step(avg['total'])

        if avg['total'] < best_loss:
            best_loss = avg['total']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Save to disk periodically (not every improvement)
        if best_state is not None and (epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS - 1):
            torch.save(best_state, 'hybrid_pinn_terminal.pt')

        if epoch % 50 == 0 or epoch == NUM_EPOCHS - 1:
            w = lc['weights']
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:5d} | Total: {avg['total']:.4e} | "
                  f"Data: {avg['data']:.4e} | PDE: {avg['pde']:.4e} | "
                  f"BC: {avg['bc']:.4e} | LR: {lr:.1e}")
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
    print(f"Best loss: {best_loss:.8f}, NaN count: {nan_count}")

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


# #############################################################################
#                                                                             #
#  STEP 3:  EVALUATION & COMPARISON                                           #
#                                                                             #
# #############################################################################

print("\n" + "=" * 70)
print("STEP 3: EVALUATION & VISUALIZATION")
print("=" * 70)

# Load best model
model_eval = HybridSUPGPINN2D(
    hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS,
    num_fourier_features=NUM_FOURIER, sigma=FOURIER_SCALE).to(device)
model_eval.load_state_dict(
    torch.load('hybrid_pinn_terminal.pt', map_location=device, weights_only=True))
model_eval.eval()


def evaluate_pinn_grid(model, pts_2d, t_val, dev):
    """Evaluate PINN at fixed time t_val on 2D spatial points."""
    n = len(pts_2d)
    t_col = np.full((n, 1), t_val)
    pts_3d = np.column_stack([t_col, pts_2d])
    tensor_in = torch.tensor(pts_3d, dtype=torch.float32).to(dev)
    with torch.no_grad():
        u_pred = model(tensor_in).cpu().numpy().flatten()
    return u_pred


# Evaluation grid
nx_eval, ny_eval = 64, 64
x_ev = np.linspace(0, 1, nx_eval)
y_ev = np.linspace(0, 1, ny_eval)
X_ev, Y_ev = np.meshgrid(x_ev, y_ev)
points_eval = np.column_stack([X_ev.ravel(), Y_ev.ravel()])

# Compute solutions on evaluation grid at t = t_terminal
u_anal_grid = analytical_solution_np(t_terminal, X_ev, Y_ev)
u_fem_grid = griddata(coords, snapshot_solutions[-1], (X_ev, Y_ev),
                      method='linear', fill_value=0.0)
u_supg_only_grid = griddata(coords, supg_only_terminal, (X_ev, Y_ev),
                            method='linear', fill_value=0.0)
u_pinn_grid = evaluate_pinn_grid(model_eval, points_eval, t_terminal, device).reshape(ny_eval, nx_eval)


# ===== PLOT 1: Solution comparison — SEPARATE figures =====
print(f"\nCreating solution comparison (separate) at t = {t_terminal:.2f} ...")

vmin = min(u_anal_grid.min(), u_fem_grid.min(), u_supg_only_grid.min(), u_pinn_grid.min())
vmax = max(u_anal_grid.max(), u_fem_grid.max(), u_supg_only_grid.max(), u_pinn_grid.max())
levels = np.linspace(vmin - 1e-10, vmax + 1e-10, 25)

for data, fname in [
    (u_anal_grid,      'contour_analytical.png'),
    (u_supg_only_grid, 'contour_supg.png'),
    (u_fem_grid,       'contour_fem.png'),
    (u_pinn_grid,      'contour_pinn.png')]:
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.contourf(X_ev, Y_ev, data, levels=levels, cmap='viridis')
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(fname, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {fname}")


# ===== PLOT 2: Error maps =====
print("Creating error maps ...")
fig, axes = plt.subplots(1, 3, figsize=(21, 5))
err_supg_only = np.abs(u_supg_only_grid - u_anal_grid)
err_fem = np.abs(u_fem_grid - u_anal_grid)
err_pinn = np.abs(u_pinn_grid - u_anal_grid)
for ax, err, title in zip(axes, [err_supg_only, err_fem, err_pinn],
    ['|SUPG-only − Analytical|', '|SUPG+YZβ − Analytical|', '|PINN − Analytical|']):
    im = ax.contourf(X_ev, Y_ev, err, 20, cmap='hot')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('terminal_error_maps.png', dpi=400, bbox_inches='tight')
plt.close()
print("  ✅ Saved: terminal_error_maps.png")


# ===== PLOT 3: Cross-section at x2 = 0.5 =====
print("Creating cross-section ...")
fig, ax = plt.subplots(figsize=(10, 6))
u_anal_line = analytical_solution_np(t_terminal, x_ev, 0.5)
u_fem_line = griddata(coords, snapshot_solutions[-1],
                      np.column_stack([x_ev, 0.5*np.ones_like(x_ev)]),
                      method='linear', fill_value=0.0)
u_pinn_line = u_pinn_grid[ny_eval // 2, :]
u_so_line_term = griddata(coords, supg_only_terminal,
                          np.column_stack([x_ev, 0.5*np.ones_like(x_ev)]),
                          method='linear', fill_value=0.0)
ax.plot(x_ev, u_anal_line, 'k-', linewidth=2.5, label='Analytical')
ax.plot(x_ev, u_so_line_term, 'b:', linewidth=2, label='SUPG-only')
ax.plot(x_ev, u_fem_line, 'g--', linewidth=2, label='SUPG+YZβ')
ax.plot(x_ev, u_pinn_line, 'r-.', linewidth=2, label='Hybrid PINN')
ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel(f'$u(x_1, 0.5)$ at $t = {t_terminal:.2f}$', fontsize=14)
ax.set_title(f'Cross-Section at $x_2 = 0.5$, $t = {t_terminal:.2f}$ — Traveling Wave', fontsize=14)
ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('terminal_cross_section_x2_05.png', dpi=400, bbox_inches='tight')
plt.close()
print("  ✅ Saved: terminal_cross_section_x2_05.png")


# ===== PLOT 4: L2 error over time + PINN point =====
print("Creating L2 error plot ...")
u_pinn_dof = evaluate_pinn_grid(model_eval, coords, t_terminal, device)
diff_pinn = u_analytical_terminal - u_pinn_dof
h_est = 1.0 / nx_mesh
l2_pinn_terminal = np.sqrt(np.sum(diff_pinn**2) * h_est**2)

diff_fem = u_analytical_terminal - snapshot_solutions[-1]
l2_fem_terminal = np.sqrt(np.sum(diff_fem**2) * h_est**2)

fig, ax = plt.subplots(figsize=(12, 5))
ax.semilogy(snapshot_times, l2_errors_all, 'g-', linewidth=1.5, label='SUPG+YZβ', alpha=0.8)
ax.semilogy(snapshot_times[::10], l2_errors_all[::10], 'go', markersize=4)
ax.semilogy(t_terminal, l2_pinn_terminal, 'r*', markersize=15,
            label=f'Hybrid PINN (t={t_terminal:.2f})', zorder=5)
ax.set_xlabel('Time t', fontsize=14); ax.set_ylabel('$L^2$ Error', fontsize=14)
ax.set_title('$L^2$ Error: FEM + PINN correction — Traveling Wave', fontsize=14)
ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('l2_error_over_time_with_pinn.png', dpi=400, bbox_inches='tight')
plt.close()
print("  ✅ Saved: l2_error_over_time_with_pinn.png")


# ===== PLOT 5: Training history — SEPARATE figures =====
print("Creating training history (separate) ...")
epochs_hist = history_hybrid['epoch']

# ── Figure 5a: Training Losses ──
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(epochs_hist, history_hybrid['total_loss'], 'b-',  linewidth=1.5, label='Total')
ax.semilogy(epochs_hist, history_hybrid['data_loss'],  'g--', linewidth=1.5, label='Data')
ax.semilogy(epochs_hist, history_hybrid['pde_loss'],   'r-.', linewidth=1.5, label='PDE')
#ax.semilogy(epochs_hist, history_hybrid['bc_loss'],    'm:',  linewidth=1.5, label='BC')
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Loss',  fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_losses.png', dpi=400, bbox_inches='tight')
plt.close()
print("  ✅ Saved: training_losses.png")

# ── Figure 5b: Learning Rate ──
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs_hist, history_hybrid['lr'], 'k-', linewidth=1.5)
ax.set_xlabel('Epoch',         fontsize=14)
ax.set_ylabel('Learning Rate', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_lr.png', dpi=400, bbox_inches='tight')
plt.close()
print("  ✅ Saved: training_lr.png")

# ── Figure 5c: Adaptive Weights ──
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs_hist, history_hybrid['data_weight'], 'g-',   linewidth=1.5, label='$w_{\\mathrm{data}}$')
ax.plot(epochs_hist, history_hybrid['pde_weight'],  'r--',  linewidth=1.5, label='$w_{\\mathrm{pde}}$')
#ax.plot(epochs_hist, history_hybrid['bc_weight'],   'm-.',  linewidth=1.5, label='$w_{\\mathrm{bc}}$')
ax.set_xlabel('Epoch',  fontsize=14)
ax.set_ylabel('Weight', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_weights.png', dpi=400, bbox_inches='tight')
plt.close()
print("  ✅ Saved: training_weights.png")


# ===== PLOT 6: 3D surface comparison — SEPARATE figures =====
print(f"Creating 3D surface plots (separate) at t = {t_terminal:.2f} ...")

for data, fname in [
    (u_anal_grid,      '3d_surface_analytical.png'),
    (u_supg_only_grid, '3d_surface_supg.png'),
    (u_fem_grid,       '3d_surface_fem.png'),
    (u_pinn_grid,      '3d_surface_pinn.png')]:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_ev, Y_ev, data, cmap='viridis', alpha=0.85, rstride=2, cstride=2)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_zlabel('$u$',   fontsize=14)
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    plt.savefig(fname, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {fname}")


# =============================================================================
# FINAL ERROR TABLE
# =============================================================================
print("\n" + "=" * 70)
print("FINAL ERROR TABLE")
print("=" * 70)
print(f"\nTerminal-time correction (t = {t_terminal:.4f}):")
print(f"  L2(SUPG-only) = {l2_so_terminal:.6e}")
print(f"  L2(SUPG+YZβ)  = {l2_fem_terminal:.6e}")
print(f"  L2(PINN)      = {l2_pinn_terminal:.6e}")
if l2_pinn_terminal < l2_fem_terminal:
    ratio = l2_fem_terminal / max(l2_pinn_terminal, 1e-16)
    print(f"  ✓ PINN improved by {ratio:.2f}x")
else:
    ratio = l2_pinn_terminal / max(l2_fem_terminal, 1e-16)
    print(f"  ✗ PINN worse by {ratio:.2f}x")

# Pointwise comparison at center
u_anal_center = analytical_solution_np(t_terminal, 0.5, 0.5)
u_so_center = griddata(coords, supg_only_terminal, np.array([[0.5, 0.5]]),
                        method='linear')[0]
u_fem_center_t = griddata(coords, snapshot_solutions[-1], np.array([[0.5, 0.5]]),
                          method='linear')[0]
u_pinn_center = evaluate_pinn_grid(model_eval, np.array([[0.5, 0.5]]), t_terminal, device)[0]
print(f"\nPointwise at (0.5, 0.5):")
print(f"  Analytical:  {u_anal_center:.6e}")
print(f"  SUPG-only:   {u_so_center:.6e}  (err: {abs(u_anal_center - u_so_center):.4e})")
print(f"  SUPG+YZβ:    {u_fem_center_t:.6e}  (err: {abs(u_anal_center - u_fem_center_t):.4e})")
print(f"  PINN:        {u_pinn_center:.6e}  (err: {abs(u_anal_center - u_pinn_center):.4e})")

print("\n" + "=" * 70)
print("✅ ALL COMPUTATIONS COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\n📊 Generated plots:")
print("  • fem_snapshots_every5.png         (FEM grid, every 5 steps)")
print("  • analytical_snapshots_every5.png   (Analytical grid)")
print("  • error_snapshots_every5.png        (Error grid)")
print("  • l2_error_evolution.png            (L2 error curve)")
print("  • cross_section_x2_05.png           (1D slice at x2=0.5)")
print("  • 3d_surface.png                    (FEM vs Analytical 3D)")
print("  • terminal_solution_comparison.png  (FEM vs PINN contour)")
print("  • terminal_error_maps.png           (Error maps)")
print("  • terminal_cross_section_x2_05.png  (FEM vs PINN 1D slice)")
print("  • l2_error_over_time_with_pinn.png  (L2 + PINN point)")
print("  • training_history_terminal.png     (Loss curves)")
print("  • 3d_surface_terminal.png           (4-panel 3D)")
print("=" * 70)