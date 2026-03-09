# ============================================================================
# PINN-FEM Hybrid Solver for Time-Dependent 1D Burgers Equation (Example 2)
# Problem: ∂u/∂t + u(∂u/∂x) = ν(∂²u/∂x²), t ∈ (t0, tf], x ∈ (0, 1)
# ============================================================================

# NumPy 2.0+ compatibility fix for FEniCS
import numpy as np
if not hasattr(np, 'product'):
    np.product = np.prod
if not hasattr(np, 'cumproduct'):
    np.cumproduct = np.cumprod
if not hasattr(np, 'sometrue'):
    np.sometrue = np.any
if not hasattr(np, 'alltrue'):
    np.alltrue = np.all

import fenics as fe
import dolfin as df
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import gc
import os
import warnings

warnings.filterwarnings('ignore')

# FEniCS configuration
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"
df.parameters['form_compiler']['quadrature_degree'] = 9

# Random seeds
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# PROBLEM PARAMETERS (Example 2 from Paper)
# ============================================================================
Re = 1000  # Reynolds number
nu = 1.0 / Re  # Viscosity ν = 1/Re

# Spatial domain
x_min, x_max = 0.0, 1.0
mesh_resolution = 80

# Temporal domain
t_start = 1.0  # Initial time t₀ = 1
t_end = 2.0    # Final time T_final
num_time_steps = 100
dt = (t_end - t_start) / num_time_steps

print("="*70)
print("EXAMPLE 2: 1D TIME-DEPENDENT BURGERS EQUATION")
print("="*70)
print(f"\nProblem: ∂u/∂t + u(∂u/∂x) = ν(∂²u/∂x²)")
print(f"Parameters: Re = {Re}, ν = {nu:.6e}")
print(f"Domain: x ∈ [0, 1], t ∈ [{t_start}, {t_end}]")
print(f"Mesh: {mesh_resolution} elements, {num_time_steps} time steps, Δt = {dt:.4f}")
print(f"\nStrategy: FEM solves full time → PINN corrects at t = {t_end}")

# ============================================================================
# ANALYTICAL SOLUTION
# Cole-Hopf transformation:
# u(x,t) = (x/t) / (1 + √t · exp(x²/(4νt) - 1/(16ν)))
# ============================================================================

def analytical_solution(x, t):
    """
    u(x,t) = (x/t) / (1 + √t · exp(x²/(4νt) - 1/(16ν)))
    """
    x = np.asarray(x)
    t_val = float(t)

    combined_exp = x**2 / (4 * nu * t_val) - 1.0 / (16 * nu)
    combined_exp = np.clip(combined_exp, -700, 700)

    denom = 1.0 + np.sqrt(t_val) * np.exp(combined_exp)
    u_exact = (x / t_val) / denom

    return u_exact


def initial_condition(x):
    """
    IC at t=1: u(x,1) = x / (1 + exp((x²-1/4)/(4ν)))
    """
    x = np.asarray(x)
    exp_arg = (x**2 - 0.25) / (4 * nu)
    exp_arg = np.clip(exp_arg, -700, 700)
    return x / (1 + np.exp(exp_arg))


# Verify IC consistency
x_test = np.linspace(0, 1, 11)
ic_vals = initial_condition(x_test)
an_vals = analytical_solution(x_test, 1.0)
max_diff = np.max(np.abs(ic_vals - an_vals))
if max_diff > 1e-10:
    raise ValueError("Initial condition inconsistent with analytical solution!")


# Create mesh and function space
mesh = fe.IntervalMesh(mesh_resolution, x_min, x_max)
V = fe.FunctionSpace(mesh, "CG", 1)

print(f"\nMesh: {mesh.num_cells()} cells, {V.dim()} DOFs")

# ============================================================================
# FEM HELPER CLASSES
# ============================================================================
class InitialConditionExpression(fe.UserExpression):
    def __init__(self, nu_val, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu_val

    def eval(self, values, x):
        exp_arg = (x[0]**2 - 0.25) / (4 * self.nu)
        exp_arg = np.clip(exp_arg, -700, 700)
        values[0] = x[0] / (1 + np.exp(exp_arg))

    def value_shape(self):
        return ()


def boundary(x, on_boundary):
    return on_boundary


def create_solver_parameters():
    prm = {}
    prm["absolute_tolerance"] = 1E-10
    prm['relative_tolerance'] = 1E-10
    prm['maximum_iterations'] = 50
    prm['convergence_criterion'] = 'incremental'
    return prm


# ============================================================================
# FEM SOLVERS (Return only final time solution)
# ============================================================================
def solve_gfem():
    """Standard Galerkin FEM - returns solution at t=T_final"""
    print("\n--- Solving GFEM ---")
    start = time.time()

    v = fe.TestFunction(V)
    u = fe.Function(V)
    u_n = fe.Function(V)

    nu_c = fe.Constant(nu)
    dt_c = fe.Constant(dt)

    u_init = InitialConditionExpression(nu, degree=4)
    u_n.interpolate(u_init)
    u.interpolate(u_init)

    # Backward Euler weak form
    F = ((u - u_n) / dt_c) * v * fe.dx \
        + u * fe.Dx(u, 0) * v * fe.dx \
        + nu_c * fe.Dx(u, 0) * fe.Dx(v, 0) * fe.dx

    J = fe.derivative(F, u)
    bc = fe.DirichletBC(V, fe.Constant(0.0), boundary)

    problem = fe.NonlinearVariationalProblem(F, u, bc, J)
    solver = fe.NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"].update(create_solver_parameters())

    t = t_start
    for step in range(num_time_steps):
        t += dt
        solver.solve()
        u_n.assign(u)

    print(f"    Completed in {time.time()-start:.2f}s")
    return u


def solve_supg():
    """SUPG-stabilized FEM - returns solution at t=T_final"""
    print("\n--- Solving SUPG ---")
    start = time.time()

    v = fe.TestFunction(V)
    u = fe.Function(V)
    u_n = fe.Function(V)

    nu_c = fe.Constant(nu)
    dt_c = fe.Constant(dt)

    u_init = InitialConditionExpression(nu, degree=4)
    u_n.interpolate(u_init)
    u.interpolate(u_init)

    # Standard Galerkin
    F = ((u - u_n) / dt_c) * v * fe.dx \
        + u * fe.Dx(u, 0) * v * fe.dx \
        + nu_c * fe.Dx(u, 0) * fe.Dx(v, 0) * fe.dx

    # SUPG stabilization
    h = fe.CellDiameter(mesh)
    u_mag = fe.sqrt(u * u + fe.DOLFIN_EPS)

    tau = fe.Constant(1.0) / fe.sqrt(
        (2.0 / dt_c)**2 + (2.0 * u_mag / h)**2 + (4.0 * nu_c / (h * h))**2
    )

    residual = (u - u_n) / dt_c + u * fe.Dx(u, 0)
    F += tau * u * fe.Dx(v, 0) * residual * fe.dx

    J = fe.derivative(F, u)
    bc = fe.DirichletBC(V, fe.Constant(0.0), boundary)

    problem = fe.NonlinearVariationalProblem(F, u, bc, J)
    solver = fe.NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"].update(create_solver_parameters())

    t = t_start
    for step in range(num_time_steps):
        t += dt
        solver.solve()
        u_n.assign(u)

    print(f"    Completed in {time.time()-start:.2f}s")
    return u


def solve_supg_yzb():
    """SUPG + YZβ shock-capturing - returns solution at t=T_final"""
    print("\n--- Solving SUPG-YZβ ---")
    start = time.time()

    v = fe.TestFunction(V)
    u = fe.Function(V)
    u_n = fe.Function(V)

    nu_c = fe.Constant(nu)
    dt_c = fe.Constant(dt)

    u_init = InitialConditionExpression(nu, degree=4)
    u_n.interpolate(u_init)
    u.interpolate(u_init)

    # Standard Galerkin
    F = ((u - u_n) / dt_c) * v * fe.dx \
        + u * fe.Dx(u, 0) * v * fe.dx \
        + nu_c * fe.Dx(u, 0) * fe.Dx(v, 0) * fe.dx

    # SUPG
    h = fe.CellDiameter(mesh)
    u_mag = fe.sqrt(u * u + fe.DOLFIN_EPS)

    tau = fe.Constant(1.0) / fe.sqrt(
        (2.0 / dt_c)**2 + (2.0 * u_mag / h)**2 + (4.0 * nu_c / (h * h))**2
    )

    residual = (u - u_n) / dt_c + u * fe.Dx(u, 0)
    F += tau * u * fe.Dx(v, 0) * residual * fe.dx

    # YZβ shock-capturing
    Y = fe.Constant(0.5)
    Z_abs = fe.sqrt(residual * residual + fe.DOLFIN_EPS)
    nu_SHOC = (Z_abs / Y) * (h * h) / 4.0
    F += nu_SHOC * fe.Dx(u, 0) * fe.Dx(v, 0) * fe.dx

    J = fe.derivative(F, u)
    bc = fe.DirichletBC(V, fe.Constant(0.0), boundary)

    problem = fe.NonlinearVariationalProblem(F, u, bc, J)
    solver = fe.NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"].update(create_solver_parameters())

    t = t_start
    for step in range(num_time_steps):
        t += dt
        solver.solve()
        u_n.assign(u)

    print(f"    Completed in {time.time()-start:.2f}s")
    return u


# ============================================================================
# SOLVE FEM
# ============================================================================
print("\n" + "="*70)
print("PHASE 1: FEM TIME-STEPPING (t = {:.1f} → {:.1f})".format(t_start, t_end))
print("="*70)

u_gfem = solve_gfem()
u_supg = solve_supg()
u_supg_yzb = solve_supg_yzb()

# Get coordinates
coords = V.tabulate_dof_coordinates()[:, 0]
sort_idx = np.argsort(coords)
x_dof = coords[sort_idx]

# Extract solutions at final time
u_gfem_final = np.array([u_gfem(x) for x in x_dof])
u_supg_final = np.array([u_supg(x) for x in x_dof])
u_supg_yzb_final = np.array([u_supg_yzb(x) for x in x_dof])
u_analytical_final = analytical_solution(x_dof, t_end)

# Compute FEM errors
print("\n" + "="*70)
print(f"FEM ERRORS AT t = {t_end}")
print("="*70)

fem_errors = {}
for name, u_fem in [('GFEM', u_gfem_final), ('SUPG', u_supg_final), ('SUPG-YZβ', u_supg_yzb_final)]:
    l2_err = np.sqrt(np.mean((u_fem - u_analytical_final)**2))
    max_err = np.max(np.abs(u_fem - u_analytical_final))
    fem_errors[name] = {'L2': l2_err, 'Max': max_err}
    print(f"  {name:10s}: L2 = {l2_err:.6e}, Max = {max_err:.6e}")


# ============================================================================
# PINN MODEL (1D Input: x only, since we're at fixed t = T_final)
# ============================================================================
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
        return self.act(self.norm(out + identity))


class HybridPINN1D(nn.Module):
    """
    1D PINN for correcting FEM solution at t = T_final
    Input: x (spatial coordinate)
    Output: u(x, T_final)
    """
    def __init__(self, hidden_dim=128, num_blocks=8, num_fourier=24, sigma=4.0):
        super().__init__()
        self.register_buffer('B', torch.randn(1, num_fourier) * sigma)

        input_dim = 1 + 2 * num_fourier  # x + sin + cos

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU()
        )

        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Fourier features
        proj = torch.matmul(x, self.B)
        features = torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)

        h = self.input_layer(features)
        for block in self.blocks:
            h = block(h)
        return self.output_layer(h)

    def compute_pde_residual(self, x_points, t_val=t_end):
        """
        PDE residual at fixed t = T_final:
        Spatial part: u*∂u/∂x - ν*∂²u/∂x²
        """
        if not x_points.requires_grad:
            x_points = x_points.clone().detach().requires_grad_(True)

        u = self.forward(x_points).squeeze()

        # ∂u/∂x
        u_x = torch.autograd.grad(u, x_points, torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0].squeeze()

        # ∂²u/∂x²
        u_xx = torch.autograd.grad(u_x, x_points, torch.ones_like(u_x),
                                    create_graph=True, retain_graph=True)[0].squeeze()

        # Spatial residual: u*u_x - ν*u_xx
        residual = u * u_x - nu * u_xx

        return residual


# ============================================================================
# PHASE 2: PINN TRAINING (Correct SUPG-YZβ solution at t = T_final)
# ============================================================================
print("\n" + "="*70)
print(f"PHASE 2: PINN CORRECTION AT t = {t_end}")
print("="*70)

# Prepare training data (1D: x coordinates, target: SUPG-YZβ solution)
X_train = torch.tensor(x_dof.reshape(-1, 1), dtype=torch.float32)
U_train = torch.tensor(u_supg_yzb_final.reshape(-1, 1), dtype=torch.float32)

print(f"\nTraining data: {len(X_train)} points")
print(f"x range: [{x_dof.min():.4f}, {x_dof.max():.4f}]")
print(f"u range: [{u_supg_yzb_final.min():.6f}, {u_supg_yzb_final.max():.6f}]")

# Identify boundary points
tol = 1e-8
is_boundary = (np.abs(x_dof) < tol) | (np.abs(x_dof - 1.0) < tol)
X_boundary = torch.tensor(x_dof[is_boundary].reshape(-1, 1), dtype=torch.float32).to(device)

print(f"Boundary points: {len(X_boundary)}")


def train_hybrid_pinn():
    """Hybrid PINN training with data and BC losses."""

    model = HybridPINN1D(hidden_dim=64, num_blocks=4, num_fourier=16, sigma=2.0).to(device)

    dataset = TensorDataset(X_train, U_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=300, min_lr=1e-6)

    num_epochs = 5000
    history = {'epoch': [], 'total': [], 'data': [], 'pde': [], 'bc': []}

    best_loss = float('inf')
    best_epoch = 0
    best_state = None

    print("\n" + "="*60)
    print("PINN TRAINING")
    print("="*60)

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = {'total': 0, 'data': 0, 'pde': 0, 'bc': 0}
        n_batches = 0

        w_data = 1.0
        w_bc = 10.0

        for batch_x, batch_u in dataloader:
            batch_x = batch_x.to(device)
            batch_u = batch_u.to(device)

            optimizer.zero_grad()

            pred_u = model(batch_x)

            # Data loss
            loss_data = F.mse_loss(pred_u, batch_u)

            # BC loss: u(0)=0, u(1)=0
            x_bc = torch.tensor([[0.0], [1.0]], device=device)
            bc_pred = model(x_bc).squeeze()
            loss_bc = torch.mean(bc_pred**2)

            loss = w_data * loss_data + w_bc * loss_bc

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss['total'] += loss.item()
            epoch_loss['data'] += loss_data.item()
            epoch_loss['pde'] += 0.0
            epoch_loss['bc'] += loss_bc.item()
            n_batches += 1

        if n_batches == 0:
            continue

        avg_loss = {k: v/n_batches for k, v in epoch_loss.items()}
        scheduler.step(avg_loss['data'])

        if avg_loss['data'] < best_loss:
            best_loss = avg_loss['data']
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 500 == 0 or epoch == num_epochs - 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | Data Loss: {avg_loss['data']:.2e} | "
                  f"BC Loss: {avg_loss['bc']:.2e} | LR: {lr:.1e}")

            history['epoch'].append(epoch)
            history['total'].append(avg_loss['total'])
            history['data'].append(avg_loss['data'])
            history['pde'].append(0.0)
            history['bc'].append(avg_loss['bc'])

    print("-" * 60)
    print(f"Training completed in {time.time()-start_time:.1f}s")
    print(f"Best data loss: {best_loss:.6e} at epoch {best_epoch}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def predict_with_hard_bc(model, x):
    """Direct model output."""
    return model(x)


# Train
model_hybrid, history = train_hybrid_pinn()


# ============================================================================
# EVALUATION
# ============================================================================
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

model_hybrid.eval()
x_eval = np.linspace(0, 1, 200)
X_eval = torch.tensor(x_eval.reshape(-1, 1), dtype=torch.float32).to(device)

with torch.no_grad():
    u_pinn = predict_with_hard_bc(model_hybrid, X_eval).cpu().numpy().flatten()

u_exact_eval = analytical_solution(x_eval, t_end)
u_supg_yzb_eval = np.interp(x_eval, x_dof, u_supg_yzb_final)

# Errors
pinn_l2 = np.sqrt(np.mean((u_pinn - u_exact_eval)**2))
pinn_max = np.max(np.abs(u_pinn - u_exact_eval))
supg_l2 = np.sqrt(np.mean((u_supg_yzb_eval - u_exact_eval)**2))
supg_max = np.max(np.abs(u_supg_yzb_eval - u_exact_eval))

print(f"\nErrors at t = {t_end}:")
print(f"  SUPG-YZβ:    L2 = {supg_l2:.6e}, Max = {supg_max:.6e}")
print(f"  Hybrid PINN: L2 = {pinn_l2:.6e}, Max = {pinn_max:.6e}")
print(f"\nImprovement: {(supg_l2 - pinn_l2)/supg_l2 * 100:.1f}% (L2)")


# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("GENERATING PLOTS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Solution comparison
ax = axes[0, 0]
ax.plot(x_eval, u_exact_eval, 'k-', lw=2, label='Analytical')
ax.plot(x_eval, u_supg_yzb_eval, 'g--', lw=2, label='SUPG-YZβ')
ax.plot(x_eval, u_pinn, 'r-.', lw=2, label='Hybrid PINN')
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$u(x, t_{final})$', fontsize=12)
ax.set_title(f'Solution at t = {t_end}', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 2. Error comparison
ax = axes[0, 1]
err_supg = np.abs(u_supg_yzb_eval - u_exact_eval)
err_pinn = np.abs(u_pinn - u_exact_eval)
ax.semilogy(x_eval, err_supg + 1e-16, 'g--', lw=2, label='SUPG-YZβ')
ax.semilogy(x_eval, err_pinn + 1e-16, 'r-', lw=2, label='Hybrid PINN')
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('|Error|', fontsize=12)
ax.set_title('Absolute Error', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 3. Bar chart
ax = axes[1, 0]
methods = ['GFEM', 'SUPG', 'SUPG-YZβ', 'Hybrid\nPINN']
l2_errors = [fem_errors['GFEM']['L2'], fem_errors['SUPG']['L2'],
             fem_errors['SUPG-YZβ']['L2'], pinn_l2]
colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']
bars = ax.bar(methods, l2_errors, color=colors, alpha=0.7)
ax.set_ylabel('L2 Error', fontsize=12)
ax.set_title('Method Comparison', fontsize=12)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
for bar, err in zip(bars, l2_errors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{err:.1e}', ha='center', va='bottom', fontsize=9)

# 4. Training history
ax = axes[1, 1]
ax.semilogy(history['epoch'], history['total'], 'b-', lw=2, label='Total Loss')
ax.semilogy(history['epoch'], history['data'], 'g--', lw=1.5, label='Data Loss', alpha=0.7)
ax.semilogy(history['epoch'], history['pde'], 'r-.', lw=1.5, label='PDE Loss', alpha=0.7)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training History', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('burgers_1d_hybrid_results.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Plot saved: burgers_1d_hybrid_results.png")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Strategy: FEM solves time-dependent problem → PINN corrects at final time

1. FEM Phase:
   - Solved Burgers equation from t={t_start} to t={t_end}
   - Methods: GFEM, SUPG, SUPG-YZβ
   - Best FEM: SUPG-YZβ (L2 = {fem_errors['SUPG-YZβ']['L2']:.2e})

2. PINN Phase:
   - Input: SUPG-YZβ solution at t={t_end}
   - Training: Data + PDE + BC losses (adaptive weights)
   - Output: Corrected solution

3. Results at t={t_end}:
   - SUPG-YZβ L2 error: {supg_l2:.6e}
   - Hybrid PINN L2 error: {pinn_l2:.6e}
   - Improvement: {(supg_l2-pinn_l2)/supg_l2*100:.1f}%
""")
print("="*70)


# ============================================================================
# COMPREHENSIVE VISUALIZATION
# ============================================================================

def create_comprehensive_plots():
    """Create individual publication-quality plots for all methods comparison"""

    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE COMPARISON PLOTS")
    print("="*70)

    # Evaluation grid
    x_eval = np.linspace(0, 1, 300)
    X_eval_tensor = torch.tensor(x_eval.reshape(-1, 1), dtype=torch.float32).to(device)

    # Analytical solution at t = t_end
    u_exact = analytical_solution(x_eval, t_end)

    # FEM solutions (interpolated to fine grid)
    u_gfem_eval = np.interp(x_eval, x_dof, u_gfem_final)
    u_supg_eval = np.interp(x_eval, x_dof, u_supg_final)
    u_supg_yzb_eval = np.interp(x_eval, x_dof, u_supg_yzb_final)

    with torch.no_grad():
        u_pinn_eval = predict_with_hard_bc(model_hybrid, X_eval_tensor).cpu().numpy().flatten()

    FIGSIZE = (10, 7)

    # =========================================================================
    # PLOT 1: Full Domain Comparison
    # =========================================================================
    plt.figure(figsize=FIGSIZE)
    plt.plot(x_eval, u_exact, 'k-', linewidth=3, label='Analytical', alpha=0.9)
    plt.plot(x_eval, u_gfem_eval, 'b--', linewidth=2, label='GFEM', alpha=0.7)
    plt.plot(x_eval, u_supg_eval, 'c-.', linewidth=2, label='SUPG', alpha=0.7)
    plt.plot(x_eval, u_supg_yzb_eval, 'g-', linewidth=2.5, label=r'SUPG-YZ$\beta$', alpha=0.8)
    plt.plot(x_eval, u_pinn_eval, 'r--', linewidth=2.5, label='Hybrid PINN', alpha=0.8)

    plt.xlabel('$x$', fontsize=14)
    plt.ylabel(f'$u(x, t={t_end})$', fontsize=14)
    plt.title(f'Solution Comparison at $t = {t_end}$', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex2_full_domain_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: ex2_full_domain_comparison.png")

    # =========================================================================
    # PLOT 2: Left Region Detail
    # =========================================================================
    plt.figure(figsize=FIGSIZE)
    mask_left = x_eval <= 0.3

    plt.plot(x_eval[mask_left], u_exact[mask_left], 'k-', linewidth=3, label='Analytical', alpha=0.9)
    plt.plot(x_eval[mask_left], u_gfem_eval[mask_left], 'b--', linewidth=2, label='GFEM', alpha=0.7)
    plt.plot(x_eval[mask_left], u_supg_eval[mask_left], 'c-.', linewidth=2, label='SUPG', alpha=0.7)
    plt.plot(x_eval[mask_left], u_supg_yzb_eval[mask_left], 'g-', linewidth=2.5, label=r'SUPG-YZ$\beta$', alpha=0.8)
    plt.plot(x_eval[mask_left], u_pinn_eval[mask_left], 'r--', linewidth=2.5, label='Hybrid PINN', alpha=0.8)

    plt.xlabel('$x$', fontsize=14)
    plt.ylabel(f'$u(x, t={t_end})$', fontsize=14)
    plt.title('Left Region Detail ($x \leq 0.3$)', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex2_left_region_detail.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: ex2_left_region_detail.png")

    # =========================================================================
    # PLOT 3: Right Region Detail (shock region)
    # =========================================================================
    plt.figure(figsize=FIGSIZE)
    mask_right = x_eval >= 0.5

    plt.plot(x_eval[mask_right], u_exact[mask_right], 'k-', linewidth=3, label='Analytical', alpha=0.9)
    plt.plot(x_eval[mask_right], u_gfem_eval[mask_right], 'b--', linewidth=2, label='GFEM', alpha=0.7)
    plt.plot(x_eval[mask_right], u_supg_eval[mask_right], 'c-.', linewidth=2, label='SUPG', alpha=0.7)
    plt.plot(x_eval[mask_right], u_supg_yzb_eval[mask_right], 'g-', linewidth=2.5, label=r'SUPG-YZ$\beta$', alpha=0.8)
    plt.plot(x_eval[mask_right], u_pinn_eval[mask_right], 'r--', linewidth=2.5, label='Hybrid PINN', alpha=0.8)

    plt.xlabel('$x$', fontsize=14)
    plt.ylabel(f'$u(x, t={t_end})$', fontsize=14)
    plt.title('Shock Region Detail ($x \geq 0.5$)', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex2_right_region_detail.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: ex2_right_region_detail.png")

    # =========================================================================
    # PLOT 4: Absolute Error (Log Scale)
    # =========================================================================
    plt.figure(figsize=FIGSIZE)

    err_gfem = np.abs(u_gfem_eval - u_exact)
    err_supg = np.abs(u_supg_eval - u_exact)
    err_supg_yzb = np.abs(u_supg_yzb_eval - u_exact)
    err_pinn = np.abs(u_pinn_eval - u_exact)

    plt.semilogy(x_eval, err_gfem + 1e-16, 'b--', linewidth=2, label='GFEM', alpha=0.7)
    plt.semilogy(x_eval, err_supg + 1e-16, 'c-.', linewidth=2, label='SUPG', alpha=0.7)
    plt.semilogy(x_eval, err_supg_yzb + 1e-16, 'g-', linewidth=2.5, label=r'SUPG-YZ$\beta$', alpha=0.8)
    plt.semilogy(x_eval, err_pinn + 1e-16, 'r--', linewidth=2.5, label='Hybrid PINN', alpha=0.8)

    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('Absolute Error', fontsize=14)
    plt.title('Error Distribution (Log Scale)', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('ex2_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: ex2_error_distribution.png")

    # =========================================================================
    # PLOT 5: Performance Bar Chart
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    methods = ['GFEM', 'SUPG', r'SUPG-YZ$\beta$', 'Hybrid PINN']
    l2_errors = [
        np.sqrt(np.mean(err_gfem**2)),
        np.sqrt(np.mean(err_supg**2)),
        np.sqrt(np.mean(err_supg_yzb**2)),
        np.sqrt(np.mean(err_pinn**2))
    ]
    max_errors = [
        np.max(err_gfem),
        np.max(err_supg),
        np.max(err_supg_yzb),
        np.max(err_pinn)
    ]

    colors = ['#1f77b4', '#17becf', '#2ca02c', '#d62728']

    # L2 Error
    ax = axes[0]
    bars1 = ax.bar(methods, l2_errors, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('$L^2$ Error', fontsize=12)
    ax.set_title('$L^2$ Error Comparison', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, err in zip(bars1, l2_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1,
                f'{err:.2e}', ha='center', va='bottom', fontsize=10, rotation=0)

    # Max Error
    ax = axes[1]
    bars2 = ax.bar(methods, max_errors, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Max Error', fontsize=12)
    ax.set_title('Maximum Error Comparison', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, err in zip(bars2, max_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1,
                f'{err:.2e}', ha='center', va='bottom', fontsize=10, rotation=0)

    plt.tight_layout()
    plt.savefig('ex2_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: ex2_performance_comparison.png")

    # =========================================================================
    # PLOT 6: Training History
    # =========================================================================
    plt.figure(figsize=FIGSIZE)

    plt.semilogy(history['epoch'], history['total'], 'k-', linewidth=2.5, label='Total Loss')
    plt.semilogy(history['epoch'], history['data'], 'b--', linewidth=2, label='Data Loss', alpha=0.8)
    plt.semilogy(history['epoch'], history['pde'], 'g-.', linewidth=2, label='PDE Loss', alpha=0.8)
    plt.semilogy(history['epoch'], history['bc'], 'r:', linewidth=2, label='BC Loss', alpha=0.8)

    # Mark phase transitions
    plt.axvline(x=800, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.axvline(x=1500, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ymin, ymax = plt.ylim()
    plt.text(400, ymax*0.3, 'Phase I', fontsize=10, ha='center', alpha=0.7)
    plt.text(1150, ymax*0.3, 'Phase II', fontsize=10, ha='center', alpha=0.7)
    plt.text(3000, ymax*0.3, 'Phase III', fontsize=10, ha='center', alpha=0.7)

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Hybrid PINN Training History', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('ex2_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: ex2_training_history.png")

    # =========================================================================
    # PLOT 7: Gradient Comparison
    # =========================================================================
    plt.figure(figsize=FIGSIZE)

    dx = x_eval[1] - x_eval[0]
    grad_exact = np.abs(np.gradient(u_exact, dx))
    grad_gfem = np.abs(np.gradient(u_gfem_eval, dx))
    grad_supg_yzb = np.abs(np.gradient(u_supg_yzb_eval, dx))
    grad_pinn = np.abs(np.gradient(u_pinn_eval, dx))

    plt.semilogy(x_eval, grad_exact + 1e-16, 'k-', linewidth=3, label='Analytical', alpha=0.9)
    plt.semilogy(x_eval, grad_gfem + 1e-16, 'b--', linewidth=2, label='GFEM', alpha=0.7)
    plt.semilogy(x_eval, grad_supg_yzb + 1e-16, 'g-', linewidth=2.5, label=r'SUPG-YZ$\beta$', alpha=0.8)
    plt.semilogy(x_eval, grad_pinn + 1e-16, 'r--', linewidth=2.5, label='Hybrid PINN', alpha=0.8)

    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$|du/dx|$', fontsize=14)
    plt.title('Gradient Magnitude Comparison', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('ex2_gradient_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: ex2_gradient_comparison.png")

    # =========================================================================
    # PLOT 8: Combined Summary (2x2)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Full domain
    ax = axes[0, 0]
    ax.plot(x_eval, u_exact, 'k-', linewidth=2.5, label='Analytical')
    ax.plot(x_eval, u_supg_yzb_eval, 'g--', linewidth=2, label=r'SUPG-YZ$\beta$', alpha=0.8)
    ax.plot(x_eval, u_pinn_eval, 'r-.', linewidth=2, label='Hybrid PINN', alpha=0.8)
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$u$', fontsize=12)
    ax.set_title('(a) Solution Comparison', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (b) Error
    ax = axes[0, 1]
    ax.semilogy(x_eval, err_supg_yzb + 1e-16, 'g--', linewidth=2, label=r'SUPG-YZ$\beta$')
    ax.semilogy(x_eval, err_pinn + 1e-16, 'r-', linewidth=2, label='Hybrid PINN')
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('|Error|', fontsize=12)
    ax.set_title('(b) Absolute Error', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (c) Bar chart
    ax = axes[1, 0]
    bars = ax.bar(methods, l2_errors, color=colors, alpha=0.7)
    ax.set_ylabel('$L^2$ Error', fontsize=12)
    ax.set_title('(c) Performance Comparison', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # (d) Training
    ax = axes[1, 1]
    ax.semilogy(history['epoch'], history['total'], 'k-', linewidth=2, label='Total')
    ax.semilogy(history['epoch'], history['data'], 'b--', linewidth=1.5, label='Data', alpha=0.7)
    ax.semilogy(history['epoch'], history['pde'], 'g-.', linewidth=1.5, label='PDE', alpha=0.7)
    ax.axvline(x=800, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=1500, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('(d) Training History', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex2_summary_4panel.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: ex2_summary_4panel.png")

    # =========================================================================
    # PRINT NUMERICAL SUMMARY TABLE
    # =========================================================================
    print("\n" + "="*70)
    print("NUMERICAL RESULTS SUMMARY")
    print("="*70)
    print(f"\nProblem: 1D Burgers Equation at t = {t_end}")
    print(f"Re = {Re}, ν = {nu:.6e}")
    print(f"Mesh: {mesh_resolution} elements, {num_time_steps} time steps")
    print("\n" + "-"*70)
    print(f"{'Method':<20} {'L2 Error':<15} {'Max Error':<15} {'Improvement':<15}")
    print("-"*70)

    baseline_l2 = l2_errors[2]  # SUPG-YZβ as baseline
    for i, (method, l2, mx) in enumerate(zip(methods, l2_errors, max_errors)):
        if i == 3:  # Hybrid PINN
            imp = (baseline_l2 - l2) / baseline_l2 * 100
            print(f"{method:<20} {l2:<15.6e} {mx:<15.6e} {imp:>+.1f}%")
        else:
            print(f"{method:<20} {l2:<15.6e} {mx:<15.6e} {'—':<15}")

    print("-"*70)
    print(f"\n🏆 Best Method: ", end="")
    best_idx = np.argmin(l2_errors)
    print(f"{methods[best_idx]} (L2 = {l2_errors[best_idx]:.6e})")

    print("\n" + "="*70)
    print("GENERATED PLOTS:")
    print("="*70)
    print("  1. ex2_full_domain_comparison.png")
    print("  2. ex2_left_region_detail.png")
    print("  3. ex2_right_region_detail.png")
    print("  4. ex2_error_distribution.png")
    print("  5. ex2_performance_comparison.png")
    print("  6. ex2_training_history.png")
    print("  7. ex2_gradient_comparison.png")
    print("  8. ex2_summary_4panel.png")
    print("="*70)


# Run the comprehensive visualization
create_comprehensive_plots()


# ============================================================================
# 4 SOLUTION COMPARISON PLOT
# ============================================================================

def plot_4_solutions_comparison():
    """Single plot comparing GFEM, SUPG, SUPG-YZβ and Hybrid PINN"""

    # Evaluation grid
    x_eval = np.linspace(0, 1, 300)
    X_eval_tensor = torch.tensor(x_eval.reshape(-1, 1), dtype=torch.float32).to(device)

    # Analytical solution
    u_exact = analytical_solution(x_eval, t_end)

    # FEM solutions (interpolated)
    u_gfem_eval = np.interp(x_eval, x_dof, u_gfem_final)
    u_supg_eval = np.interp(x_eval, x_dof, u_supg_final)
    u_supg_yzb_eval = np.interp(x_eval, x_dof, u_supg_yzb_final)

    model_hybrid.eval()
    with torch.no_grad():
        u_pinn_eval = predict_with_hard_bc(model_hybrid, X_eval_tensor).cpu().numpy().flatten()

    # =========================================================================
    # PLOT: 4 Solution Comparison
    # =========================================================================
    plt.figure(figsize=(12, 8))

    plt.plot(x_eval, u_exact, 'k-', linewidth=3, label='Analytical', alpha=0.9)
    plt.plot(x_eval, u_gfem_eval, 'b--', linewidth=2.5, label='GFEM', alpha=0.8)
    plt.plot(x_eval, u_supg_eval, 'c-.', linewidth=2.5, label='SUPG', alpha=0.8)
    plt.plot(x_eval, u_supg_yzb_eval, 'g-', linewidth=2.5, label=r'SUPG-YZ$\beta$', alpha=0.8)
    plt.plot(x_eval, u_pinn_eval, 'r--', linewidth=2.5, label='Hybrid PINN', alpha=0.8)

    plt.xlabel('$x$', fontsize=16)
    plt.ylabel(f'$u(x, t={t_end})$', fontsize=16)
    plt.title(f'Solution Comparison at $t = {t_end}$ (Re = {Re})', fontsize=16)
    plt.legend(fontsize=14, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig('ex2_4solutions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Saved: ex2_4solutions_comparison.png")

    # =========================================================================
    # PLOT: 4 Error Comparison
    # =========================================================================
    plt.figure(figsize=(12, 8))

    err_gfem = np.abs(u_gfem_eval - u_exact)
    err_supg = np.abs(u_supg_eval - u_exact)
    err_supg_yzb = np.abs(u_supg_yzb_eval - u_exact)
    err_pinn = np.abs(u_pinn_eval - u_exact)

    plt.semilogy(x_eval, err_gfem + 1e-16, 'b--', linewidth=2.5, label='GFEM', alpha=0.8)
    plt.semilogy(x_eval, err_supg + 1e-16, 'c-.', linewidth=2.5, label='SUPG', alpha=0.8)
    plt.semilogy(x_eval, err_supg_yzb + 1e-16, 'g-', linewidth=2.5, label=r'SUPG-YZ$\beta$', alpha=0.8)
    plt.semilogy(x_eval, err_pinn + 1e-16, 'r--', linewidth=2.5, label='Hybrid PINN', alpha=0.8)

    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('Absolute Error', fontsize=16)
    plt.title('Error Comparison', fontsize=16)
    plt.legend(fontsize=14, loc='best')
    plt.grid(True, alpha=0.3, which='both')
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig('ex2_4errors_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Saved: ex2_4errors_comparison.png")

    # =========================================================================
    # PRINT L2 ERRORS
    # =========================================================================
    print("\n" + "="*50)
    print("L2 ERRORS")
    print("="*50)
    print(f"  GFEM:        {np.sqrt(np.mean(err_gfem**2)):.6e}")
    print(f"  SUPG:        {np.sqrt(np.mean(err_supg**2)):.6e}")
    print(f"  SUPG-YZβ:    {np.sqrt(np.mean(err_supg_yzb**2)):.6e}")
    print(f"  Hybrid PINN: {np.sqrt(np.mean(err_pinn**2)):.6e}")
    print("="*50)


# Run
plot_4_solutions_comparison()
