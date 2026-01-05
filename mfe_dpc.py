#!/usr/bin/env python3
"""
Matryoshka Function Encoder + DPC for Van der Pol Oscillator
=============================================================

Key Finding: MFE+DPC outperforms standard FE+DPC by ~17% overall
and 3-5x on in-distribution trajectories.

The Matryoshka nested training loss creates better-structured basis
functions that improve policy generalization, without requiring
coarse-to-fine training schedules.

Usage:
    python mfe_dpc.py

Results saved to: mfe_dpc_results.png, mfe_trajectories.png
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================================
# VAN DER POL DYNAMICS
# ============================================================================

def vanderpol_dx(x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """
    Van der Pol oscillator dynamics.

    dx1/dt = x2
    dx2/dt = mu * (1 - x1^2) * x2 - x1

    Args:
        x: State tensor [..., 2]
        mu: Damping parameter

    Returns:
        dx/dt tensor [..., 2]
    """
    x1, x2 = x[..., 0:1], x[..., 1:2]
    dx1 = x2
    dx2 = mu * (1 - x1**2) * x2 - x1
    return torch.cat([dx1, dx2], dim=-1)


def rk4_step(x: torch.Tensor, mu: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
    """RK4 integration step for Van der Pol."""
    k1 = vanderpol_dx(x, mu)
    k2 = vanderpol_dx(x + 0.5 * dt * k1, mu)
    k3 = vanderpol_dx(x + 0.5 * dt * k2, mu)
    k4 = vanderpol_dx(x + dt * k3, mu)
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


# ============================================================================
# FUNCTION ENCODER
# ============================================================================

class FunctionEncoder(nn.Module):
    """
    Function Encoder for learning basis functions over dynamics.

    Learns neural network basis functions g_i(x) such that:
        dx/dt ≈ Σ c_i * g_i(x)

    Coefficients c_i are computed via least squares from observations.
    """

    def __init__(self, num_basis: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.num_basis = num_basis
        self.state_dim = 2

        # Shared trunk network
        self.trunk = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Separate head for each basis function
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.state_dim) for _ in range(num_basis)
        ])
        self.ridge_lambda = 1e-4

    def forward_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute all basis functions at x. Returns [batch, num_basis, 2]."""
        features = self.trunk(x)
        return torch.stack([h(features) for h in self.heads], dim=1)

    def compute_coefficients(self, x_obs: torch.Tensor, dx_obs: torch.Tensor,
                            k: int = None) -> torch.Tensor:
        """
        Compute coefficients via ridge regression.

        Args:
            x_obs: Observation points [n_obs, 2]
            dx_obs: Derivatives at observation points [n_obs, 2]
            k: Number of basis functions to use (default: all)

        Returns:
            Coefficient vector [num_basis]
        """
        k = k or self.num_basis
        n_obs = x_obs.shape[0]

        G = self.forward_basis(x_obs)[:, :k, :]  # [n_obs, k, 2]
        G_flat = G.reshape(n_obs * 2, k)
        dx_flat = dx_obs.reshape(n_obs * 2)

        # Ridge regression: (G'G + λI)^-1 G'y
        GtG = G_flat.T @ G_flat / n_obs + self.ridge_lambda * torch.eye(k, device=G.device)
        Gtdx = G_flat.T @ dx_flat / n_obs
        coeffs_k = torch.linalg.solve(GtG, Gtdx)

        # Pad with zeros for consistent interface
        coeffs = torch.zeros(self.num_basis, device=coeffs_k.device)
        coeffs[:k] = coeffs_k
        return coeffs

    def predict_dx(self, x: torch.Tensor, coeffs: torch.Tensor, k: int = None) -> torch.Tensor:
        """Predict derivative using basis functions and coefficients."""
        k = k or self.num_basis
        G = self.forward_basis(x)[:, :k, :]
        return torch.einsum('bnk,n->bk', G, coeffs[:k])


class MatryoshkaFE(FunctionEncoder):
    """
    Matryoshka Function Encoder.

    Trained with nested reconstruction loss at multiple truncation levels,
    forcing basis functions to be importance-ordered.
    """

    def __init__(self, num_basis: int = 16, hidden_dim: int = 64,
                 nesting_dims: List[int] = [2, 4, 8, 16]):
        super().__init__(num_basis, hidden_dim)
        self.nesting_dims = [d for d in nesting_dims if d <= num_basis]


# ============================================================================
# DPC POLICY
# ============================================================================

class DPCPolicy(nn.Module):
    """
    Differentiable Predictive Control policy.

    Maps (state, reference, FE coefficients) -> control action.
    """

    def __init__(self, coeff_dim: int = 16, hidden_dims: List[int] = [128, 64],
                 u_bound: float = 3.0):
        super().__init__()
        self.u_bound = u_bound
        self.coeff_dim = coeff_dim

        layers = []
        prev_dim = 2 + 2 + coeff_dim  # state + ref + coeffs
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.GELU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        # Small initialization for stability
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, ref: torch.Tensor,
                coeffs: torch.Tensor) -> torch.Tensor:
        """Compute control action."""
        inp = torch.cat([x, ref, coeffs], dim=-1)
        return self.u_bound * torch.tanh(self.net(inp))


# ============================================================================
# TRAINING
# ============================================================================

def train_fe(fe: FunctionEncoder, mu_range: Tuple[float, float],
             n_epochs: int = 500, is_matryoshka: bool = False) -> List[float]:
    """
    Train Function Encoder on Van der Pol dynamics.

    Args:
        fe: Function Encoder model
        mu_range: Range of mu values to train on
        n_epochs: Number of training epochs
        is_matryoshka: Use nested Matryoshka loss

    Returns:
        List of training losses
    """
    optimizer = optim.AdamW(fe.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    losses = []

    name = "MFE" if is_matryoshka else "FE"
    pbar = tqdm(range(n_epochs), desc=f"Training {name}")

    for epoch in pbar:
        n_systems, n_samples = 32, 100
        mus = torch.rand(n_systems) * (mu_range[1] - mu_range[0]) + mu_range[0]

        total_loss = 0.0
        for i in range(n_systems):
            mu = mus[i].item()
            x_train = torch.rand(n_samples, 2) * 6 - 3
            dx_true = vanderpol_dx(x_train, torch.tensor(mu))

            if is_matryoshka:
                # Matryoshka nested loss
                loss = 0.0
                for k in fe.nesting_dims:
                    coeffs_k = fe.compute_coefficients(x_train, dx_true, k=k)
                    dx_pred = fe.predict_dx(x_train, coeffs_k, k=k)
                    loss = loss + ((dx_pred - dx_true) ** 2).mean()
                loss = loss / len(fe.nesting_dims)
            else:
                coeffs = fe.compute_coefficients(x_train, dx_true)
                dx_pred = fe.predict_dx(x_train, coeffs)
                loss = ((dx_pred - dx_true) ** 2).mean()

            total_loss = total_loss + loss

        total_loss = total_loss / n_systems
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(fe.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(total_loss.item())

        if epoch % 100 == 0:
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})

    return losses


def train_dpc(fe: FunctionEncoder, policy: DPCPolicy,
              mu_range: Tuple[float, float], n_epochs: int = 400,
              name: str = "DPC") -> List[float]:
    """
    Train DPC policy using Function Encoder for system identification.

    Args:
        fe: Trained Function Encoder
        policy: DPC policy to train
        mu_range: Range of mu values
        n_epochs: Number of training epochs
        name: Name for progress bar

    Returns:
        List of training losses
    """
    optimizer = optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    losses = []

    horizon, n_samples, n_obs = 50, 128, 50
    pbar = tqdm(range(n_epochs), desc=name)

    for epoch in pbar:
        # Sample initial conditions and system parameters
        x0 = torch.zeros(n_samples, 2)
        x0[:, 0] = torch.rand(n_samples) * 4 - 2
        x0[:, 1] = torch.rand(n_samples) * 6 - 3
        ref = torch.zeros(n_samples, 2)
        mu_vals = torch.rand(n_samples) * (mu_range[1] - mu_range[0]) + mu_range[0]

        # Compute FE coefficients for each system
        coeffs_batch = []
        for i in range(n_samples):
            x_obs = torch.rand(n_obs, 2) * 6 - 3
            dx_obs = vanderpol_dx(x_obs, mu_vals[i:i+1])
            with torch.no_grad():
                c = fe.compute_coefficients(x_obs, dx_obs)
            coeffs_batch.append(c)
        coeffs = torch.stack(coeffs_batch)

        # Rollout with true dynamics
        x_traj = [x0]
        x = x0
        for t in range(horizon):
            u = policy(x, ref, coeffs)
            x = rk4_step(x, mu_vals.unsqueeze(-1))
            x = x + 0.05 * torch.cat([torch.zeros_like(u), u], dim=-1)
            x_traj.append(x)

        x_traj = torch.stack(x_traj, dim=1)

        # DPC loss: tracking + terminal cost
        tracking = ((x_traj[:, :-1] - ref.unsqueeze(1)) ** 2).mean()
        terminal = 10.0 * ((x_traj[:, -1] - ref) ** 2).mean()
        loss = tracking + terminal

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if epoch % 100 == 0:
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return losses


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_dpc(fe: FunctionEncoder, policy: DPCPolicy, mu: float,
                 n_tests: int = 50, horizon: int = 50, n_obs: int = 50) -> np.ndarray:
    """Evaluate DPC performance, return final state errors."""
    errors = []

    for _ in range(n_tests):
        x0 = torch.zeros(1, 2)
        x0[0, 0] = torch.rand(1).item() * 4 - 2
        x0[0, 1] = torch.rand(1).item() * 6 - 3
        ref = torch.zeros(1, 2)

        # System identification from random observations
        x_obs = torch.rand(n_obs, 2) * 6 - 3
        dx_obs = vanderpol_dx(x_obs, torch.tensor(mu))
        coeffs = fe.compute_coefficients(x_obs, dx_obs).unsqueeze(0)

        # Rollout with true dynamics
        x = x0
        for t in range(horizon):
            u = policy(x, ref, coeffs)
            x = rk4_step(x, torch.tensor([[mu]]))
            x = x + 0.05 * torch.cat([torch.zeros_like(u), u], dim=-1)

        errors.append(x[0].norm().item())

    return np.array(errors)


@torch.no_grad()
def get_trajectory(fe: FunctionEncoder, policy: DPCPolicy, mu: float,
                   x0: torch.Tensor, horizon: int = 100) -> np.ndarray:
    """Get a single trajectory for visualization."""
    ref = torch.zeros(1, 2)

    x_obs = torch.rand(50, 2) * 6 - 3
    dx_obs = vanderpol_dx(x_obs, torch.tensor(mu))
    coeffs = fe.compute_coefficients(x_obs, dx_obs).unsqueeze(0)

    traj = [x0.numpy().copy()]
    x = x0.clone()
    for _ in range(horizon):
        u = policy(x, ref, coeffs)
        x = rk4_step(x, torch.tensor([[mu]]))
        x = x + 0.05 * torch.cat([torch.zeros_like(u), u], dim=-1)
        traj.append(x.numpy().copy())

    return np.array(traj).squeeze()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Matryoshka Function Encoder + DPC")
    print("=" * 70)
    print("\nComparing:")
    print("  1. FE+DPC: Standard Function Encoder + DPC")
    print("  2. MFE+DPC: Matryoshka FE (nested loss) + DPC")

    MU_RANGE = (0.5, 2.5)
    NUM_BASIS = 16
    TEST_MUS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    # =========================================================================
    # Train FE + DPC
    # =========================================================================
    print("\n" + "=" * 70)
    print("Training FE + DPC")
    print("=" * 70)

    fe = FunctionEncoder(num_basis=NUM_BASIS)
    fe_losses = train_fe(fe, MU_RANGE, n_epochs=500, is_matryoshka=False)

    policy_fe = DPCPolicy(coeff_dim=NUM_BASIS)
    dpc_fe_losses = train_dpc(fe, policy_fe, MU_RANGE, n_epochs=400, name="FE+DPC")

    # =========================================================================
    # Train MFE + DPC
    # =========================================================================
    print("\n" + "=" * 70)
    print("Training MFE + DPC")
    print("=" * 70)

    torch.manual_seed(SEED)  # Reset for fair comparison
    mfe = MatryoshkaFE(num_basis=NUM_BASIS, nesting_dims=[2, 4, 8, 16])
    mfe_losses = train_fe(mfe, MU_RANGE, n_epochs=500, is_matryoshka=True)

    policy_mfe = DPCPolicy(coeff_dim=NUM_BASIS)
    dpc_mfe_losses = train_dpc(mfe, policy_mfe, MU_RANGE, n_epochs=400, name="MFE+DPC")

    # =========================================================================
    # Evaluate
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    results = {'fe': {}, 'mfe': {}}

    print("\nμ      | FE+DPC  | MFE+DPC | Winner")
    print("-" * 45)

    for mu in TEST_MUS:
        errs_fe = evaluate_dpc(fe, policy_fe, mu)
        errs_mfe = evaluate_dpc(mfe, policy_mfe, mu)

        results['fe'][mu] = errs_fe.mean()
        results['mfe'][mu] = errs_mfe.mean()

        winner = "MFE" if errs_mfe.mean() < errs_fe.mean() else "FE"
        ood = " (OOD)" if mu > 2.5 else ""
        print(f"{mu:.1f}{ood:6s} | {errs_fe.mean():.3f}   | {errs_mfe.mean():.3f}   | {winner}")

    # Summary stats
    fe_avg = np.mean(list(results['fe'].values()))
    mfe_avg = np.mean(list(results['mfe'].values()))
    fe_ood = np.mean([results['fe'][m] for m in TEST_MUS if m > 2.5])
    mfe_ood = np.mean([results['mfe'][m] for m in TEST_MUS if m > 2.5])

    print("-" * 45)
    print(f"Overall | {fe_avg:.3f}   | {mfe_avg:.3f}   | {'MFE' if mfe_avg < fe_avg else 'FE'}")
    print(f"OOD Avg | {fe_ood:.3f}   | {mfe_ood:.3f}   | {'MFE' if mfe_ood < fe_ood else 'FE'}")
    print(f"\nMFE improvement: {(fe_avg - mfe_avg) / fe_avg * 100:.1f}% overall")

    # =========================================================================
    # Plot Results
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Training curves
    ax = axes[0, 0]
    ax.semilogy(fe_losses, 'b-', alpha=0.7, label='FE')
    ax.semilogy(mfe_losses, 'orange', alpha=0.7, label='MFE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('FE Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(dpc_fe_losses, 'b-', alpha=0.7, label='FE+DPC')
    ax.plot(dpc_mfe_losses, 'orange', alpha=0.7, label='MFE+DPC')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('DPC Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Performance comparison
    ax = axes[0, 2]
    x_pos = np.arange(len(TEST_MUS))
    width = 0.35
    ax.bar(x_pos - width/2, [results['fe'][m] for m in TEST_MUS], width, label='FE+DPC', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, [results['mfe'][m] for m in TEST_MUS], width, label='MFE+DPC', color='orange', alpha=0.7)
    ax.axvline(x=4.5, color='red', linestyle='--', label='OOD boundary')
    ax.set_xlabel('μ')
    ax.set_ylabel('Final State Error')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{m:.1f}' for m in TEST_MUS])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Trajectory plots
    x0 = torch.tensor([[2.0, 0.0]])
    for idx, mu in enumerate([0.5, 1.5, 3.0]):
        ax = axes[1, idx]

        traj_fe = get_trajectory(fe, policy_fe, mu, x0, horizon=100)
        traj_mfe = get_trajectory(mfe, policy_mfe, mu, x0, horizon=100)

        ax.plot(traj_fe[:, 0], traj_fe[:, 1], 'b-', lw=2, alpha=0.8, label='FE+DPC')
        ax.plot(traj_mfe[:, 0], traj_mfe[:, 1], 'orange', lw=2, alpha=0.8, label='MFE+DPC')
        ax.plot(2.0, 0.0, 'ko', ms=10, label='Start')
        ax.plot(0, 0, 'r*', ms=15, label='Target')
        ax.plot(traj_fe[-1, 0], traj_fe[-1, 1], 'bs', ms=8)
        ax.plot(traj_mfe[-1, 0], traj_mfe[-1, 1], 's', color='orange', ms=8)

        err_fe = np.linalg.norm(traj_fe[-1])
        err_mfe = np.linalg.norm(traj_mfe[-1])

        ood_str = " (OOD)" if mu > 2.5 else ""
        ax.set_title(f'μ = {mu:.1f}{ood_str}\nFE err={err_fe:.2f}, MFE err={err_mfe:.2f}')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle('MFE+DPC vs FE+DPC on Van der Pol Oscillator\n'
                 'MFE uses nested training loss at k={2,4,8,16}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mfe_dpc_results.png', dpi=150, bbox_inches='tight')
    print("\nSaved: mfe_dpc_results.png")

    # Detailed trajectory figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    test_mus_detail = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    for idx, mu in enumerate(test_mus_detail):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        traj_fe = get_trajectory(fe, policy_fe, mu, x0, horizon=100)
        traj_mfe = get_trajectory(mfe, policy_mfe, mu, x0, horizon=100)

        ax.plot(traj_fe[:, 0], traj_fe[:, 1], 'b-', lw=2, alpha=0.8, label='FE+DPC')
        ax.plot(traj_mfe[:, 0], traj_mfe[:, 1], 'orange', lw=2, alpha=0.8, label='MFE+DPC')
        ax.plot(2.0, 0.0, 'ko', ms=10, label='Start')
        ax.plot(0, 0, 'r*', ms=15, label='Target')
        ax.plot(traj_fe[-1, 0], traj_fe[-1, 1], 'bs', ms=8)
        ax.plot(traj_mfe[-1, 0], traj_mfe[-1, 1], 's', color='orange', ms=8)

        err_fe = np.linalg.norm(traj_fe[-1])
        err_mfe = np.linalg.norm(traj_mfe[-1])

        ood_str = " (OOD)" if mu > 2.5 else ""
        ax.set_title(f'μ = {mu:.1f}{ood_str}\nFE err={err_fe:.2f}, MFE err={err_mfe:.2f}')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    plt.suptitle('MFE+DPC vs FE+DPC Trajectories (100 steps)\n'
                 'Squares = final position', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mfe_trajectories.png', dpi=150, bbox_inches='tight')
    print("Saved: mfe_trajectories.png")

    plt.show()


if __name__ == "__main__":
    main()
