"""Training functions for Function Encoder and DPC."""
import torch
import torch.optim as optim
from typing import List, Dict, Any
from tqdm import tqdm

from ..models import (
    FunctionEncoder, MatryoshkaFE, WidthMatryoshkaFE,
    GroupedHierarchicalFE, DPCPolicy
)
from ..dynamics.base import DynamicsBase


def compute_diversity_loss(fe, x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute diversity penalty on basis function outputs.

    Penalizes off-diagonal Gram matrix energy to push bases to be
    functionally distinct (not redundant clones).

    Args:
        fe: Function Encoder model
        x: Input states [n, state_dim]
        k: Number of basis functions to use

    Returns:
        Diversity loss scalar
    """
    G = fe.forward_basis(x)[:, :k, :]           # [n, k, state_dim]
    A = G.reshape(-1, k)                         # [n*d, k]
    # Normalize columns
    A = A / (A.pow(2).mean(dim=0, keepdim=True).sqrt() + 1e-6)
    Gram = (A.T @ A) / A.shape[0]                # [k, k]
    # Penalize off-diagonal (deviation from identity)
    loss_div = (Gram - torch.eye(k, device=A.device)).pow(2).mean()
    return loss_div


def train_fe(
    fe: FunctionEncoder,
    dynamics: DynamicsBase,
    config: Dict[str, Any],
    is_matryoshka: bool = False,
) -> List[float]:
    """
    Train Function Encoder on dynamical system.

    Uses meta-style training with context/target split.
    Optionally adds diversity regularization + weighted ridge for separate MLPs.

    Args:
        fe: Function Encoder model
        dynamics: Dynamical system
        config: Training configuration
        is_matryoshka: Use nested Matryoshka loss

    Returns:
        List of training losses
    """
    n_epochs = config["training"]["fe_epochs"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    param_range = tuple(config["dynamics"]["mu_range"])

    # Meta-style training hyperparameters
    n_ctx = 50      # Context set size (matches DPC inference)
    n_tgt = 100     # Target set size (larger for stable gradients)
    lambda_cons = 0.1  # Coefficient consistency weight

    # Option 2: Diversity regularization (for separate MLPs)
    use_diversity_reg = config["training"].get("use_diversity_reg", False)
    diversity_beta = config["training"].get("diversity_beta", 0.01)
    ridge_alpha = config["training"].get("ridge_alpha", 1.1)

    # Only apply diversity reg to separate MLP models
    apply_diversity = use_diversity_reg and not fe.shared_trunk

    optimizer = optim.AdamW(fe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    losses = []

    name = "MFE" if is_matryoshka else "FE"
    if apply_diversity:
        name += "+Div"
    pbar = tqdm(range(n_epochs), desc=f"Training {name}")

    for epoch in pbar:
        n_systems = 32
        params = dynamics.sample_params(n_systems, param_range)

        total_loss = 0.0
        for i in range(n_systems):
            mu = torch.tensor(params[i].item())

            # Sample disjoint context and target sets
            x_ctx = dynamics.sample_states(n_ctx, state_range=(-3, 3))
            x_tgt = dynamics.sample_states(n_tgt, state_range=(-3, 3))
            dx_ctx = dynamics.dx(x_ctx, mu)
            dx_tgt = dynamics.dx(x_tgt, mu)

            # Second context set for coefficient consistency
            x_ctx2 = dynamics.sample_states(n_ctx, state_range=(-3, 3))
            dx_ctx2 = dynamics.dx(x_ctx2, mu)

            if is_matryoshka and hasattr(fe, 'nesting_dims'):
                # Matryoshka nested loss with context/target split
                loss = 0.0
                for k in fe.nesting_dims:
                    # Compute coeffs on context, evaluate on target
                    # Use weighted ridge if diversity reg is enabled
                    alpha = ridge_alpha if apply_diversity else None
                    c1 = fe.compute_coefficients(x_ctx, dx_ctx, k=k, ridge_alpha=alpha)
                    dx_pred = fe.predict_dx(x_tgt, c1, k=k)
                    loss_recon = ((dx_pred - dx_tgt) ** 2).mean()

                    # Coefficient consistency via cross-prediction
                    dx_cross = fe.predict_dx(x_ctx2, c1, k=k)
                    loss_cons = ((dx_cross - dx_ctx2) ** 2).mean()

                    loss_k = loss_recon + lambda_cons * loss_cons

                    # Add diversity penalty if enabled
                    if apply_diversity:
                        loss_div = compute_diversity_loss(fe, x_ctx, k)
                        loss_k = loss_k + diversity_beta * loss_div

                    loss = loss + loss_k
                loss = loss / len(fe.nesting_dims)
            else:
                # Standard FE with context/target split
                k = fe.num_basis
                alpha = ridge_alpha if apply_diversity else None
                c1 = fe.compute_coefficients(x_ctx, dx_ctx, ridge_alpha=alpha)
                dx_pred = fe.predict_dx(x_tgt, c1)
                loss_recon = ((dx_pred - dx_tgt) ** 2).mean()

                # Coefficient consistency via cross-prediction
                dx_cross = fe.predict_dx(x_ctx2, c1)
                loss_cons = ((dx_cross - dx_ctx2) ** 2).mean()

                loss = loss_recon + lambda_cons * loss_cons

                # Add diversity penalty if enabled
                if apply_diversity:
                    loss_div = compute_diversity_loss(fe, x_ctx, k)
                    loss = loss + diversity_beta * loss_div

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


def train_width_matryoshka(
    fe: WidthMatryoshkaFE,
    dynamics: DynamicsBase,
    config: Dict[str, Any],
) -> List[float]:
    """
    Train Width Masking Matryoshka FE with coupled basis-width truncation.

    At each nesting level k, uses width_schedule[k] hidden units.
    This couples capacity to basis count, forcing coarse-to-fine learning.

    Args:
        fe: Width Matryoshka Function Encoder
        dynamics: Dynamical system
        config: Training configuration

    Returns:
        List of training losses
    """
    n_epochs = config["training"]["fe_epochs"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    param_range = tuple(config["dynamics"]["mu_range"])

    optimizer = optim.AdamW(fe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    losses = []

    pbar = tqdm(range(n_epochs), desc="Training WidthMFE")

    for epoch in pbar:
        n_systems, n_samples = 32, 100
        params = dynamics.sample_params(n_systems, param_range)

        total_loss = 0.0
        for i in range(n_systems):
            param = params[i].item()
            x_train = dynamics.sample_states(n_samples, state_range=(-3, 3))
            dx_true = dynamics.dx(x_train, torch.tensor(param))

            # Coupled basis-width Matryoshka loss
            loss = 0.0
            for k in fe.nesting_dims:
                width_k = fe.width_schedule[k]  # Get coupled width

                coeffs_k = fe.compute_coefficients(
                    x_train, dx_true, k=k, active_width=width_k
                )
                dx_pred = fe.predict_dx(
                    x_train, coeffs_k, k=k, active_width=width_k
                )
                loss = loss + ((dx_pred - dx_true) ** 2).mean()

            loss = loss / len(fe.nesting_dims)
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


def train_grouped_hierarchical(
    fe: GroupedHierarchicalFE,
    dynamics: DynamicsBase,
    config: Dict[str, Any],
) -> List[float]:
    """
    Train Grouped Hierarchical Matryoshka FE with block-level nesting.

    Uses meta-style training with context/target split.
    Optionally adds diversity regularization + weighted ridge.

    Args:
        fe: Grouped Hierarchical Function Encoder
        dynamics: Dynamical system
        config: Training configuration

    Returns:
        List of training losses
    """
    n_epochs = config["training"]["fe_epochs"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    param_range = tuple(config["dynamics"]["mu_range"])

    # Meta-style training hyperparameters
    n_ctx = 50      # Context set size (matches DPC inference)
    n_tgt = 100     # Target set size (larger for stable gradients)
    lambda_cons = 0.1  # Coefficient consistency weight

    # Option 2: Diversity regularization
    use_diversity_reg = config["training"].get("use_diversity_reg", False)
    diversity_beta = config["training"].get("diversity_beta", 0.01)
    ridge_alpha = config["training"].get("ridge_alpha", 1.1)

    optimizer = optim.AdamW(fe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    losses = []

    name = "GMFE+Div" if use_diversity_reg else "GMFE"
    pbar = tqdm(range(n_epochs), desc=f"Training {name}")

    for epoch in pbar:
        n_systems = 32
        params = dynamics.sample_params(n_systems, param_range)

        total_loss = 0.0
        for i in range(n_systems):
            mu = torch.tensor(params[i].item())

            # Sample disjoint context and target sets
            x_ctx = dynamics.sample_states(n_ctx, state_range=(-3, 3))
            x_tgt = dynamics.sample_states(n_tgt, state_range=(-3, 3))
            dx_ctx = dynamics.dx(x_ctx, mu)
            dx_tgt = dynamics.dx(x_tgt, mu)

            # Second context set for coefficient consistency
            x_ctx2 = dynamics.sample_states(n_ctx, state_range=(-3, 3))
            dx_ctx2 = dynamics.dx(x_ctx2, mu)

            # Block-level Matryoshka loss with context/target split
            loss = 0.0
            for k in fe.nesting_dims:  # [4, 8, 12, 16]
                # Compute coeffs on context, evaluate on target
                # Use weighted ridge if diversity reg is enabled
                alpha = ridge_alpha if use_diversity_reg else None
                c1 = fe.compute_coefficients(x_ctx, dx_ctx, k=k, ridge_alpha=alpha)
                dx_pred = fe.predict_dx(x_tgt, c1, k=k)
                loss_recon = ((dx_pred - dx_tgt) ** 2).mean()

                # Coefficient consistency via cross-prediction
                dx_cross = fe.predict_dx(x_ctx2, c1, k=k)
                loss_cons = ((dx_cross - dx_ctx2) ** 2).mean()

                loss_k = loss_recon + lambda_cons * loss_cons

                # Add diversity penalty if enabled
                if use_diversity_reg:
                    loss_div = compute_diversity_loss(fe, x_ctx, k)
                    loss_k = loss_k + diversity_beta * loss_div

                loss = loss + loss_k

            loss = loss / len(fe.nesting_dims)
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


def train_dpc(
    fe: FunctionEncoder,
    policy: DPCPolicy,
    dynamics: DynamicsBase,
    config: Dict[str, Any],
    name: str = "DPC",
) -> List[float]:
    """
    Train DPC policy using Function Encoder for system identification.

    Args:
        fe: Trained Function Encoder
        policy: DPC policy to train
        dynamics: Dynamical system
        config: Training configuration
        name: Name for progress bar

    Returns:
        List of training losses
    """
    n_epochs = config["training"]["dpc_epochs"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    param_range = tuple(config["dynamics"]["mu_range"])

    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    losses = []

    horizon, n_samples, n_obs = 50, 128, 50
    pbar = tqdm(range(n_epochs), desc=name)

    for epoch in pbar:
        # Sample initial conditions and system parameters
        x0 = dynamics.sample_initial_conditions(n_samples)
        ref = torch.zeros(n_samples, dynamics.state_dim)
        params = dynamics.sample_params(n_samples, param_range)

        # Compute FE coefficients for each system
        coeffs_batch = []
        for i in range(n_samples):
            x_obs = dynamics.sample_states(n_obs, state_range=(-3, 3))
            dx_obs = dynamics.dx(x_obs, params[i:i+1])
            with torch.no_grad():
                c = fe.compute_coefficients(x_obs, dx_obs)
            coeffs_batch.append(c)
        coeffs = torch.stack(coeffs_batch)

        # Rollout with true dynamics
        x_traj = [x0]
        x = x0
        for t in range(horizon):
            u = policy(x, ref, coeffs)
            x = dynamics.rk4_step(x, params.unsqueeze(-1))
            # Add control input (affects second state component)
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
