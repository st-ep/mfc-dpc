"""Evaluation functions."""
import torch
import numpy as np
from typing import Dict, Any

from ..models import FunctionEncoder, DPCPolicy
from ..dynamics.base import DynamicsBase


@torch.no_grad()
def evaluate_dpc(
    fe: FunctionEncoder,
    policy: DPCPolicy,
    dynamics: DynamicsBase,
    param: float,
    config: Dict[str, Any],
) -> np.ndarray:
    """
    Evaluate DPC performance, return final state errors.

    Args:
        fe: Trained Function Encoder
        policy: Trained DPC policy
        dynamics: Dynamical system
        param: System parameter (e.g., mu for Van der Pol)
        config: Evaluation configuration

    Returns:
        Array of final state errors
    """
    n_tests = config["evaluation"]["n_tests"]
    horizon = config["evaluation"]["horizon"]
    n_obs = 50  # Fixed for system identification

    errors = []

    for _ in range(n_tests):
        x0 = torch.zeros(1, dynamics.state_dim)
        x0[0, 0] = torch.rand(1).item() * 4 - 2
        x0[0, 1] = torch.rand(1).item() * 6 - 3
        ref = torch.zeros(1, dynamics.state_dim)

        # System identification from random observations
        x_obs = dynamics.sample_states(n_obs, state_range=(-3, 3))
        dx_obs = dynamics.dx(x_obs, torch.tensor(param))
        coeffs = fe.compute_coefficients(x_obs, dx_obs).unsqueeze(0)

        # Rollout with true dynamics
        x = x0
        for t in range(horizon):
            u = policy(x, ref, coeffs)
            x = dynamics.rk4_step(x, torch.tensor([[param]]))
            x = x + 0.05 * torch.cat([torch.zeros_like(u), u], dim=-1)

        errors.append(x[0].norm().item())

    return np.array(errors)


@torch.no_grad()
def get_trajectory(
    fe: FunctionEncoder,
    policy: DPCPolicy,
    dynamics: DynamicsBase,
    param: float,
    x0: torch.Tensor,
    horizon: int = 100,
) -> np.ndarray:
    """
    Get a single trajectory for visualization.

    Args:
        fe: Trained Function Encoder
        policy: Trained DPC policy
        dynamics: Dynamical system
        param: System parameter
        x0: Initial state [1, state_dim]
        horizon: Number of steps

    Returns:
        Trajectory array [horizon+1, state_dim]
    """
    ref = torch.zeros(1, dynamics.state_dim)

    # System identification
    x_obs = dynamics.sample_states(50, state_range=(-3, 3))
    dx_obs = dynamics.dx(x_obs, torch.tensor(param))
    coeffs = fe.compute_coefficients(x_obs, dx_obs).unsqueeze(0)

    traj = [x0.numpy().copy()]
    x = x0.clone()
    for _ in range(horizon):
        u = policy(x, ref, coeffs)
        x = dynamics.rk4_step(x, torch.tensor([[param]]))
        x = x + 0.05 * torch.cat([torch.zeros_like(u), u], dim=-1)
        traj.append(x.numpy().copy())

    return np.array(traj).squeeze()
