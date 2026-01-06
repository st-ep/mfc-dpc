"""Differentiable Predictive Control policy."""
import torch
import torch.nn as nn
from typing import List


class DPCPolicy(nn.Module):
    """
    Differentiable Predictive Control policy.

    Maps (state, reference, FE coefficients) -> control action.
    """

    def __init__(
        self,
        state_dim: int = 2,
        coeff_dim: int = 16,
        hidden_dims: List[int] = None,
        u_bound: float = 3.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.state_dim = state_dim
        self.u_bound = u_bound
        self.coeff_dim = coeff_dim

        layers = []
        prev_dim = state_dim + state_dim + coeff_dim  # state + ref + coeffs
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.GELU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        # Small initialization for stability
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        ref: torch.Tensor,
        coeffs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute control action."""
        inp = torch.cat([x, ref, coeffs], dim=-1)
        return self.u_bound * torch.tanh(self.net(inp))
