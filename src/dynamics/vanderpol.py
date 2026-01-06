"""Van der Pol oscillator dynamics."""
import torch
from .base import DynamicsBase


class VanDerPol(DynamicsBase):
    """
    Van der Pol oscillator.

    Dynamics:
        dx1/dt = x2
        dx2/dt = mu * (1 - x1^2) * x2 - x1

    where mu is the damping parameter.
    """

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def param_dim(self) -> int:
        return 1

    def dx(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Compute Van der Pol derivative.

        Args:
            x: State tensor [..., 2]
            params: Damping parameter mu

        Returns:
            dx/dt tensor [..., 2]
        """
        x1, x2 = x[..., 0:1], x[..., 1:2]
        dx1 = x2
        dx2 = params * (1 - x1**2) * x2 - x1
        return torch.cat([dx1, dx2], dim=-1)

    def rk4_step(self, x: torch.Tensor, params: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        """RK4 integration step for Van der Pol."""
        k1 = self.dx(x, params)
        k2 = self.dx(x + 0.5 * dt * k1, params)
        k3 = self.dx(x + 0.5 * dt * k2, params)
        k4 = self.dx(x + dt * k3, params)
        return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def sample_params(self, n: int, param_range: tuple) -> torch.Tensor:
        """Sample random mu values."""
        return torch.rand(n) * (param_range[1] - param_range[0]) + param_range[0]

    def sample_states(self, n: int, state_range: tuple = (-3, 3)) -> torch.Tensor:
        """Sample random states in the given range."""
        width = state_range[1] - state_range[0]
        return torch.rand(n, self.state_dim) * width + state_range[0]

    def sample_initial_conditions(self, n: int) -> torch.Tensor:
        """Sample initial conditions for control (x1 in [-2,2], x2 in [-3,3])."""
        x0 = torch.zeros(n, 2)
        x0[:, 0] = torch.rand(n) * 4 - 2
        x0[:, 1] = torch.rand(n) * 6 - 3
        return x0
