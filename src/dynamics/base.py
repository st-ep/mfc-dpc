"""Abstract base class for dynamical systems."""
from abc import ABC, abstractmethod
import torch


class DynamicsBase(ABC):
    """
    Abstract base class for dynamical systems.

    Subclasses must implement:
    - dx(): compute derivative
    - rk4_step(): single RK4 integration step
    - sample_params(): sample random system parameters
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state space."""
        pass

    @property
    @abstractmethod
    def param_dim(self) -> int:
        """Dimension of the parameter space."""
        pass

    @abstractmethod
    def dx(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Compute derivative dx/dt.

        Args:
            x: State tensor [..., state_dim]
            params: System parameters

        Returns:
            dx/dt tensor [..., state_dim]
        """
        pass

    @abstractmethod
    def rk4_step(self, x: torch.Tensor, params: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        """
        Perform single RK4 integration step.

        Args:
            x: Current state [..., state_dim]
            params: System parameters
            dt: Time step

        Returns:
            Next state [..., state_dim]
        """
        pass

    @abstractmethod
    def sample_params(self, n: int, param_range: tuple) -> torch.Tensor:
        """
        Sample random system parameters.

        Args:
            n: Number of parameter sets to sample
            param_range: (min, max) tuple for parameter range

        Returns:
            Parameter tensor [n, param_dim] or [n]
        """
        pass

    @abstractmethod
    def sample_states(self, n: int, state_range: tuple = (-3, 3)) -> torch.Tensor:
        """
        Sample random states.

        Args:
            n: Number of states to sample
            state_range: (min, max) tuple for state range

        Returns:
            State tensor [n, state_dim]
        """
        pass
