"""Function Encoder and Matryoshka Function Encoder models."""
import torch
import torch.nn as nn
from typing import List, Dict, Optional


class FunctionEncoder(nn.Module):
    """
    Function Encoder for learning basis functions over dynamics.

    Learns neural network basis functions g_i(x) such that:
        dx/dt ≈ Σ c_i * g_i(x)

    Coefficients c_i are computed via least squares from observations.
    """

    def __init__(
        self,
        state_dim: int = 2,
        num_basis: int = 16,
        hidden_dim: int = 64,
        shared_trunk: bool = True,
        ridge_lambda: float = 1e-4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_basis = num_basis
        self.hidden_dim = hidden_dim
        self.shared_trunk = shared_trunk
        self.ridge_lambda = ridge_lambda

        if shared_trunk:
            # Shared trunk network with separate heads
            self.trunk = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.heads = nn.ModuleList([
                nn.Linear(hidden_dim, state_dim) for _ in range(num_basis)
            ])
        else:
            # Separate MLP for each basis function
            self.mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, state_dim),
                ) for _ in range(num_basis)
            ])

    def forward_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute all basis functions at x. Returns [batch, num_basis, state_dim]."""
        if self.shared_trunk:
            features = self.trunk(x)
            return torch.stack([h(features) for h in self.heads], dim=1)
        else:
            return torch.stack([mlp(x) for mlp in self.mlps], dim=1)

    def compute_coefficients(
        self,
        x_obs: torch.Tensor,
        dx_obs: torch.Tensor,
        k: int = None,
    ) -> torch.Tensor:
        """
        Compute coefficients via ridge regression.

        Args:
            x_obs: Observation points [n_obs, state_dim]
            dx_obs: Derivatives at observation points [n_obs, state_dim]
            k: Number of basis functions to use (default: all)

        Returns:
            Coefficient vector [num_basis]
        """
        k = k or self.num_basis
        n_obs = x_obs.shape[0]

        G = self.forward_basis(x_obs)[:, :k, :]  # [n_obs, k, state_dim]
        G_flat = G.reshape(n_obs * self.state_dim, k)
        dx_flat = dx_obs.reshape(n_obs * self.state_dim)

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

    def __init__(
        self,
        state_dim: int = 2,
        num_basis: int = 16,
        hidden_dim: int = 64,
        shared_trunk: bool = True,
        ridge_lambda: float = 1e-4,
        nesting_dims: List[int] = None,
    ):
        super().__init__(state_dim, num_basis, hidden_dim, shared_trunk, ridge_lambda)
        if nesting_dims is None:
            nesting_dims = [2, 4, 8, 16]
        self.nesting_dims = [d for d in nesting_dims if d <= num_basis]


class WidthMaskingMLP(nn.Module):
    """MLP that supports width masking for Matryoshka training."""

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, active_width: Optional[int] = None) -> torch.Tensor:
        active_width = active_width or self.hidden_dim

        h = torch.tanh(self.fc1(x))
        if active_width < self.hidden_dim:
            h = h.clone()
            h[:, active_width:] = 0  # Mask hidden units beyond active_width

        h = torch.tanh(self.fc2(h))
        if active_width < self.hidden_dim:
            h = h.clone()
            h[:, active_width:] = 0  # Mask again after second layer

        return self.fc3(h)


class WidthMatryoshkaFE(nn.Module):
    """
    Width Masking Matryoshka Function Encoder.

    Couples basis truncation with hidden width truncation:
    - Fewer bases → narrower hidden layers
    - Forces importance ordering with capacity constraints

    For separate MLPs only - applies Matryoshka principle within each MLP's width.
    """

    def __init__(
        self,
        state_dim: int = 2,
        num_basis: int = 16,
        hidden_dim: int = 64,
        ridge_lambda: float = 1e-4,
        nesting_dims: Optional[List[int]] = None,
        width_schedule: Optional[Dict[int, int]] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_basis = num_basis
        self.hidden_dim = hidden_dim
        self.ridge_lambda = ridge_lambda
        self.shared_trunk = False  # Always separate MLPs

        if nesting_dims is None:
            nesting_dims = [2, 4, 8, 16]
        self.nesting_dims = [d for d in nesting_dims if d <= num_basis]

        # Default: width grows with basis count
        if width_schedule is None:
            width_schedule = {2: 16, 4: 32, 8: 48, 16: 64}
        self.width_schedule = width_schedule

        # Width-maskable MLPs
        self.mlps = nn.ModuleList([
            WidthMaskingMLP(state_dim, hidden_dim) for _ in range(num_basis)
        ])

    def forward_basis(
        self,
        x: torch.Tensor,
        active_width: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute basis functions with optional width masking."""
        active_width = active_width or self.hidden_dim
        return torch.stack([mlp(x, active_width) for mlp in self.mlps], dim=1)

    def compute_coefficients(
        self,
        x_obs: torch.Tensor,
        dx_obs: torch.Tensor,
        k: Optional[int] = None,
        active_width: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute coefficients via ridge regression with width masking."""
        k = k or self.num_basis
        active_width = active_width or self.hidden_dim
        n_obs = x_obs.shape[0]

        G = self.forward_basis(x_obs, active_width)[:, :k, :]  # [n_obs, k, state_dim]
        G_flat = G.reshape(n_obs * self.state_dim, k)
        dx_flat = dx_obs.reshape(n_obs * self.state_dim)

        # Ridge regression: (G'G + λI)^-1 G'y
        GtG = G_flat.T @ G_flat / n_obs + self.ridge_lambda * torch.eye(k, device=G.device)
        Gtdx = G_flat.T @ dx_flat / n_obs
        coeffs_k = torch.linalg.solve(GtG, Gtdx)

        # Pad with zeros for consistent interface
        coeffs = torch.zeros(self.num_basis, device=coeffs_k.device)
        coeffs[:k] = coeffs_k
        return coeffs

    def predict_dx(
        self,
        x: torch.Tensor,
        coeffs: torch.Tensor,
        k: Optional[int] = None,
        active_width: Optional[int] = None,
    ) -> torch.Tensor:
        """Predict derivative with width masking."""
        k = k or self.num_basis
        active_width = active_width or self.hidden_dim
        G = self.forward_basis(x, active_width)[:, :k, :]
        return torch.einsum('bnk,n->bk', G, coeffs[:k])


class SharedTrunkBlock(nn.Module):
    """
    Independent shared-trunk block producing multiple basis functions.

    Each block has its own trunk (shared within the block) and multiple heads.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_outputs: int = 4,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.state_dim = state_dim

        # Shared trunk within this block
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Separate heads for each basis function
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, state_dim) for _ in range(num_outputs)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: State input [batch, state_dim]

        Returns:
            outputs: Basis function outputs [batch, num_outputs, state_dim]
        """
        features = self.trunk(x)
        outputs = torch.stack([head(features) for head in self.heads], dim=1)
        return outputs


class GroupedHierarchicalFE(nn.Module):
    """
    Grouped Hierarchical Matryoshka Function Encoder.

    Architecture: 4 INDEPENDENT shared-trunk blocks, each producing 4 basis functions.
        Block_1: x → trunk_1 → [g_1, g_2, g_3, g_4]     (most important)
        Block_2: x → trunk_2 → [g_5, g_6, g_7, g_8]
        Block_3: x → trunk_3 → [g_9, g_10, g_11, g_12]
        Block_4: x → trunk_4 → [g_13, g_14, g_15, g_16] (least important)

    Key difference from Cascaded: NO hidden feature passing between blocks.
    Each block learns independently, but Block 1 gets 4x training signal
    (used at k=4,8,12,16) creating importance ordering at block level.

    Benefits:
    - No conflicting objectives (no h passing)
    - Each block benefits from shared trunk internally
    - Block-level Matryoshka creates importance ordering
    """

    def __init__(
        self,
        state_dim: int = 2,
        num_basis: int = 16,
        hidden_dim: int = 64,
        ridge_lambda: float = 1e-4,
        num_blocks: int = 4,
        bases_per_block: int = 4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_basis = num_basis
        self.hidden_dim = hidden_dim
        self.ridge_lambda = ridge_lambda
        self.num_blocks = num_blocks
        self.bases_per_block = bases_per_block
        self.shared_trunk = False  # For compatibility

        # Block-level nesting dims: [4, 8, 12, 16] for 4 blocks with 4 bases each
        self.nesting_dims = [
            (i + 1) * bases_per_block
            for i in range(num_blocks)
            if (i + 1) * bases_per_block <= num_basis
        ]
        # Fine-grained nesting within Block 1: [1, 2, 3, 4]
        self.fine_nesting_dims = list(range(1, bases_per_block + 1))

        # Create independent blocks (no cascading)
        self.blocks = nn.ModuleList([
            SharedTrunkBlock(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                num_outputs=bases_per_block,
            )
            for _ in range(num_blocks)
        ])

    def forward_basis(
        self,
        x: torch.Tensor,
        num_blocks: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute basis functions using first num_blocks blocks.

        Args:
            x: State input [batch, state_dim]
            num_blocks: Number of blocks to use (default: all)

        Returns:
            Basis outputs [batch, k, state_dim] where k = num_blocks * bases_per_block
        """
        num_blocks = num_blocks or self.num_blocks
        all_outputs = []

        for i in range(num_blocks):
            outputs = self.blocks[i](x)  # [batch, 4, state_dim]
            all_outputs.append(outputs)

        return torch.cat(all_outputs, dim=1)

    def compute_coefficients(
        self,
        x_obs: torch.Tensor,
        dx_obs: torch.Tensor,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute coefficients via ridge regression.

        Args:
            x_obs: Observation points [n_obs, state_dim]
            dx_obs: Derivatives at observation points [n_obs, state_dim]
            k: Number of basis functions to use (default: all)

        Returns:
            Coefficient vector [num_basis]
        """
        k = k or self.num_basis
        num_blocks = (k + self.bases_per_block - 1) // self.bases_per_block
        num_blocks = min(num_blocks, self.num_blocks)
        n_obs = x_obs.shape[0]

        G = self.forward_basis(x_obs, num_blocks)[:, :k, :]
        G_flat = G.reshape(n_obs * self.state_dim, k)
        dx_flat = dx_obs.reshape(n_obs * self.state_dim)

        # Ridge regression: (G'G + λI)^-1 G'y
        GtG = G_flat.T @ G_flat / n_obs + self.ridge_lambda * torch.eye(k, device=G.device)
        Gtdx = G_flat.T @ dx_flat / n_obs
        coeffs_k = torch.linalg.solve(GtG, Gtdx)

        # Pad with zeros for consistent interface
        coeffs = torch.zeros(self.num_basis, device=coeffs_k.device)
        coeffs[:k] = coeffs_k
        return coeffs

    def predict_dx(
        self,
        x: torch.Tensor,
        coeffs: torch.Tensor,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """Predict derivative using basis functions and coefficients."""
        k = k or self.num_basis
        num_blocks = (k + self.bases_per_block - 1) // self.bases_per_block
        num_blocks = min(num_blocks, self.num_blocks)

        G = self.forward_basis(x, num_blocks)[:, :k, :]
        return torch.einsum('bnk,n->bk', G, coeffs[:k])
