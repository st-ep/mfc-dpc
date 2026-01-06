"""Visualization utilities."""
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Any

from ..models import FunctionEncoder, DPCPolicy
from ..dynamics.base import DynamicsBase
from ..training.evaluation import get_trajectory


# Color scheme for variants
COLORS = {
    'fe_shared': 'blue',
    'fe_sep': 'cyan',
    'mfe_shared': 'orange',
    'mfe_sep': 'red',
    'gmfe_grouped': 'green',
}

LABELS = {
    'fe_shared': 'FE-Shared',
    'fe_sep': 'FE-Sep',
    'mfe_shared': 'MFE-Shared',
    'mfe_sep': 'MFE-Sep',
    'gmfe_grouped': 'GMFE',
}


def plot_results(
    results: Dict[str, Dict[float, float]],
    fe_losses: Dict[str, List[float]],
    dpc_losses: Dict[str, List[float]],
    models: Dict[str, FunctionEncoder],
    policies: Dict[str, DPCPolicy],
    dynamics: DynamicsBase,
    test_mus: List[float],
    output_path: str,
) -> None:
    """
    Plot training curves and performance comparison.

    Args:
        results: Evaluation results {variant: {mu: error}}
        fe_losses: FE training losses {variant: [losses]}
        dpc_losses: DPC training losses {variant: [losses]}
        models: Trained FE models
        policies: Trained DPC policies
        dynamics: Dynamical system
        test_mus: List of test mu values
        output_path: Path to save figure
    """
    variant_names = list(results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Training curves - FE
    ax = axes[0, 0]
    for name in variant_names:
        ax.semilogy(fe_losses[name], color=COLORS[name], alpha=0.7, label=LABELS[name])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('FE Training')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Training curves - DPC
    ax = axes[0, 1]
    for name in variant_names:
        ax.plot(dpc_losses[name], color=COLORS[name], alpha=0.7, label=LABELS[name])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('DPC Training')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Performance comparison - bar chart
    ax = axes[0, 2]
    x_pos = np.arange(len(test_mus))
    n_variants = len(variant_names)
    width = 0.8 / n_variants  # Adjust width based on number of variants
    for i, name in enumerate(variant_names):
        offset = (i - (n_variants - 1) / 2) * width
        ax.bar(x_pos + offset, [results[name][m] for m in test_mus], width,
               label=LABELS[name], color=COLORS[name], alpha=0.7)
    ax.axvline(x=4.5, color='gray', linestyle='--', lw=2, label='OOD boundary')
    ax.set_xlabel('μ')
    ax.set_ylabel('Final State Error')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{m:.1f}' for m in test_mus])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Trajectory plots for 3 different mu values
    x0 = torch.tensor([[2.0, 0.0]])
    for idx, mu in enumerate([0.5, 1.5, 3.0]):
        ax = axes[1, idx]

        for name in variant_names:
            traj = get_trajectory(models[name], policies[name], dynamics, mu, x0, horizon=100)
            ax.plot(traj[:, 0], traj[:, 1], color=COLORS[name], lw=2, alpha=0.8, label=LABELS[name])
            ax.plot(traj[-1, 0], traj[-1, 1], 's', color=COLORS[name], ms=6)

        ax.plot(2.0, 0.0, 'ko', ms=10, label='Start')
        ax.plot(0, 0, 'r*', ms=15, label='Target')

        ood_str = " (OOD)" if mu > 2.5 else ""
        ax.set_title(f'μ = {mu:.1f}{ood_str}')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.legend(fontsize=6, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle('5-Way Comparison: FE vs MFE × Shared/Separate + GMFE\n'
                 'Van der Pol Oscillator Control',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_trajectories(
    models: Dict[str, FunctionEncoder],
    policies: Dict[str, DPCPolicy],
    dynamics: DynamicsBase,
    test_mus: List[float],
    output_path: str,
) -> None:
    """
    Plot detailed trajectory comparisons.

    Args:
        models: Trained FE models
        policies: Trained DPC policies
        dynamics: Dynamical system
        test_mus: List of mu values to plot
        output_path: Path to save figure
    """
    variant_names = list(models.keys())
    n_mus = len(test_mus)
    n_cols = 3
    n_rows = (n_mus + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_mus == 1 else axes

    x0 = torch.tensor([[2.0, 0.0]])

    for idx, mu in enumerate(test_mus):
        ax = axes[idx]

        for name in variant_names:
            traj = get_trajectory(models[name], policies[name], dynamics, mu, x0, horizon=100)
            ax.plot(traj[:, 0], traj[:, 1], color=COLORS[name], lw=2, alpha=0.8, label=LABELS[name])
            ax.plot(traj[-1, 0], traj[-1, 1], 's', color=COLORS[name], ms=6)

        ax.plot(2.0, 0.0, 'ko', ms=10, label='Start')
        ax.plot(0, 0, 'r*', ms=15, label='Target')

        ood_str = " (OOD)" if mu > 2.5 else ""
        ax.set_title(f'μ = {mu:.1f}{ood_str}')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.legend(fontsize=6, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    # Hide unused axes
    for idx in range(len(test_mus), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('5-Way Trajectory Comparison (100 steps)\n'
                 'Squares = final position', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
