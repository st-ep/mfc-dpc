"""5-way comparison experiment: FE vs MFE × Shared vs Separate + GMFE."""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from ..dynamics import VanDerPol
from ..models import (
    FunctionEncoder, MatryoshkaFE, GroupedHierarchicalFE, DPCPolicy
)
from ..training import (
    train_fe, train_dpc, train_grouped_hierarchical, evaluate_dpc
)
from ..utils.logging import ExperimentLogger
from ..utils.visualization import plot_results, plot_trajectories


def get_checkpoint_path(output_dir: str, variant_name: str) -> Path:
    """Get checkpoint path for a variant."""
    return Path(output_dir) / f"{variant_name}.pt"


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    policy: DPCPolicy,
    fe_losses: list,
    dpc_losses: list,
) -> None:
    """Save model, policy, and training losses to checkpoint."""
    torch.save({
        'model_state': model.state_dict(),
        'policy_state': policy.state_dict(),
        'fe_losses': fe_losses,
        'dpc_losses': dpc_losses,
    }, path)
    print(f"  Saved checkpoint: {path}")


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    policy: DPCPolicy,
) -> tuple:
    """Load model, policy, and training losses from checkpoint."""
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    policy.load_state_dict(ckpt['policy_state'])
    return ckpt['fe_losses'], ckpt['dpc_losses']


def create_model(
    variant_type: str,
    shared_trunk: bool,
    dynamics,
    num_basis: int,
    hidden_dim: int,
    nesting_dims: list,
) -> torch.nn.Module:
    """Create model based on variant type."""
    if variant_type == "gmfe":
        # 4 blocks × 4 bases, hidden_dim=128 to match FE-Sep params
        return GroupedHierarchicalFE(
            state_dim=dynamics.state_dim,
            num_basis=num_basis,
            hidden_dim=128,
            num_blocks=4,
            bases_per_block=4,
        )
    elif variant_type == "mfe":
        return MatryoshkaFE(
            state_dim=dynamics.state_dim,
            num_basis=num_basis,
            hidden_dim=hidden_dim,
            shared_trunk=shared_trunk,
            nesting_dims=nesting_dims,
        )
    else:
        return FunctionEncoder(
            state_dim=dynamics.state_dim,
            num_basis=num_basis,
            hidden_dim=hidden_dim,
            shared_trunk=shared_trunk,
        )


def train_model(
    model: torch.nn.Module,
    variant_type: str,
    dynamics,
    config: Dict[str, Any],
) -> list:
    """Train model based on variant type."""
    if variant_type == "gmfe":
        return train_grouped_hierarchical(model, dynamics, config)
    elif variant_type == "mfe":
        return train_fe(model, dynamics, config, is_matryoshka=True)
    else:
        return train_fe(model, dynamics, config, is_matryoshka=False)


def run_comparison(config: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run 6-way comparison experiment with checkpoint support.

    Args:
        config: Experiment configuration
        config_path: Path to config file (used for output directory naming)

    Returns:
        Results dictionary
    """
    # Setup
    seed = config["seed"]
    logger = ExperimentLogger(config_path)
    logger.log_config(config)

    print("=" * 70)
    print("Matryoshka Function Encoder + DPC")
    print("=" * 70)
    print(f"\nOutput directory: {logger.output_dir}")
    print("\nComparing 5 variants:")
    print("  1. FE-Shared:    Standard FE (shared trunk) + DPC")
    print("  2. FE-Sep:       Standard FE (separate MLPs) + DPC")
    print("  3. MFE-Shared:   Matryoshka FE (shared trunk) + DPC")
    print("  4. MFE-Sep:      Matryoshka FE (separate MLPs) + DPC")
    print("  5. GMFE:         Grouped Matryoshka FE (independent blocks) + DPC")

    # Initialize dynamics
    dynamics = VanDerPol()

    # Model config
    num_basis = config["model"]["num_basis"]
    hidden_dim = config["model"]["hidden_dim"]
    nesting_dims = config["model"]["nesting_dims"]
    test_mus = config["dynamics"]["test_mus"]

    # Storage
    models = {}
    policies = {}
    fe_losses = {}
    dpc_losses = {}

    # Variant definitions: (name, variant_type, shared_trunk)
    variants = [
        ("fe_shared", "fe", True),
        ("fe_sep", "fe", False),
        ("mfe_shared", "mfe", True),
        ("mfe_sep", "mfe", False),
        ("gmfe_grouped", "gmfe", False),
    ]

    variant_labels = {
        "fe_shared": "FE-Shared",
        "fe_sep": "FE-Sep",
        "mfe_shared": "MFE-Shared",
        "mfe_sep": "MFE-Sep",
        "gmfe_grouped": "GMFE",
    }

    for name, variant_type, shared_trunk in variants:
        print("\n" + "=" * 70)
        ckpt_path = get_checkpoint_path(logger.output_dir, name)

        # Create model and policy
        torch.manual_seed(seed)
        model = create_model(
            variant_type, shared_trunk, dynamics,
            num_basis, hidden_dim, nesting_dims
        )
        policy = DPCPolicy(state_dim=dynamics.state_dim, coeff_dim=num_basis)

        if ckpt_path.exists():
            # Load from checkpoint
            print(f"Loading {variant_labels[name]} from checkpoint...")
            fe_losses[name], dpc_losses[name] = load_checkpoint(ckpt_path, model, policy)
            print(f"  Loaded: {ckpt_path}")
        else:
            # Train from scratch
            if variant_type == "gmfe":
                print(f"Training GMFE + DPC (independent blocks)")
            else:
                label = "MFE" if variant_type == "mfe" else "FE"
                trunk = "shared trunk" if shared_trunk else "separate MLPs"
                print(f"Training {label} + DPC ({trunk})")
            print("=" * 70)

            # Train FE
            fe_losses[name] = train_model(model, variant_type, dynamics, config)

            # Train DPC
            dpc_losses[name] = train_dpc(
                model, policy, dynamics, config, name=f"{name}+DPC"
            )

            # Save checkpoint
            save_checkpoint(ckpt_path, model, policy, fe_losses[name], dpc_losses[name])

        models[name] = model
        policies[name] = policy

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    variant_names = list(models.keys())
    results = {name: {} for name in variant_names}

    # Build header dynamically
    header_labels = [variant_labels[n][:8] for n in variant_names]
    header = "μ      | " + " | ".join(f"{l:>7}" for l in header_labels) + " | Best"
    print(f"\n{header}")
    print("-" * len(header))

    for mu in test_mus:
        errs = {}
        for name in variant_names:
            errs[name] = evaluate_dpc(models[name], policies[name], dynamics, mu, config).mean()
            results[name][mu] = errs[name]

        best = min(variant_names, key=lambda n: errs[n])
        ood = " (OOD)" if mu > config["dynamics"]["mu_range"][1] else ""
        row = f"{mu:.1f}{ood:6s} | " + " | ".join(f"{errs[n]:.3f}  " for n in variant_names)
        row += f" | {variant_labels[best]}"
        print(row)

    # Summary stats
    print("-" * len(header))
    avgs = {name: np.mean(list(results[name].values())) for name in variant_names}
    ood_threshold = config["dynamics"]["mu_range"][1]
    oods = {name: np.mean([results[name][m] for m in test_mus if m > ood_threshold])
            for name in variant_names}

    best_overall = min(variant_names, key=lambda n: avgs[n])
    best_ood = min(variant_names, key=lambda n: oods[n])

    row = "Overall | " + " | ".join(f"{avgs[n]:.3f}  " for n in variant_names)
    row += f" | {variant_labels[best_overall]}"
    print(row)

    row = "OOD Avg | " + " | ".join(f"{oods[n]:.3f}  " for n in variant_names)
    row += f" | {variant_labels[best_ood]}"
    print(row)

    # Improvement stats
    print(f"\n=== Improvement Analysis ===")
    if "fe_shared" in avgs and "mfe_shared" in avgs:
        print(f"MFE-Shared vs FE-Shared: "
              f"{(avgs['fe_shared'] - avgs['mfe_shared']) / avgs['fe_shared'] * 100:.1f}% improvement")
    if "fe_sep" in avgs and "mfe_sep" in avgs:
        print(f"MFE-Sep vs FE-Sep: "
              f"{(avgs['fe_sep'] - avgs['mfe_sep']) / avgs['fe_sep'] * 100:.1f}% improvement")
    if "fe_sep" in avgs and "gmfe_grouped" in avgs:
        print(f"GMFE vs FE-Sep: "
              f"{(avgs['fe_sep'] - avgs['gmfe_grouped']) / avgs['fe_sep'] * 100:.1f}% improvement")
    if "mfe_sep" in avgs and "gmfe_grouped" in avgs:
        print(f"GMFE vs MFE-Sep: "
              f"{(avgs['mfe_sep'] - avgs['gmfe_grouped']) / avgs['mfe_sep'] * 100:.1f}% improvement")

    # Plot results
    plot_results(
        results, fe_losses, dpc_losses,
        models, policies, dynamics, test_mus,
        logger.get_output_path('mfe_dpc_results.png')
    )

    # Plot detailed trajectories
    detail_mus = [mu for mu in test_mus if mu <= 3.0]
    plot_trajectories(
        models, policies, dynamics, detail_mus,
        logger.get_output_path('mfe_trajectories.png')
    )

    import matplotlib.pyplot as plt
    plt.show()

    return {
        'results': results,
        'fe_losses': fe_losses,
        'dpc_losses': dpc_losses,
        'avgs': avgs,
        'oods': oods,
    }
