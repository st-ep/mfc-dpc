#!/usr/bin/env python3
"""
Matryoshka Function Encoder + DPC
=================================

Run experiments comparing FE vs MFE with shared vs separate MLP architectures.

Usage:
    python main.py                              # Use default config
    python main.py --config configs/default.yaml
    python main.py --config configs/experiments/vanderpol_quick.yaml
"""
import argparse
from src.utils.config import load_config
from src.experiments import run_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Matryoshka Function Encoder + DPC experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_comparison(config, config_path=args.config)


if __name__ == "__main__":
    main()
