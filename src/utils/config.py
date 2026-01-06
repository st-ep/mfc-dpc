"""Configuration loading utilities."""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Path to save YAML file
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
