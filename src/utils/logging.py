"""Logging utilities (placeholder for future experiment tracking)."""
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentLogger:
    """
    Simple experiment logger.

    Placeholder for future integration with wandb, mlflow, etc.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize logger.

        Args:
            config_path: Path to config file. Directory named after config stem.
                        e.g., "configs/experiments/vanderpol_quick.yaml" -> "logs/vanderpol_quick/"
        """
        if config_path:
            config_name = Path(config_path).stem
            output_dir = os.path.join("logs", config_name)
        else:
            output_dir = os.path.join("logs", "default")

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        from .config import save_config
        save_config(config, os.path.join(self.output_dir, "config.yaml"))

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number
        """
        # TODO: Add wandb/mlflow integration
        pass

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """
        Log artifact (file).

        Args:
            path: Path to artifact
            name: Optional name for artifact
        """
        # TODO: Add wandb/mlflow integration
        pass

    def get_output_path(self, filename: str) -> str:
        """Get full path for output file."""
        return os.path.join(self.output_dir, filename)
