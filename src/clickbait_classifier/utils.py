"""Utility functions for configuration management."""

from pathlib import Path

from omegaconf import OmegaConf


def save_config(cfg: OmegaConf, output_path: Path) -> None:
    """
    Save OmegaConf configuration to a YAML file.

    Args:
        cfg: OmegaConf configuration object
        output_path: Path where the config should be saved
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_path)