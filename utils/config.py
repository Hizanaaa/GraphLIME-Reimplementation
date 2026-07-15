"""
Configuration loader.
"""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
