"""Loads configs/config.yaml. Same pattern as sagemaker_finetuning/utils/config.py
-- kept separate (not cross-imported) so this folder stays fully standalone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config.yaml, relative to the project root (tiled_detection_probe/), to an absolute Path."""
    return (PROJECT_ROOT / relative_path).resolve()
