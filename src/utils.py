"""Shared helpers: config loading, logging, run directories."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for CLI scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    config_path = Path(path)
    logging.debug("Loading config from %s", config_path)
    with config_path.open() as f:
        return yaml.safe_load(f)


def resolve_output_dir(config: dict[str, Any]) -> Path:
    """Build a versioned output directory for this clustering run."""
    export_cfg = config.get("export", {})
    base = Path(export_cfg.get("output_dir", "outputs"))
    run_name = export_cfg.get("run_name")
    if not run_name:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = REPO_ROOT / base / run_name
    out.mkdir(parents=True, exist_ok=True)
    logging.debug("Output directory: %s", out)
    return out


def get_cache_dir(config: dict[str, Any]) -> str | None:
    """Return HF datasets cache dir from config, if set."""
    cache = config.get("dataset", {}).get("cache_dir")
    if cache:
        return os.path.expanduser(cache)
    return None
