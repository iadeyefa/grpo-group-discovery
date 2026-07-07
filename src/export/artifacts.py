"""Export cluster artifacts for grpo-reproduction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def export_assignments(assignments: pd.DataFrame, output_dir: Path) -> Path:
    """Write country → cluster_id mapping."""
    path = output_dir / "cluster_assignments.parquet"
    assignments.to_parquet(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(assignments))
    return path


def export_metadata(metadata: dict[str, Any], output_dir: Path) -> Path:
    """Write run metadata JSON."""
    path = output_dir / "metadata.json"
    with path.open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    logger.info("Wrote %s", path)
    return path


def export_cluster_map(
    assignments: pd.DataFrame,
    dataset_prefix: str,
    output_dir: Path,
) -> Path:
    """
    Write grpo-reproduction dataset name mapping.

    Maps goqa_cluster_0 -> [list of countries], etc.
    """
    mapping: dict[str, list[str]] = {}
    for cluster_id, group in assignments.groupby("cluster_id"):
        key = f"{dataset_prefix}_{int(cluster_id)}"
        mapping[key] = sorted(group["country"].tolist())

    path = output_dir / "dataset_group_map.json"
    with path.open("w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    logger.info("Wrote %s (%d groups)", path, len(mapping))
    return path


def plot_cluster_sizes(assignments: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart of countries per cluster."""
    sizes = assignments.groupby("cluster_id").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    sizes.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Countries per discovered cluster")
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("count")
    fig.tight_layout()
    path = output_dir / "cluster_sizes.png"
    fig.savefig(path)
    plt.close(fig)
    logger.debug("Wrote plot %s", path)
    return path


def export_all(
    assignments: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
    dataset_prefix: str,
) -> dict[str, Path]:
    """Write all standard artifacts for a clustering run."""
    paths = {
        "assignments": export_assignments(assignments, output_dir),
        "metadata": export_metadata(metadata, output_dir),
        "dataset_map": export_cluster_map(assignments, dataset_prefix, output_dir),
        "plot": plot_cluster_sizes(assignments, output_dir),
    }
    return paths
