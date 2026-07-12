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
    """Write entity → cluster_id mapping (clustering output only)."""
    path = output_dir / "cluster_assignments.parquet"
    assignments[["entity_id", "cluster_id"]].to_parquet(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(assignments))
    return path


def export_cluster_records(
    assignments: pd.DataFrame,
    entity_registry: pd.DataFrame,
    preference_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write per-cluster preference records with question text and distributions."""
    merged = assignments.merge(
        entity_registry[["entity_id"]].merge(
            preference_df[["question", "options", "prob_y"]].reset_index(drop=True),
            left_index=True,
            right_index=True,
        ),
        on="entity_id",
        how="left",
    )
    merged = merged.sort_values(["cluster_id", "entity_id"]).reset_index(drop=True)

    clusters: dict[str, list[dict[str, Any]]] = {}
    for cluster_id, group in merged.groupby("cluster_id"):
        clusters[str(int(cluster_id))] = [
            {
                "entity_id": row.entity_id,
                "question": row.question,
                "options": row.options,
                "prob_y": row.prob_y.tolist() if hasattr(row.prob_y, "tolist") else row.prob_y,
            }
            for row in group.itertuples(index=False)
        ]

    path = output_dir / "cluster_records.json"
    with path.open("w") as f:
        json.dump(clusters, f, indent=2, sort_keys=True)
    logger.info("Wrote %s (%d clusters)", path, len(clusters))
    return path


def export_metadata(metadata: dict[str, Any], output_dir: Path) -> Path:
    """Write run metadata JSON."""
    path = output_dir / "metadata.json"
    with path.open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    logger.info("Wrote %s", path)
    return path


def plot_cluster_sizes(assignments: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart of entities per cluster."""
    sizes = assignments.groupby("cluster_id").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    sizes.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Entities per discovered cluster")
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
    entity_registry: pd.DataFrame,
    preference_df: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write all standard artifacts for a clustering run."""
    paths = {
        "assignments": export_assignments(assignments, output_dir),
        "cluster_records": export_cluster_records(assignments, entity_registry, preference_df, output_dir),
        "metadata": export_metadata(metadata, output_dir),
        "plot": plot_cluster_sizes(assignments, output_dir),
    }
    return paths
