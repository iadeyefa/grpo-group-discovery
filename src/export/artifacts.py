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


def export_entity_registry(entity_registry: pd.DataFrame, output_dir: Path) -> Path:
    """
    Map opaque entity_id to HF source_group label.

    Used for downstream training filters and post-hoc evaluation — not clustering.
    """
    mapping = (
        entity_registry[["entity_id", "source_group"]]
        .drop_duplicates()
        .sort_values("entity_id")
        .set_index("entity_id")["source_group"]
        .to_dict()
    )
    path = output_dir / "entity_registry.json"
    with path.open("w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    logger.info("Wrote %s (%d entities)", path, len(mapping))
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
    entity_registry: pd.DataFrame,
    dataset_prefix: str,
    output_dir: Path,
) -> Path:
    """
    Write grpo-reproduction dataset name mapping.

    Maps goqa_cluster_N -> unique HF source_group labels for entities in that cluster.
    Downstream training filters GOQA rows by these source_group values.
    """
    merged = assignments.merge(
        entity_registry[["entity_id", "source_group"]].drop_duplicates(),
        on="entity_id",
        how="left",
    )
    mapping: dict[str, list[str]] = {}
    for cluster_id, group in merged.groupby("cluster_id"):
        key = f"{dataset_prefix}_{int(cluster_id)}"
        mapping[key] = sorted(group["source_group"].unique().tolist())

    path = output_dir / "dataset_group_map.json"
    with path.open("w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    logger.info("Wrote %s (%d groups)", path, len(mapping))
    return path


def export_evaluation_source_groups(
    assignments: pd.DataFrame,
    entity_registry: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Post-hoc source_group breakdown per cluster for hidden-structure evaluation.

    Clustering never uses this; it compares discovered groups against salient
    population structure after the fact.
    """
    merged = assignments.merge(
        entity_registry[["entity_id", "source_group"]].drop_duplicates(),
        on="entity_id",
        how="left",
    )
    breakdown: dict[str, dict[str, int]] = {}
    for cluster_id, group in merged.groupby("cluster_id"):
        counts = group["source_group"].value_counts().to_dict()
        breakdown[str(int(cluster_id))] = {k: int(v) for k, v in counts.items()}

    path = output_dir / "evaluation_source_groups.json"
    with path.open("w") as f:
        json.dump(breakdown, f, indent=2, sort_keys=True)
    logger.info(
        "Wrote %s (post-hoc source_group breakdown for %d clusters)",
        path,
        len(breakdown),
    )
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
    metadata: dict[str, Any],
    output_dir: Path,
    dataset_prefix: str,
) -> dict[str, Path]:
    """Write all standard artifacts for a clustering run."""
    paths = {
        "assignments": export_assignments(assignments, output_dir),
        "entity_registry": export_entity_registry(entity_registry, output_dir),
        "metadata": export_metadata(metadata, output_dir),
        "dataset_map": export_cluster_map(
            assignments,
            entity_registry,
            dataset_prefix,
            output_dir,
        ),
        "evaluation_source_groups": export_evaluation_source_groups(
            assignments,
            entity_registry,
            output_dir,
        ),
        "plot": plot_cluster_sizes(assignments, output_dir),
    }
    return paths
