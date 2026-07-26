"""Export cluster artifacts for grpo-reproduction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def export_assignments(
    assignments: pd.DataFrame,
    preference_with_entities: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write entity -> cluster_id mapping in Parquet and CSV formats."""
    # Merge source_group if present
    cols = ["entity_id", "cluster_id"]
    if "source_group" in preference_with_entities.columns:
        entity_sg = preference_with_entities[["entity_id", "source_group"]].drop_duplicates()
        merged_assignments = assignments.merge(entity_sg, on="entity_id", how="left")
        cols.append("source_group")
    else:
        merged_assignments = assignments.copy()

    parquet_path = output_dir / "cluster_assignments.parquet"
    merged_assignments[["entity_id", "cluster_id"]].to_parquet(parquet_path, index=False)

    csv_path = output_dir / "cluster_assignments.csv"
    merged_assignments[cols].to_csv(csv_path, index=False)

    logger.info("Wrote %s and %s (%d rows)", parquet_path.name, csv_path.name, len(assignments))
    return {"parquet": parquet_path, "csv": csv_path}


def export_cluster_records(
    assignments: pd.DataFrame,
    preference_with_entities: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write per-cluster preference records with question text and distributions."""
    merged = assignments.merge(
        preference_with_entities[["entity_id", "question", "options", "prob_y"]],
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


def export_cluster_summary(
    assignments: pd.DataFrame,
    preference_with_entities: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write human-readable CSV and Markdown summary of cluster sizes and composition."""
    total_entities = len(assignments)

    if "source_group" in preference_with_entities.columns:
        entity_sg = preference_with_entities[["entity_id", "source_group"]].drop_duplicates()
        merged = assignments.merge(entity_sg, on="entity_id", how="left")
    else:
        merged = assignments.copy()
        merged["source_group"] = "unknown"

    summary_rows = []
    for cluster_id, group in merged.groupby("cluster_id"):
        count = len(group)
        pct = (count / total_entities) * 100.0

        # Find top source groups
        sg_counts = group["source_group"].value_counts()
        top_sgs = [
            f"{sg} ({cnt}, {cnt/count*100:.1f}%)"
            for sg, cnt in sg_counts.head(5).items()
        ]
        top_sg_str = "; ".join(top_sgs)

        summary_rows.append(
            {
                "cluster_id": int(cluster_id),
                "entity_count": count,
                "percentage": round(pct, 2),
                "top_source_groups": top_sg_str,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("cluster_id")

    csv_path = output_dir / "cluster_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    md_path = output_dir / "cluster_summary.md"
    with md_path.open("w") as f:
        f.write("# Cluster Discovery Summary\n\n")
        f.write(f"**Total Entities**: {total_entities}\n\n")
        f.write("| Cluster ID | Entity Count | Percentage | Top Source Groups |\n")
        f.write("| --- | --- | --- | --- |\n")
        for row in summary_df.itertuples(index=False):
            f.write(
                f"| {row.cluster_id} | {row.entity_count} | {row.percentage:.1f}% | {row.top_source_groups} |\n"
            )

    logger.info("Wrote %s and %s", csv_path.name, md_path.name)
    return {"csv": csv_path, "md": md_path}


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
    preference_with_entities: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write all standard artifacts for a clustering run."""
    assignment_paths = export_assignments(assignments, preference_with_entities, output_dir)
    summary_paths = export_cluster_summary(assignments, preference_with_entities, output_dir)

    paths = {
        "assignments": assignment_paths["parquet"],
        "assignments_csv": assignment_paths["csv"],
        "cluster_records": export_cluster_records(assignments, preference_with_entities, output_dir),
        "cluster_summary_csv": summary_paths["csv"],
        "cluster_summary_md": summary_paths["md"],
        "metadata": export_metadata(metadata, output_dir),
        "plot": plot_cluster_sizes(assignments, output_dir),
    }
    return paths
