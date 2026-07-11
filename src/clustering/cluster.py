"""Cluster entities by preference similarity."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def run_clustering(
    features_df: pd.DataFrame,
    algorithm: str = "kmeans",
    n_clusters: int = 5,
    random_state: int = 42,
    n_init: int = 10,
) -> pd.DataFrame:
    """
    Assign each entity to a cluster using similarity in preference feature space.

    KMeans partitions entities by distance in the L2-normalized opinion vector
    space (related to cosine similarity between preference profiles).

    Returns DataFrame with columns: entity_id, cluster_id
    """
    if algorithm != "kmeans":
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    X = features_df.to_numpy()
    logger.info(
        "Running KMeans on preference features: k=%d, n_entities=%d, random_state=%d",
        n_clusters,
        len(features_df),
        random_state,
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = model.fit_predict(X)

    assignments = pd.DataFrame(
        {
            "entity_id": features_df.index.tolist(),
            "cluster_id": labels.astype(int),
        }
    ).sort_values(["cluster_id", "entity_id"])

    for cid in sorted(assignments["cluster_id"].unique()):
        members = assignments.loc[assignments["cluster_id"] == cid, "entity_id"].tolist()
        logger.debug("Cluster %d (%d entities): %s", cid, len(members), members)

    logger.info(
        "Clustered %d entities into %d groups by preference similarity",
        len(assignments),
        n_clusters,
    )
    return assignments


def cluster_sizes(assignments: pd.DataFrame) -> dict[int, int]:
    """Count entities per cluster."""
    return assignments.groupby("cluster_id").size().to_dict()


def clustering_metadata(
    config: dict[str, Any],
    assignments: pd.DataFrame,
    feature_dim: int,
    *,
    baseline: str | None = None,
    uses_preference_features: bool = True,
) -> dict[str, Any]:
    """Serializable metadata for reproducibility."""
    metadata: dict[str, Any] = {
        "dataset": config.get("dataset", {}),
        "entities": config.get("entities", {}),
        "features": {**config.get("features", {}), "dim": feature_dim},
        "clustering": config.get("clustering", {}),
        "uses_preference_features": uses_preference_features,
        "n_entities": int(len(assignments)),
        "cluster_sizes": cluster_sizes(assignments),
        "dataset_names": [
            f"{config['export']['dataset_prefix']}_{cid}"
            for cid in sorted(assignments["cluster_id"].unique())
        ],
    }
    if baseline is not None:
        metadata["baseline"] = baseline
    return metadata
