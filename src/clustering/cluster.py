"""Cluster countries into preference groups."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
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
    Assign each country (row) to a cluster id.

    Returns DataFrame with columns: country, cluster_id
    """
    if algorithm != "kmeans":
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    X = features_df.to_numpy()
    logger.info(
        "Running KMeans: k=%d, n_countries=%d, random_state=%d",
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
        {"country": features_df.index.tolist(), "cluster_id": labels.astype(int)}
    ).sort_values(["cluster_id", "country"])

    for cid in sorted(assignments["cluster_id"].unique()):
        members = assignments.loc[assignments["cluster_id"] == cid, "country"].tolist()
        logger.debug("Cluster %d (%d countries): %s", cid, len(members), members)

    return assignments


def cluster_sizes(assignments: pd.DataFrame) -> dict[int, int]:
    """Count countries per cluster."""
    return assignments.groupby("cluster_id").size().to_dict()


def clustering_metadata(
    config: dict[str, Any],
    assignments: pd.DataFrame,
    feature_dim: int,
) -> dict[str, Any]:
    """Serializable metadata for reproducibility."""
    return {
        "dataset": config.get("dataset", {}),
        "features": {**config.get("features", {}), "dim": feature_dim},
        "clustering": config.get("clustering", {}),
        "n_countries": int(len(assignments)),
        "cluster_sizes": cluster_sizes(assignments),
        "dataset_names": [
            f"{config['export']['dataset_prefix']}_{cid}"
            for cid in sorted(assignments["cluster_id"].unique())
        ],
    }
