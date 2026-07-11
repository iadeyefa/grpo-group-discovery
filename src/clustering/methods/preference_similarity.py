"""Minimum baseline: cluster entities by mean-opinion-vector similarity (KMeans)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.cluster import KMeans

from src.data.entities import attach_entity_ids
from src.features.preference_vectors import build_preference_feature_matrix

logger = logging.getLogger(__name__)

METHOD_NAME = "preference_similarity"


def run(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Floor discovery method — simplest preference-aware clustering.

    Builds per-entity mean opinion vectors, L2-normalizes, clusters with KMeans
    in cosine-related distance space.
    """
    feature_cfg = config.get("features", {})
    cluster_cfg = config.get("clustering", {})
    n_clusters = cluster_cfg.get("n_clusters", 5)
    random_state = cluster_cfg.get("random_state", 42)
    n_init = cluster_cfg.get("n_init", 10)

    logger.info("Building preference feature vectors for %s", METHOD_NAME)
    preference_with_entities = attach_entity_ids(preference_df, entity_registry)
    features_df = build_preference_feature_matrix(
        preference_with_entities,
        method=feature_cfg.get("method", "mean_opinion_vector"),
        normalize_vectors=feature_cfg.get("normalize", True),
    )

    X = features_df.to_numpy()
    logger.info(
        "Running KMeans: k=%d, n_entities=%d, random_state=%d",
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
        "Discovered %d groups via %s (%d entities)",
        n_clusters,
        METHOD_NAME,
        len(assignments),
    )

    extras = {
        "features_df": features_df,
        "feature_dim": features_df.shape[1],
        "uses_preference_features": True,
        "discovery_method": METHOD_NAME,
        "discovery_tier": "baseline",
    }
    return assignments, extras
