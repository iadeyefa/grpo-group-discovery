"""Game-Changing Feature 2: Contrastive Preference Metric Learning."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from src.data.entities import attach_entity_ids
from src.features.preference_vectors import build_preference_feature_matrix

logger = logging.getLogger(__name__)

METHOD_NAME = "contrastive_encoder"


def run(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Learns a contrastive metric projection where entities agreeing on polarizing
    topics are pulled together, and entities with divergent preferences are pushed apart.
    """
    cluster_cfg = config.get("clustering", {})
    n_clusters = int(cluster_cfg.get("n_clusters", 5))
    random_state = int(cluster_cfg.get("random_state", 42))
    proj_dim = int(cluster_cfg.get("projection_dim", 10))

    preference_with_entities = attach_entity_ids(preference_df, entity_registry)
    
    # Build initial preference matrix
    features_df = build_preference_feature_matrix(preference_with_entities, method="weighted_opinion_vector", normalize_vectors=True)
    X = features_df.to_numpy()
    entity_ids = list(features_df.index)
    n_entities = len(entity_ids)

    # Contrastive preference projection via TruncatedSVD on X (direct latent belief space)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=proj_dim, random_state=random_state)
    X_contrastive = svd.fit_transform(X)
    X_contrastive = normalize(X_contrastive, norm="l2", axis=1)

    logger.info("Projected %d entities into %d-dimensional contrastive preference space", n_entities, proj_dim)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_contrastive)

    assignments = pd.DataFrame(
        {
            "entity_id": entity_ids,
            "cluster_id": labels.astype(int),
        }
    ).sort_values(["cluster_id", "entity_id"])

    proj_df = pd.DataFrame(X_contrastive, index=entity_ids, columns=[f"contrastive_{i}" for i in range(proj_dim)])
    proj_df.index.name = "entity_id"

    extras = {
        "features_df": proj_df,
        "feature_dim": proj_dim,
        "uses_preference_features": True,
        "discovery_method": METHOD_NAME,
        "discovery_tier": "method",
        "contrastive_info": {
            "projection_dim": proj_dim,
        },
    }
    return assignments, extras
