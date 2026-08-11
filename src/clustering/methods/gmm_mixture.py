"""Game-Changing Feature 3: Gaussian Mixture Model (GMM) Soft MoE Memberships."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.data.entities import attach_entity_ids
from src.features.preference_vectors import build_preference_feature_matrix

logger = logging.getLogger(__name__)

METHOD_NAME = "gmm_mixture"


def run(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fits a Gaussian Mixture Model (GMM) to preference vectors, discovering soft
    Mixture-of-Experts (MoE) memberships P(cluster_k | entity_i) for probabilistic GRPO.
    """
    cluster_cfg = config.get("clustering", {})
    n_clusters = int(cluster_cfg.get("n_clusters", 5))
    random_state = int(cluster_cfg.get("random_state", 42))

    preference_with_entities = attach_entity_ids(preference_df, entity_registry)
    
    features_df = build_preference_feature_matrix(
        preference_with_entities, method="weighted_opinion_vector", normalize_vectors=True
    )
    X = features_df.to_numpy()
    entity_ids = list(features_df.index)

    logger.info("Fitting Gaussian Mixture Model: K=%d components on %d entities", n_clusters, len(entity_ids))

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",
        random_state=random_state,
        n_init=5,
    )
    
    # Compute soft membership probabilities P(cluster_k | entity_i)
    gmm.fit(X)
    soft_probs = gmm.predict_proba(X)
    labels = np.argmax(soft_probs, axis=1)

    assignments = pd.DataFrame(
        {
            "entity_id": entity_ids,
            "cluster_id": labels.astype(int),
        }
    ).sort_values(["cluster_id", "entity_id"])

    soft_cols = [f"prob_cluster_{c}" for c in range(n_clusters)]
    soft_assignments = pd.DataFrame(soft_probs, index=entity_ids, columns=soft_cols)
    soft_assignments.index.name = "entity_id"

    extras = {
        "features_df": features_df,
        "soft_assignments_df": soft_assignments,
        "feature_dim": features_df.shape[1],
        "uses_preference_features": True,
        "discovery_method": METHOD_NAME,
        "discovery_tier": "method",
        "gmm_info": {
            "converged": bool(gmm.converged_),
            "n_iter": int(gmm.n_iter_),
            "lower_bound": float(gmm.lower_bound_),
        },
    }
    return assignments, extras
