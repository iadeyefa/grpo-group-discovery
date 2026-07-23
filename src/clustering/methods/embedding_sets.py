"""Method 2: Chosen-rejected embedding sets (Fain)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

from src.data.entities import attach_entity_ids

logger = logging.getLogger(__name__)

METHOD_NAME = "embedding_sets"


def _chamfer_distance(set_a: np.ndarray, set_b: np.ndarray) -> float:
    """
    Computes symmetric Chamfer distance between two multisets of vectors set_a (M_A x D) and set_b (M_B x D).
    Uses fast C-level cdist with squared euclidean distances.
    """
    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0

    # Fast pairwise squared Euclidean distance matrix via C-level cdist
    sq_dist = cdist(set_a, set_b, metric="sqeuclidean")

    # Min distance for each point in set_a to set_b (sqrt only on min values)
    a_to_b = float(np.mean(np.sqrt(np.min(sq_dist, axis=1))))

    # Min distance for each point in set_b to set_a (sqrt only on min values)
    b_to_a = float(np.mean(np.sqrt(np.min(sq_dist, axis=0))))

    return a_to_b + b_to_a


def run(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Per entity: multiset of difference vectors delta_v = embed(chosen) - embed(rejected).
    Distance = Chamfer distance between multisets; clusters using Agglomerative Clustering.
    """
    cluster_cfg = config.get("clustering", {})
    n_clusters = int(cluster_cfg.get("n_clusters", 5))
    random_state = int(cluster_cfg.get("random_state", 42))
    linkage = str(cluster_cfg.get("linkage", "average")).lower()

    logger.info("Attaching entity IDs for %s discovery", METHOD_NAME)
    preference_with_entities = attach_entity_ids(preference_df, entity_registry)

    # Find max options count across questions for padding difference vectors
    max_options = int(preference_with_entities["prob_y"].apply(len).max())
    logger.debug("Max options count for vector padding: %d", max_options)

    # Construct multiset of difference vectors delta_v per entity
    entity_sets: dict[str, list[np.ndarray]] = {}

    for entity_id, group in preference_with_entities.groupby("entity_id"):
        diff_vectors: list[np.ndarray] = []
        for row in group.itertuples(index=False):
            prob_y = np.asarray(row.prob_y, dtype=np.float64)
            n_opts = len(prob_y)
            if n_opts <= 1:
                continue

            chosen_idx = int(np.argmax(prob_y))

            # Chosen option vector: 1.0 at chosen_idx
            v_chosen = np.zeros(n_opts, dtype=np.float64)
            v_chosen[chosen_idx] = 1.0

            # Rejected options vector: uniform probability across non-chosen options
            v_rejected = np.ones(n_opts, dtype=np.float64) / (n_opts - 1)
            v_rejected[chosen_idx] = 0.0

            # Difference vector delta_v = v_chosen - v_rejected
            delta_v = v_chosen - v_rejected

            # Pad to max_options so vectors across all questions share compatible dimensions
            if n_opts < max_options:
                delta_v = np.pad(delta_v, (0, max_options - n_opts))

            diff_vectors.append(delta_v)

        entity_sets[entity_id] = diff_vectors

    entity_ids = sorted(entity_sets.keys())
    n_entities = len(entity_ids)
    if n_entities == 0:
        raise ValueError("No entities available for embedding sets discovery")
    if n_clusters > n_entities:
        raise ValueError(
            f"n_clusters={n_clusters} cannot exceed number of entities={n_entities}"
        )

    # Format entity sets as 2D numpy arrays
    formatted_sets: list[np.ndarray] = [
        np.array(entity_sets[eid], dtype=np.float64)
        if len(entity_sets[eid]) > 0
        else np.zeros((1, max_options), dtype=np.float64)
        for eid in entity_ids
    ]

    logger.info(
        "Building %d x %d pairwise Chamfer distance matrix (fast cdist)",
        n_entities,
        n_entities,
    )
    dist_matrix = np.zeros((n_entities, n_entities), dtype=np.float64)

    for i in range(n_entities):
        set_i = formatted_sets[i]
        for j in range(i + 1, n_entities):
            set_j = formatted_sets[j]
            d = _chamfer_distance(set_i, set_j)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    features_df = pd.DataFrame(dist_matrix, index=entity_ids, columns=entity_ids)
    features_df.index.name = "entity_id"

    logger.info(
        "Running Agglomerative Clustering on Chamfer distance matrix: K=%d, linkage=%s",
        n_clusters,
        linkage,
    )
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage=linkage,
    )
    labels = model.fit_predict(dist_matrix)

    assignments = pd.DataFrame(
        {
            "entity_id": entity_ids,
            "cluster_id": labels.astype(int),
        }
    ).sort_values(["cluster_id", "entity_id"])

    logger.info(
        "Discovered %d groups via %s (%d entities)",
        n_clusters,
        METHOD_NAME,
        len(assignments),
    )

    extras = {
        "features_df": features_df,
        "feature_dim": n_entities,
        "uses_preference_features": True,
        "discovery_method": METHOD_NAME,
        "discovery_tier": "method",
        "embedding_sets_params": {
            "linkage": linkage,
            "max_options": max_options,
        },
    }
    return assignments, extras
