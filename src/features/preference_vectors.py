"""Build per-entity preference feature vectors from opinion distributions."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def build_preference_feature_matrix(
    preference_df: pd.DataFrame,
    method: str = "mean_opinion_vector",
    normalize_vectors: bool = True,
) -> pd.DataFrame:
    """
    Aggregate question-level opinion vectors into one feature vector per entity.

    Clustering algorithms use cosine/Euclidean similarity in this space to group
    entities with similar preference profiles.

    Args:
        preference_df: Must include entity_id and prob_y columns.

    Returns:
        features_df indexed by entity_id with columns feature_0..feature_n
    """
    if "entity_id" not in preference_df.columns:
        raise ValueError("preference_df must include entity_id — call attach_entity_ids first")
    if method != "mean_opinion_vector":
        raise ValueError(f"Unsupported feature method: {method}")

    max_len = int(preference_df["prob_y"].apply(len).max())
    logger.debug("Max option count across questions: %d", max_len)

    entity_vectors: dict[str, list[np.ndarray]] = {}
    for entity_id, group in preference_df.groupby("entity_id"):
        vectors = []
        for prob_y in group["prob_y"]:
            arr = np.asarray(prob_y, dtype=np.float64)
            if len(arr) < max_len:
                arr = np.pad(arr, (0, max_len - len(arr)))
            vectors.append(arr)
        entity_vectors[entity_id] = vectors

    entity_ids = sorted(entity_vectors.keys())
    matrix = np.stack([np.mean(entity_vectors[e], axis=0) for e in entity_ids])

    if normalize_vectors:
        matrix = normalize(matrix, norm="l2", axis=1)
        logger.debug("L2-normalized preference feature matrix")

    columns = [f"feature_{i}" for i in range(matrix.shape[1])]
    features_df = pd.DataFrame(matrix, index=entity_ids, columns=columns)
    features_df.index.name = "entity_id"
    logger.info(
        "Built preference feature matrix: %d entities x %d dims",
        *matrix.shape,
    )
    return features_df
