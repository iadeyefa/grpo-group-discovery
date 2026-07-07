"""Build per-country feature vectors from opinion distributions."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def build_country_feature_matrix(
    df: pd.DataFrame,
    method: str = "mean_opinion_vector",
    normalize_vectors: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Aggregate question-level opinion vectors into one feature vector per country.

    Returns:
        features_df: index=country, columns=feature_0..feature_n
        feature_names: ordered country list (index order)
    """
    if method != "mean_opinion_vector":
        raise ValueError(f"Unsupported feature method: {method}")

    # Pad opinion vectors to max option count per question before averaging.
    max_len = int(df["prob_y"].apply(len).max())
    logger.debug("Max option count across questions: %d", max_len)

    country_vectors: dict[str, list[np.ndarray]] = {}
    for country, group in df.groupby("group"):
        vectors = []
        for prob_y in group["prob_y"]:
            arr = np.asarray(prob_y, dtype=np.float64)
            if len(arr) < max_len:
                arr = np.pad(arr, (0, max_len - len(arr)))
            vectors.append(arr)
        country_vectors[country] = vectors

    countries = sorted(country_vectors.keys())
    matrix = np.stack([np.mean(country_vectors[c], axis=0) for c in countries])

    if normalize_vectors:
        matrix = normalize(matrix, norm="l2", axis=1)
        logger.debug("L2-normalized country feature matrix")

    columns = [f"feature_{i}" for i in range(matrix.shape[1])]
    features_df = pd.DataFrame(matrix, index=countries, columns=columns)
    logger.info("Built feature matrix: %d countries x %d dims", *matrix.shape)
    return features_df, countries
