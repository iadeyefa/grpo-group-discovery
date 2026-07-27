"""Held-out preference prediction validation for discovered clusters.

The gold-standard behavioral test: does knowing an entity's cluster assignment
improve prediction of their choices on unseen prompts?
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def train_test_split_by_entity(
    preference_df: pd.DataFrame,
    test_frac: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split preference records per entity into train/test partitions.

    For each entity, reserves ``test_frac`` of their question responses as
    held-out test data.  This ensures every entity appears in both splits.
    """
    rng = np.random.default_rng(random_state)
    train_rows = []
    test_rows = []

    for entity_id, group in preference_df.groupby("entity_id"):
        n = len(group)
        n_test = max(1, int(n * test_frac))
        indices = rng.permutation(n)
        test_indices = set(indices[:n_test])

        for i, (_, row) in enumerate(group.iterrows()):
            if i in test_indices:
                test_rows.append(row)
            else:
                train_rows.append(row)

    train_df = pd.DataFrame(train_rows).reset_index(drop=True)
    test_df = pd.DataFrame(test_rows).reset_index(drop=True)

    logger.info(
        "Train/test split: %d train rows, %d test rows (%.1f%% held out)",
        len(train_df),
        len(test_df),
        100.0 * len(test_df) / max(1, len(train_df) + len(test_df)),
    )
    return train_df, test_df


def _build_cluster_centroids(
    train_df: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict[int, np.ndarray]:
    """Build mean preference vector per cluster from training data."""
    merged = train_df.merge(assignments, on="entity_id", how="inner")

    if "prob_y" not in merged.columns:
        logger.warning("No prob_y column — cannot build centroids from aggregate data")
        return {}

    max_len = int(merged["prob_y"].apply(len).max())
    centroids: dict[int, np.ndarray] = {}

    for cluster_id, group in merged.groupby("cluster_id"):
        vectors = []
        for prob_y in group["prob_y"]:
            arr = np.asarray(prob_y, dtype=np.float64)
            if len(arr) < max_len:
                arr = np.pad(arr, (0, max_len - len(arr)))
            vectors.append(arr)
        centroids[int(cluster_id)] = np.mean(vectors, axis=0)

    return centroids


def compute_cluster_prediction_score(
    assignments: pd.DataFrame,
    preference_df: pd.DataFrame,
    test_frac: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate whether cluster assignment predicts held-out preference choices.

    For each test record:
        1. Look up the entity's cluster.
        2. Use that cluster's training centroid as a predicted distribution.
        3. Check if the centroid's argmax matches the test record's argmax.

    Compares against a naive baseline (pooled centroid — no cluster info).

    Returns dict with cluster_accuracy, baseline_accuracy, prediction_lift.
    """
    if "prob_y" not in preference_df.columns:
        logger.warning("prob_y not in preference_df — skipping prediction score")
        return {
            "cluster_accuracy": 0.0,
            "baseline_accuracy": 0.0,
            "prediction_lift": 0.0,
        }

    train_df, test_df = train_test_split_by_entity(
        preference_df, test_frac=test_frac, random_state=random_state
    )

    # Build cluster centroids from training data
    cluster_centroids = _build_cluster_centroids(train_df, assignments)
    if not cluster_centroids:
        return {
            "cluster_accuracy": 0.0,
            "baseline_accuracy": 0.0,
            "prediction_lift": 0.0,
        }

    # Build pooled baseline centroid (no cluster info)
    max_len = int(train_df["prob_y"].apply(len).max())
    all_vectors = []
    for prob_y in train_df["prob_y"]:
        arr = np.asarray(prob_y, dtype=np.float64)
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)))
        all_vectors.append(arr)
    pooled_centroid = np.mean(all_vectors, axis=0) if all_vectors else np.zeros(max_len)

    # Merge test data with assignments to get cluster_id
    test_with_clusters = test_df.merge(assignments, on="entity_id", how="inner")

    cluster_correct = 0
    baseline_correct = 0
    total = 0

    for _, row in test_with_clusters.iterrows():
        prob_y = np.asarray(row["prob_y"], dtype=np.float64)
        true_choice = int(np.argmax(prob_y))
        cluster_id = int(row["cluster_id"])

        # Cluster-informed prediction
        centroid = cluster_centroids.get(cluster_id)
        if centroid is not None:
            pred_choice = int(np.argmax(centroid[: len(prob_y)]))
            if pred_choice == true_choice:
                cluster_correct += 1

        # Baseline prediction (pooled, no cluster info)
        baseline_pred = int(np.argmax(pooled_centroid[: len(prob_y)]))
        if baseline_pred == true_choice:
            baseline_correct += 1

        total += 1

    cluster_acc = cluster_correct / max(total, 1)
    baseline_acc = baseline_correct / max(total, 1)
    lift = cluster_acc - baseline_acc

    logger.info(
        "Prediction validation: cluster_acc=%.4f, baseline_acc=%.4f, lift=%.4f (%d test records)",
        cluster_acc,
        baseline_acc,
        lift,
        total,
    )

    return {
        "cluster_accuracy": round(cluster_acc, 4),
        "baseline_accuracy": round(baseline_acc, 4),
        "prediction_lift": round(lift, 4),
        "n_test_records": total,
    }
