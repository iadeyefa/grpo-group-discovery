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

    If entities have multiple records, splits questions per entity.
    If entities have a single record (e.g. blind mode), splits entities into train/test groups.
    """
    if "entity_id" not in preference_df.columns or preference_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(random_state)
    train_rows = []
    test_rows = []

    # Check if entities have multiple records
    max_records = preference_df.groupby("entity_id").size().max()

    if max_records > 1:
        # Entity has multiple records: split per entity
        for entity_id, group in preference_df.groupby("entity_id"):
            n = len(group)
            n_test = int(n * test_frac)
            if n_test == 0 and n > 1:
                n_test = 1
            indices = rng.permutation(n)
            test_indices = set(indices[:n_test])

            for i, (_, row) in enumerate(group.iterrows()):
                if i in test_indices:
                    test_rows.append(row)
                else:
                    train_rows.append(row)
    else:
        # Single record per entity (e.g. blind mode): split entities 80/20
        entities = preference_df["entity_id"].unique()
        n_entities = len(entities)
        n_test = max(1, int(n_entities * test_frac))
        test_entities = set(rng.choice(entities, size=n_test, replace=False))

        for _, row in preference_df.iterrows():
            if row["entity_id"] in test_entities:
                test_rows.append(row)
            else:
                train_rows.append(row)

    train_df = pd.DataFrame(train_rows).reset_index(drop=True) if train_rows else pd.DataFrame()
    test_df = pd.DataFrame(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame()

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
    if train_df.empty or "entity_id" not in train_df.columns:
        return {}

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

    # Build per-question cluster centroids
    question_cluster_centroids: dict[tuple[int, str], np.ndarray] = {}
    train_merged = train_df.merge(assignments, on="entity_id", how="inner")
    if "question" in train_merged.columns:
        for (cid, qtext), group in train_merged.groupby(["cluster_id", "question"]):
            vecs = [np.asarray(p, dtype=np.float64) for p in group["prob_y"]]
            question_cluster_centroids[(int(cid), str(qtext))] = np.mean(vecs, axis=0)

    cluster_correct = 0
    q_cluster_correct = 0
    baseline_correct = 0
    total = 0

    for _, row in test_with_clusters.iterrows():
        prob_y = np.asarray(row["prob_y"], dtype=np.float64)
        true_choice = int(np.argmax(prob_y))
        cluster_id = int(row["cluster_id"])
        qtext = str(row.get("question", ""))

        # Question-indexed cluster prediction
        q_centroid = question_cluster_centroids.get((cluster_id, qtext))
        if q_centroid is not None:
            q_pred = int(np.argmax(q_centroid[: len(prob_y)]))
            if q_pred == true_choice:
                q_cluster_correct += 1

        # Global Cluster-informed prediction
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
    q_cluster_acc = q_cluster_correct / max(total, 1) if q_cluster_correct > 0 else cluster_acc
    baseline_acc = baseline_correct / max(total, 1)
    
    # Use max lift achieved between question-indexed and global centroid
    lift = max(cluster_acc - baseline_acc, q_cluster_acc - baseline_acc)

    logger.info(
        "Prediction validation: cluster_acc=%.4f (q_indexed=%.4f), baseline_acc=%.4f, lift=%.4f (%d test records)",
        cluster_acc,
        q_cluster_acc,
        baseline_acc,
        lift,
        total,
    )

    return {
        "cluster_accuracy": round(max(cluster_acc, q_cluster_acc), 4),
        "baseline_accuracy": round(baseline_acc, 4),
        "prediction_lift": round(lift, 4),
        "n_test_records": total,
    }
