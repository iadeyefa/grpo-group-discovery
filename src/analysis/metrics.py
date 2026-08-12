"""Intrinsic metrics for evaluating discovered preference groups."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

from src.features.preference_vectors import build_preference_feature_matrix

logger = logging.getLogger(__name__)


def compute_shannon_entropy(prob_vector: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Shannon entropy H(p) in bits for a probability distribution vector."""
    p = np.asarray(prob_vector, dtype=np.float64)
    p = p / (np.sum(p) + eps)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p + eps)))


def compute_entropy_reduction(
    preference_with_entities: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate entropy reduction (delta H) after group discovery.

    Measures how much choice uncertainty decreases when conditioning on discovered groups.

    Returns:
        dict containing pooled_entropy, weighted_cluster_entropy, entropy_reduction, relative_entropy_drop
    """
    merged = preference_with_entities.merge(assignments, on="entity_id", how="inner")
    if merged.empty:
        logger.warning("Empty intersection between preference records and assignments")
        return {
            "pooled_entropy": 0.0,
            "weighted_cluster_entropy": 0.0,
            "entropy_reduction": 0.0,
            "relative_entropy_drop": 0.0,
        }

    # Compute per-question entropy reduction on shared support
    qkeys = merged["qkey"].unique()
    total_entities = max(assignments["entity_id"].nunique(), 1)
    
    q_pooled_entropies = []
    q_cluster_entropies = []
    
    for qkey, q_records in merged.groupby("qkey"):
        # Pooled entropy for this question
        probs_pooled = np.mean(np.stack(q_records["prob_y"].to_numpy()), axis=0)
        h_pooled = compute_shannon_entropy(probs_pooled)
        
        # Weighted cluster entropy for this question
        h_cluster_weighted = 0.0
        q_total_entities = q_records["entity_id"].nunique()
        
        for c_id, c_q_records in q_records.groupby("cluster_id"):
            n_c = c_q_records["entity_id"].nunique()
            w_c = n_c / max(q_total_entities, 1)
            probs_c = np.mean(np.stack(c_q_records["prob_y"].to_numpy()), axis=0)
            h_c = compute_shannon_entropy(probs_c)
            h_cluster_weighted += w_c * h_c
            
        q_pooled_entropies.append(h_pooled)
        q_cluster_entropies.append(h_cluster_weighted)
        
    pooled_entropy = float(np.mean(q_pooled_entropies)) if q_pooled_entropies else 0.0
    weighted_cluster_entropy = float(np.mean(q_cluster_entropies)) if q_cluster_entropies else 0.0
    entropy_reduction = max(0.0, pooled_entropy - weighted_cluster_entropy)
    relative_drop = (
        entropy_reduction / pooled_entropy if pooled_entropy > 1e-9 else 0.0
    )

    logger.info(
        "Entropy reduction: pooled=%.4f bits, weighted_cluster=%.4f bits, delta_H=%.4f (%.2f%% drop)",
        pooled_entropy,
        weighted_cluster_entropy,
        entropy_reduction,
        relative_drop * 100.0,
    )

    return {
        "pooled_entropy": round(pooled_entropy, 4),
        "weighted_cluster_entropy": round(weighted_cluster_entropy, 4),
        "entropy_reduction": round(entropy_reduction, 4),
        "relative_entropy_drop": round(relative_drop, 4),
    }


def compute_cluster_cohesion(
    assignments: pd.DataFrame,
    preference_with_entities: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute intra-cluster consensus / cohesion metrics.

    Evaluates mean pairwise cosine similarity among preference vectors within each cluster.
    """
    features_df = build_preference_feature_matrix(
        preference_with_entities, normalize_vectors=True
    )
    merged = assignments.merge(features_df.reset_index(), on="entity_id", how="inner")

    feature_cols = [c for c in features_df.columns]
    per_cluster_cohesion: dict[int, float] = {}

    total_entities = len(assignments)
    overall_cohesion = 0.0

    for cluster_id, group in merged.groupby("cluster_id"):
        X_k = group[feature_cols].to_numpy()
        n_k = len(X_k)

        if n_k <= 1:
            mean_sim = 1.0
        else:
            # O(N) fast mean cosine similarity: sum(x_i . x_j for i != j) / (n_k * (n_k - 1))
            sum_vec = np.sum(X_k, axis=0)
            norm_sq = float(np.dot(sum_vec, sum_vec))
            row_norms_sq = float(np.sum(np.square(X_k)))
            mean_sim = float((norm_sq - row_norms_sq) / max(n_k * (n_k - 1), 1))
            mean_sim = float(np.clip(mean_sim, -1.0, 1.0))

        per_cluster_cohesion[int(cluster_id)] = round(mean_sim, 4)
        overall_cohesion += (n_k / max(total_entities, 1)) * mean_sim

    logger.info("Overall cluster cohesion (mean cosine sim): %.4f", overall_cohesion)
    return {
        "overall_cohesion": round(overall_cohesion, 4),
        "per_cluster_cohesion": per_cluster_cohesion,
    }


def compute_cluster_separation(
    assignments: pd.DataFrame,
    preference_with_entities: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute inter-cluster separation metrics: JSD matrix, Calinski-Harabasz, and Davies-Bouldin scores.
    """
    features_df = build_preference_feature_matrix(
        preference_with_entities, normalize_vectors=True
    )
    merged = assignments.merge(features_df.reset_index(), on="entity_id", how="inner")
    feature_cols = [c for c in features_df.columns]

    cluster_ids = sorted(merged["cluster_id"].unique())
    k = len(cluster_ids)

    X = merged[feature_cols].to_numpy()
    labels = merged["cluster_id"].to_numpy()

    # 1. Calinski-Harabasz & Davies-Bouldin
    if k > 1 and len(X) > k:
        try:
            ch_score = float(calinski_harabasz_score(X, labels))
            db_score = float(davies_bouldin_score(X, labels))
        except Exception as e:
            logger.warning("Error computing CH/DB scores: %s", e)
            ch_score, db_score = 0.0, 0.0
    else:
        ch_score, db_score = 0.0, 0.0

    # 2. Pairwise JSD between cluster centroids
    centroids = []
    for c_id in cluster_ids:
        c_mask = labels == c_id
        centroid = np.mean(X[c_mask], axis=0)
        # Ensure centroid is non-negative and sums to 1 for JSD calculation
        centroid = np.maximum(centroid, 0.0)
        s = np.sum(centroid)
        if s > 0:
            centroid = centroid / s
        centroids.append(centroid)

    jsd_matrix = np.zeros((k, k), dtype=np.float64)
    off_diag_values = []

    for i in range(k):
        for j in range(i + 1, k):
            dist = float(jensenshannon(centroids[i], centroids[j]))
            jsd_matrix[i, j] = dist
            jsd_matrix[j, i] = dist
            off_diag_values.append(dist)

    mean_jsd = float(np.mean(off_diag_values)) if off_diag_values else 0.0

    logger.info(
        "Cluster separation: mean JSD=%.4f, Calinski-Harabasz=%.2f, Davies-Bouldin=%.4f",
        mean_jsd,
        ch_score,
        db_score,
    )

    return {
        "mean_inter_cluster_jsd": round(mean_jsd, 4),
        "calinski_harabasz_score": round(ch_score, 2),
        "davies_bouldin_score": round(db_score, 4),
        "jsd_matrix": [[round(val, 4) for val in row] for row in jsd_matrix.tolist()],
        "cluster_ids": [int(c) for c in cluster_ids],
    }


def compute_demographic_overlay(
    assignments: pd.DataFrame,
    preference_with_entities: pd.DataFrame,
) -> dict[str, Any] | None:
    """Compute ARI/NMI between discovered clusters and hidden source_group labels.

    Low alignment is not failure — clusters may represent cross-country
    political, religious, or social attitudes.
    """
    if "source_group" not in preference_with_entities.columns:
        logger.debug("No source_group column — skipping demographic overlay")
        return None

    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    entity_sg = (
        preference_with_entities[["entity_id", "source_group"]]
        .drop_duplicates()
    )
    merged = assignments.merge(entity_sg, on="entity_id", how="inner")
    if merged.empty:
        return None

    ari = float(adjusted_rand_score(merged["source_group"], merged["cluster_id"]))
    nmi = float(adjusted_mutual_info_score(merged["source_group"], merged["cluster_id"]))

    # Country composition per cluster
    composition: dict[str, dict[str, int]] = {}
    for c_id, group in merged.groupby("cluster_id"):
        composition[str(int(c_id))] = (
            group["source_group"].value_counts().to_dict()
        )

    logger.info("Demographic overlay: ARI=%.4f, NMI=%.4f", ari, nmi)
    return {
        "adjusted_rand_index": round(ari, 4),
        "normalized_mutual_info": round(nmi, 4),
        "cluster_country_composition": composition,
    }
