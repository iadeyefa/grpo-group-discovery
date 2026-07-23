"""Method 4: Latent class preference model (Mixture Bradley-Terry / Multinomial Logit via EM)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from src.data.entities import attach_entity_ids

logger = logging.getLogger(__name__)

METHOD_NAME = "latent_class"


def run(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fit K latent sub-populations via Expectation-Maximization (EM) on entity preference distributions.

    Models entity preference responses as generated from a mixture of K latent
    preference profiles (multinomial logit distributions per question).
    """
    cluster_cfg = config.get("clustering", {})
    n_clusters = int(cluster_cfg.get("n_clusters", 5))
    random_state = int(cluster_cfg.get("random_state", 42))
    max_iter = int(cluster_cfg.get("max_iter", 100))
    tol = float(cluster_cfg.get("tol", 1e-4))
    smoothing = float(cluster_cfg.get("smoothing", 1e-3))

    logger.info("Attaching entity IDs for %s discovery", METHOD_NAME)
    preference_with_entities = attach_entity_ids(preference_df, entity_registry)

    entity_ids = sorted(entity_registry["entity_id"].unique())
    n_entities = len(entity_ids)
    if n_entities == 0:
        raise ValueError("No entities available for latent class discovery")
    if n_clusters > n_entities:
        raise ValueError(
            f"n_clusters={n_clusters} cannot exceed number of entities={n_entities}"
        )

    entity_to_idx = {eid: idx for idx, eid in enumerate(entity_ids)}

    # Group responses by question key (qkey)
    qkeys = sorted(preference_with_entities["qkey"].unique())
    logger.info(
        "Building entity preference matrices across %d questions and %d entities",
        len(qkeys),
        n_entities,
    )

    # For each question q, construct matrix X_q (n_entities x num_options) and mask M_q (n_entities)
    question_matrices: list[np.ndarray] = []
    question_masks: list[np.ndarray] = []

    for qkey in qkeys:
        sub_df = preference_with_entities[preference_with_entities["qkey"] == qkey]
        if sub_df.empty:
            continue

        sample_prob = sub_df.iloc[0]["prob_y"]
        n_opts = len(sample_prob)
        if n_opts == 0:
            continue

        X_q = np.zeros((n_entities, n_opts), dtype=np.float64)
        M_q = np.zeros(n_entities, dtype=bool)

        for row in sub_df.itertuples(index=False):
            eid = row.entity_id
            if eid in entity_to_idx:
                idx = entity_to_idx[eid]
                prob_y = np.asarray(row.prob_y, dtype=np.float64)
                prob_sum = float(np.sum(prob_y))
                if prob_sum > 0:
                    X_q[idx] = prob_y / prob_sum
                    M_q[idx] = True

        question_matrices.append(X_q)
        question_masks.append(M_q)

    n_questions = len(question_matrices)
    if n_questions == 0:
        raise ValueError("No valid question preference distributions found")

    rng = np.random.default_rng(random_state)

    # Initialize responsibilities W (n_entities x n_clusters) using Dirichlet distribution
    W = rng.dirichlet(alpha=np.ones(n_clusters), size=n_entities)

    pi = np.full(n_clusters, 1.0 / n_clusters, dtype=np.float64)
    # Theta: list per question q of (n_clusters x n_opts) probability parameters
    theta: list[np.ndarray] = [
        np.full((n_clusters, q_mat.shape[1]), 1.0 / q_mat.shape[1], dtype=np.float64)
        for q_mat in question_matrices
    ]

    prev_ll = -np.inf
    final_iter = 0

    logger.info(
        "Running EM for Latent Class Model: K=%d, max_iter=%d, tol=%.1e",
        n_clusters,
        max_iter,
        tol,
    )

    for iteration in range(1, max_iter + 1):
        final_iter = iteration

        # --- M-Step ---
        # 1. Update class priors pi_k
        N_k = np.sum(W, axis=0)  # shape (K,)
        pi = (N_k + smoothing) / np.sum(N_k + smoothing)

        # 2. Update question option distributions theta_{k, q, c}
        for q_idx in range(n_questions):
            X_q = question_matrices[q_idx]  # (N, C_q)
            M_q = question_masks[q_idx]      # (N,)
            C_q = X_q.shape[1]

            # Weighted sum of observed preferences for each cluster k
            # W[M_q, :] is (N_obs, K), X_q[M_q] is (N_obs, C_q)
            if np.any(M_q):
                counts_k_c = W[M_q].T @ X_q[M_q]  # (K, C_q)
            else:
                counts_k_c = np.zeros((n_clusters, C_q), dtype=np.float64)

            # Apply additive smoothing and normalize per cluster
            smoothed_counts = counts_k_c + smoothing
            theta[q_idx] = smoothed_counts / np.sum(smoothed_counts, axis=1, keepdims=True)

        # --- E-Step ---
        # Compute log P(data_i | class k) for each entity i and cluster k
        log_prob = np.zeros((n_entities, n_clusters), dtype=np.float64)

        for q_idx in range(n_questions):
            X_q = question_matrices[q_idx]  # (N, C_q)
            M_q = question_masks[q_idx]      # (N,)
            log_theta_q = np.log(theta[q_idx])  # (K, C_q)

            if np.any(M_q):
                # (N_obs, C_q) @ (C_q, K) -> (N_obs, K)
                log_prob[M_q] += X_q[M_q] @ log_theta_q.T

        # Unnormalized log posterior: log(pi_k) + log P(data_i | class k)
        log_joint = np.log(pi)[np.newaxis, :] + log_prob  # (N, K)

        # Compute log-likelihood using logsumexp
        log_marginal = logsumexp(log_joint, axis=1)  # (N,)
        current_ll = float(np.sum(log_marginal))

        # Update posterior probabilities (responsibilities W)
        W = np.exp(log_joint - log_marginal[:, np.newaxis])

        # Check convergence
        ll_change = current_ll - prev_ll
        if iteration > 1 and abs(ll_change) < tol:
            logger.info("EM converged at iteration %d (LL=%.4f, change=%.2e)", iteration, current_ll, ll_change)
            break
        prev_ll = current_ll

    labels = np.argmax(W, axis=1)

    assignments = pd.DataFrame(
        {
            "entity_id": entity_ids,
            "cluster_id": labels.astype(int),
        }
    ).sort_values(["cluster_id", "entity_id"])

    features_df = pd.DataFrame(
        W,
        index=entity_ids,
        columns=[f"cluster_prob_{k}" for k in range(n_clusters)],
    )
    features_df.index.name = "entity_id"

    logger.info(
        "Discovered %d groups via %s (%d entities, %d EM iterations)",
        n_clusters,
        METHOD_NAME,
        len(assignments),
        final_iter,
    )

    extras = {
        "features_df": features_df,
        "feature_dim": n_clusters,
        "uses_preference_features": True,
        "discovery_method": METHOD_NAME,
        "discovery_tier": "method",
        "em_stats": {
            "n_iterations": final_iter,
            "final_log_likelihood": prev_ll,
            "class_priors": pi.tolist(),
        },
    }
    return assignments, extras
