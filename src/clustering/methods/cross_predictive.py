"""Method 1: Cross-predictive similarity clustering (Fain)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

from src.data.entities import attach_entity_ids

logger = logging.getLogger(__name__)

METHOD_NAME = "cross_predictive"


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _top_choice_sim(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 if int(np.argmax(a)) == int(np.argmax(b)) else 0.0


def run(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Computes cross-predictive transfer gain sim(A,B) = avg(Acc(A|B)-Acc(A), Acc(B|A)-Acc(B)),
    scaled by prompt coverage, and clusters the affinity matrix using Spectral Clustering.
    """
    cluster_cfg = config.get("clustering", {})
    n_clusters = int(cluster_cfg.get("n_clusters", 5))
    random_state = int(cluster_cfg.get("random_state", 42))
    min_shared_prompts = int(cluster_cfg.get("min_shared_prompts", 2))
    metric = str(cluster_cfg.get("similarity_metric", "cosine")).lower()
    impute_first = bool(cluster_cfg.get("impute_first", False))

    sim_func = _top_choice_sim if metric == "top_choice" else _cosine_sim

    logger.info("Attaching entity IDs for %s discovery (impute_first=%s)", METHOD_NAME, impute_first)
    preference_with_entities = attach_entity_ids(preference_df, entity_registry)

    if impute_first:
        from sklearn.decomposition import NMF
        # Stage 1: Build sparse matrix (entities x question_options) and run NMF matrix completion
        all_qkeys = sorted(preference_with_entities["qkey"].unique())
        qkey_to_opts = preference_with_entities.groupby("qkey")["options"].first().to_dict()
        
        # Map (qkey, opt_idx) -> col_idx
        col_mapping = {}
        col_idx = 0
        for qk in all_qkeys:
            opts = qkey_to_opts[qk]
            for opt_i in range(len(opts)):
                col_mapping[(qk, opt_i)] = col_idx
                col_idx += 1
                
        entity_ids = sorted(preference_with_entities["entity_id"].unique())
        eid_to_row = {eid: idx for idx, eid in enumerate(entity_ids)}
        
        sparse_mat = np.zeros((len(entity_ids), col_idx), dtype=np.float64)
        mask_mat = np.zeros((len(entity_ids), col_idx), dtype=bool)
        
        for row in preference_with_entities.itertuples(index=False):
            r_i = eid_to_row[row.entity_id]
            qk = row.qkey
            prob_y = row.prob_y
            for opt_i, p_val in enumerate(prob_y):
                c_i = col_mapping.get((qk, opt_i))
                if c_i is not None:
                    sparse_mat[r_i, c_i] = p_val
                    mask_mat[r_i, c_i] = True
                    
        # Fit NMF on non-zero matrix to impute missing cells
        nmf_model = NMF(n_components=min(10, len(entity_ids)), random_state=random_state, max_iter=200)
        W = nmf_model.fit_transform(sparse_mat)
        H = nmf_model.components_
        imputed_mat = np.dot(W, H)
        
        # Build imputed entity_prompt_map
        entity_prompt_map: dict[str, dict[str, np.ndarray]] = {}
        for r_i, eid in enumerate(entity_ids):
            prompts: dict[str, np.ndarray] = {}
            for qk in all_qkeys:
                opts = qkey_to_opts[qk]
                c_indices = [col_mapping[(qk, opt_i)] for opt_i in range(len(opts))]
                arr = imputed_mat[r_i, c_indices]
                arr = np.maximum(arr, 0.0)
                s = float(np.sum(arr))
                prompts[qk] = arr / s if s > 0 else np.ones(len(opts)) / len(opts)
            entity_prompt_map[eid] = prompts
        logger.info("Stage 1 NMF Matrix Completion finished for %d entities x %d features", len(entity_ids), col_idx)
    else:
        # Build map per entity: entity_id -> {qkey: prob_y_array}
        entity_prompt_map: dict[str, dict[str, np.ndarray]] = {}
        for entity_id, group in preference_with_entities.groupby("entity_id"):
            prompts: dict[str, np.ndarray] = {}
            for row in group.itertuples(index=False):
                arr = np.asarray(row.prob_y, dtype=np.float64)
                p_sum = float(np.sum(arr))
                if p_sum > 0:
                    prompts[row.qkey] = arr / p_sum
            entity_prompt_map[entity_id] = prompts

    entity_ids = sorted(entity_prompt_map.keys())
    n_entities = len(entity_ids)
    if n_entities == 0:
        raise ValueError("No entities available for cross-predictive clustering")
    if n_clusters > n_entities:
        raise ValueError(
            f"n_clusters={n_clusters} cannot exceed number of entities={n_entities}"
        )

    # Compute baseline self-accuracy Acc(A) relative to entity A's marginal opinion
    entity_marginal_acc: dict[str, float] = {}
    for eid, prompt_dict in entity_prompt_map.items():
        if not prompt_dict:
            entity_marginal_acc[eid] = 0.0
            continue
        max_opts = max(len(v) for v in prompt_dict.values())
        padded = [
            np.pad(v, (0, max_opts - len(v))) if len(v) < max_opts else v
            for v in prompt_dict.values()
        ]
        mean_vector = np.mean(padded, axis=0)

        accs = [sim_func(v, mean_vector[: len(v)]) for v in prompt_dict.values()]
        entity_marginal_acc[eid] = float(np.mean(accs)) if accs else 0.0

    logger.info(
        "Building %d x %d cross-predictive similarity matrix (metric=%s)",
        n_entities,
        n_entities,
        metric,
    )
    similarity = np.eye(n_entities, dtype=np.float64)

    for i in range(n_entities):
        eid_a = entity_ids[i]
        prompts_a = entity_prompt_map[eid_a]
        base_acc_a = entity_marginal_acc[eid_a]

        for j in range(i + 1, n_entities):
            eid_b = entity_ids[j]
            prompts_b = entity_prompt_map[eid_b]
            base_acc_b = entity_marginal_acc[eid_b]

            shared_keys = sorted(set(prompts_a).intersection(prompts_b))
            shared_count = len(shared_keys)

            if shared_count < (1 if impute_first else min_shared_prompts):
                score = 0.0
            else:
                agreements = [
                    sim_func(prompts_a[qk], prompts_b[qk]) for qk in shared_keys
                ]
                acc_ab = float(np.mean(agreements))

                gain_a = max(0.0, acc_ab - base_acc_a)
                gain_b = max(0.0, acc_ab - base_acc_b)

                transfer_gain = 0.5 * (gain_a + gain_b)
                coverage = shared_count / max(len(prompts_a), len(prompts_b))
                score = transfer_gain * coverage

            similarity[i, j] = score
            similarity[j, i] = score

    features_df = pd.DataFrame(similarity, index=entity_ids, columns=entity_ids)
    features_df.index.name = "entity_id"

    logger.info(
        "Running spectral clustering on cross-predictive similarity: K=%d, random_state=%d",
        n_clusters,
        random_state,
    )
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_state,
        assign_labels="kmeans",
    )
    labels = model.fit_predict(similarity)

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
        "cross_predictive_params": {
            "metric": metric,
            "min_shared_prompts": min_shared_prompts,
        },
    }
    return assignments, extras
