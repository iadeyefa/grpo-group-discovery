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

    sim_func = _top_choice_sim if metric == "top_choice" else _cosine_sim

    logger.info("Attaching entity IDs for %s discovery", METHOD_NAME)
    preference_with_entities = attach_entity_ids(preference_df, entity_registry)

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

            if shared_count < min_shared_prompts:
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
