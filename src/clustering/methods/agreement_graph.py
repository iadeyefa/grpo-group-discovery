"""Method 5: Agreement graph on overlapping prompts."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

from src.data.entities import attach_entity_ids

logger = logging.getLogger(__name__)

METHOD_NAME = "agreement_graph"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _build_entity_prompt_map(preference_with_entities: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    entity_prompt_map: dict[str, dict[str, np.ndarray]] = {}
    for entity_id, group in preference_with_entities.groupby("entity_id"):
        prompt_map: dict[str, np.ndarray] = {}
        for row in group.itertuples(index=False):
            prompt_map[row.qkey] = np.asarray(row.prob_y, dtype=np.float64)
        entity_prompt_map[entity_id] = prompt_map
    return entity_prompt_map


def _pairwise_agreement(
    prompts_a: dict[str, np.ndarray],
    prompts_b: dict[str, np.ndarray],
    *,
    top_choice_weight: float,
    min_shared_prompts: int,
) -> tuple[float, int]:
    shared_qkeys = sorted(set(prompts_a).intersection(prompts_b))
    shared_count = len(shared_qkeys)
    if shared_count < min_shared_prompts:
        return 0.0, shared_count

    prompt_scores: list[float] = []
    for qkey in shared_qkeys:
        a = prompts_a[qkey]
        b = prompts_b[qkey]
        a_top = int(np.argmax(a))
        b_top = int(np.argmax(b))
        top_agreement = 1.0 if a_top == b_top else 0.0
        cosine = max(0.0, _cosine_similarity(a, b))
        prompt_scores.append(top_choice_weight * top_agreement + (1.0 - top_choice_weight) * cosine)

    mean_prompt_score = float(np.mean(prompt_scores)) if prompt_scores else 0.0
    coverage = shared_count / max(len(prompts_a), len(prompts_b))
    return mean_prompt_score * coverage, shared_count


def run(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Edge weight = agreement on shared prompts (top choice + cosine);
    community detection / spectral clustering on graph.
    """
    cluster_cfg = config.get("clustering", {})
    n_clusters = cluster_cfg.get("n_clusters", 5)
    random_state = cluster_cfg.get("random_state", 42)
    top_choice_weight = float(cluster_cfg.get("top_choice_weight", 0.5))
    min_shared_prompts = int(cluster_cfg.get("min_shared_prompts", 2))

    logger.info("Building entity-attached preference frame for %s", METHOD_NAME)
    preference_with_entities = attach_entity_ids(preference_df, entity_registry)
    entity_prompt_map = _build_entity_prompt_map(preference_with_entities)
    entity_ids = sorted(entity_prompt_map.keys())

    n_entities = len(entity_ids)
    if n_entities == 0:
        raise ValueError("No entities available for agreement graph clustering")
    if n_clusters > n_entities:
        raise ValueError(
            f"n_clusters={n_clusters} cannot exceed number of entities={n_entities}"
        )

    similarity = np.eye(n_entities, dtype=np.float64)
    for i in range(n_entities):
        for j in range(i + 1, n_entities):
            score, shared_count = _pairwise_agreement(
                entity_prompt_map[entity_ids[i]],
                entity_prompt_map[entity_ids[j]],
                top_choice_weight=top_choice_weight,
                min_shared_prompts=min_shared_prompts,
            )
            similarity[i, j] = score
            similarity[j, i] = score

    features_df = pd.DataFrame(similarity, index=entity_ids, columns=entity_ids)
    features_df.index.name = "entity_id"

    logger.info(
        "Running spectral clustering on agreement graph: k=%d, n_entities=%d, random_state=%d",
        n_clusters,
        n_entities,
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
        "Discovered %d groups via %s (%d entities, min_shared_prompts=%d)",
        n_clusters,
        METHOD_NAME,
        len(assignments),
        min_shared_prompts,
    )

    extras = {
        "features_df": features_df,
        "feature_dim": features_df.shape[1],
        "uses_preference_features": True,
        "discovery_method": METHOD_NAME,
        "discovery_tier": "method",
        "agreement_params": {
            "top_choice_weight": top_choice_weight,
            "min_shared_prompts": min_shared_prompts,
            "n_entities": n_entities,
        },
    }
    return assignments, extras
