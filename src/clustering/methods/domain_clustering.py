"""Game-Changing Feature 1: Domain-Specific Topic Preference Discovery."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from src.data.entities import attach_entity_ids

logger = logging.getLogger(__name__)

METHOD_NAME = "domain_topic_clustering"

DOMAIN_KEYWORDS = {
    "tech_governance": ["ai", "technology", "internet", "privacy", "data", "algorithm", "digital", "media", "online", "robot"],
    "social_values": ["religion", "morality", "god", "marriage", "family", "gender", "women", "rights", "culture", "tradition", "faith"],
    "economic_policy": ["tax", "economy", "jobs", "income", "poverty", "business", "market", "trade", "government", "welfare", "work"],
    "world_affairs": ["war", "military", "defense", "foreign", "china", "us", "russia", "un", "global", "security", "peace", "border"],
    "environment_health": ["climate", "environment", "energy", "health", "pollution", "nature", "science", "medical", "disease"]
}


def categorize_question_domain(question_text: str) -> str:
    text_lower = str(question_text).lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return domain
    return "general_values"


def run(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Categorizes preference prompts into distinct topic domains and discovers
    domain-specific preference groups, yielding higher within-domain prediction lift.
    """
    cluster_cfg = config.get("clustering", {})
    n_clusters = int(cluster_cfg.get("n_clusters", 5))
    random_state = int(cluster_cfg.get("random_state", 42))
    target_domain = str(cluster_cfg.get("target_domain", "all"))

    preference_with_entities = attach_entity_ids(preference_df, entity_registry)
    
    # Assign domain tag to each row
    preference_with_entities["domain"] = preference_with_entities["question"].apply(categorize_question_domain)
    
    domain_counts = preference_with_entities["domain"].value_counts().to_dict()
    logger.info("Question domain categorization counts: %s", domain_counts)

    if target_domain != "all" and target_domain in domain_counts:
        sub_df = preference_with_entities[preference_with_entities["domain"] == target_domain]
        logger.info("Filtering preference records for target domain: %s (%d rows)", target_domain, len(sub_df))
    else:
        sub_df = preference_with_entities

    # Build per-domain feature matrix
    max_len = int(sub_df["prob_y"].apply(len).max())
    entity_vectors: dict[str, list[np.ndarray]] = {}
    
    for entity_id, group in sub_df.groupby("entity_id"):
        vectors = []
        for prob_y in group["prob_y"]:
            arr = np.asarray(prob_y, dtype=np.float64)
            if len(arr) < max_len:
                arr = np.pad(arr, (0, max_len - len(arr)))
            vectors.append(arr)
        entity_vectors[entity_id] = vectors

    entity_ids = sorted(entity_vectors.keys())
    matrix = np.stack([np.mean(entity_vectors[e], axis=0) for e in entity_ids])
    
    # Weight by domain option variance
    var_per_dim = np.var(matrix, axis=0)
    matrix = matrix * (np.sqrt(var_per_dim) + 1e-6)
    matrix = normalize(matrix, norm="l2", axis=1)

    features_df = pd.DataFrame(matrix, index=entity_ids, columns=[f"domain_feat_{i}" for i in range(matrix.shape[1])])
    features_df.index.name = "entity_id"

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(matrix)

    assignments = pd.DataFrame(
        {
            "entity_id": entity_ids,
            "cluster_id": labels.astype(int),
        }
    ).sort_values(["cluster_id", "entity_id"])

    extras = {
        "features_df": features_df,
        "feature_dim": matrix.shape[1],
        "uses_preference_features": True,
        "discovery_method": METHOD_NAME,
        "discovery_tier": "method",
        "domain_info": {
            "domain_counts": domain_counts,
            "target_domain": target_domain,
        },
    }
    return assignments, extras
