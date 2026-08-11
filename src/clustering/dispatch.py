"""Dispatch discovery methods by name."""

from __future__ import annotations

import logging
from typing import Any, Callable

import pandas as pd

from src.clustering.methods import (
    baselines,
    cross_predictive,
    domain_clustering,
    embedding_sets,
    gmm_mixture,
    matrix_factorization,
    preference_similarity,
)
from src.features import contrastive

logger = logging.getLogger(__name__)

BASELINE_METHOD = preference_similarity.METHOD_NAME

DISCOVERY_METHODS: dict[str, Callable[..., tuple[pd.DataFrame, dict[str, Any]]]] = {
    # Floor baseline
    preference_similarity.METHOD_NAME: preference_similarity.run,
    # Comparison baselines
    baselines.SINGLE_GROUP_METHOD: baselines.run_single_group,
    baselines.RANDOM_ASSIGNMENT_METHOD: baselines.run_random_assignment,
    baselines.COUNTRY_ORACLE_METHOD: baselines.run_country_oracle,
    # Discovery methods 1-3
    cross_predictive.METHOD_NAME: cross_predictive.run,
    embedding_sets.METHOD_NAME: embedding_sets.run,
    matrix_factorization.METHOD_NAME: matrix_factorization.run,
    # Advanced discovery methods
    domain_clustering.METHOD_NAME: domain_clustering.run,
    contrastive.METHOD_NAME: contrastive.run,
    gmm_mixture.METHOD_NAME: gmm_mixture.run,
}


def resolve_method(config: dict[str, Any]) -> str:
    """Read discovery method from config (clustering.method or legacy clustering.algorithm)."""
    cluster_cfg = config.get("clustering", {})
    method = cluster_cfg.get("method") or cluster_cfg.get("algorithm", BASELINE_METHOD)
    if method == "kmeans":
        logger.debug("Mapping legacy algorithm 'kmeans' -> '%s'", BASELINE_METHOD)
        return BASELINE_METHOD
    return method


def run_discovery(
    method: str,
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the requested discovery method and return assignments + extras."""
    if method not in DISCOVERY_METHODS:
        raise ValueError(
            f"Unknown discovery method {method!r}. Known: {sorted(DISCOVERY_METHODS)}"
        )

    logger.info("Running discovery method '%s'", method)
    return DISCOVERY_METHODS[method](preference_df, entity_registry, config)


def cluster_sizes(assignments: pd.DataFrame) -> dict[int, int]:
    """Count entities per cluster."""
    return assignments.groupby("cluster_id").size().to_dict()


def clustering_metadata(
    config: dict[str, Any],
    assignments: pd.DataFrame,
    feature_dim: int,
    *,
    discovery_method: str,
    discovery_tier: str,
    uses_preference_features: bool,
) -> dict[str, Any]:
    """Serializable metadata for reproducibility."""
    return {
        "dataset": config.get("dataset", {}),
        "entities": config.get("entities", {}),
        "features": {**config.get("features", {}), "dim": feature_dim},
        "clustering": config.get("clustering", {}),
        "discovery_method": discovery_method,
        "discovery_tier": discovery_tier,
        "uses_preference_features": uses_preference_features,
        "n_entities": int(len(assignments)),
        "cluster_sizes": cluster_sizes(assignments),
        "dataset_names": [
            f"{config['export']['dataset_prefix']}_{cid}"
            for cid in sorted(assignments["cluster_id"].unique())
        ],
    }
