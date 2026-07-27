"""Baseline discovery methods for comparison bracketing.

Provides three baselines:
    single_group      — all entities in one cluster (lower bound)
    random_assignment  — random K-way split (stochastic baseline)
    country_oracle     — uses hidden source_group labels (approximate upper bound)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.entities import attach_entity_ids

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline 1: Single Group
# ---------------------------------------------------------------------------

SINGLE_GROUP_METHOD = "single_group"


def run_single_group(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign all entities to cluster 0 — the no-discovery lower bound."""
    entity_ids = sorted(entity_registry["entity_id"].unique())

    assignments = pd.DataFrame(
        {"entity_id": entity_ids, "cluster_id": 0}
    )

    logger.info(
        "Baseline '%s': assigned %d entities to 1 group",
        SINGLE_GROUP_METHOD,
        len(assignments),
    )

    extras = {
        "features_df": None,
        "feature_dim": 0,
        "uses_preference_features": False,
        "discovery_method": SINGLE_GROUP_METHOD,
        "discovery_tier": "baseline",
    }
    return assignments, extras


# ---------------------------------------------------------------------------
# Baseline 2: Random Assignment
# ---------------------------------------------------------------------------

RANDOM_ASSIGNMENT_METHOD = "random_assignment"


def run_random_assignment(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Randomly assign entities to K groups — stochastic baseline."""
    cluster_cfg = config.get("clustering", {})
    n_clusters = int(cluster_cfg.get("n_clusters", 5))
    random_state = int(cluster_cfg.get("random_state", 42))

    entity_ids = sorted(entity_registry["entity_id"].unique())
    rng = np.random.default_rng(random_state)

    labels = rng.integers(0, n_clusters, size=len(entity_ids))

    assignments = pd.DataFrame(
        {"entity_id": entity_ids, "cluster_id": labels.astype(int)}
    ).sort_values(["cluster_id", "entity_id"])

    logger.info(
        "Baseline '%s': assigned %d entities to %d random groups (seed=%d)",
        RANDOM_ASSIGNMENT_METHOD,
        len(assignments),
        n_clusters,
        random_state,
    )

    extras = {
        "features_df": None,
        "feature_dim": 0,
        "uses_preference_features": False,
        "discovery_method": RANDOM_ASSIGNMENT_METHOD,
        "discovery_tier": "baseline",
    }
    return assignments, extras


# ---------------------------------------------------------------------------
# Baseline 3: Country Oracle
# ---------------------------------------------------------------------------

COUNTRY_ORACLE_METHOD = "country_oracle"


def run_country_oracle(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use hidden source_group (country) labels as cluster assignment.

    This is the approximate upper bound — it shows how well known demographic
    grouping performs.  Source_group labels are never used in feature construction
    for real discovery methods.
    """
    if "source_group" not in entity_registry.columns:
        raise ValueError(
            "country_oracle baseline requires source_group in entity_registry. "
            "Use entities.mode = 'observed' or 'simulated'."
        )

    # Deterministic mapping: sorted unique source_groups → cluster_id 0..N-1
    source_groups = sorted(entity_registry["source_group"].unique())
    sg_to_cluster = {sg: idx for idx, sg in enumerate(source_groups)}

    entity_ids = entity_registry["entity_id"].tolist()
    cluster_ids = [
        sg_to_cluster[sg]
        for sg in entity_registry["source_group"].tolist()
    ]

    assignments = pd.DataFrame(
        {"entity_id": entity_ids, "cluster_id": cluster_ids}
    ).sort_values(["cluster_id", "entity_id"])

    n_clusters = len(source_groups)
    logger.info(
        "Baseline '%s': assigned %d entities to %d country-based groups",
        COUNTRY_ORACLE_METHOD,
        len(assignments),
        n_clusters,
    )

    extras = {
        "features_df": None,
        "feature_dim": 0,
        "uses_preference_features": False,
        "discovery_method": COUNTRY_ORACLE_METHOD,
        "discovery_tier": "baseline",
        "country_oracle_params": {
            "source_group_to_cluster": sg_to_cluster,
            "n_source_groups": n_clusters,
        },
    }
    return assignments, extras
