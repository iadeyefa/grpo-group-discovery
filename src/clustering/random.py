"""Random cluster assignment baseline.

Assigns entities to clusters without consulting preference features or source
population labels. Null hypothesis baseline for comparing preference-based
discovery methods.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def run_random_clustering(
    entities: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Randomly assign each entity to a cluster id in [0, n_clusters).

    Only entity_id is used; source_group and preference data are ignored.

    Returns DataFrame with columns: entity_id, cluster_id
    """
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if entities.empty:
        raise ValueError("entities must be non-empty")
    if "entity_id" not in entities.columns:
        raise ValueError("entities must include an entity_id column")

    shuffled = entities[["entity_id"]].sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)

    # Round-robin after shuffle keeps group sizes balanced for small N.
    shuffled["cluster_id"] = [i % n_clusters for i in range(len(shuffled))]
    assignments = shuffled.sort_values(["cluster_id", "entity_id"]).reset_index(drop=True)

    for cid in range(n_clusters):
        members = assignments.loc[assignments["cluster_id"] == cid, "entity_id"].tolist()
        logger.debug("Random cluster %d (%d entities): %s", cid, len(members), members)

    logger.info(
        "Random baseline: %d entities -> %d clusters (seed=%d, no preference features used)",
        len(assignments),
        n_clusters,
        random_state,
    )
    return assignments
