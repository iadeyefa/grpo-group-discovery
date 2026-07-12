"""Derive clustering entities from preference records.

Entities are opaque IDs assigned to preference profiles. Source population labels
from the HF dataset are stored separately for post-hoc evaluation only.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

ENTITY_MODE_OBSERVED = "observed"
ENTITY_MODE_SIMULATED = "simulated"
ENTITY_MODE_BLIND = "blind"


def build_entity_table(
    preference_df: pd.DataFrame,
    mode: str = ENTITY_MODE_OBSERVED,
    n_simulated_per_group: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build entity registry from preference data.

    Returns DataFrame with columns:
        entity_id    — opaque clustering unit (used by all algorithms)
        source_group — HF population label (evaluation only, never clustered on)
    """
    if mode == ENTITY_MODE_OBSERVED:
        return _build_observed_entities(preference_df)
    if mode == ENTITY_MODE_SIMULATED:
        return _build_simulated_entities(
            preference_df,
            n_simulated_per_group=n_simulated_per_group,
            random_state=random_state,
        )
    if mode == ENTITY_MODE_BLIND:
        return _build_blind_entities(preference_df)
    raise ValueError(
        f"Unsupported entity mode: {mode!r}. "
        f"Expected {ENTITY_MODE_OBSERVED!r}, {ENTITY_MODE_SIMULATED!r}, or {ENTITY_MODE_BLIND!r}."
    )


def _build_observed_entities(preference_df: pd.DataFrame) -> pd.DataFrame:
    """One entity per distinct source group that has preference records."""
    source_groups = sorted(preference_df["source_group"].unique())
    entities = pd.DataFrame(
        {
            "entity_id": [_opaque_id(i) for i in range(len(source_groups))],
            "source_group": source_groups,
        }
    )
    logger.info(
        "Registered %d observed entities from preference records",
        len(entities),
    )
    logger.debug("Entity registry: %s", entities.to_dict("records"))
    return entities


def _build_simulated_entities(
    preference_df: pd.DataFrame,
    n_simulated_per_group: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Synthetic individuals sampled per source group for individual-level experiments.

    Preference-based methods will attach sampled opinion records to each entity.
    """
    if n_simulated_per_group <= 0:
        raise ValueError("n_simulated_per_group must be positive")

    source_groups = sorted(preference_df["source_group"].unique())
    rows: list[dict[str, str]] = []
    counter = 0
    for source_group in source_groups:
        for _ in range(n_simulated_per_group):
            rows.append(
                {
                    "entity_id": _opaque_id(counter),
                    "source_group": source_group,
                }
            )
            counter += 1

    entities = pd.DataFrame(rows)
    logger.info(
        "Registered %d simulated entities (%d per source group, %d groups)",
        len(entities),
        n_simulated_per_group,
        len(source_groups),
    )
    return entities


def _build_blind_entities(preference_df: pd.DataFrame) -> pd.DataFrame:
    """One anonymous entity per preference row for blind clustering."""
    entities = pd.DataFrame(
        {
            "entity_id": [_opaque_id(i) for i in range(len(preference_df))],
        }
    )
    logger.info("Registered %d blind entities from preference rows", len(entities))
    logger.debug("Blind entity registry: %s", entities.to_dict("records"))
    return entities


def _opaque_id(index: int) -> str:
    """Stable opaque entity identifier — no demographic information encoded."""
    return f"entity_{index:04d}"


def attach_entity_ids(
    preference_df: pd.DataFrame,
    entities: pd.DataFrame,
) -> pd.DataFrame:
    """Join entity_id onto preference rows via source_group (for feature building)."""
    if "source_group" in entities.columns:
        registry = entities[["entity_id", "source_group"]].drop_duplicates()
        merged = preference_df.merge(registry, on="source_group", how="inner")
        logger.debug(
            "Attached entity_id to %d / %d preference rows via source_group",
            len(merged),
            len(preference_df),
        )
        return merged

    if len(entities) != len(preference_df):
        raise ValueError(
            "Blind entity registry must have the same number of rows as preference_df"
        )

    merged = preference_df.copy()
    merged["entity_id"] = entities["entity_id"].to_numpy()
    logger.debug(
        "Attached entity_id to %d preference rows via positional blind mapping",
        len(merged),
    )
    return merged
