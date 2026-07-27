"""Canonical preference record schema and validation utilities.

All discovery methods and evaluation tools consume data conforming to these schemas.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required columns for the two internal data representations
# ---------------------------------------------------------------------------

# Population-level aggregate schema (GOQA-style: prob_y distributions)
AGGREGATE_REQUIRED_COLS = frozenset(
    {"entity_id", "qkey", "question", "options", "prob_y"}
)

# Individual-level pairwise schema (HH-RLHF / WildChat / UltraFeedback style)
PAIRWISE_REQUIRED_COLS = frozenset(
    {"entity_id", "qkey", "prompt", "chosen", "rejected"}
)

# Cluster assignment schema
ASSIGNMENT_REQUIRED_COLS = frozenset({"entity_id", "cluster_id"})


def validate_aggregate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that *df* conforms to the aggregate preference schema.

    Raises ValueError with specifics on missing columns.
    Returns the dataframe unchanged for chaining.
    """
    missing = AGGREGATE_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Aggregate preference DataFrame is missing columns: {sorted(missing)}. "
            f"Required: {sorted(AGGREGATE_REQUIRED_COLS)}"
        )
    logger.debug(
        "Validated aggregate preference frame: %d rows, %d entities",
        len(df),
        df["entity_id"].nunique(),
    )
    return df


def validate_pairwise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that *df* conforms to the pairwise preference schema.

    Raises ValueError with specifics on missing columns.
    Returns the dataframe unchanged for chaining.
    """
    missing = PAIRWISE_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Pairwise preference DataFrame is missing columns: {sorted(missing)}. "
            f"Required: {sorted(PAIRWISE_REQUIRED_COLS)}"
        )
    logger.debug(
        "Validated pairwise preference frame: %d rows, %d entities",
        len(df),
        df["entity_id"].nunique(),
    )
    return df


def validate_assignments(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that *df* conforms to the cluster assignment schema."""
    missing = ASSIGNMENT_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Assignment DataFrame is missing columns: {sorted(missing)}. "
            f"Required: {sorted(ASSIGNMENT_REQUIRED_COLS)}"
        )
    return df


def aggregate_to_pairwise(
    aggregate_df: pd.DataFrame,
    n_samples_per_question: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Convert aggregate prob_y distributions into pairwise (chosen, rejected) records.

    For each entity × question, samples chosen/rejected option pairs from the
    probability distribution.  This enables aggregate datasets like GOQA to be
    used with pairwise-native methods (embedding sets, cross-predictive).

    Returns a DataFrame with columns matching PAIRWISE_REQUIRED_COLS plus
    ``prob_chosen`` (the probability weight of the chosen option).
    """
    validate_aggregate_df(aggregate_df)
    rng = np.random.default_rng(random_state)

    rows: list[dict[str, Any]] = []
    for _, record in aggregate_df.iterrows():
        prob_y = np.asarray(record["prob_y"], dtype=np.float64)
        options = record["options"]
        n_opts = len(prob_y)
        if n_opts < 2:
            continue

        # Normalise
        total = prob_y.sum()
        if total <= 0:
            continue
        probs = prob_y / total

        question_text = record["question"]
        # Build prompt from the question text and option list
        options_str = " | ".join(str(o) for o in options)
        prompt = f"{question_text}\nOptions: {options_str}"

        for _ in range(n_samples_per_question):
            # Sample chosen weighted by prob_y
            chosen_idx = int(rng.choice(n_opts, p=probs))

            # Sample rejected from remaining options (uniform)
            remaining = [i for i in range(n_opts) if i != chosen_idx]
            if not remaining:
                continue
            rejected_idx = int(rng.choice(remaining))

            rows.append(
                {
                    "entity_id": record["entity_id"],
                    "qkey": record["qkey"],
                    "prompt": prompt,
                    "chosen": str(options[chosen_idx]),
                    "rejected": str(options[rejected_idx]),
                    "prob_chosen": float(probs[chosen_idx]),
                }
            )

    out = pd.DataFrame(rows)
    logger.info(
        "Converted aggregate → pairwise: %d aggregate rows → %d pairwise records",
        len(aggregate_df),
        len(out),
    )
    return out
