"""Method 2: Chosen-rejected embedding sets (Fain)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

METHOD_NAME = "embedding_sets"


def run(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Per entity: set of embed(chosen)-embed(rejected) vectors (not averaged).
    Distance = mean pairwise distance between sets; cluster on distance matrix.
    """
    raise NotImplementedError(
        f"{METHOD_NAME}: requires embedding model and chosen/rejected pairs. "
        "See docs/methods.md."
    )
