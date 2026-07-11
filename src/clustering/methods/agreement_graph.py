"""Method 5: Agreement graph on overlapping prompts."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

METHOD_NAME = "agreement_graph"


def run(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Edge weight = agreement on shared prompts (top choice, cosine, rank corr);
    community detection / spectral clustering on graph.
    """
    raise NotImplementedError(
        f"{METHOD_NAME}: requires agreement graph construction. See docs/methods.md."
    )
