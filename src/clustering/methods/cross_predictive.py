"""Method 1: Cross-predictive similarity clustering (Fain)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

METHOD_NAME = "cross_predictive"


def run(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Score sim(A,B) = avg(Acc(pred A|B)-Acc(pred A), Acc(pred B|A)-Acc(pred B)),
    cluster the similarity matrix (spectral / hierarchical).
    """
    raise NotImplementedError(
        f"{METHOD_NAME}: requires LM-based cross-prediction similarity matrix. "
        "See docs/methods.md."
    )
