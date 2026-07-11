"""Method 4: Latent class preference model (mixture Bradley-Terry / logit)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

METHOD_NAME = "latent_class"


def run(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit K hidden subpopulations via EM on observed choices/pairs."""
    raise NotImplementedError(
        f"{METHOD_NAME}: requires mixture preference model (EM). See docs/methods.md."
    )
