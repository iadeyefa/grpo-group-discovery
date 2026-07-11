"""Method 3: Sparse matrix factorization + latent clustering."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

METHOD_NAME = "matrix_factorization"


def run(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Sparse entity x prompt matrix with preference signals; ALS/implicit MF
    latent vectors, then cluster latent space.
    """
    raise NotImplementedError(
        f"{METHOD_NAME}: requires sparse preference matrix + factorization. "
        "See docs/methods.md."
    )
