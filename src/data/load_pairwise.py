"""Generic loader for individual-level pairwise preference datasets.

Provides adapters for common RLHF datasets (Anthropic HH-RLHF, UltraFeedback, etc.)
and a unified output schema: entity_id, qkey, prompt, chosen, rejected.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_pairwise_csv(
    path: str | Path,
    entity_col: str = "individual_id",
    prompt_col: str = "prompt",
    chosen_col: str = "chosen",
    rejected_col: str = "rejected",
    metadata_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load pairwise preferences from a local CSV file.

    Maps column names to the canonical schema used by the discovery pipeline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pairwise CSV not found: {path}")

    raw = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(raw), path)

    # Rename columns to canonical schema
    rename_map = {
        entity_col: "entity_id",
        prompt_col: "prompt",
        chosen_col: "chosen",
        rejected_col: "rejected",
    }
    df = raw.rename(columns=rename_map)

    # Add qkey as row index if not present
    if "qkey" not in df.columns:
        df["qkey"] = range(len(df))

    required = {"entity_id", "prompt", "chosen", "rejected", "qkey"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"After column mapping, pairwise DataFrame is missing: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )

    # Carry through metadata columns
    keep_cols = list(required)
    if metadata_cols:
        for mc in metadata_cols:
            if mc in df.columns and mc not in keep_cols:
                keep_cols.append(mc)

    logger.info(
        "Pairwise frame: %d records, %d unique entities",
        len(df),
        df["entity_id"].nunique(),
    )
    return df[keep_cols].reset_index(drop=True)


def load_pairwise_jsonl(
    path: str | Path,
    entity_col: str = "individual_id",
    prompt_col: str = "prompt",
    chosen_col: str = "chosen",
    rejected_col: str = "rejected",
) -> pd.DataFrame:
    """Load pairwise preferences from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pairwise JSONL not found: {path}")

    raw = pd.read_json(path, lines=True)
    logger.info("Loaded %d rows from %s", len(raw), path)

    rename_map = {
        entity_col: "entity_id",
        prompt_col: "prompt",
        chosen_col: "chosen",
        rejected_col: "rejected",
    }
    df = raw.rename(columns=rename_map)

    if "qkey" not in df.columns:
        df["qkey"] = range(len(df))

    required = {"entity_id", "prompt", "chosen", "rejected", "qkey"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"After column mapping, pairwise DataFrame is missing: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )

    logger.info(
        "Pairwise frame: %d records, %d unique entities",
        len(df),
        df["entity_id"].nunique(),
    )
    return df[list(required)].reset_index(drop=True)


def load_pairwise_preferences(
    config: dict[str, Any],
) -> pd.DataFrame:
    """Dispatch to the correct pairwise loader based on config.

    Config keys:
        dataset.format: "csv" | "jsonl"
        dataset.path: path to data file
        dataset.column_mapping: optional dict of column name overrides
    """
    dataset_cfg = config.get("dataset", {})
    fmt = dataset_cfg.get("format", "csv")
    path = dataset_cfg.get("path")

    if path is None:
        raise ValueError("dataset.path is required for pairwise loaders")

    col_map = dataset_cfg.get("column_mapping", {})
    entity_col = col_map.get("entity", "individual_id")
    prompt_col = col_map.get("prompt", "prompt")
    chosen_col = col_map.get("chosen", "chosen")
    rejected_col = col_map.get("rejected", "rejected")

    if fmt == "csv":
        return load_pairwise_csv(
            path,
            entity_col=entity_col,
            prompt_col=prompt_col,
            chosen_col=chosen_col,
            rejected_col=rejected_col,
        )
    elif fmt == "jsonl":
        return load_pairwise_jsonl(
            path,
            entity_col=entity_col,
            prompt_col=prompt_col,
            chosen_col=chosen_col,
            rejected_col=rejected_col,
        )
    else:
        raise ValueError(
            f"Unsupported pairwise format {fmt!r}. Supported: 'csv', 'jsonl'."
        )
