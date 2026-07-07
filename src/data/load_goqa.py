"""Load Anthropic Global Opinion QA for group-discovery preprocessing."""

from __future__ import annotations

import ast
import logging
from typing import Any

import datasets
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mirrors grpo-reproduction/src/groupstuff/global_opinion_data_processing.py
DEFAULT_COUNTRIES = [
    "Nigeria",
    "Egypt",
    "India (Current national sample)",
    "China",
    "Japan",
    "Germany",
    "France",
    "Spain",
    "United States",
    "Canada",
    "Brazil",
    "Argentina",
    "Australia",
    "New Zealand",
]


def load_raw_goqa(
    dataset_name: str = "Anthropic/llm_global_opinions",
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Load the Global Opinion QA dataset from Hugging Face."""
    logger.info("Loading dataset %s", dataset_name)
    dataset = datasets.load_dataset(dataset_name, cache_dir=cache_dir)["train"]
    df = pd.DataFrame(dataset)
    df["qkey"] = df.index
    logger.debug("Loaded %d raw questions", len(df))
    return df


def _parse_selections(selections_str: str) -> dict[str, list[float]]:
    """Parse the selections dict embedded in the HF row."""
    inner = "{" + selections_str.split("{")[1].split("}")[0] + "}"
    return ast.literal_eval(inner)


def build_country_question_frame(
    df: pd.DataFrame,
    countries: list[str] | None = None,
) -> pd.DataFrame:
    """
    Explode each question into one row per (question, country) with opinion vector.

    Returns columns: qkey, question, options, group, prob_y
    """
    countries = countries or DEFAULT_COUNTRIES
    rows: list[dict[str, Any]] = []

    for i in range(len(df)):
        question = df.loc[i, "question"]
        options_raw = df.loc[i, "options"]
        if not question or not options_raw:
            continue

        selections = _parse_selections(df.loc[i, "selections"])
        options = [str(opt) for opt in ast.literal_eval(options_raw)]

        for country in countries:
            if country not in selections:
                continue
            prob_y = selections[country]
            if prob_y is None or len(prob_y) == 0 or np.sum(prob_y) == 0:
                continue
            rows.append(
                {
                    "qkey": df.loc[i, "qkey"],
                    "question": question,
                    "options": options,
                    "group": country,
                    "prob_y": np.asarray(prob_y, dtype=np.float64),
                }
            )

    out = pd.DataFrame(rows)
    logger.info(
        "Built country-question frame: %d rows, %d countries, %d questions",
        len(out),
        out["group"].nunique() if len(out) else 0,
        out["qkey"].nunique() if len(out) else 0,
    )
    return out


def load_goqa_for_clustering(
    dataset_name: str = "Anthropic/llm_global_opinions",
    countries: list[str] | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """End-to-end loader used by the discovery pipeline."""
    raw = load_raw_goqa(dataset_name=dataset_name, cache_dir=cache_dir)
    return build_country_question_frame(raw, countries=countries)
