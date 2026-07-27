"""Load Anthropic Global Opinion QA preference records for group discovery."""

from __future__ import annotations

import ast
import logging
from typing import Any

import datasets
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


def _discover_source_groups(df: pd.DataFrame) -> list[str]:
    """Collect all population group keys present across GOQA questions."""
    groups: set[str] = set()
    for i in range(len(df)):
        selections = _parse_selections(df.loc[i, "selections"])
        groups.update(selections.keys())
    return sorted(groups)


def build_preference_frame(
    df: pd.DataFrame,
    source_groups: list[str] | None = None,
) -> pd.DataFrame:
    """
    Explode each question into one row per (question, population group) with opinion vector.

    The HF dataset stores population labels under per-question selection keys; we keep
    that as `source_group` only for downstream evaluation — clustering never uses it.

    Returns columns: qkey, question, options, source_group, prob_y
    """
    available = _discover_source_groups(df)
    if source_groups is None:
        source_groups = available
    else:
        missing = set(source_groups) - set(available)
        if missing:
            logger.warning("Requested source_groups not in dataset: %s", sorted(missing))

    rows: list[dict[str, Any]] = []

    for i in range(len(df)):
        question = df.loc[i, "question"]
        options_raw = df.loc[i, "options"]
        if not question or not options_raw:
            continue

        selections = _parse_selections(df.loc[i, "selections"])
        options = [str(opt) for opt in ast.literal_eval(options_raw)]

        for original_source_group in source_groups:
            if original_source_group not in selections:
                continue
            prob_y = selections[original_source_group]
            if prob_y is None or len(prob_y) == 0 or np.sum(prob_y) == 0:
                continue
            rows.append(
                {
                    "qkey": df.loc[i, "qkey"],
                    "question": question,
                    "options": options,
                    "source_group": original_source_group,
                    "prob_y": np.asarray(prob_y, dtype=np.float64),
                }
            )

    out = pd.DataFrame(rows)
    logger.info(
        "Built preference frame: %d rows, %d source groups, %d questions",
        len(out),
        out["source_group"].nunique() if len(out) else 0,
        out["qkey"].nunique() if len(out) else 0,
    )
    return out

def load_goqa_preferences(
    dataset_name: str = "Anthropic/llm_global_opinions",
    source_groups: list[str] | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """End-to-end loader used by the discovery pipeline (aggregate format)."""
    raw = load_raw_goqa(dataset_name=dataset_name, cache_dir=cache_dir)
    return build_preference_frame(raw, source_groups=source_groups)


def build_pairwise_frame(
    df: pd.DataFrame,
    source_groups: list[str] | None = None,
    n_simulated_per_group: int = 10,
    n_samples_per_question: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Convert GOQA aggregate distributions into simulated individual-level pairwise records.

    For each source_group, creates ``n_simulated_per_group`` synthetic individuals.
    Each individual samples ``n_samples_per_question`` (chosen, rejected) pairs per question
    from that group's probability distribution.

    Returns columns: entity_id, qkey, prompt, chosen, rejected, source_group, prob_chosen
    """
    available = _discover_source_groups(df)
    if source_groups is None:
        source_groups = available

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    entity_counter = 0

    for sg in source_groups:
        for sim_idx in range(n_simulated_per_group):
            entity_id = f"entity_{entity_counter:04d}"
            entity_counter += 1

            for i in range(len(df)):
                question = df.loc[i, "question"]
                options_raw = df.loc[i, "options"]
                if not question or not options_raw:
                    continue

                selections = _parse_selections(df.loc[i, "selections"])
                if sg not in selections:
                    continue

                prob_y = selections[sg]
                if prob_y is None or len(prob_y) == 0 or np.sum(prob_y) == 0:
                    continue

                options = [str(opt) for opt in ast.literal_eval(options_raw)]
                probs = np.asarray(prob_y, dtype=np.float64)
                probs = probs / probs.sum()
                n_opts = len(probs)
                if n_opts < 2:
                    continue

                options_str = " | ".join(options)
                prompt = f"{question}\nOptions: {options_str}"

                for _ in range(n_samples_per_question):
                    chosen_idx = int(rng.choice(n_opts, p=probs))
                    remaining = [j for j in range(n_opts) if j != chosen_idx]
                    rejected_idx = int(rng.choice(remaining))

                    rows.append(
                        {
                            "entity_id": entity_id,
                            "qkey": df.loc[i, "qkey"],
                            "prompt": prompt,
                            "chosen": options[chosen_idx],
                            "rejected": options[rejected_idx],
                            "source_group": sg,
                            "prob_chosen": float(probs[chosen_idx]),
                        }
                    )

    out = pd.DataFrame(rows)
    logger.info(
        "Built pairwise frame: %d records, %d entities, %d source groups",
        len(out),
        out["entity_id"].nunique() if len(out) else 0,
        out["source_group"].nunique() if len(out) else 0,
    )
    return out


def load_goqa_pairwise(
    dataset_name: str = "Anthropic/llm_global_opinions",
    source_groups: list[str] | None = None,
    n_simulated_per_group: int = 10,
    n_samples_per_question: int = 3,
    random_state: int = 42,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """End-to-end loader for GOQA in pairwise format (simulated individuals)."""
    raw = load_raw_goqa(dataset_name=dataset_name, cache_dir=cache_dir)
    return build_pairwise_frame(
        raw,
        source_groups=source_groups,
        n_simulated_per_group=n_simulated_per_group,
        n_samples_per_question=n_samples_per_question,
        random_state=random_state,
    )

