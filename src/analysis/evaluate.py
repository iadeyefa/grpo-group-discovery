"""End-to-end evaluation runner for preference group discovery runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.eval_prediction import compute_cluster_prediction_score
from src.analysis.metrics import (
    compute_cluster_cohesion,
    compute_cluster_separation,
    compute_demographic_overlay,
    compute_entropy_reduction,
)
from src.analysis.polarization import extract_top_polarizing_questions

logger = logging.getLogger(__name__)


def evaluate_discovery_run(
    assignments: pd.DataFrame,
    preference_with_entities: pd.DataFrame,
    output_dir: Path | None = None,
    top_n_polarizing: int = 10,
) -> dict[str, Any]:
    """Run all evaluation metrics for a clustering run.

    Computes intrinsic preference metrics, held-out prediction validation,
    demographic overlay, and polarizing question extraction.
    """
    logger.info("Evaluating discovery run metrics...")

    entropy_metrics = compute_entropy_reduction(preference_with_entities, assignments)
    cohesion_metrics = compute_cluster_cohesion(assignments, preference_with_entities)
    separation_metrics = compute_cluster_separation(assignments, preference_with_entities)
    prediction_metrics = compute_cluster_prediction_score(
        assignments, preference_with_entities
    )
    demographic_metrics = compute_demographic_overlay(
        assignments, preference_with_entities
    )
    polarization_results = extract_top_polarizing_questions(
        preference_with_entities, assignments, top_n=top_n_polarizing
    )

    evaluation_summary: dict[str, Any] = {
        "entropy": entropy_metrics,
        "cohesion": cohesion_metrics,
        "separation": separation_metrics,
        "prediction": prediction_metrics,
        "top_polarizing_questions": polarization_results["top_questions"],
    }
    if demographic_metrics:
        evaluation_summary["demographic_overlay"] = demographic_metrics

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with (output_dir / "evaluation_metrics.json").open("w") as f:
            json.dump(evaluation_summary, f, indent=2)
        logger.info("Wrote %s", output_dir / "evaluation_metrics.json")

        with (output_dir / "polarizing_questions.md").open("w") as f:
            f.write(polarization_results["markdown_report"])
        logger.info("Wrote %s", output_dir / "polarizing_questions.md")

    return evaluation_summary
