#!/usr/bin/env python3
"""Discover preference groups via clustering and export artifacts for GRPO training."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.clustering.dispatch import (
    BASELINE_METHOD,
    clustering_metadata,
    resolve_method,
    run_discovery,
)
from src.clustering.stability import bootstrap_stability
from src.data.entities import ENTITY_MODE_BLIND, attach_entity_ids, build_entity_table
from src.data.load_goqa import load_goqa_preferences
from src.export.artifacts import export_all
from src.utils import get_cache_dir, load_config, resolve_output_dir, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover preference groups for GRPO training."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/baselines/preference_similarity.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    entity_cfg = config.get("entities", {})
    cluster_cfg = config.get("clustering", {})
    stability_cfg = config.get("stability", {})
    export_cfg = config.get("export", {})

    method = resolve_method(config)
    logger.info("Discovery method: %s (floor baseline: %s)", method, BASELINE_METHOD)

    logger.info("Stage 1/4: loading preference records")
    preference_df = load_goqa_preferences(
        dataset_name=dataset_cfg.get("name", "Anthropic/llm_global_opinions"),
        source_groups=dataset_cfg.get("source_groups"),
        cache_dir=get_cache_dir(config),
    )
    if preference_df.empty:
        logger.error("No preference data loaded — check source_groups filter and dataset access")
        return 1

    logger.info("Stage 2/4: registering entities from preference data")
    if entity_cfg.get("mode", "observed") == ENTITY_MODE_BLIND and method != BASELINE_METHOD:
        raise ValueError(
            "Blind entity mode is currently supported only for the preference_similarity baseline"
        )
    entity_registry = build_entity_table(
        preference_df,
        mode=entity_cfg.get("mode", "observed"),
        n_simulated_per_group=entity_cfg.get("n_simulated_per_group", 10),
        random_state=entity_cfg.get("random_state", cluster_cfg.get("random_state", 42)),
    )
    preference_with_entities = attach_entity_ids(preference_df, entity_registry)

    logger.info("Stage 3/4: running discovery (%s)", method)
    assignments, extras = run_discovery(
        method=method,
        preference_df=preference_df,
        entity_registry=entity_registry,
        config=config,
    )

    metadata = clustering_metadata(
        config,
        assignments,
        feature_dim=extras["feature_dim"],
        discovery_method=extras["discovery_method"],
        discovery_tier=extras["discovery_tier"],
        uses_preference_features=extras["uses_preference_features"],
    )

    features_df = extras.get("features_df")
    if stability_cfg.get("enabled", False) and features_df is not None:
        logger.info("Running bootstrap stability check")
        metadata["stability"] = bootstrap_stability(
            features_df,
            n_clusters=cluster_cfg.get("n_clusters", 5),
            n_bootstrap=stability_cfg.get("n_bootstrap", 20),
            subsample_frac=stability_cfg.get("subsample_frac", 0.8),
            random_state=cluster_cfg.get("random_state", 42),
            n_init=cluster_cfg.get("n_init", 10),
        )

    logger.info("Stage 4/4: exporting artifacts")
    if not export_cfg.get("run_name"):
        export_cfg["run_name"] = method

    output_dir = resolve_output_dir(config, overwrite=True)
    paths = export_all(
        assignments=assignments,
        preference_with_entities=preference_with_entities,
        metadata=metadata,
        output_dir=output_dir,
    )

    logger.info("Done. Artifacts written to %s", output_dir)
    for name, path in paths.items():
        logger.info("  %s: %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
