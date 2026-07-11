#!/usr/bin/env python3
"""Discover preference groups via clustering and export artifacts for GRPO training."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.clustering.cluster import clustering_metadata, run_clustering
from src.clustering.random import run_random_clustering
from src.clustering.stability import bootstrap_stability
from src.data.entities import attach_entity_ids, build_entity_table
from src.data.load_goqa import load_goqa_preferences
from src.export.artifacts import export_all
from src.features.preference_vectors import build_preference_feature_matrix
from src.utils import get_cache_dir, load_config, resolve_output_dir, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster preference profiles into groups for GRPO training."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
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
    feature_cfg = config.get("features", {})
    cluster_cfg = config.get("clustering", {})
    stability_cfg = config.get("stability", {})
    export_cfg = config.get("export", {})

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
    entity_registry = build_entity_table(
        preference_df,
        mode=entity_cfg.get("mode", "observed"),
        n_simulated_per_group=entity_cfg.get("n_simulated_per_group", 10),
        random_state=entity_cfg.get("random_state", cluster_cfg.get("random_state", 42)),
    )

    algorithm = cluster_cfg.get("algorithm", "kmeans")
    n_clusters = cluster_cfg.get("n_clusters", 5)
    random_state = cluster_cfg.get("random_state", 42)

    if algorithm == "random":
        logger.info(
            "Stage 3/4: random baseline — %d entities -> %d clusters "
            "(preference-blind null hypothesis)",
            len(entity_registry),
            n_clusters,
        )
        assignments = run_random_clustering(
            entities=entity_registry,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        metadata = clustering_metadata(
            config,
            assignments,
            feature_dim=0,
            baseline="random",
            uses_preference_features=False,
        )
    else:
        logger.info("Stage 3/4: building preference feature vectors")
        preference_with_entities = attach_entity_ids(preference_df, entity_registry)
        features_df = build_preference_feature_matrix(
            preference_with_entities,
            method=feature_cfg.get("method", "mean_opinion_vector"),
            normalize_vectors=feature_cfg.get("normalize", True),
        )

        logger.info("Stage 3/4: clustering by preference similarity (%s)", algorithm)
        assignments = run_clustering(
            features_df,
            algorithm=algorithm,
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=cluster_cfg.get("n_init", 10),
        )

        metadata = clustering_metadata(
            config,
            assignments,
            feature_dim=features_df.shape[1],
            uses_preference_features=True,
        )

        if stability_cfg.get("enabled", False):
            logger.info("Running bootstrap stability check")
            metadata["stability"] = bootstrap_stability(
                features_df,
                n_clusters=n_clusters,
                n_bootstrap=stability_cfg.get("n_bootstrap", 20),
                subsample_frac=stability_cfg.get("subsample_frac", 0.8),
                random_state=random_state,
                n_init=cluster_cfg.get("n_init", 10),
            )

    logger.info("Stage 4/4: exporting artifacts")
    output_dir = resolve_output_dir(config)
    paths = export_all(
        assignments=assignments,
        entity_registry=entity_registry,
        metadata=metadata,
        output_dir=output_dir,
        dataset_prefix=export_cfg.get("dataset_prefix", "goqa_cluster"),
    )

    logger.info("Done. Artifacts written to %s", output_dir)
    for name, path in paths.items():
        logger.info("  %s: %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
