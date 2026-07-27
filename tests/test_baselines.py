"""Tests for baseline discovery methods."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.clustering.methods.baselines import (
    run_country_oracle,
    run_random_assignment,
    run_single_group,
)


def _mock_preference_and_entities():
    """Create minimal mock data for baseline tests."""
    pref_rows = []
    for i in range(6):
        sg = "CountryA" if i < 3 else "CountryB"
        pref_rows.append(
            {
                "qkey": 0,
                "question": "Test question?",
                "options": ["Yes", "No"],
                "source_group": sg,
                "prob_y": np.array([0.7, 0.3] if sg == "CountryA" else [0.3, 0.7]),
            }
        )

    pref_df = pd.DataFrame(pref_rows)

    entity_registry = pd.DataFrame(
        {
            "entity_id": ["entity_0000", "entity_0001", "entity_0002"],
            "source_group": ["CountryA", "CountryA", "CountryB"],
        }
    )

    config = {
        "clustering": {"n_clusters": 2, "random_state": 42},
        "export": {"dataset_prefix": "goqa_cluster"},
    }

    return pref_df, entity_registry, config


class TestSingleGroup(unittest.TestCase):
    def test_all_entities_in_cluster_zero(self):
        pref_df, entity_registry, config = _mock_preference_and_entities()
        assignments, extras = run_single_group(pref_df, entity_registry, config)

        self.assertEqual(len(assignments), 3)
        self.assertTrue((assignments["cluster_id"] == 0).all())
        self.assertEqual(extras["discovery_method"], "single_group")
        self.assertEqual(extras["discovery_tier"], "baseline")


class TestRandomAssignment(unittest.TestCase):
    def test_produces_k_groups(self):
        pref_df, entity_registry, config = _mock_preference_and_entities()
        assignments, extras = run_random_assignment(pref_df, entity_registry, config)

        self.assertEqual(len(assignments), 3)
        # Should have at most K unique cluster ids
        self.assertLessEqual(assignments["cluster_id"].nunique(), 2)
        self.assertEqual(extras["discovery_method"], "random_assignment")

    def test_reproducible_with_seed(self):
        pref_df, entity_registry, config = _mock_preference_and_entities()
        a1, _ = run_random_assignment(pref_df, entity_registry, config)
        a2, _ = run_random_assignment(pref_df, entity_registry, config)

        pd.testing.assert_frame_equal(a1.reset_index(drop=True), a2.reset_index(drop=True))


class TestCountryOracle(unittest.TestCase):
    def test_maps_source_groups_to_clusters(self):
        pref_df, entity_registry, config = _mock_preference_and_entities()
        assignments, extras = run_country_oracle(pref_df, entity_registry, config)

        self.assertEqual(len(assignments), 3)
        # 2 countries → 2 clusters
        self.assertEqual(assignments["cluster_id"].nunique(), 2)
        self.assertEqual(extras["discovery_method"], "country_oracle")

        # All CountryA entities should be in the same cluster
        merged = assignments.merge(entity_registry, on="entity_id")
        for _, group in merged.groupby("source_group"):
            self.assertEqual(group["cluster_id"].nunique(), 1)

    def test_raises_without_source_group(self):
        pref_df, _, config = _mock_preference_and_entities()
        entity_registry_no_sg = pd.DataFrame(
            {"entity_id": ["entity_0000", "entity_0001", "entity_0002"]}
        )
        with self.assertRaises(ValueError):
            run_country_oracle(pref_df, entity_registry_no_sg, config)


if __name__ == "__main__":
    unittest.main()
