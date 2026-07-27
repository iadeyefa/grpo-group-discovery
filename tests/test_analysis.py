"""Unit tests for intrinsic preference evaluation module."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.evaluate import evaluate_discovery_run
from src.analysis.metrics import (
    compute_cluster_cohesion,
    compute_cluster_separation,
    compute_entropy_reduction,
    compute_shannon_entropy,
)
from src.analysis.polarization import extract_top_polarizing_questions


class TestAnalysisModule(unittest.TestCase):
    def setUp(self):
        """Create mock preference records and cluster assignments with 2 clear groups."""
        rows = []
        assignments_rows = []

        for i in range(4):
            entity_id = f"entity_{i:04d}"
            cluster_id = 0 if i < 2 else 1
            assignments_rows.append({"entity_id": entity_id, "cluster_id": cluster_id})

            for q in range(3):
                if q == 0:
                    prob = [0.9, 0.1] if cluster_id == 0 else [0.1, 0.9]
                elif q == 1:
                    prob = [0.7, 0.3] if cluster_id == 0 else [0.3, 0.7]
                else:
                    prob = [0.5, 0.5]

                rows.append(
                    {
                        "entity_id": entity_id,
                        "qkey": q,
                        "question": f"Question {q} text?",
                        "options": ["Option A", "Option B"],
                        "source_group": "Group_X" if i < 2 else "Group_Y",
                        "prob_y": np.array(prob, dtype=np.float64),
                    }
                )

        self.pref_df = pd.DataFrame(rows)
        self.assign_df = pd.DataFrame(assignments_rows)

    def test_compute_shannon_entropy(self):
        prob = np.array([0.5, 0.5])
        self.assertAlmostEqual(compute_shannon_entropy(prob), 1.0, places=3)

        prob_det = np.array([1.0, 0.0])
        self.assertAlmostEqual(compute_shannon_entropy(prob_det), 0.0, places=3)

    def test_compute_entropy_reduction(self):
        results = compute_entropy_reduction(self.pref_df, self.assign_df)
        self.assertIn("pooled_entropy", results)
        self.assertIn("weighted_cluster_entropy", results)
        self.assertIn("entropy_reduction", results)
        self.assertGreaterEqual(results["entropy_reduction"], 0.0)

    def test_compute_cluster_cohesion(self):
        results = compute_cluster_cohesion(self.assign_df, self.pref_df)
        self.assertIn("overall_cohesion", results)
        self.assertIn("per_cluster_cohesion", results)
        self.assertTrue(0.0 <= results["overall_cohesion"] <= 1.0)

    def test_compute_cluster_separation(self):
        results = compute_cluster_separation(self.assign_df, self.pref_df)
        self.assertIn("mean_inter_cluster_jsd", results)
        self.assertIn("jsd_matrix", results)
        self.assertGreater(results["mean_inter_cluster_jsd"], 0.0)

    def test_extract_top_polarizing_questions(self):
        results = extract_top_polarizing_questions(self.pref_df, self.assign_df, top_n=2)
        top_qs = results["top_questions"]
        self.assertEqual(len(top_qs), 2)
        self.assertEqual(top_qs[0]["qkey"], 0)
        self.assertIn("markdown_report", results)
        self.assertIn("Question 0 text?", results["markdown_report"])

    def test_evaluate_discovery_run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            metrics = evaluate_discovery_run(self.assign_df, self.pref_df, output_dir=out_path)

            self.assertTrue((out_path / "evaluation_metrics.json").exists())
            self.assertTrue((out_path / "polarizing_questions.md").exists())
            self.assertIn("entropy", metrics)
            self.assertIn("cohesion", metrics)
            self.assertIn("separation", metrics)
            self.assertIn("top_polarizing_questions", metrics)


if __name__ == "__main__":
    unittest.main()
