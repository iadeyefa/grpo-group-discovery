"""Identify and extract top polarizing questions across discovered preference clusters."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_top_polarizing_questions(
    preference_with_entities: pd.DataFrame,
    assignments: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Find questions with the largest preference divergence between discovered clusters.

    Args:
        preference_with_entities: Must contain entity_id, qkey, question, options, prob_y
        assignments: Must contain entity_id, cluster_id
        top_n: Number of top polarizing questions to return

    Returns:
        dict containing:
            top_questions: list of question dicts with cluster probability breakdown
            markdown_report: formatted markdown string for report export
    """
    merged = preference_with_entities.merge(assignments, on="entity_id", how="inner")
    if merged.empty:
        logger.warning("No records found when merging preferences with assignments")
        return {"top_questions": [], "markdown_report": "# Top Polarizing Questions\n\nNo records."}

    cluster_ids = sorted(merged["cluster_id"].unique())
    k = len(cluster_ids)

    if k <= 1:
        logger.info("Only 1 cluster discovered; no polarizing questions can be computed.")
        return {
            "top_questions": [],
            "markdown_report": "# Top Polarizing Questions\n\nOnly 1 cluster discovered.",
        }

    question_analysis = []

    for qkey, q_group in merged.groupby("qkey"):
        question_text = q_group["question"].iloc[0]
        options = q_group["options"].iloc[0]

        # Compute per-cluster average probability distribution for this question
        cluster_probs: dict[int, np.ndarray] = {}
        for c_id in cluster_ids:
            c_records = q_group[q_group["cluster_id"] == c_id]
            if not c_records.empty:
                arr = np.mean(np.stack(c_records["prob_y"].to_numpy()), axis=0)
            else:
                arr = np.zeros(len(options), dtype=np.float64)
            cluster_probs[int(c_id)] = arr

        # Compute max pairwise L1 divergence between clusters with valid observations
        max_l1 = 0.0
        max_pair = (cluster_ids[0], cluster_ids[1] if k > 1 else cluster_ids[0])
        valid_c_ids = [c_id for c_id in cluster_ids if not q_group[q_group["cluster_id"] == c_id].empty]
        if len(valid_c_ids) >= 2:
            for i in range(len(valid_c_ids)):
                for j in range(i + 1, len(valid_c_ids)):
                    c_i, c_j = valid_c_ids[i], valid_c_ids[j]
                    p_i = cluster_probs[c_i]
                    p_j = cluster_probs[c_j]

                    max_len = max(len(p_i), len(p_j))
                    if len(p_i) < max_len:
                        p_i = np.pad(p_i, (0, max_len - len(p_i)))
                    if len(p_j) < max_len:
                        p_j = np.pad(p_j, (0, max_len - len(p_j)))

                    l1_dist = float(0.5 * np.sum(np.abs(p_i - p_j)))
                    if l1_dist > max_l1:
                        max_l1 = l1_dist
                        max_pair = (c_i, c_j)

        # Convert numpy distributions to lists
        cluster_dist_dict = {
            str(c_id): [round(float(val), 4) for val in arr]
            for c_id, arr in cluster_probs.items()
        }

        question_analysis.append(
            {
                "qkey": int(qkey) if isinstance(qkey, (int, np.integer)) else str(qkey),
                "question": question_text,
                "options": options if isinstance(options, list) else list(options),
                "polarization_score": round(max_l1, 4),
                "max_divergent_clusters": [int(max_pair[0]), int(max_pair[1])],
                "cluster_distributions": cluster_dist_dict,
            }
        )

    # Sort descending by polarization_score
    question_analysis.sort(key=lambda x: x["polarization_score"], reverse=True)
    top_questions = question_analysis[:top_n]

    # Generate Markdown report
    md_lines = [
        "# Top Polarizing Questions Across Discovered Groups\n",
        f"Ranked by maximum inter-cluster preference divergence ($L_1$ distribution distance).\n",
    ]

    for rank, q in enumerate(top_questions, 1):
        md_lines.append(f"### {rank}. {q['question']}")
        md_lines.append(f"**Polarization Score**: `{q['polarization_score']:.4f}` | **Max Divergence Between**: Cluster {q['max_divergent_clusters'][0]} vs Cluster {q['max_divergent_clusters'][1]}\n")
        md_lines.append("| Option | " + " | ".join([f"Cluster {c_id}" for c_id in cluster_ids]) + " |")
        md_lines.append("| --- | " + " | ".join(["---"] * len(cluster_ids)) + " |")

        options = q["options"]
        for opt_idx, opt_text in enumerate(options):
            probs_str = []
            for c_id in cluster_ids:
                dist = q["cluster_distributions"].get(str(c_id), [])
                p_val = dist[opt_idx] if opt_idx < len(dist) else 0.0
                probs_str.append(f"{p_val * 100:.1f}%")
            md_lines.append(f"| {opt_text} | " + " | ".join(probs_str) + " |")
        md_lines.append("\n---\n")

    markdown_report = "\n".join(md_lines)
    logger.info("Extracted top %d polarizing questions (max score: %.4f)", len(top_questions), top_questions[0]["polarization_score"] if top_questions else 0.0)

    return {
        "top_questions": top_questions,
        "markdown_report": markdown_report,
    }
