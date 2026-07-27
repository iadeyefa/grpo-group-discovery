# Artifact format

Exports from `discover.py` are written to `outputs/<run_name>/`.

## Files

| File | Description |
|------|-------------|
| `cluster_assignments.parquet` | `entity_id`, `cluster_id` — primary clustering output |
| `cluster_assignments.csv` | Same as above plus `source_group` if present |
| `cluster_records.json` | Per-cluster preference records with questions and distributions |
| `cluster_summary.md` | Human-readable summary of cluster sizes |
| `evaluation_metrics.json` | All evaluation metrics (see schema below) |
| `polarizing_questions.md` | Top questions with maximum inter-cluster preference divergence |
| `metadata.json` | Run config, method, cluster sizes, evaluation summary |
| `cluster_sizes.png` | Bar chart of entities per cluster |

## `cluster_assignments.parquet`

```
entity_id    cluster_id
entity_0000  0
entity_0001  2
entity_0002  1
```

## `evaluation_metrics.json`

```json
{
  "entropy": {
    "pooled_entropy": 1.5669,
    "weighted_cluster_entropy": 1.4120,
    "entropy_reduction": 0.1549,
    "relative_entropy_drop": 0.0989
  },
  "cohesion": {
    "overall_cohesion": 0.8421,
    "per_cluster_cohesion": {"0": 0.865, "1": 0.819}
  },
  "separation": {
    "mean_inter_cluster_jsd": 0.4215,
    "calinski_harabasz_score": 1254.32,
    "davies_bouldin_score": 1.12
  },
  "prediction": {
    "cluster_accuracy": 0.72,
    "baseline_accuracy": 0.58,
    "prediction_lift": 0.14
  },
  "demographic_overlay": {
    "adjusted_rand_index": 0.23,
    "normalized_mutual_info": 0.31
  }
}
```

## Integration with grpo-reproduction

1. Copy `cluster_assignments.parquet` into `grpo-reproduction`.
2. Group training samples by `cluster_id`.
3. Train with GR-IPO per discovered preference cluster.

## Versioning

- Commit `metadata.json` alongside W&B runs.
- Re-clustering should produce a new `run_name`; do not overwrite prior artifacts.
