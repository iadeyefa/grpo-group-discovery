# Artifact format

Exports from `discover.py` are written to `outputs/<run_name>/`.

## Files

| File | Description |
|------|-------------|
| `cluster_assignments.parquet` | One row per country: `country`, `cluster_id` |
| `metadata.json` | Run config, cluster sizes, dataset names |
| `dataset_group_map.json` | Maps `goqa_cluster_N` → list of countries |
| `cluster_sizes.png` | Bar chart of countries per cluster |

## `cluster_assignments.parquet`

```
country                              cluster_id
Nigeria                              0
Egypt                                0
China                                1
...
```

## `dataset_group_map.json`

```json
{
  "goqa_cluster_0": ["Egypt", "Nigeria"],
  "goqa_cluster_1": ["China", "Japan"]
}
```

## Integration with grpo-reproduction

This repo produces **group definitions only**. Training still happens in [grpo-reproduction](https://github.com/...).

To consume discovered groups:

1. Copy `dataset_group_map.json` into the training repo (or reference by path).
2. Add a loader branch in `src/preference_datasets.py` that:
   - Reads the cluster map
   - Filters `get_goqa` by country lists per `goqa_cluster_N`
3. Train with `datasets=[goqa_cluster_0, goqa_cluster_1, ...]` as in the fixed-country setup.

The per-prompt data contract in grpo-reproduction is unchanged:

```python
{
  prompt: {
    "responses": [...],
    "pairs": [(chosen_idx, rejected_idx), ...],
    "sft_target": "A",
  }
}
```

## Versioning

- Commit or store `metadata.json` alongside W&B runs.
- Re-clustering should produce a new `run_name`; do not overwrite prior artifacts.
