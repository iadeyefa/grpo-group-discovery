# Artifact format

Exports from `discover.py` are written to `outputs/<run_name>/`.

## Files

| File | Description |
|------|-------------|
| `cluster_assignments.parquet` | `entity_id`, `cluster_id` — clustering output only |
| `entity_registry.json` | `entity_id` → HF `source_group` (downstream + eval, not clustering) |
| `metadata.json` | Run config, `discovery_method`, `discovery_tier`, cluster sizes |
| `dataset_group_map.json` | `goqa_cluster_N` → HF source_group labels for grpo-reproduction |
| `evaluation_source_groups.json` | Post-hoc source_group counts per cluster |
| `cluster_sizes.png` | Bar chart of entities per cluster |

## Clustering contract

1. **Load** preference records (opinion distributions per question per source group).
2. **Register** opaque `entity_id` values from preference data.
3. **Discover** groups via `preference_similarity` (floor baseline) or methods 1–5.
4. **Evaluate** using `evaluation_source_groups.json` — hidden population structure never enters discovery.

## `cluster_assignments.parquet`

```
entity_id    cluster_id
entity_0000  0
entity_0001  2
entity_0002  1
```

## `entity_registry.json`

```json
{
  "entity_0000": "Nigeria",
  "entity_0001": "Egypt"
}
```

## `dataset_group_map.json`

```json
{
  "goqa_cluster_0": ["Egypt", "Nigeria"],
  "goqa_cluster_1": ["China", "Japan"]
}
```

Values are HF source_group labels used by grpo-reproduction to filter training data.

## `evaluation_source_groups.json`

```json
{
  "0": {"Egypt": 1, "Nigeria": 1},
  "1": {"China": 1, "Japan": 1}
}
```

## Integration with grpo-reproduction

1. Copy `dataset_group_map.json` and `entity_registry.json` into the training repo.
2. Filter `get_goqa` by source_group lists per `goqa_cluster_N`.
3. Train with `datasets=[goqa_cluster_0, goqa_cluster_1, ...]`.

## Versioning

- Commit `metadata.json` alongside W&B runs.
- Re-clustering should produce a new `run_name`; do not overwrite prior artifacts.
