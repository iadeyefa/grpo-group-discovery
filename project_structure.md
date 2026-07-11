# Project structure

```
grpo-group-discovery/
├── discover.py              # CLI: load preferences → features → cluster → export
├── config/
│   ├── default.yaml         # KMeans on preference similarity
│   └── random.yaml          # Random baseline (preference-blind null)
├── scripts/
│   ├── run_clustering.sh
│   └── run_random_baseline.sh
├── src/
│   ├── data/
│   │   ├── load_goqa.py     # Load GOQA preference records
│   │   └── entities.py      # Opaque entity_id registry from preference data
│   ├── features/
│   │   └── preference_vectors.py  # Per-entity opinion feature matrix
│   ├── clustering/
│   │   ├── cluster.py       # KMeans on preference similarity
│   │   ├── random.py        # Random baseline
│   │   └── stability.py     # Bootstrap stability checks
│   ├── export/
│   │   └── artifacts.py     # Parquet + JSON for grpo-reproduction
│   └── utils.py
├── docs/
│   └── artifact-format.md
└── outputs/
```

## Pipeline

1. **Load** — `load_goqa.py` → preference records (`source_group`, `prob_y`).
2. **Register** — `entities.py` → opaque `entity_id` per preference profile.
3. **Features** — `preference_vectors.py` → similarity-ready vectors (skipped for random).
4. **Cluster** — `cluster.py` (preference similarity) or `random.py` (null baseline).
5. **Export** — assignments + `entity_registry.json` + evaluation breakdown.

## Baselines

| Baseline | Config | Uses preferences? |
|----------|--------|-------------------|
| Random K groups | `config/random.yaml` | No |
| Single-group IPO/DPO | grpo-reproduction | Yes |
| Preference similarity | `config/default.yaml` | Yes |

Post-hoc population alignment: `evaluation_source_groups.json`.
