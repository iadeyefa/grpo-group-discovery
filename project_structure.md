# Project structure

```
grpo-group-discovery/
├── discover.py              # CLI entry: load → feature → cluster → export
├── config/
│   └── default.yaml         # Clustering hyperparameters
├── scripts/
│   └── run_clustering.sh    # Convenience wrapper
├── src/
│   ├── data/
│   │   └── load_goqa.py     # Load Global Opinion QA from Hugging Face
│   ├── features/
│   │   └── opinion_vectors.py  # Build country-level feature matrices
│   ├── clustering/
│   │   ├── cluster.py       # K-means (extensible to other algorithms)
│   │   └── stability.py     # Bootstrap stability checks
│   ├── export/
│   │   └── artifacts.py     # Write parquet + metadata for grpo-reproduction
│   └── utils.py             # Config loading, logging helpers
├── docs/
│   └── artifact-format.md   # Export contract for grpo-reproduction
└── outputs/                 # Generated artifacts (gitignored)
```

## Pipeline

1. **Load** — `src/data/load_goqa.py` pulls Anthropic/llm_global_opinions.
2. **Features** — `src/features/opinion_vectors.py` builds per-country vectors.
3. **Cluster** — `src/clustering/cluster.py` assigns countries to groups.
4. **Export** — `src/export/artifacts.py` writes `cluster_assignments.parquet` + `metadata.json`.

Downstream: point `grpo-reproduction` at exported group names (e.g. `goqa_cluster_0`).
