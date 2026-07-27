# Project structure

```
grpo-group-discovery/
├── discover.py                         # Main CLI entry point
├── config/
│   ├── baselines/                      # Reference anchors & controls
│   │   ├── single_group.yaml           # Lower bound (pooled data)
│   │   ├── random_assignment.yaml      # Stochastic chance control
│   │   ├── preference_similarity.yaml  # Floor baseline (KMeans)
│   │   └── country_oracle.yaml         # Demographic upper bound
│   └── discovery/                      # Core discovery algorithms (1–3)
│       ├── cross_predictive.yaml       # Method 1: Log-prob conditioning transfer
│       ├── embedding_sets.yaml         # Method 2: Chamfer distance on vector sets
│       └── matrix_factorization.yaml   # Method 3: Sparse collaborative filtering
├── src/
│   ├── data/
│   │   ├── load_goqa.py                # GOQA aggregate + pairwise loaders
│   │   ├── load_pairwise.py            # Generic CSV/JSONL pairwise loader
│   │   ├── entities.py                 # Entity registry (observed/simulated/blind/individual)
│   │   └── schema.py                   # Canonical schema + validation
│   ├── features/
│   │   └── preference_vectors.py       # Aggregate feature matrix builder
│   ├── clustering/
│   │   ├── dispatch.py                 # Method dispatch + metadata
│   │   ├── stability.py                # Bootstrap stability checks
│   │   └── methods/
│   │       ├── baselines.py            # single_group, random_assignment, country_oracle
│   │       ├── preference_similarity.py
│   │       ├── cross_predictive.py
│   │       ├── embedding_sets.py
│   │       └── matrix_factorization.py
│   ├── analysis/
│   │   ├── metrics.py                  # Entropy, cohesion, separation, demographic overlay
│   │   ├── polarization.py             # Top polarizing question extraction
│   │   ├── eval_prediction.py          # Held-out preference prediction
│   │   └── evaluate.py                 # End-to-end evaluation runner
│   ├── export/
│   │   └── artifacts.py                # Parquet/CSV/JSON/MD/PNG export
│   └── utils.py
├── tests/
│   ├── test_analysis.py
│   └── test_baselines.py
├── docs/
│   ├── artifact-format.md
│   └── methods.md
└── outputs/
```

See [docs/methods.md](docs/methods.md) for method descriptions and evaluation details.
