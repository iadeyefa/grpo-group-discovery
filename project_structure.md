# Project structure

```
grpo-group-discovery/
├── discover.py
├── config/
│   ├── preference_similarity.yaml   # Floor baseline
│   └── methods/                     # Advanced discovery methods 1–5
├── scripts/
│   └── run_baseline.sh
├── src/
│   ├── data/
│   │   ├── load_goqa.py
│   │   └── entities.py
│   ├── features/
│   │   └── preference_vectors.py
│   ├── clustering/
│   │   ├── dispatch.py
│   │   ├── stability.py
│   │   └── methods/
│   ├── export/
│   │   └── artifacts.py
│   └── utils.py
├── docs/
│   ├── artifact-format.md
│   └── methods.md
└── outputs/
```

See [docs/methods.md](docs/methods.md).
