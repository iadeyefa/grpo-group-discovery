# grpo-group-discovery

Discover and export preference groups for GRPO training from opinion/preference data.

Pipeline: load GOQA preference records → build per-entity feature vectors → cluster by similarity → export artifacts for [grpo-reproduction](https://github.com/) (SFT → IPO / GR-IPO).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Preference-similarity clustering (KMeans on opinion vectors)
sh scripts/run_clustering.sh

# Random baseline (preference-blind null hypothesis)
sh scripts/run_random_baseline.sh

# Custom config / debug logs
python3 discover.py --config config/default.yaml --verbose
```

Outputs land in `outputs/<run_name>/` — see [docs/artifact-format.md](docs/artifact-format.md).

## Pipeline

```
Anthropic/llm_global_opinions
        │
        ▼
  load_goqa.py              # preference records per source group
        │
        ▼
  entities.py               # opaque entity_id registry
        │
        ▼
  preference_vectors.py     # per-entity opinion feature matrix
        │
        ▼
  cluster.py / random.py    # similarity clustering or random baseline
        │
        ▼
  artifacts.py              # parquet + JSON + evaluation breakdown
        │
        ▼
  grpo-reproduction         # train on goqa_cluster_0, ...
```

## Config

Edit `config/default.yaml`:

- `clustering.n_clusters` — number of groups (default 5)
- `dataset.source_groups` — filter HF population groups (`null` = all)
- `export.dataset_prefix` — prefix for downstream dataset names

## Project layout

See [project_structure.md](project_structure.md).

## License

See [LICENSE](LICENSE).
