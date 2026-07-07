# grpo-group-discovery

Discover and export preference groups for GRPO training from opinion/preference data.

Pre-GRPO clustering pipeline: load Global Opinion QA → build country features → cluster → export artifacts for [grpo-reproduction](https://github.com/) (SFT → IPO / GR-IPO).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Default config (K=5 KMeans on mean opinion vectors)
sh scripts/run_clustering.sh

# Custom config / debug logs
python discover.py --config config/default.yaml --verbose
```

Outputs land in `outputs/<run_name>/` — see [docs/artifact-format.md](docs/artifact-format.md).

## Pipeline

```
Anthropic/llm_global_opinions
        │
        ▼
  load_goqa.py          # per-country opinion rows
        │
        ▼
  opinion_vectors.py    # country feature matrix
        │
        ▼
  cluster.py            # KMeans → cluster_id per country
        │
        ▼
  artifacts.py          # parquet + JSON + plot
        │
        ▼
  grpo-reproduction     # train on goqa_cluster_0, ...
```

## Config

Edit `config/default.yaml`:

- `clustering.n_clusters` — number of groups (default 5, matching paper)
- `countries` — restrict which countries to cluster (`null` = all)
- `export.dataset_prefix` — prefix for downstream dataset names

## Project layout

See [project_structure.md](project_structure.md).

## License

See [LICENSE](LICENSE).
