# grpo-group-discovery

Discover preference groups for GRPO training — runs **before** [grpo-reproduction](https://github.com/).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Floor baseline: preference-similarity discovery
sh scripts/run_baseline.sh

# Future methods (stubs until implemented)
python3 discover.py --config config/methods/cross_predictive.yaml
```

Outputs: `outputs/<run_name>/` — see [docs/artifact-format.md](docs/artifact-format.md).

Method hierarchy: [docs/methods.md](docs/methods.md).

## Pipeline

```
GOQA preference records
        ↓
opaque entity_id registry
        ↓
discovery method (preference_similarity → methods 1–5)
        ↓
export cluster assignments
        ↓
grpo-reproduction (GR-IPO per discovered group)
```

## Config

- `config/preference_similarity.yaml` — floor baseline
- `config/methods/*.yaml` — advanced discovery methods

## Project layout

See [project_structure.md](project_structure.md).
