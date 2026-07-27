# grpo-group-discovery

Discover preference groups for GRPO training — runs **before** [grpo-reproduction](https://github.com/iadeyefa/grpo-reproduction).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Reference Baselines
python3 discover.py --config config/baselines/single_group.yaml
python3 discover.py --config config/baselines/random_assignment.yaml
python3 discover.py --config config/baselines/preference_similarity.yaml
python3 discover.py --config config/baselines/country_oracle.yaml

# Core Discovery Methods (Methods 1–3)
python3 discover.py --config config/discovery/cross_predictive.yaml
python3 discover.py --config config/discovery/embedding_sets.yaml
python3 discover.py --config config/discovery/matrix_factorization.yaml
```

Outputs: `outputs/<run_name>/` — see [docs/artifact-format.md](docs/artifact-format.md).

## Pipeline

```
Preference records (GOQA / pairwise dataset)
        ↓
Opaque entity_id registry
        ↓
Discovery method (baselines → discovery methods 1–3)
        ↓
Evaluation (entropy, cohesion, separation, held-out prediction, demographic overlay)
        ↓
Export cluster assignments + polarizing questions report
        ↓
grpo-reproduction (GR-IPO per discovered group)
```

## Config Organization

- `config/baselines/` — Control anchors (single_group, random_assignment, preference_similarity, country_oracle)
- `config/discovery/` — Core discovery algorithms (cross_predictive, embedding_sets, matrix_factorization)

## Docs

- [docs/methods.md](docs/methods.md) — method descriptions, comparison ladder, evaluation metrics
- [docs/artifact-format.md](docs/artifact-format.md) — output file schemas
- [project_structure.md](project_structure.md) — codebase layout

## Tests

```bash
.venv/bin/python -m unittest discover tests/
```
