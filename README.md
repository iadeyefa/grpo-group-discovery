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
# Baselines
python3 discover.py --config config/methods/single_group.yaml
python3 discover.py --config config/methods/random_assignment.yaml
python3 discover.py --config config/methods/country_oracle.yaml

# Floor baseline
python3 discover.py --config config/preference_similarity.yaml

# Discovery methods 1–5
python3 discover.py --config config/methods/embedding_sets.yaml
python3 discover.py --config config/methods/cross_predictive.yaml
python3 discover.py --config config/methods/matrix_factorization.yaml
python3 discover.py --config config/methods/latent_class.yaml
python3 discover.py --config config/methods/agreement_graph.yaml
```

Outputs: `outputs/<run_name>/` — see [docs/artifact-format.md](docs/artifact-format.md).

## Pipeline

```
Preference records (GOQA / pairwise dataset)
        ↓
Opaque entity_id registry
        ↓
Discovery method (baselines → methods 1–5)
        ↓
Evaluation (entropy, cohesion, separation, held-out prediction, demographic overlay)
        ↓
Export cluster assignments + polarizing questions report
        ↓
grpo-reproduction (GR-IPO per discovered group)
```

## Config

- `config/preference_similarity.yaml` — floor baseline
- `config/methods/*.yaml` — baselines + discovery methods

## Docs

- [docs/methods.md](docs/methods.md) — method descriptions, comparison ladder, evaluation metrics
- [docs/artifact-format.md](docs/artifact-format.md) — output file schemas
- [project_structure.md](project_structure.md) — codebase layout

## Tests

```bash
.venv/bin/python -m unittest discover tests/
```
