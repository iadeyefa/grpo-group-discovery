# Discovery methods

Pre-GRPO group discovery runs in this repo **before** grpo-reproduction. This repo discovers K groups; grpo-reproduction trains GR-IPO on those groups.

## Comparison ladder

| Tier | Method | Where | Uses preferences? |
|------|--------|-------|-------------------|
| **No discovery** | Single-group IPO/DPO | grpo-reproduction | Yes (pooled) |
| **Floor baseline** | `preference_similarity` | This repo | Yes |
| **Methods 1–5** | See below | This repo | Yes |
| **Post-hoc eval** | Hidden source_group alignment | `evaluation_source_groups.json` | N/A |

Single-group IPO/DPO skips this repo entirely.

## Floor baseline: `preference_similarity`

Config: `config/preference_similarity.yaml`

- Build mean opinion vector per entity from GOQA `prob_y`
- L2-normalize
- KMeans in similarity space

Simplest preference-aware discovery. Methods 1–5 should beat this.

## Methods 1–5 (implement on top of floor baseline)

| # | Method | Config | Module | Status |
|---|--------|--------|--------|--------|
| 1 | Cross-predictive similarity | `config/methods/cross_predictive.yaml` | `methods/cross_predictive.py` | Stub |
| 2 | Chosen-rejected embedding sets | `config/methods/embedding_sets.yaml` | `methods/embedding_sets.py` | Stub |
| 3 | Sparse matrix factorization | `config/methods/matrix_factorization.yaml` | `methods/matrix_factorization.py` | Stub |
| 4 | Latent class preference model | `config/methods/latent_class.yaml` | `methods/latent_class.py` | Stub |
| 5 | Agreement graph | `config/methods/agreement_graph.yaml` | `methods/agreement_graph.py` | Stub |

Run a method once implemented:

```bash
python3 discover.py --config config/methods/cross_predictive.yaml
```

## GOQA data note

Methods assume individual-level preference records. GOQA provides population-level aggregates.

- **`entities.mode: observed`** — one entity per HF source group (pilot)
- **`entities.mode: simulated`** — synthetic individuals per source group (for methods 1–2)

## Evaluation

- **Discovered groups:** cohesion, stability, worst-group GRPO performance
- **Hidden structure:** `evaluation_source_groups.json` — source_group counts per cluster (never used in clustering)
