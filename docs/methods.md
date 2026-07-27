# Discovery methods

Pre-GRPO group discovery runs in this repo **before** grpo-reproduction. This repo discovers K groups; grpo-reproduction trains GR-IPO on those groups.

## Comparison ladder

### 1. Reference Baselines (`config/baselines/`)

| Tier | Method | Config | Purpose |
| --- | --- | --- | --- |
| **Lower bound** | `single_group` | `config/baselines/single_group.yaml` | All entities pooled into 1 group (no discovery) |
| **Stochastic** | `random_assignment` | `config/baselines/random_assignment.yaml` | Random K-way split — chance/noise control |
| **Floor baseline** | `preference_similarity` | `config/baselines/preference_similarity.yaml` | KMeans on L2-normalized mean opinion vectors |
| **Oracle** | `country_oracle` | `config/baselines/country_oracle.yaml` | Hidden source_group (country) labels — approx upper bound |

---

### 2. Core Discovery Methods (`config/discovery/`)

| # | Method | Config | Module | Algorithm |
| --- | --- | --- | --- | --- |
| 1 | Cross-predictive similarity | `config/discovery/cross_predictive.yaml` | `methods/cross_predictive.py` | Transfer gain on shared prompts → Spectral Clustering |
| 2 | Chosen-rejected embedding sets | `config/discovery/embedding_sets.yaml` | `methods/embedding_sets.py` | Chamfer distance on vector sets → Agglomerative / K-Medoids |
| 3 | Sparse matrix factorization | `config/discovery/matrix_factorization.yaml` | `methods/matrix_factorization.py` | Sparse NMF/SVD/ALS on entity×option matrix → KMeans |

```bash
# Run a discovery method:
python3 discover.py --config config/discovery/embedding_sets.yaml

# Run a baseline check:
python3 discover.py --config config/baselines/single_group.yaml
```

## Entity modes

| Mode | Description |
| --- | --- |
| `observed` | One entity per HF source group (pilot) |
| `simulated` | Synthetic individuals sampled from group distributions |
| `blind` | One anonymous entity per preference row |
| `individual` | Pre-existing entity_id from pairwise datasets |

## Evaluation

- **Entropy Reduction (ΔH):** Reduction in choice uncertainty when conditioning on groups.
- **Cohesion:** Mean intra-cluster cosine similarity.
- **Separation:** Inter-cluster JSD, Calinski-Harabasz, Davies-Bouldin.
- **Held-out Prediction Lift:** Does cluster assignment predict unseen choices above a naive baseline?
- **Demographic Overlay:** ARI/NMI against hidden source_group labels (low is not failure).
- **Polarizing Prompt Audit:** Top questions with maximum inter-group divergence.
- **Downstream:** Worst-group GRPO reward under group-specific policy optimization.
