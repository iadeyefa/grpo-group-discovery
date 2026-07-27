# Discovery methods

Pre-GRPO group discovery runs in this repo **before** grpo-reproduction. This repo discovers K groups; grpo-reproduction trains GR-IPO on those groups.

## Comparison ladder

| Tier | Method | Config | Purpose |
| --- | --- | --- | --- |
| **Lower bound** | `single_group` | `config/methods/single_group.yaml` | All entities pooled — no discovery |
| **Stochastic** | `random_assignment` | `config/methods/random_assignment.yaml` | Random K-way split — noise baseline |
| **Floor baseline** | `preference_similarity` | `config/preference_similarity.yaml` | KMeans on mean opinion vectors |
| **Methods 1–5** | See below | `config/methods/*.yaml` | Real discovery methods |
| **Oracle** | `country_oracle` | `config/methods/country_oracle.yaml` | Hidden source_group labels — approx upper bound |

## Discovery methods

| # | Method | Module | Algorithm |
| --- | --- | --- | --- |
| 1 | Cross-predictive similarity | `methods/cross_predictive.py` | Transfer gain on shared prompts → Spectral Clustering |
| 2 | Chosen-rejected embedding sets | `methods/embedding_sets.py` | Chamfer distance on diff vectors → Agglomerative |
| 3 | Sparse matrix factorization | `methods/matrix_factorization.py` | NMF/SVD on entity×option matrix → KMeans |
| 4 | Latent class preference model | `methods/latent_class.py` | Mixture multinomial logit via EM |
| 5 | Agreement graph | `methods/agreement_graph.py` | Pairwise agreement on overlapping prompts → Spectral |

```bash
python3 discover.py --config config/methods/embedding_sets.yaml
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
