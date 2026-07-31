# Pre-GRPO Group Discovery: Methods, Baselines, and Metrics

This document provides a complete guide to the pre-GRPO group discovery framework. It is divided into two sections:
1. **Conceptual Overview:** Simple, intuitive explanations of how each method, baseline, and evaluation metric works.
2. **Implementation Details:** Specific technical mechanics of how these algorithms are implemented for the Global Opinions QA dataset within the `grpo-group-discovery` repository.

---

## Part 1: Conceptual Overview (Simple Explanations)

### 1. Discovery Methods (How we group people)

* **Embedding-Set Approach (Chamfer Distance):**
  * **Concept:** For every prompt a person answered, we subtract the rejected response vector from the chosen response vector $(\text{chosen} - \text{rejected})$ to find their preferred direction in text embedding space.
  * **Chamfer Distance:** Since different individuals answer different prompts, we cannot compare their vectors 1-to-1. Chamfer distance measures how close person A's set of preference vectors is to person B's set of preference vectors by finding the nearest vector match for each prompt.
  * **Goal:** Group people together if their preferred text directions are close in vector space.

* **Cross-Predictive Similarity (Accuracy Margins + Spectral Clustering):**
  * **Concept:** Tests how well one person's preference choices help predict another person's choices using a language model: *"If I show the model Person B's choices as context, does it get better at predicting Person A's choices on shared prompts?"*
  * **Accuracy Margin:** Calculates $\text{Accuracy}(A \mid B) - \text{Accuracy}(A \text{ baseline})$. A positive margin means Person B's data improves prediction of Person A's choices.
  * **Spectral Clustering:** Builds a similarity matrix out of these pairwise prediction transfer scores and uses graph/spectral clustering to find natural preference communities.

* **Sparse Matrix Factorization (NMF / SVD):**
  * **Concept:** Imagine a large table where rows are individuals and columns are specific option choices across prompts (with empty cells for prompts a person didn't see).
  * **Factorization:** Decomposes this incomplete table into a small number of latent preference factors (e.g., Factor 1 = "Privacy orientation", Factor 2 = "Risk tolerance").
  * **Goal:** Clusters individuals who score similarly on these hidden preference factors using KMeans.

---

### 2. Baselines (What we compare against)

* **Random Assignment (`random_assignment`):** Assigns entities to $K$ groups completely at random. Acts as a stochastic control to prove algorithms beat pure chance.
* **Single-Group (`single_group`):** Pools all individuals into 1 group (no clustering). Represents standard global alignment (DPO/IPO).
* **Preference Vector Similarity (`preference_similarity`):** A floor baseline that averages a person's option choice probabilities into a single vector and clusters them using standard KMeans.
* **Country Oracle (`country_oracle`):** Groups individuals by their actual country of origin. Serves as our demographic benchmark / upper bound anchor.

---

### 3. Evaluation Metrics (How we evaluate clusters)

* **Entropy Reduction ($\Delta H$):** Measures how much choice uncertainty about an individual's preferences decreases once you condition on their assigned cluster group. Higher reduction = more cohesive groups.
* **Choice Prediction Lift:** Measures how much choice prediction accuracy increases when guessing an individual's answers on unseen prompts using their group's average preference vs. a global average.
* **Cluster Cohesion & Separation:** 
  * *Cohesion:* How tight/similar individuals within the same cluster are.
  * *Separation:* How distinct different clusters are from each other (e.g., Calinski-Harabasz and Davies-Bouldin indices).
* **Country Label Overlay (ARI / NMI):** Measures how closely discovered clusters align with country borders using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI). Low overlap is not a failure—it shows preference groups cut across geographic lines.

---

## Part 2: Implementation Details in `grpo-group-discovery`

### 1. Data Pipeline & Entity Modes
The pipeline processes the **Global Opinions QA (GOQA)** dataset containing survey questions ($qkey$), option sets ($prob\_y$), and demographic source groups (countries). The framework supports four entity modes (`src/data/entities.py`):
* `observed`: One entity per source group/country (used for pilot runs).
* `simulated`: Synthetic individuals sampled from group multinomial choice distributions.
* `blind`: One anonymous entity per preference record row.
* `individual`: Pre-existing individual IDs from pairwise interaction datasets.

---

### 2. Specific Method Implementations

#### Method 1: Embedding Sets (`src/clustering/methods/embedding_sets.py`)
1. **Vector Construction:** For each preference record, the text of the chosen response and rejected response are embedded using a sentence transformer model. The preference vector is computed as $\Delta v = \text{embed}(\text{chosen}) - \text{embed}(\text{rejected})$.
2. **Symmetric Chamfer Distance:** For two entities $A$ and $B$ with vector sets $S_A$ and $S_B$:
   $$D_{\text{Chamfer}}(S_A, S_B) = \frac{1}{|S_A|} \sum_{a \in S_A} \min_{b \in S_B} \|a - b\|_2 + \frac{1}{|S_B|} \sum_{b \in S_B} \min_{a \in S_A} \|b - a\|_2$$
   Implemented efficiently via C-accelerated `scipy.spatial.distance.cdist`.
3. **Clustering:** Agglomerative Clustering with average linkage on the precomputed Chamfer distance matrix.

#### Method 2: Cross-Predictive Similarity (`src/clustering/methods/cross_predictive.py`)
1. **Shared Prompt Extraction:** For every pair of entities $(A, B)$, identifies overlapping prompts ($qkey$).
2. **Transfer Score:** Computes choice distribution similarity (cosine or top-choice match) between $A$ and $B$ across shared prompts:
   $$S(A, B) = \frac{1}{|Q_{A \cap B}|} \sum_{q \in Q_{A \cap B}} \text{CosineSim}(p_A^{(q)}, p_B^{(q)})$$
3. **Coverage Scaling & Spectral Clustering:** Multiplies similarity by a coverage penalty if shared prompts fall below a minimum threshold (`min_shared_prompts`). Applies `sklearn.cluster.SpectralClustering` to the affinity matrix.

#### Method 3: Sparse Matrix Factorization (`src/clustering/methods/matrix_factorization.py`)
1. **Sparse Matrix Construction:** Constructs a sparse matrix of size $\text{Entities} \times \text{Unique(Prompt, Option)}$ where entries correspond to observed choice probabilities or sampled choices.
2. **Decomposition:** Fits Non-Negative Matrix Factorization (`NMF`) or Truncated Singular Value Decomposition (`TruncatedSVD`) to reduce features to $N$ latent components (default: 10 components).
3. **Clustering:** Runs `sklearn.cluster.KMeans` on the low-dimensional latent entity representation.

---

### 3. Evaluation Framework Implementation (`src/analysis/`)

* **Entropy Reduction (`metrics.py`):** Calculates global preference entropy $H(Y)$ vs. conditional entropy $H(Y \mid C)$ across cluster assignments $C$:
  $$\Delta H = H(Y) - \sum_{k} P(C=k) H(Y \mid C=k)$$
* **Held-out Choice Prediction (`eval_prediction.py`):** Performs train/test splits per entity across prompts. Evaluates whether cluster-conditioned mode choices achieve higher top-1 accuracy on held-out prompts than pooled global choices.
* **Polarizing Prompt Audit (`polarization.py`):** Calculates inter-cluster Jensen-Shannon Divergence (JSD) per prompt to identify the top survey questions driving maximum group separation.
* **Demographic Overlay (`metrics.py`):** Uses `sklearn.metrics.adjusted_rand_score` (ARI) and `normalized_mutual_info_score` (NMI) between cluster labels and hidden ground-truth country source labels.
