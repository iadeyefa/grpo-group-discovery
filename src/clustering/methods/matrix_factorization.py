"""Method 3: Sparse matrix factorization + latent clustering."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD

from src.data.entities import attach_entity_ids

logger = logging.getLogger(__name__)

METHOD_NAME = "matrix_factorization"


def run(
    preference_df: pd.DataFrame,
    entity_registry: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Constructs a sparse entity x option matrix, performs matrix factorization
    (NMF or TruncatedSVD) to get latent vectors, and clusters them using KMeans.
    """
    cluster_cfg = config.get("clustering", {})
    n_clusters = cluster_cfg.get("n_clusters", 5)
    random_state = cluster_cfg.get("random_state", 42)
    n_components = cluster_cfg.get("n_components", 10)
    factorization_method = str(cluster_cfg.get("factorization_method", "nmf")).lower()

    # Automatically set defaults based on entity mode if not explicitly config'd
    entity_mode = config.get("entities", {}).get("mode", "observed")
    default_subsample = 0.3 if entity_mode == "simulated" else 1.0
    default_sample_choices = True if entity_mode == "simulated" else False

    subsample_fraction = float(cluster_cfg.get("subsample_fraction", default_subsample))
    sample_choices = bool(cluster_cfg.get("sample_choices", default_sample_choices))

    logger.info("Building sparse matrix column mapping for %s", METHOD_NAME)
    col_mapping = {}
    col_counter = 0

    # Determistically map each unique (qkey, option_index) to a column index
    qkeys = sorted(preference_df["qkey"].unique())
    for qkey in qkeys:
        q_rows = preference_df[preference_df["qkey"] == qkey]
        if q_rows.empty:
            continue
        num_options = len(q_rows.iloc[0]["prob_y"])
        for opt_idx in range(num_options):
            col_mapping[(qkey, opt_idx)] = col_counter
            col_counter += 1

    n_features = col_counter
    n_entities = len(entity_registry)
    logger.info(
        "Sparse matrix dimensions: %d entities x %d features",
        n_entities,
        n_features,
    )

    logger.info("Attaching entity IDs and simulating sparsity")
    preference_with_entities = attach_entity_ids(preference_df, entity_registry)

    # Initialize sparse matrix
    X = lil_matrix((n_entities, n_features), dtype=np.float64)
    entity_ids = sorted(entity_registry["entity_id"].unique())
    entity_to_row = {eid: idx for idx, eid in enumerate(entity_ids)}

    rng = np.random.default_rng(random_state)
    entity_groups = preference_with_entities.groupby("entity_id")

    for eid, group in entity_groups:
        row_idx = entity_to_row[eid]
        num_questions = len(group)
        if num_questions == 0:
            continue

        if subsample_fraction < 1.0:
            keep_count = max(1, int(num_questions * subsample_fraction))
            keep_indices = rng.choice(num_questions, size=keep_count, replace=False)
            sub_group = group.iloc[keep_indices]
        else:
            sub_group = group

        for row in sub_group.itertuples(index=False):
            qkey = row.qkey
            prob_y = np.asarray(row.prob_y, dtype=np.float64)
            n_opts = len(prob_y)
            if n_opts == 0:
                continue

            if sample_choices:
                p_sum = np.sum(prob_y)
                if p_sum > 0:
                    p_norm = prob_y / p_sum
                    chosen_opt = rng.choice(n_opts, p=p_norm)
                    col_idx = col_mapping.get((qkey, chosen_opt))
                    if col_idx is not None:
                        X[row_idx, col_idx] = 1.0
            else:
                for opt_idx, p in enumerate(prob_y):
                    col_idx = col_mapping.get((qkey, opt_idx))
                    if col_idx is not None:
                        X[row_idx, col_idx] = p

    # Convert to CSR format for fast linear algebra
    X_csr = X.tocsr()
    nnz = X_csr.nnz
    density = nnz / (n_entities * n_features) if n_entities * n_features > 0 else 0
    logger.info(
        "Sparse matrix built: %d non-zeros, density = %.4f",
        nnz,
        density,
    )

    actual_components = min(n_components, n_entities, n_features)
    if actual_components < n_components:
        logger.warning(
            "n_components changed from %d to %d to fit matrix dimensions (%d x %d)",
            n_components,
            actual_components,
            n_entities,
            n_features,
        )

    logger.info(
        "Running matrix factorization: method=%s, components=%d",
        factorization_method,
        actual_components,
    )

    if factorization_method == "nmf":
        model = NMF(
            n_components=actual_components,
            random_state=random_state,
            init="random",
            max_iter=500,
        )
        W = model.fit_transform(X_csr)
    elif factorization_method == "svd":
        model = TruncatedSVD(
            n_components=actual_components,
            random_state=random_state,
        )
        W = model.fit_transform(X_csr)
    else:
        raise ValueError(
            f"Unsupported factorization method {factorization_method!r}. "
            "Supported methods are: 'nmf', 'svd'."
        )

    features_df = pd.DataFrame(
        W,
        index=entity_ids,
        columns=[f"latent_feat_{i}" for i in range(actual_components)],
    )
    features_df.index.name = "entity_id"

    logger.info(
        "Running KMeans clustering on latent features: k=%d, random_state=%d",
        n_clusters,
        random_state,
    )
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(W)

    # Compute soft membership probabilities P(cluster_k | entity_i) via softmax over centroid distances
    distances = kmeans.transform(W)  # N x K Euclidean distances to centroids
    scaled_neg_dist = -1.0 * distances
    # Numerically stable softmax
    max_neg_dist = np.max(scaled_neg_dist, axis=1, keepdims=True)
    exp_dist = np.exp(scaled_neg_dist - max_neg_dist)
    soft_probs = exp_dist / np.sum(exp_dist, axis=1, keepdims=True)

    assignments = pd.DataFrame(
        {
            "entity_id": entity_ids,
            "cluster_id": labels.astype(int),
        }
    ).sort_values(["cluster_id", "entity_id"])

    soft_cols = [f"prob_cluster_{c}" for c in range(n_clusters)]
    soft_assignments = pd.DataFrame(soft_probs, index=entity_ids, columns=soft_cols)
    soft_assignments.index.name = "entity_id"

    extras = {
        "features_df": features_df,
        "feature_dim": features_df.shape[1],
        "uses_preference_features": True,
        "discovery_method": METHOD_NAME,
        "discovery_tier": "method",
        "factorization_params": {
            "factorization_method": factorization_method,
            "n_components": actual_components,
            "subsample_fraction": subsample_fraction,
            "sample_choices": sample_choices,
            "n_entities": n_entities,
            "n_features": n_features,
            "density": density,
        },
    }
    return assignments, extras
