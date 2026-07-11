"""Bootstrap stability checks for cluster assignments."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

logger = logging.getLogger(__name__)


def bootstrap_stability(
    features_df: pd.DataFrame,
    n_clusters: int,
    n_bootstrap: int = 20,
    subsample_frac: float = 0.8,
    random_state: int = 42,
    n_init: int = 10,
) -> dict[str, float]:
    """
    Measure clustering stability via bootstrap ARI against a reference fit.

    Subsamples entities (not features) and compares label agreement.
    """
    rng = np.random.default_rng(random_state)
    entity_ids = features_df.index.tolist()
    X = features_df.to_numpy()

    reference = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=n_init
    ).fit(X)
    ref_labels = dict(zip(entity_ids, reference.labels_))

    scores: list[float] = []
    for b in range(n_bootstrap):
        n_sample = max(n_clusters, int(len(entity_ids) * subsample_frac))
        idx = rng.choice(len(entity_ids), size=n_sample, replace=False)
        sample_ids = [entity_ids[i] for i in idx]
        X_sub = X[idx]

        model = KMeans(n_clusters=n_clusters, random_state=random_state + b, n_init=n_init)
        sub_labels = model.fit_predict(X_sub)

        ref_sub = [ref_labels[e] for e in sample_ids]
        ari = adjusted_rand_score(ref_sub, sub_labels)
        scores.append(ari)
        logger.debug("Bootstrap %d/%d ARI=%.3f", b + 1, n_bootstrap, ari)

    summary = {
        "mean_ari": float(np.mean(scores)),
        "std_ari": float(np.std(scores)),
        "n_bootstrap": n_bootstrap,
    }
    logger.info(
        "Stability: mean ARI=%.3f (std=%.3f) over %d bootstraps",
        summary["mean_ari"],
        summary["std_ari"],
        n_bootstrap,
    )
    return summary
