from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> Tuple[KMeans, np.ndarray, np.ndarray]:
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2")
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init, max_iter=max_iter)
    labels = model.fit_predict(X)
    logger.info("KMeans fit complete. Inertia: %.4f", float(model.inertia_))
    return model, labels, model.cluster_centers_


def compute_elbow_inertia(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    if k_min < 2 or k_max < k_min:
        raise ValueError("Invalid k_min/k_max for elbow method")
    ks = np.arange(k_min, k_max + 1)
    inertias = np.array([
        KMeans(n_clusters=int(k), random_state=random_state, n_init=n_init, max_iter=max_iter).fit(X).inertia_
        for k in ks
    ], dtype=float)
    return ks, inertias
