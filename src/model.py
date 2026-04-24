from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KMeansResult:
    """
    Container for KMeans outputs.
    """

    model: KMeans
    labels: np.ndarray
    centroids: np.ndarray


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> Tuple[KMeans, np.ndarray, np.ndarray]:
    """
    Fit KMeans clustering on scaled features.

    Args:
        X: Scaled feature matrix.
        n_clusters: Number of clusters.
        random_state: Seed for reproducibility.
        n_init: Number of initializations.
        max_iter: Maximum iterations per run.

    Returns:
        model: Fitted KMeans model.
        labels: Cluster label per row.
        centroids: Cluster centers in scaled feature space.
    """
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2")

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_
    logger.info("KMeans fit complete. Inertia: %.4f", float(model.inertia_))
    return model, labels, centroids


def compute_elbow_inertia(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optional elbow helper: compute inertias for k in [k_min, k_max].

    Args:
        X: Scaled feature matrix.
        k_min: Minimum clusters.
        k_max: Maximum clusters.
        random_state: Seed.
        n_init: KMeans n_init.
        max_iter: KMeans max_iter.

    Returns:
        ks: Array of k values.
        inertias: Inertia per k.
    """
    if k_min < 2 or k_max < k_min:
        raise ValueError("Invalid k_min/k_max for elbow method")

    ks = np.arange(k_min, k_max + 1)
    inertias = np.zeros_like(ks, dtype=float)
    for i, k in enumerate(ks):
        km = KMeans(
            n_clusters=int(k),
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
        km.fit(X)
        inertias[i] = float(km.inertia_)
    return ks, inertias
