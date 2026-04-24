from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def compute_distances_to_centroids(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean distance from each point to its assigned cluster centroid.

    Args:
        X_scaled: Scaled feature matrix.
        labels: Cluster labels for each row.
        centroids: Centroids in scaled feature space.

    Returns:
        distances: 1D array of distances per row.
    """
    if X_scaled.shape[0] != labels.shape[0]:
        raise ValueError("X_scaled and labels must have the same number of rows")
    if centroids.ndim != 2:
        raise ValueError("centroids must be a 2D array")

    assigned = centroids[labels]
    diffs = X_scaled - assigned
    distances = np.linalg.norm(diffs, axis=1)
    return distances


def flag_top_fraction(distances: np.ndarray, anomaly_fraction: float) -> np.ndarray:
    """
    Flag anomalies as the top fraction farthest points.

    Args:
        distances: Distance per row.
        anomaly_fraction: Fraction in (0, 1) to flag as anomalies (e.g., 0.05).

    Returns:
        Boolean mask, True for anomalies.
    """
    if not (0.0 < anomaly_fraction < 1.0):
        raise ValueError("anomaly_fraction must be between 0 and 1 (exclusive)")
    if distances.size == 0:
        return np.array([], dtype=bool)

    threshold = np.quantile(distances, 1.0 - anomaly_fraction)
    return distances >= threshold


def detect_anomalies(
    raw_df: pd.DataFrame,
    X_scaled: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    id_col: str,
    feature_cols: List[str],
    anomaly_fraction: float = 0.05,
) -> pd.DataFrame:
    """
    Detect anomalies based on distance to KMeans centroid within assigned cluster.

    Args:
        raw_df: Cleaned input DataFrame.
        X_scaled: Scaled matrix for feature_cols.
        labels: KMeans cluster labels.
        centroids: KMeans centroids (scaled space).
        id_col: Customer ID column name.
        feature_cols: Features used, expected length 2 per project scope.
        anomaly_fraction: Fraction of farthest points to flag.

    Returns:
        DataFrame with:
            CustomerID (if present)
            Income
            Spending Score
            Distance
            Is_Anomaly
            Cluster
    """
    if len(feature_cols) != 2:
        logger.warning("Expected 2 features; got %d", len(feature_cols))

    distances = compute_distances_to_centroids(X_scaled, labels, centroids)
    is_anomaly = flag_top_fraction(distances, anomaly_fraction=anomaly_fraction)

    out = pd.DataFrame(
        {
            "Distance": distances,
            "Is_Anomaly": is_anomaly.astype(bool),
            "Cluster": labels.astype(int),
        }
    )

    if id_col in raw_df.columns:
        out[id_col] = raw_df[id_col].values
    else:
        out[id_col] = np.arange(1, len(raw_df) + 1)

    out["Income"] = raw_df[feature_cols[0]].values
    out["Spending Score"] = raw_df[feature_cols[1]].values

    cols = [id_col, "Income", "Spending Score", "Distance", "Is_Anomaly", "Cluster"]
    out = out[cols].sort_values(["Is_Anomaly", "Distance"], ascending=[False, False]).reset_index(drop=True)

    logger.info(
        "Anomaly detection complete. Flagged %d/%d (%.2f%%).",
        int(out["Is_Anomaly"].sum()),
        len(out),
        100.0 * float(out["Is_Anomaly"].mean()),
    )
    return out

