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
    if X_scaled.shape[0] != labels.shape[0]:
        raise ValueError("X_scaled and labels must have the same number of rows")
    if centroids.ndim != 2:
        raise ValueError("centroids must be a 2D array")
    return np.linalg.norm(X_scaled - centroids[labels], axis=1)


def flag_top_fraction(distances: np.ndarray, anomaly_fraction: float) -> np.ndarray:
    if not (0.0 < anomaly_fraction < 1.0):
        raise ValueError("anomaly_fraction must be between 0 and 1 (exclusive)")
    if distances.size == 0:
        return np.array([], dtype=bool)
    return distances >= np.quantile(distances, 1.0 - anomaly_fraction)


def detect_anomalies(
    raw_df: pd.DataFrame,
    X_scaled: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    id_col: str,
    feature_cols: List[str],
    anomaly_fraction: float = 0.05,
) -> pd.DataFrame:
    if len(feature_cols) != 2:
        logger.warning("Expected 2 features; got %d", len(feature_cols))

    distances = compute_distances_to_centroids(X_scaled, labels, centroids)
    is_anomaly = flag_top_fraction(distances, anomaly_fraction=anomaly_fraction)

    out = pd.DataFrame({
        "Distance": distances,
        "Is_Anomaly": is_anomaly.astype(bool),
        "Cluster": labels.astype(int),
    })

    out[id_col] = raw_df[id_col].values if id_col in raw_df.columns else np.arange(1, len(raw_df) + 1)
    out["Income"] = raw_df[feature_cols[0]].values
    out["Spending Score"] = raw_df[feature_cols[1]].values

    cols = [id_col, "Income", "Spending Score", "Distance", "Is_Anomaly", "Cluster"]
    out = out[cols].sort_values(["Is_Anomaly", "Distance"], ascending=[False, False]).reset_index(drop=True)

    logger.info(
        "Anomaly detection complete. Flagged %d/%d (%.2f%%).",
        int(out["Is_Anomaly"].sum()), len(out), 100.0 * float(out["Is_Anomaly"].mean()),
    )
    return out
