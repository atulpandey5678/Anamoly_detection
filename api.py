from __future__ import annotations

import base64
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.anomaly import detect_anomalies
from src.config import Settings
from src.model import fit_kmeans
from src.preprocessing import load_and_preprocess
from src.utils import ensure_dirs, save_anomalies_csv, save_cluster_anomaly_plot, setup_logging

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Anomaly Detection API")

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.post("/api/run")
def run_pipeline():
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    settings = Settings.from_env()
    ensure_dirs([settings.data_dir, settings.outputs_dir, settings.plots_dir])

    try:
        raw_df, features_scaled, feature_cols = load_and_preprocess(
            csv_path=settings.data_csv_path,
            gender_col=settings.gender_col,
            id_col=settings.id_col,
            feature_cols=settings.feature_cols,
            use_kagglehub=settings.use_kagglehub,
            kaggle_dataset=settings.kaggle_dataset,
            kaggle_file_path=settings.kaggle_file_path,
        )
    except Exception as e:
        logger.error("Data loading failed: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

    kmeans, labels, centroids = fit_kmeans(
        X=features_scaled,
        n_clusters=settings.n_clusters,
        random_state=settings.random_state,
        n_init=settings.kmeans_n_init,
        max_iter=settings.kmeans_max_iter,
    )

    results_df = detect_anomalies(
        raw_df=raw_df,
        X_scaled=features_scaled,
        labels=labels,
        centroids=centroids,
        id_col=settings.id_col,
        feature_cols=feature_cols,
        anomaly_fraction=settings.anomaly_fraction,
    )

    anomalies_only = results_df[results_df["Is_Anomaly"]].copy()
    save_anomalies_csv(anomalies_only, settings.anomalies_csv_path)

    plot_path = settings.plots_dir / "clusters_anomalies.png"
    save_cluster_anomaly_plot(
        raw_df=raw_df,
        feature_cols=feature_cols,
        labels=labels,
        is_anomaly=results_df["Is_Anomaly"].to_numpy(),
        plot_path=plot_path,
        title="Customer Spending Behavior: Clusters & Anomalies",
    )

    anomaly_rows = anomalies_only.to_dict(orient="records")
    for row in anomaly_rows:
        row["Is_Anomaly"] = bool(row["Is_Anomaly"])
        row["Distance"] = round(float(row["Distance"]), 4)

    cluster_counts = results_df.groupby("Cluster").size().to_dict()
    cluster_counts = {int(k): int(v) for k, v in cluster_counts.items()}

    plot_b64 = None
    if plot_path.exists():
        with open(plot_path, "rb") as f:
            plot_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "total_customers": int(len(results_df)),
        "total_anomalies": int(len(anomalies_only)),
        "anomaly_fraction": settings.anomaly_fraction,
        "n_clusters": settings.n_clusters,
        "cluster_counts": cluster_counts,
        "anomalies": anomaly_rows,
        "plot_base64": plot_b64,
    }
