from __future__ import annotations

import logging

import pandas as pd

from src.anomaly import detect_anomalies
from src.config import Settings
from src.model import fit_kmeans
from src.preprocessing import load_and_preprocess
from src.utils import ensure_dirs, save_anomalies_csv, save_cluster_anomaly_plot, setup_logging, try_generate_insights


def run_pipeline(settings: Settings) -> pd.DataFrame:
    ensure_dirs([settings.data_dir, settings.outputs_dir, settings.plots_dir])

    raw_df, features_scaled, feature_cols = load_and_preprocess(
        csv_path=settings.data_csv_path,
        gender_col=settings.gender_col,
        id_col=settings.id_col,
        feature_cols=settings.feature_cols,
        use_kagglehub=settings.use_kagglehub,
        kaggle_dataset=settings.kaggle_dataset,
        kaggle_file_path=settings.kaggle_file_path,
    )

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

    save_anomalies_csv(results_df[results_df["Is_Anomaly"]].copy(), settings.anomalies_csv_path)
    save_cluster_anomaly_plot(
        raw_df=raw_df,
        feature_cols=feature_cols,
        labels=labels,
        is_anomaly=results_df["Is_Anomaly"].to_numpy(),
        plot_path=settings.plots_dir / "clusters_anomalies.png",
        title="Customer Spending Behavior: Clusters & Anomalies",
    )
    try_generate_insights(results_df=results_df, settings=settings, output_path=settings.outputs_dir / "insights.txt")
    return results_df


def main() -> None:
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting pipeline — data: %s | clusters: %d | anomaly_fraction: %.3f",
                settings.data_csv_path, settings.n_clusters, settings.anomaly_fraction)

    try:
        results_df = run_pipeline(settings)
    except FileNotFoundError as e:
        logger.error("%s — place Mall_Customers.csv in ./data/", e)
        raise
    except Exception:
        logger.exception("Pipeline failed")
        raise

    logger.info("Done. Anomalies flagged: %d | CSV: %s | Plot: %s",
                int(results_df["Is_Anomaly"].sum()),
                settings.anomalies_csv_path,
                settings.plots_dir / "clusters_anomalies.png")


if __name__ == "__main__":
    main()
