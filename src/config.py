from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv
import os


@dataclass(frozen=True)
class Settings:
    """
    Centralized configuration for reproducible, production-friendly runs.
    """

    project_root: Path
    data_dir: Path
    outputs_dir: Path
    plots_dir: Path
    data_csv_path: Path
    anomalies_csv_path: Path

    id_col: str
    gender_col: str
    feature_cols: List[str]

    use_kagglehub: bool
    kaggle_dataset: str
    kaggle_file_path: str

    n_clusters: int
    anomaly_fraction: float
    random_state: int
    kmeans_n_init: int
    kmeans_max_iter: int

    log_level: str

    openai_api_key: str | None
    model_name: str

    @staticmethod
    def from_env() -> "Settings":
        """
        Load settings from environment (.env supported) with safe defaults.
        """
        load_dotenv(override=False)

        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / "data"
        outputs_dir = project_root / "outputs"
        plots_dir = outputs_dir / "plots"

        data_csv_path = data_dir / "Mall_Customers.csv"
        anomalies_csv_path = outputs_dir / "anomalies.csv"

        id_col = "CustomerID"
        gender_col = "Gender"
        feature_cols = ["Annual Income (k$)", "Spending Score (1-100)"]

        use_kagglehub = os.getenv("USE_KAGGLEHUB", "1").strip().lower() in {"1", "true", "yes", "y"}
        kaggle_dataset = os.getenv(
            "KAGGLE_DATASET",
            "vjchoudhary7/customer-segmentation-tutorial-in-python",
        )
        kaggle_file_path = os.getenv("KAGGLE_FILE_PATH", "Mall_Customers.csv")

        n_clusters = int(os.getenv("N_CLUSTERS", "5"))
        anomaly_fraction = float(os.getenv("ANOMALY_FRACTION", "0.05"))
        random_state = int(os.getenv("RANDOM_STATE", "42"))
        kmeans_n_init = int(os.getenv("KMEANS_N_INIT", "10"))
        kmeans_max_iter = int(os.getenv("KMEANS_MAX_ITER", "300"))

        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_key = openai_api_key if openai_api_key and "your_openai_key_here" not in openai_api_key else None
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

        return Settings(
            project_root=project_root,
            data_dir=data_dir,
            outputs_dir=outputs_dir,
            plots_dir=plots_dir,
            data_csv_path=data_csv_path,
            anomalies_csv_path=anomalies_csv_path,
            id_col=id_col,
            gender_col=gender_col,
            feature_cols=feature_cols,
            use_kagglehub=use_kagglehub,
            kaggle_dataset=kaggle_dataset,
            kaggle_file_path=kaggle_file_path,
            n_clusters=n_clusters,
            anomaly_fraction=anomaly_fraction,
            random_state=random_state,
            kmeans_n_init=kmeans_n_init,
            kmeans_max_iter=kmeans_max_iter,
            log_level=log_level,
            openai_api_key=openai_api_key,
            model_name=model_name,
        )
