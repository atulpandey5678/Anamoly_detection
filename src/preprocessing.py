from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load the Mall Customers CSV dataset.

    Args:
        csv_path: Path to `Mall_Customers.csv`.

    Returns:
        DataFrame with raw customer data.

    Raises:
        FileNotFoundError: If the CSV path does not exist.
        ValueError: If the CSV cannot be parsed.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV at {csv_path}: {e}") from e
    if df.shape[0] == 0:
        raise ValueError(
            f"CSV at {csv_path} has no rows. Download the Kaggle dataset "
            "and place the full Mall_Customers.csv into ./data/."
        )
    return df


def load_dataset_with_kagglehub(dataset: str, file_path: str) -> pd.DataFrame:
    """
    Load the dataset via KaggleHub.

    This supports the KaggleHub snippet provided by the user and allows running
    the project without manually downloading the CSV (if KaggleHub is configured).

    Args:
        dataset: Kaggle dataset slug (e.g. "vjchoudhary7/customer-segmentation-tutorial-in-python").
        file_path: File path inside the dataset (e.g. "Mall_Customers.csv").

    Returns:
        Loaded DataFrame.

    Raises:
        ImportError: If kagglehub is not installed.
        RuntimeError: If loading fails for any reason.
    """
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except Exception as e:
        raise ImportError(
            "kagglehub is not installed. Install with: pip install kagglehub[pandas-datasets]"
        ) from e

    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            dataset,
            file_path,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset via KaggleHub: {e}") from e

    if not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        raise RuntimeError("KaggleHub returned an empty dataset.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in a conservative, production-friendly way.

    - Numeric columns: fill with median
    - Non-numeric columns: fill with mode

    Args:
        df: Raw input DataFrame.

    Returns:
        DataFrame with missing values handled.
    """
    out = df.copy()
    if out.isna().sum().sum() == 0:
        return out

    for col in out.columns:
        if out[col].isna().any():
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].fillna(out[col].median())
            else:
                mode = out[col].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else ""
                out[col] = out[col].fillna(fill_value)
    return out


def encode_gender(df: pd.DataFrame, gender_col: str = "Gender") -> pd.DataFrame:
    """
    Encode the `Gender` column as numeric values.

    Mapping:
        Female -> 0
        Male   -> 1

    If the column is missing, the function is a no-op.

    Args:
        df: Input DataFrame.
        gender_col: Name of gender column.

    Returns:
        DataFrame with `Gender` encoded.
    """
    out = df.copy()
    if gender_col not in out.columns:
        return out

    mapping = {"Female": 0, "Male": 1}
    out[gender_col] = out[gender_col].map(mapping).astype("Int64")
    if out[gender_col].isna().any():
        mode = out[gender_col].mode(dropna=True)
        fill_value = int(mode.iloc[0]) if not mode.empty else 0
        out[gender_col] = out[gender_col].fillna(fill_value).astype(int)
    else:
        out[gender_col] = out[gender_col].astype(int)
    return out


def select_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Select relevant numeric features for modeling.

    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.

    Returns:
        DataFrame with selected features.

    Raises:
        KeyError: If required columns are missing.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    return df[feature_cols].copy()


def scale_features(X: pd.DataFrame) -> np.ndarray:
    """
    Normalize features using StandardScaler.

    Args:
        X: Feature DataFrame.

    Returns:
        Scaled feature matrix as numpy array.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X.values)


def load_and_preprocess(
    csv_path: Path,
    gender_col: str,
    id_col: str,
    feature_cols: List[str],
    *,
    use_kagglehub: bool = False,
    kaggle_dataset: Optional[str] = None,
    kaggle_file_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load dataset, handle missing values, encode gender, select & scale features.

    Args:
        csv_path: Path to dataset CSV.
        gender_col: Gender column name.
        id_col: Customer ID column name.
        feature_cols: Columns used for anomaly detection.

    Returns:
        raw_df: Cleaned raw DataFrame (including original columns).
        X_scaled: Scaled matrix for selected features.
        feature_cols: The feature columns used (echoed for convenience).
    """
    df: pd.DataFrame
    try:
        df = load_csv(csv_path)
    except (FileNotFoundError, ValueError) as e:
        if not use_kagglehub:
            raise
        if not kaggle_dataset or not kaggle_file_path:
            raise ValueError("KaggleHub is enabled but kaggle_dataset/kaggle_file_path were not provided.") from e
        logger.info(
            "Local CSV unavailable/empty (%s). Loading via KaggleHub: %s (%s)",
            e,
            kaggle_dataset,
            kaggle_file_path,
        )
        df = load_dataset_with_kagglehub(kaggle_dataset, kaggle_file_path)
    df = handle_missing_values(df)
    df = encode_gender(df, gender_col=gender_col)

    if id_col not in df.columns:
        logger.warning("ID column '%s' not found; downstream output may be limited.", id_col)

    X = select_features(df, feature_cols=feature_cols)
    X_scaled = scale_features(X)
    return df, X_scaled, feature_cols
