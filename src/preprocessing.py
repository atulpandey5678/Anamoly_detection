from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV at {csv_path}: {e}") from e
    if df.shape[0] == 0:
        raise ValueError(f"CSV at {csv_path} has no rows.")
    return df


def load_dataset_with_kagglehub(dataset: str, file_path: str) -> pd.DataFrame:
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except Exception as e:
        raise ImportError(
            "kagglehub is not installed. Run: pip install kagglehub[pandas-datasets]"
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
    out = df.copy()
    if out.isna().sum().sum() == 0:
        return out
    for col in out.columns:
        if out[col].isna().any():
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].fillna(out[col].median())
            else:
                mode = out[col].mode(dropna=True)
                out[col] = out[col].fillna(mode.iloc[0] if not mode.empty else "")
    return out


def encode_gender(df: pd.DataFrame, gender_col: str = "Gender") -> pd.DataFrame:
    out = df.copy()
    if gender_col not in out.columns:
        return out
    out[gender_col] = out[gender_col].map({"Female": 0, "Male": 1}).astype("Int64")
    if out[gender_col].isna().any():
        mode = out[gender_col].mode(dropna=True)
        out[gender_col] = out[gender_col].fillna(int(mode.iloc[0]) if not mode.empty else 0).astype(int)
    else:
        out[gender_col] = out[gender_col].astype(int)
    return out


def select_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    return df[feature_cols].copy()


def scale_features(X: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(X.values)


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
    try:
        df = load_csv(csv_path)
    except (FileNotFoundError, ValueError) as e:
        if not use_kagglehub:
            raise
        if not kaggle_dataset:
            raise ValueError("KaggleHub enabled but kaggle_dataset not provided.") from e
        logger.info("Local CSV unavailable (%s). Loading via KaggleHub: %s", e, kaggle_dataset)
        df = load_dataset_with_kagglehub(kaggle_dataset, kaggle_file_path or "")

    df = handle_missing_values(df)
    df = encode_gender(df, gender_col=gender_col)

    if id_col not in df.columns:
        logger.warning("ID column '%s' not found.", id_col)

    X_scaled = scale_features(select_features(df, feature_cols=feature_cols))
    return df, X_scaled, feature_cols
