from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def save_anomalies_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_cluster_anomaly_plot(
    raw_df: pd.DataFrame,
    feature_cols: list[str],
    labels: np.ndarray,
    is_anomaly: np.ndarray,
    plot_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if len(feature_cols) != 2:
        raise ValueError("feature_cols must contain exactly 2 columns")

    income_col, spend_col = feature_cols
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = raw_df[[income_col, spend_col]].copy()
    plot_df["Cluster"] = labels.astype(int)
    plot_df["Is_Anomaly"] = is_anomaly.astype(bool)

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df, x=income_col, y=spend_col,
        hue="Cluster", palette="tab10", alpha=0.75, s=60, ax=ax, legend="brief",
    )
    anomalies = plot_df[plot_df["Is_Anomaly"]]
    ax.scatter(
        anomalies[income_col], anomalies[spend_col],
        c="red", s=120, edgecolors="black", linewidths=0.7, label="Anomaly", zorder=5,
    )
    ax.set_title(title)
    ax.set_xlabel(income_col)
    ax.set_ylabel(spend_col)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def try_generate_insights(
    results_df: pd.DataFrame,
    settings: "Settings",
    output_path: Path,
) -> Optional[str]:
    logger = logging.getLogger(__name__)
    if not getattr(settings, "openai_api_key", None):
        logger.info("OPENAI_API_KEY not set. Skipping insight generation.")
        return None

    try:
        from openai import OpenAI
    except Exception as e:
        logger.warning("OpenAI SDK not available (%s). Skipping insights.", e)
        return None

    client = OpenAI(api_key=settings.openai_api_key)
    anomalies = results_df[results_df["Is_Anomaly"]].copy()
    top_rows = (
        anomalies.sort_values("Distance", ascending=False)
        .head(10)[["CustomerID", "Income", "Spending Score", "Distance", "Cluster"]]
        .to_dict(orient="records")
    )
    prompt = (
        f"You are a data analyst. Summarize {len(anomalies)} anomalous customers out of {len(results_df)} total.\n"
        "Anomalies are the top 5% farthest from their KMeans centroid in standardized feature space.\n\n"
        "Provide: 1) A concise summary. 2) Business interpretations. 3) Method limitations.\n\n"
        f"Top 10 anomalies: {top_rows}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=settings.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content
    except Exception:
        logger.error("Insight generation failed. Continuing without insights.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Saved insights to %s", output_path)
    return text
