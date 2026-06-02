from __future__ import annotations

import dataclasses

import matplotlib.pyplot as plt
import streamlit as st

from src.anomaly import detect_anomalies
from src.config import Settings
from src.model import fit_kmeans
from src.preprocessing import load_and_preprocess
from src.utils import ensure_dirs, save_anomalies_csv, save_cluster_anomaly_plot

st.set_page_config(
    page_title="AnomalyIQ — Customer Spending Anomaly Detection",
    page_icon="◈",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0b0d14; }
[data-testid="stSidebar"] { background: #131720; border-right: 1px solid #252d40; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
h1, h2, h3 { color: #f1f5f9 !important; }
p, li { color: #94a3b8; }
.metric-label { color: #64748b !important; }
[data-testid="metric-container"] {
    background: #131720;
    border: 1px solid #252d40;
    border-radius: 12px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## ◈ AnomalyIQ")
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)
    st.caption("Sorts customers into N groups by spending behavior. More clusters = smaller, more specific groups.")

    anomaly_pct = st.slider("Anomaly Threshold (%)", min_value=1, max_value=20, value=5)
    st.caption(
        f"Flags the top {anomaly_pct}% of customers who sit farthest from their group's center. "
        f"e.g. at 5% → ~10 outliers out of 200. Lower = stricter. Higher = catches more borderline cases."
    )
    anomaly_fraction = anomaly_pct / 100

    st.markdown("---")
    run = st.button("▶ Run Analysis", use_container_width=True, type="primary")

st.title("Customer Spending Anomaly Detection")
st.markdown("Surfaces outlier customers whose income-to-spending ratio deviates significantly from their peer group.")
st.divider()

if not run:
    col1, col2, col3 = st.columns(3)
    col1.info("**Step 1** — Configure clusters & threshold in the sidebar")
    col2.info("**Step 2** — Click **▶ Run Analysis** to run the ML pipeline")
    col3.info("**Step 3** — Explore clusters, the scatter plot, and flagged anomalies")
    st.stop()

with st.spinner("Loading data and running pipeline…"):
    try:
        settings = Settings.from_env()
        settings = dataclasses.replace(
            settings, n_clusters=n_clusters, anomaly_fraction=anomaly_fraction
        )
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
        _, labels, centroids = fit_kmeans(
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
        anomalies_df = results_df[results_df["Is_Anomaly"]].copy()
        save_anomalies_csv(anomalies_df, settings.anomalies_csv_path)

        plot_path = settings.plots_dir / "clusters_anomalies.png"
        save_cluster_anomaly_plot(
            raw_df=raw_df,
            feature_cols=feature_cols,
            labels=labels,
            is_anomaly=results_df["Is_Anomaly"].to_numpy(),
            plot_path=plot_path,
            title="Customer Spending Behavior: Clusters & Anomalies",
        )

    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

st.success(f"Analysis complete — {len(anomalies_df)} anomalies detected out of {len(results_df)} customers.")
st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", len(results_df))
c2.metric("Anomalies Found", len(anomalies_df), delta=f"{anomaly_pct}% threshold", delta_color="off")
c3.metric("Clusters", n_clusters)
c4.metric("Anomaly Threshold", f"{anomaly_pct}%")

st.divider()

col_plot, col_table = st.columns([1.6, 1], gap="large")

with col_plot:
    st.subheader("Cluster Visualization")
    st.caption("Income vs Spending Score — colored by cluster, anomalies in red")
    if plot_path.exists():
        st.image(str(plot_path), use_container_width=True)

with col_table:
    st.subheader("Cluster Breakdown")
    cluster_counts = results_df.groupby("Cluster").size().reset_index(name="Customers")
    cluster_counts["Share (%)"] = (cluster_counts["Customers"] / len(results_df) * 100).round(1)
    st.dataframe(cluster_counts, use_container_width=True, hide_index=True)

st.divider()
st.subheader("🚨 Flagged Anomalies")
st.caption("Top customers sorted by distance from cluster centroid")

display_df = anomalies_df.copy()
display_df["Distance"] = display_df["Distance"].round(4)
display_df["Is_Anomaly"] = display_df["Is_Anomaly"].map({True: "🚨 Anomaly"})
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.download_button(
    label="⬇ Download anomalies.csv",
    data=anomalies_df.to_csv(index=False).encode("utf-8"),
    file_name="anomalies.csv",
    mime="text/csv",
)
