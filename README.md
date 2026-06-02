# ◈ AnomalyIQ — Customer Spending Anomaly Detection

🚀 **Live App:** [https://atulpandey5678-anamoly-detection-streamlit-app-lbczrc.streamlit.app/](https://atulpandey5678-anamoly-detection-streamlit-app-lbczrc.streamlit.app/)

---

## What Did I Build?

A web app that **automatically finds unusual customers** in a mall dataset — customers whose spending behavior doesn't match their income group.

You give it 200 mall customers. It groups them, finds the odd ones out, and shows you exactly who they are and why they're flagged — all with an interactive dashboard.

---

## What Problem Does It Solve?

Imagine you run a mall. You have 200 customers with their:
- 💰 **Annual Income** (how much they earn)
- 🛍️ **Spending Score** (how much they spend, rated 1–100)

Most customers behave predictably. But some are **weird**:
- Someone earns **$137k/year** but spends almost nothing *(miser? saving up?)*
- Someone earns very little but has a sky-high spending score *(living beyond means?)*

These are **anomalies** — and finding them manually across thousands of customers is impossible. This app does it in seconds.

**Real-world use cases:**
- 🏦 Banks detecting fraud (unusual transactions)
- 🏪 Retail identifying at-risk or VIP customers
- 📦 Supply chains spotting defective products
- 🏥 Healthcare flagging unusual patient readings

---

## How Does It Work? (Simple Version)

Think of it in 3 steps:

**Step 1 — Group customers**
The app sorts all 200 customers into N groups (you pick 2–10) based on how similar their income and spending are. Each group has a "center" (the average of that group).

**Step 2 — Measure how far each customer is from their group's center**
A customer who fits perfectly sits close to the center. A weird customer sits far away.

**Step 3 — Flag the farthest ones**
The top X% farthest customers are marked as anomalies. You control this threshold (1%–20%).

---

## Live Demo

👉 **[Open the App](https://atulpandey5678-anamoly-detection-streamlit-app-lbczrc.streamlit.app/)**

1. Use the sidebar sliders to configure clusters and anomaly threshold
2. Click **▶ Run Analysis**
3. See the scatter plot, cluster breakdown, and flagged anomalies table
4. Download `anomalies.csv` with one click

---

## What You See in the App

| Section | Description |
|---|---|
| **Metrics Row** | Total customers, anomalies found, clusters, threshold |
| **Cluster Visualization** | Scatter plot — Income vs Spending Score, clusters in colors, anomalies in red |
| **Cluster Breakdown** | Table showing how many customers fall into each group |
| **Flagged Anomalies Table** | List of unusual customers sorted by how far they are from normal |
| **Download Button** | Export anomalies as CSV |

---

## Sidebar Controls Explained

**Number of Clusters (2–10)**
Decides how many groups to split customers into. Think of it like sorting people into buckets. 5 buckets = "high earner high spender", "high earner low spender", "average", "low earner high spender", "low earner low spender". More clusters = more precise groups.

**Anomaly Threshold (%)**
Controls how strict you are. At 5%, the app flags the 10 weirdest customers out of 200. At 10%, it flags 20. Lower = stricter, only catches the most obvious outliers.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **UI / Frontend** | Streamlit |
| **ML Model** | scikit-learn (K-Means Clustering) |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Dataset Source** | KaggleHub (`shwetabh123/mall-customers`) |
| **Config Management** | python-dotenv |
| **Optional AI Insights** | OpenAI API (`gpt-4o-mini`) |
| **Language** | Python 3.10+ |

---

## Project Structure

```
Anamoly_detection/
│
├── streamlit_app.py        ← Main app (run this)
├── main.py                 ← CLI pipeline runner
├── requirements.txt        ← All dependencies
│
├── src/
│   ├── config.py           ← All settings in one place
│   ├── preprocessing.py    ← Load, clean & scale data
│   ├── model.py            ← K-Means clustering
│   ← anomaly.py            ← Distance-based anomaly detection
│   └── utils.py            ← Plots, CSV saving, AI insights
│
├── data/
│   └── Mall_Customers.csv  ← Auto-populated on first run
│
└── outputs/
    ├── anomalies.csv        ← Flagged customers
    └── plots/
        └── clusters_anomalies.png
```

---

## How the ML Works (For the Curious)

**Algorithm:** K-Means Clustering + Euclidean Distance Scoring

1. **StandardScaler** normalizes income and spending to the same scale so neither dominates
2. **K-Means** assigns each customer to the nearest cluster centroid
3. **Euclidean distance** is computed from each customer to their assigned centroid
4. The top `anomaly_fraction` % of farthest points are flagged as anomalies

This is **unsupervised learning** — no one told the model what an anomaly looks like. It figures out "normal" on its own, then surfaces what doesn't fit.

**Limitation:** K-Means assumes roughly spherical clusters. For non-linear patterns, consider Isolation Forest or Local Outlier Factor.

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/atulpandey5678/Anamoly_detection.git
cd Anamoly_detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run streamlit_app.py
```

The dataset auto-downloads from Kaggle on the first run — no manual download needed.

**Optional:** Add an OpenAI key for AI-generated insights:
```
OPENAI_API_KEY=your_key_here
```

---

## Environment Variables (Optional)

| Variable | Default | Description |
|---|---|---|
| `N_CLUSTERS` | `5` | Number of K-Means clusters |
| `ANOMALY_FRACTION` | `0.05` | Fraction of customers to flag |
| `RANDOM_STATE` | `42` | Seed for reproducibility |
| `KAGGLE_DATASET` | `shwetabh123/mall-customers` | Kaggle dataset slug |
| `OPENAI_API_KEY` | *(none)* | Enables AI insight generation |
| `MODEL_NAME` | `gpt-4o-mini` | OpenAI model for insights |

---

## Dataset

**Mall Customer Segmentation Data** — 200 customers with:
- `CustomerID` — Unique identifier
- `Gender` — Male / Female
- `Age` — Customer age
- `Annual Income (k$)` — Yearly income in thousands
- `Spending Score (1–100)` — Mall-assigned score based on spending behavior

Source: [Kaggle — shwetabh123/mall-customers](https://www.kaggle.com/datasets/shwetabh123/mall-customers)
