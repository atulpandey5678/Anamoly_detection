# Anamoly_detection

# Anomaly Detection in Customer Spending Behavior (ML)

Production-ready Python project to detect **outlier customers with highly erratic spending behavior** based on:

- **Annual Income (k$)**
- **Spending Score (1-100)**

Using **K-Means clustering** + **distance-to-centroid** anomaly scoring (top 5% farthest points).

## Dataset (Required)

This project expects the exact Kaggle dataset:

- Kaggle: `https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python`

Download `Mall_Customers.csv` and place it at:

```
data/Mall_Customers.csv
```

### Option B: Auto-load via KaggleHub (no manual download)

This project also supports loading the dataset using KaggleHub (as in your snippet).

Install:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

By default, KaggleHub fallback is enabled when `data/Mall_Customers.csv` is missing/empty.
You can control it via environment variables:

- `USE_KAGGLEHUB=1` (default) or `0`
- `KAGGLE_DATASET=vjchoudhary7/customer-segmentation-tutorial-in-python`
- `KAGGLE_FILE_PATH=Mall_Customers.csv`

## Folder Structure

```
/project-root
├── data/
│   └── Mall_Customers.csv
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── anomaly.py
│   ├── utils.py
│   └── config.py
├── outputs/
│   ├── anomalies.csv
│   └── plots/
├── main.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Setup

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` (already included as a placeholder) and optionally set:

```
OPENAI_API_KEY=your_openai_key_here
MODEL_NAME=gpt-4o-mini
```

If `OPENAI_API_KEY` is not set, insight generation will be skipped gracefully.

## Run

```bash
python main.py
```

## Outputs

After running:

- **CSV**: `outputs/anomalies.csv`
  - Contains the **flagged anomalous customers** (top 5% by distance).
- **Plot**: `outputs/plots/clusters_anomalies.png`
  - Income vs Spending Score scatter plot
  - Colored by cluster, anomalies highlighted in red
- **Optional insights** (if OpenAI key is set): `outputs/insights.txt`

## Example Output Columns

`outputs/anomalies.csv` includes:

- `CustomerID`
- `Income`
- `Spending Score`
- `Distance`
- `Is_Anomaly`
- `Cluster`

## Configuration

Core parameters live in `src/config.py` and can also be set via environment variables:

- `N_CLUSTERS` (default: 5)
- `ANOMALY_FRACTION` (default: 0.05)
- `RANDOM_STATE` (default: 42)
- `LOG_LEVEL` (default: INFO)

## Notes / Method Limitations

- K-Means assumes roughly spherical clusters in feature space; outliers are distance-based, not density-based.
- Results depend on scaling and the chosen `n_clusters`.
- For more robust anomaly detection, consider Isolation Forest or Local Outlier Factor (not implemented here by design).

