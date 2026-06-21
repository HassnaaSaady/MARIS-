# MLOps — Maritime Navigation AI System

## Why MLflow improves production ML reliability

The original training scripts (`src/ml/train_*.py`) write `.pkl` files and
print metrics to stdout.  That works for a first prototype but creates
real operational problems at scale:

| Problem without MLflow | Consequence in production |
|---|---|
| No experiment history | Can't compare "did contamination=0.01 beat 0.005 on last week's data?" |
| No parameter tracking | Retrained model may silently change behaviour; no audit trail |
| No model versioning | Rolling back to last-good model requires manual file management |
| Metrics only in logs | Can't plot MAE trend over 10 retraining runs |
| Single artifact location | Two engineers training simultaneously overwrite each other's `.pkl` |
| No model staging | No way to promote a candidate model through dev → staging → production safely |

MLflow solves all of these with four components:

```
┌─────────────────────────────────────────────────────────────┐
│  MLflow Tracking Server                                      │
│  (local file store or remote PostgreSQL+S3)                  │
│                                                              │
│  Experiments ──► Runs ──► { params, metrics, artifacts }    │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  MLflow Model Registry                                       │
│                                                              │
│  maritime-anomaly-detector    v1 Staging → v2 Production     │
│  maritime-congestion-cls      v1 Production                  │
│  maritime-position-predictor-5min   v3 Production            │
│  maritime-position-predictor-10min  v2 Production            │
│  maritime-position-predictor-15min  v2 Production            │
└─────────────────────────────────────────────────────────────┘
```

---

## Design principles for this integration

### 1. MLflow is optional — system never breaks without it

Every MLflow call is wrapped in `try/except ImportError` and
`try/except MlflowException`.  When `MLFLOW_TRACKING_URI` is unset or
the library is not installed:

- Training scripts fall back to writing `.pkl` files to `MODELS_PATH`
  (exactly what the original scripts do)
- `model_loader.py` falls back to loading those same `.pkl` files
- `scorer.py` and `live_scorer.py` are **never modified** — they
  continue to load `.pkl` files as always

### 2. `.pkl` fallback is always written

Every MLflow training script writes the same `.pkl` artifact paths as
the original (`isolation_forest.pkl`, `congestion_rf.pkl`, etc.) so
`scorer.py` works whether models came from MLflow or not.

### 3. Existing training scripts are never touched

`src/ml/train_anomaly.py`, `train_congestion.py`, `train_predictor.py`,
`scorer.py`, `live_scorer.py` are read-only from this layer's perspective.
The `mlops/experiments/` scripts are parallel, additive paths to the same
model artifacts — not replacements.

---

## File map

```
mlops/
├── README.md                          ← this file
├── configs/
│   └── mlflow_config.py               ← tracking URI, experiment names
├── experiments/
│   ├── train_anomaly_mlflow.py        ← IsolationForest + MLflow logging
│   ├── train_congestion_mlflow.py     ← RandomForest + MLflow logging
│   ├── train_predictor_mlflow.py      ← XGBoost 5/10/15 min + MLflow logging
│   └── feature_importance.py          ← visualisations saved to artifacts/
├── model_registry/
│   ├── model_loader.py                ← MLflow registry → .pkl fallback
│   └── evaluate_models.py             ← anomaly rate, MAE, F1 report
└── artifacts/                         ← generated plots + evaluation JSON
    (created on first run)
```

---

## Quick start

### Option A: Local file-based tracking (zero infrastructure)

```bash
# No server needed — MLflow writes to mlops/mlruns/
pip install mlflow scikit-learn xgboost

python mlops/experiments/train_anomaly_mlflow.py
python mlops/experiments/train_congestion_mlflow.py
python mlops/experiments/train_predictor_mlflow.py

mlflow ui --backend-store-uri mlops/mlruns --port 5001
# Open http://localhost:5001
```

### Option B: Full tracking server with Docker

```bash
# Start MLflow server alongside the main stack
docker compose -f docker-compose.yml \
               -f docker/docker-compose.mlflow.yml \
               up -d mlflow

# Point experiments at the server
export MLFLOW_TRACKING_URI=http://localhost:5000
python mlops/experiments/train_anomaly_mlflow.py
```

### Option C: No MLflow at all (unchanged behaviour)

```bash
# Just run the original scripts — nothing in mlops/ is affected
docker compose exec producer python src/ml/train_anomaly.py
docker compose exec producer python src/ml/train_congestion.py
```

---

## Model lifecycle

```
train_*_mlflow.py
      │  logs params + metrics
      │  saves model artifact
      ▼
MLflow Tracking Server
      │  auto-registers run
      ▼
Model Registry (Staging)
      │  manual or CI promotion
      ▼
Model Registry (Production)
      │
      ▼
model_loader.py ──► scorer.py / live_scorer.py
```

Promotion command:
```python
import mlflow
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="maritime-anomaly-detector",
    version=2,
    stage="Production",
)
```

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `mlops/mlruns` | File store (no server) or `http://host:5000` |
| `MLFLOW_EXPERIMENT_NAME` | `maritime-ais-models` | Parent experiment bucket |
| `MLFLOW_S3_ENDPOINT_URL` | — | MinIO / localstack endpoint for artifact store |
| `AWS_ACCESS_KEY_ID` | — | S3 artifact store credentials |
| `MODELS_PATH` | `/app/models` | `.pkl` fallback path (from `config.py`) |
