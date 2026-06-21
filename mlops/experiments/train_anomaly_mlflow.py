"""
train_anomaly_mlflow.py — Maritime Navigation AI System
IsolationForest anomaly detector training with full MLflow tracking.

Mirrors the logic in src/ml/train_anomaly.py — DO NOT modify that file.
This script adds experiment tracking, parameter logging, metric logging,
model registration, and artifact saving on top of identical training code.

Fallback behaviour when MLflow is unavailable:
  - Training runs normally
  - Model saved to MODELS_PATH/*.pkl (same paths as train_anomaly.py)
  - No experiment logged; no exception raised

Run:
    python mlops/experiments/train_anomaly_mlflow.py
  or
    MLFLOW_TRACKING_URI=http://localhost:5000 \
    python mlops/experiments/train_anomaly_mlflow.py
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — supports Docker (/app/src) and local dev
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
for _p in [
    "/app/src/common", "/app/src",
    str(_PROJECT_ROOT / "src" / "common"),
    str(_PROJECT_ROOT / "src"),
    str(_PROJECT_ROOT / "mlops"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs.mlflow_config import (
    is_mlflow_available, setup_experiment, register_model,
    ANOMALY_EXPERIMENT, ANOMALY_MODEL_NAME, BASE_TAGS,
    get_tracking_uri,
)

try:
    from config import DELTA_SILVER_PATH, MODELS_PATH, ANOMALY_CONTAMINATION
except ImportError:
    DELTA_SILVER_PATH     = os.getenv("DELTA_SILVER_PATH", "/delta/silver/ais_clean")
    MODELS_PATH           = os.getenv("MODELS_PATH", "/app/models")
    ANOMALY_CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", "0.01"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature set — identical to train_anomaly.py
# ---------------------------------------------------------------------------
FEATURES = [
    "sog", "cog", "heading",
    "lat", "lon",
    "sog_change", "heading_change",
    "time_delta_sec", "distance_nm",
    "length", "width", "draft",
]

HYPERPARAMS = {
    "n_estimators":    200,
    "contamination":   ANOMALY_CONTAMINATION,
    "max_samples":     "auto",
    "random_state":    42,
    "n_jobs":          -1,
}

RULES_CONFIG = {
    "sudden_stop_threshold":    5.0,
    "sharp_turn_threshold":    45.0,
    "speed_max_threshold":     30.0,
    "ais_gap_seconds":        300,
}


# ---------------------------------------------------------------------------
# Data loading — identical to train_anomaly.py
# ---------------------------------------------------------------------------

def load_train_data(max_rows: int = 2_000_000) -> pd.DataFrame:
    parquet_path = Path("/app/data/parquet")
    silver_path  = Path(DELTA_SILVER_PATH)
    load_path    = silver_path if silver_path.exists() else parquet_path

    all_files = sorted(load_path.rglob("*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No parquet files under {load_path}")

    sample_per_file = max_rows // max(len(all_files), 1)
    chunks, total   = [], 0

    for f in all_files:
        try:
            chunk = pd.read_parquet(f)
            if "data_split" in chunk.columns:
                chunk = chunk[chunk["data_split"] == "train"]
            if chunk.empty:
                continue
            if len(chunk) > sample_per_file:
                chunk = chunk.sample(n=sample_per_file, random_state=42)
            chunks.append(chunk)
            total += len(chunk)
            if total >= max_rows:
                break
        except Exception as exc:
            logger.warning("Skipping %s: %s", f.name, exc)

    if not chunks:
        raise FileNotFoundError("No train rows found in parquet files")

    df = pd.concat(chunks, ignore_index=True)
    logger.info("Train data: %d rows, %d vessels", len(df), df["mmsi"].nunique())
    return df


def apply_rule_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rule_anomaly"] = False
    df["rule_type"]    = ""

    if "sog_change" in df.columns:
        mask = df["sog_change"].fillna(0) < -RULES_CONFIG["sudden_stop_threshold"]
        df.loc[mask, ["rule_anomaly", "rule_type"]] = [True, "SUDDEN_STOP"]

    if "heading_change" in df.columns:
        mask = (df["heading_change"].fillna(0).abs() > RULES_CONFIG["sharp_turn_threshold"]) & (df["sog"] > 2.0)
        df.loc[mask, ["rule_anomaly", "rule_type"]] = [True, "SHARP_TURN"]

    mask = df["sog"] > RULES_CONFIG["speed_max_threshold"]
    df.loc[mask, ["rule_anomaly", "rule_type"]] = [True, "UNUSUAL_SPEED"]

    if "time_delta_sec" in df.columns:
        mask = df["time_delta_sec"] > RULES_CONFIG["ais_gap_seconds"]
        df.loc[mask, ["rule_anomaly", "rule_type"]] = [True, "AIS_GAP"]

    logger.info("Rule anomalies: %d (%.2f%%)",
                df["rule_anomaly"].sum(),
                df["rule_anomaly"].mean() * 100)
    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        classification_report, precision_score, recall_score, f1_score
    )

    feat_cols = [c for c in FEATURES if c in df.columns]
    X_raw     = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    model = IsolationForest(**HYPERPARAMS)
    model.fit(X_scaled)

    # Evaluate against rule-based ground truth
    metrics = {}
    if "rule_anomaly" in df.columns:
        preds      = model.predict(X_scaled)
        pred_bin   = preds == -1
        true_bin   = df["rule_anomaly"].values

        metrics["precision"]    = float(precision_score(true_bin, pred_bin, zero_division=0))
        metrics["recall"]       = float(recall_score(true_bin, pred_bin, zero_division=0))
        metrics["f1"]           = float(f1_score(true_bin, pred_bin, zero_division=0))
        metrics["anomaly_rate"] = float(pred_bin.mean())
        metrics["rule_rate"]    = float(true_bin.mean())
        metrics["train_samples"]= int(len(X_raw))
        metrics["feature_count"]= int(len(feat_cols))

        logger.info("Metrics: precision=%.3f recall=%.3f f1=%.3f",
                    metrics["precision"], metrics["recall"], metrics["f1"])

    return model, scaler, feat_cols, metrics


# ---------------------------------------------------------------------------
# Save artefacts (always — MLflow + pkl fallback)
# ---------------------------------------------------------------------------

def save_models(model, scaler, feat_cols):
    """Write .pkl files to MODELS_PATH so scorer.py can load them."""
    out = Path(MODELS_PATH)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,       out / "isolation_forest.pkl")
    joblib.dump(scaler,      out / "scaler_anomaly.pkl")
    joblib.dump(feat_cols,   out / "anomaly_features.pkl")
    joblib.dump(RULES_CONFIG, out / "rules_config.pkl")
    logger.info("Models saved to %s", out)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    start = datetime.now()
    logger.info("=" * 55)
    logger.info("Maritime AIS — Anomaly Detector Training (MLflow)")
    logger.info("=" * 55)

    df      = load_train_data()
    df      = apply_rule_labels(df)
    model, scaler, feat_cols, metrics = train(df)

    # Always save .pkl so scorer.py fallback works
    save_models(model, scaler, feat_cols)

    elapsed = (datetime.now() - start).total_seconds()

    if not is_mlflow_available():
        logger.warning(
            "[MLflow] Not available — models saved as .pkl only. "
            "Install mlflow or set MLFLOW_TRACKING_URI to enable tracking."
        )
        return

    # ── MLflow logging ────────────────────────────────────────────────────────
    import mlflow
    import mlflow.sklearn

    setup_experiment(ANOMALY_EXPERIMENT)
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(ANOMALY_EXPERIMENT)

    with mlflow.start_run(run_name=f"isolation-forest-{datetime.now():%Y%m%d-%H%M}") as run:
        # Parameters
        mlflow.log_params({
            **HYPERPARAMS,
            "features":      ",".join(feat_cols),
            "feature_count": len(feat_cols),
            "train_samples": metrics.get("train_samples", 0),
            "scaler":        "StandardScaler",
        })

        # Metrics
        if metrics:
            mlflow.log_metrics({
                "precision":    metrics.get("precision",    0.0),
                "recall":       metrics.get("recall",       0.0),
                "f1_score":     metrics.get("f1",           0.0),
                "anomaly_rate": metrics.get("anomaly_rate", 0.0),
                "rule_rate":    metrics.get("rule_rate",    0.0),
                "training_time_s": elapsed,
            })

        # Tags
        mlflow.set_tags({**BASE_TAGS, "model_type": "IsolationForest",
                         "target": "anomaly_detection"})

        # Log models as MLflow artifacts
        mlflow.sklearn.log_model(model,  "isolation_forest",
                                 registered_model_name=ANOMALY_MODEL_NAME)
        mlflow.sklearn.log_model(scaler, "scaler_anomaly")

        # Log pkl paths as artifacts too (for audit)
        mlflow.log_artifact(str(Path(MODELS_PATH) / "isolation_forest.pkl"),
                            "pkl_artifacts")

        run_id = run.info.run_id
        logger.info("[MLflow] Run ID: %s", run_id)
        logger.info("[MLflow] Experiment: %s", ANOMALY_EXPERIMENT)

    # Promote to Staging in registry
    register_model(run_id, "isolation_forest", ANOMALY_MODEL_NAME, "Staging")

    logger.info("Training complete in %.1fs", elapsed)
    logger.info("View results: mlflow ui --backend-store-uri %s",
                get_tracking_uri())


if __name__ == "__main__":
    main()
