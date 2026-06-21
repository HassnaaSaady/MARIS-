"""
train_congestion_mlflow.py — Maritime Navigation AI System
RandomForest congestion classifier training with full MLflow tracking.

Mirrors src/ml/train_congestion.py — DO NOT modify that file.
Adds MLflow experiment logging, feature importance artifacts, and model
registration on top of identical training logic.

Fallback: if MLflow is unavailable, saves .pkl files to MODELS_PATH and
exits cleanly — no exception raised.

Run:
    python mlops/experiments/train_congestion_mlflow.py
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
# Path setup
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).resolve()
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
    CONGESTION_EXPERIMENT, CONGESTION_MODEL_NAME, BASE_TAGS,
    get_tracking_uri, ARTIFACTS_DIR,
)

try:
    from config import DELTA_GOLD_DENSITY_PATH, MODELS_PATH
except ImportError:
    DELTA_GOLD_DENSITY_PATH = os.getenv("DELTA_GOLD_DENSITY_PATH",
                                        "/delta/gold/traffic_density")
    MODELS_PATH = os.getenv("MODELS_PATH", "/app/models")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature set — identical to train_congestion.py
# ---------------------------------------------------------------------------
FEATURES = [
    "vessel_count", "avg_sog", "stopped_count",
    "hour", "day_of_week", "is_weekend",
    "lat_bin", "lon_bin",
]

HYPERPARAMS = {
    "n_estimators":    200,
    "max_depth":       10,
    "min_samples_leaf": 5,
    "random_state":    42,
    "n_jobs":         -1,
    "class_weight":   "balanced",
}

CONGESTION_HIGH   = int(os.getenv("CONGESTION_HIGH",   "15"))
CONGESTION_MEDIUM = int(os.getenv("CONGESTION_MEDIUM", "5"))


# ---------------------------------------------------------------------------
# Data loading — identical to train_congestion.py
# ---------------------------------------------------------------------------

def load_density_data() -> pd.DataFrame:
    gold_path    = Path(DELTA_GOLD_DENSITY_PATH)
    parquet_path = Path("/app/data/parquet")

    if gold_path.exists():
        logger.info("Loading from Gold Delta: %s", gold_path)
        df = pd.read_parquet(gold_path)
        if "data_split" in df.columns:
            df = df[df["data_split"] == "train"].copy()
        logger.info("Density data: %d rows", len(df))
        return df

    logger.info("Building density from Parquet: %s", parquet_path)
    all_files = sorted(parquet_path.rglob("*.parquet"))
    MAX_ROWS   = 1_000_000
    sample_per = MAX_ROWS // max(len(all_files), 1)

    chunks, total = [], 0
    for f in all_files:
        try:
            chunk = pd.read_parquet(f)
            if "data_split" in chunk.columns:
                chunk = chunk[chunk["data_split"] == "train"]
            if chunk.empty:
                continue
            if len(chunk) > sample_per:
                chunk = chunk.sample(n=sample_per, random_state=42)
            chunks.append(chunk)
            total += len(chunk)
            if total >= MAX_ROWS:
                break
        except Exception as exc:
            logger.warning("Skipping %s: %s", f.name, exc)

    if not chunks:
        raise FileNotFoundError("No density training data found")

    df = pd.concat(chunks, ignore_index=True)
    df["base_datetime"] = pd.to_datetime(df["base_datetime"], errors="coerce")
    df["hour_bucket"] = df["base_datetime"].dt.floor("H")
    df["lat_bin"]     = (df["lat"] / 0.1).astype(int) * 0.1
    df["lon_bin"]     = (df["lon"] / 0.1).astype(int) * 0.1

    density = (
        df.groupby(["lat_bin", "lon_bin", "hour_bucket"])
        .agg(
            vessel_count  = ("mmsi", "nunique"),
            avg_sog       = ("sog",  "mean"),
            stopped_count = ("sog",  lambda x: (x < 0.5).sum()),
        )
        .reset_index()
    )
    density["hour"]           = density["hour_bucket"].dt.hour
    density["day_of_week"]    = density["hour_bucket"].dt.dayofweek
    density["is_weekend"]     = (density["day_of_week"] >= 5).astype(int)
    density["congestion_level"] = density["vessel_count"].apply(
        lambda n: "HIGH" if n >= CONGESTION_HIGH
                  else ("MEDIUM" if n >= CONGESTION_MEDIUM else "LOW")
    )
    logger.info("Built density: %d grid-hour cells", len(density))
    return density


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        classification_report, accuracy_score,
        precision_score, recall_score, f1_score
    )
    from sklearn.model_selection import train_test_split

    feat_cols = [c for c in FEATURES if c in df.columns]
    X  = df[feat_cols].fillna(0).values
    le = LabelEncoder()
    y  = le.fit_transform(df["congestion_level"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(**HYPERPARAMS)
    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)

    # Per-class metrics
    report = classification_report(y_te, preds,
                                   target_names=le.classes_,
                                   zero_division=0,
                                   output_dict=True)

    metrics = {
        "accuracy":          float(accuracy_score(y_te, preds)),
        "macro_precision":   float(report["macro avg"]["precision"]),
        "macro_recall":      float(report["macro avg"]["recall"]),
        "macro_f1":          float(report["macro avg"]["f1-score"]),
        "train_samples":     int(len(X_tr)),
        "test_samples":      int(len(X_te)),
        "feature_count":     int(len(feat_cols)),
    }

    # Per-class metrics
    for cls in le.classes_:
        if cls in report:
            metrics[f"{cls.lower()}_f1"] = float(report[cls]["f1-score"])

    # Feature importance dict (also used by feature_importance.py)
    feat_importance = dict(zip(feat_cols, model.feature_importances_.tolist()))

    logger.info("Accuracy=%.3f  Macro-F1=%.3f", metrics["accuracy"], metrics["macro_f1"])
    for feat, imp in sorted(feat_importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        logger.info("  %-20s %s %.3f", feat, bar, imp)

    return model, le, feat_cols, feat_importance, metrics


# ---------------------------------------------------------------------------
# Save artefacts
# ---------------------------------------------------------------------------

def save_models(model, le, feat_cols):
    out = Path(MODELS_PATH)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,     out / "congestion_rf.pkl")
    joblib.dump(le,        out / "congestion_encoder.pkl")
    joblib.dump(feat_cols, out / "congestion_features.pkl")
    logger.info("Models saved to %s", out)


def _save_importance_csv(feat_importance: dict) -> Path:
    """Save feature importance to CSV for artifact logging."""
    rows = sorted(feat_importance.items(), key=lambda x: -x[1])
    df   = pd.DataFrame(rows, columns=["feature", "importance"])
    out  = ARTIFACTS_DIR / "congestion_feature_importance.csv"
    df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start = datetime.now()
    logger.info("=" * 55)
    logger.info("Maritime AIS — Congestion Classifier Training (MLflow)")
    logger.info("=" * 55)

    df = load_density_data()
    model, le, feat_cols, feat_importance, metrics = train(df)

    # Always save .pkl (scorer.py fallback)
    save_models(model, le, feat_cols)
    elapsed = (datetime.now() - start).total_seconds()

    if not is_mlflow_available():
        logger.warning(
            "[MLflow] Not available — models saved as .pkl only."
        )
        return

    # ── MLflow logging ────────────────────────────────────────────────────────
    import mlflow
    import mlflow.sklearn

    setup_experiment(CONGESTION_EXPERIMENT)
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(CONGESTION_EXPERIMENT)

    importance_csv = _save_importance_csv(feat_importance)

    with mlflow.start_run(
            run_name=f"random-forest-{datetime.now():%Y%m%d-%H%M}") as run:

        mlflow.log_params({
            **HYPERPARAMS,
            "features":      ",".join(feat_cols),
            "feature_count": len(feat_cols),
            "congestion_high":   CONGESTION_HIGH,
            "congestion_medium": CONGESTION_MEDIUM,
            "classes":       ",".join(le.classes_),
        })

        mlflow.log_metrics({**metrics, "training_time_s": elapsed})

        mlflow.set_tags({**BASE_TAGS, "model_type": "RandomForestClassifier",
                         "target": "congestion_level"})

        # Feature importance as param dict (also queryable in UI)
        for feat, imp in feat_importance.items():
            mlflow.log_metric(f"importance_{feat}", imp)

        mlflow.sklearn.log_model(model, "congestion_rf",
                                 registered_model_name=CONGESTION_MODEL_NAME)
        mlflow.sklearn.log_model(le, "congestion_encoder")

        # CSV artifact
        mlflow.log_artifact(str(importance_csv), "feature_importance")
        mlflow.log_artifact(str(Path(MODELS_PATH) / "congestion_rf.pkl"),
                            "pkl_artifacts")

        run_id = run.info.run_id

    register_model(run_id, "congestion_rf", CONGESTION_MODEL_NAME, "Staging")

    logger.info("Training complete in %.1fs", elapsed)
    logger.info("View results: mlflow ui --backend-store-uri %s",
                get_tracking_uri())


if __name__ == "__main__":
    main()
