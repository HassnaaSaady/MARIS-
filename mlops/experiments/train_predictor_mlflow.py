"""
train_predictor_mlflow.py — Maritime Navigation AI System
XGBoost position predictor (5 / 10 / 15 min) with full MLflow tracking.

Mirrors src/ml/train_predictor.py — DO NOT modify that file.
Adds per-horizon MLflow runs, MAE/RMSE in nautical miles, feature
importance logging, and model registration.

Fallback: if MLflow unavailable, saves .pkl files to MODELS_PATH and exits
cleanly.

Run:
    python mlops/experiments/train_predictor_mlflow.py
  or train a single horizon:
    PREDICT_MINUTES=5 python mlops/experiments/train_predictor_mlflow.py
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
    PREDICTOR_EXPERIMENT, PREDICTOR_MODEL_NAME, BASE_TAGS,
    get_tracking_uri, ARTIFACTS_DIR,
)

try:
    from config import DELTA_SILVER_PATH, MODELS_PATH
except ImportError:
    DELTA_SILVER_PATH = os.getenv("DELTA_SILVER_PATH", "/delta/silver/ais_clean")
    MODELS_PATH       = os.getenv("MODELS_PATH", "/app/models")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — identical to train_predictor.py
# ---------------------------------------------------------------------------
FEATURES = [
    "lat", "lon",
    "sog", "cog", "heading",
    "sog_change", "heading_change",
    "time_delta_sec",
    "hour", "month",
]

PREDICT_MINUTES = [
    int(m) for m in os.getenv("PREDICT_MINUTES", "5,10,15").split(",")
]

XGB_PARAMS = {
    "n_estimators":   300,
    "max_depth":       6,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "random_state":   42,
    "n_jobs":         -1,
}

NM_PER_DEGREE = 60.0   # approximate: 1° ≈ 60 nautical miles


# ---------------------------------------------------------------------------
# Data loading — identical to train_predictor.py
# ---------------------------------------------------------------------------

def load_train_data(max_rows: int = 2_000_000) -> pd.DataFrame:
    parquet_path = Path("/app/data/parquet")
    silver_path  = Path(DELTA_SILVER_PATH)
    load_path    = silver_path if silver_path.exists() else parquet_path

    all_files   = sorted(load_path.rglob("*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No parquet files under {load_path}")

    sample_per  = max_rows // max(len(all_files), 1)
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
            if total >= max_rows:
                break
        except Exception as exc:
            logger.warning("Skipping %s: %s", f.name, exc)

    if not chunks:
        raise FileNotFoundError("No train data found")

    df = pd.concat(chunks, ignore_index=True)
    df["base_datetime"] = pd.to_datetime(df["base_datetime"], errors="coerce")
    df = df.sort_values(["mmsi", "base_datetime"])

    # Temporal features
    df["hour"]  = df["base_datetime"].dt.hour
    df["month"] = df["base_datetime"].dt.month

    logger.info("Train data: %d rows, %d vessels", len(df), df["mmsi"].nunique())
    return df


def build_targets(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Build delta_lat / delta_lon targets N minutes ahead."""
    # ~1 AIS ping every 28 s → steps = (minutes * 60) / 28
    steps = max(1, int(minutes * 60 / 28))
    df    = df.copy()
    grp   = df.groupby("mmsi")

    df["next_lat"]  = grp["lat"].shift(-steps)
    df["next_lon"]  = grp["lon"].shift(-steps)
    df["delta_lat"] = df["next_lat"] - df["lat"]
    df["delta_lon"] = df["next_lon"] - df["lon"]

    df = df.dropna(subset=["delta_lat", "delta_lon"])
    # Remove teleportation artefacts
    df = df[(df["delta_lat"].abs() < 1.0) & (df["delta_lon"].abs() < 1.0)]
    return df


# ---------------------------------------------------------------------------
# XGBoost training for one horizon
# ---------------------------------------------------------------------------

def train_horizon(df: pd.DataFrame, feat_cols: list, minutes: int) -> dict:
    """
    Train XGBoost lat + lon regressors for one prediction horizon.
    Returns metrics dict.
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed — pip install xgboost")
        return {}

    df_t = build_targets(df, minutes)
    if len(df_t) < 1000:
        logger.warning("%d-min: only %d samples — skipping", minutes, len(df_t))
        return {}

    X     = df_t[feat_cols].fillna(0).values
    y_lat = df_t["delta_lat"].values
    y_lon = df_t["delta_lon"].values

    split  = int(len(X) * 0.8)
    X_tr,  X_val  = X[:split],      X[split:]
    yl_tr, yl_val = y_lat[:split],   y_lat[split:]
    yn_tr, yn_val = y_lon[:split],   y_lon[split:]

    model_lat = xgb.XGBRegressor(**XGB_PARAMS)
    model_lon = xgb.XGBRegressor(**XGB_PARAMS)

    model_lat.fit(X_tr, yl_tr, eval_set=[(X_val, yl_val)], verbose=False)
    model_lon.fit(X_tr, yn_tr, eval_set=[(X_val, yn_val)], verbose=False)

    # Metrics in nautical miles
    pred_lat = model_lat.predict(X_val)
    pred_lon = model_lon.predict(X_val)

    mae_lat_nm  = float(np.mean(np.abs(pred_lat - yl_val)) * NM_PER_DEGREE)
    mae_lon_nm  = float(np.mean(np.abs(pred_lon - yn_val)) * NM_PER_DEGREE)
    rmse_lat_nm = float(np.sqrt(np.mean((pred_lat - yl_val)**2)) * NM_PER_DEGREE)
    rmse_lon_nm = float(np.sqrt(np.mean((pred_lon - yn_val)**2)) * NM_PER_DEGREE)
    total_mae   = float(np.sqrt(mae_lat_nm**2 + mae_lon_nm**2))

    metrics = {
        "mae_lat_nm":   mae_lat_nm,
        "mae_lon_nm":   mae_lon_nm,
        "rmse_lat_nm":  rmse_lat_nm,
        "rmse_lon_nm":  rmse_lon_nm,
        "total_mae_nm": total_mae,
        "train_samples":int(len(X_tr)),
        "val_samples":  int(len(X_val)),
    }

    logger.info("%d-min → lat MAE=%.3f nm  lon MAE=%.3f nm  total=%.3f nm",
                minutes, mae_lat_nm, mae_lon_nm, total_mae)

    # Feature importance
    feat_imp = dict(zip(feat_cols, model_lat.feature_importances_.tolist()))

    # Save .pkl (scorer.py fallback — same paths as train_predictor.py)
    out = Path(MODELS_PATH)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_lat, out / f"xgb_lat_{minutes}min.pkl")
    joblib.dump(model_lon, out / f"xgb_lon_{minutes}min.pkl")

    return {
        "metrics":     metrics,
        "feat_imp":    feat_imp,
        "model_lat":   model_lat,
        "model_lon":   model_lon,
        "run_samples": len(df_t),
    }


def _save_importance_csv(feat_imp: dict, minutes: int) -> Path:
    rows = sorted(feat_imp.items(), key=lambda x: -x[1])
    df   = pd.DataFrame(rows, columns=["feature", "importance"])
    out  = ARTIFACTS_DIR / f"predictor_{minutes}min_feature_importance.csv"
    df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start = datetime.now()
    logger.info("=" * 55)
    logger.info("Maritime AIS — Position Predictor Training (MLflow)")
    logger.info("Horizons: %s minutes", PREDICT_MINUTES)
    logger.info("=" * 55)

    df        = load_train_data()
    feat_cols = [c for c in FEATURES if c in df.columns]

    # Save feature list (scorer.py reads this)
    out = Path(MODELS_PATH)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(feat_cols, out / "predictor_features.pkl")

    horizon_results = {}
    for minutes in PREDICT_MINUTES:
        logger.info("── %d-minute prediction ──", minutes)
        result = train_horizon(df, feat_cols, minutes)
        if result:
            horizon_results[minutes] = result

    elapsed = (datetime.now() - start).total_seconds()

    if not is_mlflow_available():
        logger.warning(
            "[MLflow] Not available — models saved as .pkl only."
        )
        return

    # ── MLflow logging — one run per horizon ─────────────────────────────────
    import mlflow
    import mlflow.xgboost

    setup_experiment(PREDICTOR_EXPERIMENT)
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(PREDICTOR_EXPERIMENT)

    for minutes, result in horizon_results.items():
        model_name = f"{PREDICTOR_MODEL_NAME}-{minutes}min"

        importance_csv = _save_importance_csv(result["feat_imp"], minutes)

        with mlflow.start_run(
                run_name=f"xgboost-{minutes}min-{datetime.now():%Y%m%d-%H%M}") as run:

            mlflow.log_params({
                **XGB_PARAMS,
                "horizon_minutes":  minutes,
                "features":         ",".join(feat_cols),
                "feature_count":    len(feat_cols),
                "train_samples":    result["metrics"].get("train_samples", 0),
            })

            mlflow.log_metrics({
                **result["metrics"],
                "training_time_s": elapsed / len(horizon_results),
            })

            # Feature importance as metrics (filterable in UI)
            for feat, imp in result["feat_imp"].items():
                mlflow.log_metric(f"importance_{feat}", imp)

            mlflow.set_tags({
                **BASE_TAGS,
                "model_type":      "XGBRegressor",
                "target":          f"position_{minutes}min",
                "horizon_minutes": str(minutes),
            })

            # Log both lat and lon models
            mlflow.xgboost.log_model(result["model_lat"], f"xgb_lat_{minutes}min",
                                     registered_model_name=f"{model_name}-lat")
            mlflow.xgboost.log_model(result["model_lon"], f"xgb_lon_{minutes}min",
                                     registered_model_name=f"{model_name}-lon")

            mlflow.log_artifact(str(importance_csv), "feature_importance")
            mlflow.log_artifact(str(out / f"xgb_lat_{minutes}min.pkl"),
                                "pkl_artifacts")

            run_id = run.info.run_id

        register_model(run_id, f"xgb_lat_{minutes}min",
                       f"{model_name}-lat", "Staging")
        register_model(run_id, f"xgb_lon_{minutes}min",
                       f"{model_name}-lon", "Staging")

    logger.info("All horizons complete in %.1fs", elapsed)
    logger.info("View results: mlflow ui --backend-store-uri %s",
                get_tracking_uri())


if __name__ == "__main__":
    main()
