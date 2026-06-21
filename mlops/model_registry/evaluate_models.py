"""
evaluate_models.py — Maritime Navigation AI System
Evaluate all three trained models on the TEST split and produce a
structured report saved to mlops/artifacts/evaluation_report.json.

Metrics computed:
  Anomaly detector   — anomaly rate, precision/recall vs rule-based labels,
                       estimated false positive rate on known-normal vessels
  Congestion model   — accuracy, macro F1, per-class precision/recall
  Position predictor — MAE and RMSE in nautical miles per horizon (5/10/15 min)

Fallback behaviour:
  - Works entirely without MLflow (loads .pkl via model_loader.py)
  - If a model is missing, that section is skipped and report shows "not_available"
  - If MLflow is available, metrics are logged to a dedicated "evaluation" run

Run:
    python mlops/model_registry/evaluate_models.py
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

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
    is_mlflow_available, get_tracking_uri, ARTIFACTS_DIR,
    ANOMALY_EXPERIMENT, CONGESTION_EXPERIMENT, PREDICTOR_EXPERIMENT,
    BASE_TAGS,
)
from model_registry.model_loader import load_anomaly_model, load_congestion_model, load_predictor_models

try:
    from config import DELTA_SILVER_PATH, MODELS_PATH
    from config import CONGESTION_HIGH, CONGESTION_MEDIUM
except ImportError:
    DELTA_SILVER_PATH = os.getenv("DELTA_SILVER_PATH", "/delta/silver/ais_clean")
    MODELS_PATH       = os.getenv("MODELS_PATH", "/app/models")
    CONGESTION_HIGH   = int(os.getenv("CONGESTION_HIGH", "15"))
    CONGESTION_MEDIUM = int(os.getenv("CONGESTION_MEDIUM", "5"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NM_PER_DEGREE = 60.0
MAX_TEST_ROWS = 500_000


# ---------------------------------------------------------------------------
# Test data loader
# ---------------------------------------------------------------------------

def load_test_data(max_rows: int = MAX_TEST_ROWS) -> pd.DataFrame:
    parquet_path = Path("/app/data/parquet")
    silver_path  = Path(DELTA_SILVER_PATH)
    load_path    = silver_path if silver_path.exists() else parquet_path

    all_files   = sorted(load_path.rglob("*.parquet"))
    if not all_files:
        logger.warning("No parquet files found under %s", load_path)
        return pd.DataFrame()

    sample_per = max_rows // max(len(all_files), 1)
    chunks, total = [], 0

    for f in all_files:
        try:
            chunk = pd.read_parquet(f)
            if "data_split" in chunk.columns:
                chunk = chunk[chunk["data_split"] == "test"]
            if chunk.empty:
                continue
            if len(chunk) > sample_per:
                chunk = chunk.sample(n=sample_per, random_state=0)
            chunks.append(chunk)
            total += len(chunk)
            if total >= max_rows:
                break
        except Exception as exc:
            logger.warning("Skipping %s: %s", f.name, exc)

    if not chunks:
        logger.warning("No test split rows found")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df["base_datetime"] = pd.to_datetime(df["base_datetime"], errors="coerce")
    df = df.sort_values(["mmsi", "base_datetime"])
    df["hour"]  = df["base_datetime"].dt.hour
    df["month"] = df["base_datetime"].dt.month

    logger.info("Test data: %d rows, %d vessels", len(df), df["mmsi"].nunique())
    return df


# ---------------------------------------------------------------------------
# Rule-based ground truth (mirrors train_anomaly.py)
# ---------------------------------------------------------------------------

def _rule_labels(df: pd.DataFrame) -> np.ndarray:
    labels = np.zeros(len(df), dtype=bool)
    if "sog_change" in df.columns:
        labels |= df["sog_change"].fillna(0) < -5.0
    if "heading_change" in df.columns:
        labels |= (df["heading_change"].fillna(0).abs() > 45) & (df["sog"] > 2)
    labels |= df["sog"] > 30.0
    if "time_delta_sec" in df.columns:
        labels |= df["time_delta_sec"] > 300
    return labels


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def evaluate_anomaly(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"status": "no_test_data"}

    model, scaler, features, source = load_anomaly_model()
    if model is None:
        logger.warning("Anomaly model not available — skipping evaluation.")
        return {"status": "not_available"}

    feat_cols = [c for c in features if c in df.columns]
    if not feat_cols:
        return {"status": "no_features_in_test_data"}

    X_raw   = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
    if scaler is not None:
        X = scaler.transform(X_raw)
    else:
        X = X_raw.values

    try:
        preds  = model.predict(X)
        scores = model.score_samples(X)
    except Exception as exc:
        # MLflow pyfunc wrapper — use predict directly
        pred_df = model.predict(pd.DataFrame(X_raw))
        preds   = np.array(pred_df).flatten()
        scores  = preds.copy()

    pred_bin  = preds == -1
    true_bin  = _rule_labels(df)
    anomaly_rate = float(pred_bin.mean())

    result = {
        "status":         "ok",
        "source":         source,
        "test_samples":   int(len(df)),
        "anomaly_rate":   round(anomaly_rate, 4),
        "rule_rate":      round(float(true_bin.mean()), 4),
    }

    # Precision/recall against rule-based ground truth
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score
        result["precision"]  = round(float(precision_score(true_bin, pred_bin, zero_division=0)), 4)
        result["recall"]     = round(float(recall_score(true_bin, pred_bin, zero_division=0)), 4)
        result["f1_score"]   = round(float(f1_score(true_bin, pred_bin, zero_division=0)), 4)
    except Exception as exc:
        logger.warning("sklearn metrics failed: %s", exc)

    # False positive estimate: vessels with sog > 5 AND no rule anomaly
    fp_mask      = pred_bin & ~true_bin & (df["sog"] > 5)
    result["estimated_false_positive_rate"] = round(float(fp_mask.mean()), 4)

    logger.info("Anomaly  precision=%.3f recall=%.3f f1=%.3f rate=%.4f",
                result.get("precision", 0), result.get("recall", 0),
                result.get("f1_score", 0), anomaly_rate)
    return result


def evaluate_congestion(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"status": "no_test_data"}

    model, encoder, features, source = load_congestion_model()
    if model is None:
        logger.warning("Congestion model not available — skipping.")
        return {"status": "not_available"}

    # Build density grid from raw test positions
    needed = ["lat", "lon", "sog", "mmsi", "base_datetime"]
    if not all(c in df.columns for c in needed):
        return {"status": "missing_columns"}

    df_d = df.copy()
    df_d["hour_bucket"] = df_d["base_datetime"].dt.floor("H")
    df_d["lat_bin"]     = (df_d["lat"] / 0.1).astype(int) * 0.1
    df_d["lon_bin"]     = (df_d["lon"] / 0.1).astype(int) * 0.1

    density = (
        df_d.groupby(["lat_bin", "lon_bin", "hour_bucket"])
        .agg(
            vessel_count  = ("mmsi", "nunique"),
            avg_sog       = ("sog",  "mean"),
            stopped_count = ("sog",  lambda x: (x < 0.5).sum()),
        )
        .reset_index()
    )
    density["hour"]        = density["hour_bucket"].dt.hour
    density["day_of_week"] = density["hour_bucket"].dt.dayofweek
    density["is_weekend"]  = (density["day_of_week"] >= 5).astype(int)
    density["true_label"]  = density["vessel_count"].apply(
        lambda n: "HIGH" if n >= CONGESTION_HIGH
                  else ("MEDIUM" if n >= CONGESTION_MEDIUM else "LOW")
    )

    feat_cols = [c for c in features if c in density.columns]
    X         = density[feat_cols].fillna(0).values
    y_true    = encoder.transform(density["true_label"])

    try:
        y_pred = model.predict(X)
    except Exception as exc:
        logger.warning("Congestion predict failed: %s", exc)
        return {"status": "predict_error", "detail": str(exc)}

    try:
        from sklearn.metrics import (
            accuracy_score, classification_report,
            precision_score, recall_score, f1_score
        )
        report = classification_report(y_true, y_pred,
                                       target_names=encoder.classes_,
                                       zero_division=0, output_dict=True)
        result = {
            "status":          "ok",
            "source":          source,
            "test_grid_cells": int(len(density)),
            "accuracy":        round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1":        round(float(report["macro avg"]["f1-score"]), 4),
            "macro_precision": round(float(report["macro avg"]["precision"]), 4),
            "macro_recall":    round(float(report["macro avg"]["recall"]), 4),
            "per_class":       {
                cls: {
                    "precision": round(report[cls]["precision"], 4),
                    "recall":    round(report[cls]["recall"],    4),
                    "f1":        round(report[cls]["f1-score"],  4),
                }
                for cls in encoder.classes_ if cls in report
            },
        }
        logger.info("Congestion accuracy=%.3f macro-F1=%.3f",
                    result["accuracy"], result["macro_f1"])
        return result
    except Exception as exc:
        logger.warning("Congestion metrics failed: %s", exc)
        return {"status": "metrics_error", "detail": str(exc)}


def evaluate_predictor(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"status": "no_test_data"}

    predictors = load_predictor_models()
    results    = {}

    feat_path = Path(MODELS_PATH) / "predictor_features.pkl"
    try:
        import joblib
        features = joblib.load(feat_path)
    except Exception:
        logger.warning("predictor_features.pkl not found")
        return {"status": "features_not_found"}

    for minutes, models in predictors.items():
        if models["lat"] is None:
            results[f"{minutes}min"] = {"status": "not_available"}
            continue

        steps  = max(1, int(minutes * 60 / 28))
        df_t   = df.copy()
        grp    = df_t.groupby("mmsi")
        df_t["next_lat"]  = grp["lat"].shift(-steps)
        df_t["next_lon"]  = grp["lon"].shift(-steps)
        df_t["delta_lat"] = df_t["next_lat"] - df_t["lat"]
        df_t["delta_lon"] = df_t["next_lon"] - df_t["lon"]

        df_t = df_t.dropna(subset=["delta_lat", "delta_lon"])
        df_t = df_t[(df_t["delta_lat"].abs() < 1.0) & (df_t["delta_lon"].abs() < 1.0)]

        if len(df_t) < 100:
            results[f"{minutes}min"] = {"status": "insufficient_samples",
                                         "samples": len(df_t)}
            continue

        feat_cols = [c for c in features if c in df_t.columns]
        X         = df_t[feat_cols].fillna(0).values
        y_lat     = df_t["delta_lat"].values
        y_lon     = df_t["delta_lon"].values

        try:
            pred_lat = models["lat"].predict(X)
            pred_lon = models["lon"].predict(X)
        except Exception as exc:
            results[f"{minutes}min"] = {"status": "predict_error",
                                         "detail": str(exc)}
            continue

        mae_lat_nm  = float(np.mean(np.abs(pred_lat - y_lat)) * NM_PER_DEGREE)
        mae_lon_nm  = float(np.mean(np.abs(pred_lon - y_lon)) * NM_PER_DEGREE)
        rmse_lat_nm = float(np.sqrt(np.mean((pred_lat - y_lat)**2)) * NM_PER_DEGREE)
        rmse_lon_nm = float(np.sqrt(np.mean((pred_lon - y_lon)**2)) * NM_PER_DEGREE)
        total_mae   = float(np.sqrt(mae_lat_nm**2 + mae_lon_nm**2))

        results[f"{minutes}min"] = {
            "status":       "ok",
            "source":       models["source"],
            "test_samples": int(len(df_t)),
            "mae_lat_nm":   round(mae_lat_nm,  4),
            "mae_lon_nm":   round(mae_lon_nm,  4),
            "rmse_lat_nm":  round(rmse_lat_nm, 4),
            "rmse_lon_nm":  round(rmse_lon_nm, 4),
            "total_mae_nm": round(total_mae,   4),
        }
        logger.info("%d-min predictor  lat_MAE=%.3f nm  total=%.3f nm",
                    minutes, mae_lat_nm, total_mae)

    return results


# ---------------------------------------------------------------------------
# MLflow logging helper
# ---------------------------------------------------------------------------

def _log_report_to_mlflow(report: dict) -> None:
    if not is_mlflow_available():
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(get_tracking_uri())
        mlflow.set_experiment("maritime-model-evaluation")
        with mlflow.start_run(
                run_name=f"evaluation-{datetime.now():%Y%m%d-%H%M}"):
            mlflow.set_tags({**BASE_TAGS, "evaluation_type": "test_split"})

            # Anomaly metrics
            a = report.get("anomaly", {})
            if a.get("status") == "ok":
                mlflow.log_metrics({
                    "anomaly_rate":     a.get("anomaly_rate", 0),
                    "anomaly_precision":a.get("precision",    0),
                    "anomaly_recall":   a.get("recall",       0),
                    "anomaly_f1":       a.get("f1_score",     0),
                    "anomaly_fp_rate":  a.get("estimated_false_positive_rate", 0),
                })

            # Congestion metrics
            c = report.get("congestion", {})
            if c.get("status") == "ok":
                mlflow.log_metrics({
                    "cong_accuracy":   c.get("accuracy",        0),
                    "cong_macro_f1":   c.get("macro_f1",        0),
                    "cong_macro_prec": c.get("macro_precision",  0),
                    "cong_macro_rec":  c.get("macro_recall",     0),
                })

            # Predictor metrics
            for key, val in report.get("predictor", {}).items():
                if isinstance(val, dict) and val.get("status") == "ok":
                    prefix = f"pred_{key}"
                    mlflow.log_metrics({
                        f"{prefix}_mae_nm":   val.get("total_mae_nm", 0),
                        f"{prefix}_lat_mae":  val.get("mae_lat_nm",   0),
                        f"{prefix}_lon_mae":  val.get("mae_lon_nm",   0),
                    })

            mlflow.log_artifact(str(ARTIFACTS_DIR / "evaluation_report.json"))
            logger.info("[MLflow] Evaluation metrics logged")
    except Exception as exc:
        logger.warning("[MLflow] Could not log evaluation: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 55)
    logger.info("Maritime AIS — Model Evaluation")
    logger.info("=" * 55)

    df = load_test_data()

    report = {
        "generated_at":  datetime.utcnow().isoformat(),
        "test_rows":     len(df),
        "anomaly":       evaluate_anomaly(df),
        "congestion":    evaluate_congestion(df),
        "predictor":     evaluate_predictor(df),
    }

    # Save report
    report_path = ARTIFACTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report saved → %s", report_path)

    # Log to MLflow if available
    _log_report_to_mlflow(report)

    # Print summary
    print("\n" + "=" * 55)
    print("  Evaluation Summary")
    print("=" * 55)

    a = report["anomaly"]
    if a.get("status") == "ok":
        print(f"  Anomaly    precision={a.get('precision',0):.3f}  "
              f"recall={a.get('recall',0):.3f}  "
              f"f1={a.get('f1_score',0):.3f}  "
              f"rate={a.get('anomaly_rate',0):.4f}")
    else:
        print(f"  Anomaly    {a.get('status','?')}")

    c = report["congestion"]
    if c.get("status") == "ok":
        print(f"  Congestion accuracy={c.get('accuracy',0):.3f}  "
              f"macro-f1={c.get('macro_f1',0):.3f}")
    else:
        print(f"  Congestion {c.get('status','?')}")

    for key, val in report["predictor"].items():
        if isinstance(val, dict) and val.get("status") == "ok":
            print(f"  Predictor {key:<5}  total_mae={val.get('total_mae_nm',0):.3f} nm")
        else:
            print(f"  Predictor {key:<5}  {val.get('status','?')}")

    print(f"\n  Report → {report_path}")


if __name__ == "__main__":
    main()
