"""
feature_importance.py — Maritime Navigation AI System
Visualise and export feature importance for all three trained models.

Loads models from MODELS_PATH (.pkl) — works with or without MLflow.
Saves plots to mlops/artifacts/ as PNG and an optional interactive HTML.

Supported model types:
  IsolationForest  — mean impurity decrease averaged across estimator trees
  RandomForest     — native feature_importances_
  XGBoost          — native feature_importances_ (weight / gain / cover)

Run:
    python mlops/experiments/feature_importance.py
    python mlops/experiments/feature_importance.py --no-plots  # CSV only
"""

import argparse
import logging
import os
import sys
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
    is_mlflow_available, get_tracking_uri, ARTIFACTS_DIR,
    ANOMALY_EXPERIMENT, CONGESTION_EXPERIMENT, PREDICTOR_EXPERIMENT,
)

try:
    from config import MODELS_PATH
except ImportError:
    MODELS_PATH = os.getenv("MODELS_PATH", "/app/models")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PREDICT_MINUTES = [5, 10, 15]


# ---------------------------------------------------------------------------
# Feature importance extractors
# ---------------------------------------------------------------------------

def _isolation_forest_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Approximate feature importance for IsolationForest by averaging the
    impurity-based importance across all constituent ExtraTree estimators.
    Sklearn's IsolationForest does not expose feature_importances_ directly.
    """
    importances = np.zeros(len(feature_names))
    for tree in model.estimators_:
        # Each estimator is an ExtraTreeRegressor; average its importances
        if hasattr(tree, "feature_importances_"):
            importances += tree.feature_importances_
    importances /= len(model.estimators_)

    return pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def _rf_importance(model, feature_names: list) -> pd.DataFrame:
    return pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def _xgb_importance(model, feature_names: list,
                    importance_type: str = "gain") -> pd.DataFrame:
    """
    XGBoost supports three importance types:
      weight — number of times a feature is used to split
      gain   — average gain of splits using the feature (best for magnitude)
      cover  — average coverage of splits using the feature
    """
    try:
        scores = model.get_booster().get_score(importance_type=importance_type)
    except Exception:
        # Fallback for older XGBoost versions
        scores = dict(zip(feature_names, model.feature_importances_))

    # Normalise to [0, 1]
    total  = sum(scores.values()) or 1.0
    rows   = [{"feature": f, "importance": v / total}
              for f, v in scores.items()]
    df     = pd.DataFrame(rows).sort_values("importance", ascending=False)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _plot_importance(df: pd.DataFrame, title: str,
                     output_path: Path, top_n: int = 15) -> None:
    """Save a horizontal bar chart to output_path as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot for %s", title)
        return

    df_top = df.head(top_n).iloc[::-1]  # reverse for bottom-to-top bars

    fig, ax = plt.subplots(figsize=(9, max(4, len(df_top) * 0.45)))
    bars = ax.barh(df_top["feature"], df_top["importance"],
                   color="#3B82F6", edgecolor="none")

    # Value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, df_top["importance"].max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", output_path)


# ---------------------------------------------------------------------------
# Per-model extraction
# ---------------------------------------------------------------------------

def analyse_anomaly(save_plots: bool = True) -> pd.DataFrame:
    """IsolationForest feature importance."""
    mp = Path(MODELS_PATH)
    try:
        model    = joblib.load(mp / "isolation_forest.pkl")
        features = joblib.load(mp / "anomaly_features.pkl")
    except FileNotFoundError:
        logger.warning("Anomaly model not found at %s — skipping.", mp)
        return pd.DataFrame()

    df = _isolation_forest_importance(model, features)
    csv_path = ARTIFACTS_DIR / "anomaly_feature_importance.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Anomaly importance saved → %s", csv_path)

    if save_plots:
        _plot_importance(df, "IsolationForest — Feature Importance\n(Anomaly Detection)",
                         ARTIFACTS_DIR / "anomaly_feature_importance.png")
    return df


def analyse_congestion(save_plots: bool = True) -> pd.DataFrame:
    """RandomForest feature importance."""
    mp = Path(MODELS_PATH)
    try:
        model    = joblib.load(mp / "congestion_rf.pkl")
        features = joblib.load(mp / "congestion_features.pkl")
    except FileNotFoundError:
        logger.warning("Congestion model not found at %s — skipping.", mp)
        return pd.DataFrame()

    df = _rf_importance(model, features)
    csv_path = ARTIFACTS_DIR / "congestion_feature_importance.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Congestion importance saved → %s", csv_path)

    if save_plots:
        _plot_importance(df, "RandomForest — Feature Importance\n(Congestion Classification)",
                         ARTIFACTS_DIR / "congestion_feature_importance.png")
    return df


def analyse_predictor(save_plots: bool = True) -> dict:
    """XGBoost feature importance for each horizon."""
    mp     = Path(MODELS_PATH)
    result = {}
    try:
        features = joblib.load(mp / "predictor_features.pkl")
    except FileNotFoundError:
        logger.warning("Predictor features not found — skipping.")
        return result

    for minutes in PREDICT_MINUTES:
        try:
            model_lat = joblib.load(mp / f"xgb_lat_{minutes}min.pkl")
        except FileNotFoundError:
            logger.warning("xgb_lat_%dmin.pkl not found — skipping.", minutes)
            continue

        df = _xgb_importance(model_lat, features, importance_type="gain")
        csv_path = ARTIFACTS_DIR / f"predictor_{minutes}min_feature_importance.csv"
        df.to_csv(csv_path, index=False)
        result[minutes] = df
        logger.info("%d-min predictor importance → %s", minutes, csv_path)

        if save_plots:
            _plot_importance(
                df,
                f"XGBoost (lat) — Feature Importance\n({minutes}-min Position Prediction)",
                ARTIFACTS_DIR / f"predictor_{minutes}min_feature_importance.png",
            )

    return result


# ---------------------------------------------------------------------------
# Optional: log artifacts to the latest MLflow run per experiment
# ---------------------------------------------------------------------------

def _log_to_mlflow(csv_path: Path, experiment_name: str) -> None:
    """Attach a CSV artifact to the most recent run of an experiment."""
    if not is_mlflow_available():
        return
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(get_tracking_uri())
        client = MlflowClient()
        exp    = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return
        runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"],
                                  max_results=1)
        if not runs:
            return
        client.log_artifact(runs[0].info.run_id, str(csv_path),
                             "feature_importance")
        logger.info("[MLflow] Logged %s to run %s",
                    csv_path.name, runs[0].info.run_id)
    except Exception as exc:
        logger.warning("[MLflow] Could not log artifact: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate feature importance visualisations for all models")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip PNG generation (CSV only)")
    args = parser.parse_args()

    save_plots = not args.no_plots

    logger.info("Output directory: %s", ARTIFACTS_DIR)
    logger.info("Models path:      %s", MODELS_PATH)

    # ── Anomaly ──────────────────────────────────────────────────────────────
    logger.info("── IsolationForest (Anomaly) ──")
    df_anomaly = analyse_anomaly(save_plots)
    if not df_anomaly.empty:
        print("\nTop 5 anomaly features:")
        print(df_anomaly.head(5).to_string(index=False))
        _log_to_mlflow(ARTIFACTS_DIR / "anomaly_feature_importance.csv",
                       ANOMALY_EXPERIMENT)

    # ── Congestion ───────────────────────────────────────────────────────────
    logger.info("── RandomForest (Congestion) ──")
    df_congestion = analyse_congestion(save_plots)
    if not df_congestion.empty:
        print("\nTop 5 congestion features:")
        print(df_congestion.head(5).to_string(index=False))
        _log_to_mlflow(ARTIFACTS_DIR / "congestion_feature_importance.csv",
                       CONGESTION_EXPERIMENT)

    # ── Predictor ────────────────────────────────────────────────────────────
    logger.info("── XGBoost (Position Predictor) ──")
    predictor_results = analyse_predictor(save_plots)
    for minutes, df in predictor_results.items():
        print(f"\nTop 5 features for {minutes}-min predictor:")
        print(df.head(5).to_string(index=False))
        _log_to_mlflow(ARTIFACTS_DIR / f"predictor_{minutes}min_feature_importance.csv",
                       PREDICTOR_EXPERIMENT)

    # ── Summary ──────────────────────────────────────────────────────────────
    artifacts = list(ARTIFACTS_DIR.glob("*.csv")) + list(ARTIFACTS_DIR.glob("*.png"))
    logger.info("\nAll artifacts (%d files):", len(artifacts))
    for a in sorted(artifacts):
        logger.info("  %s", a.name)


if __name__ == "__main__":
    main()
