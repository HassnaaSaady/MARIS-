"""
model_loader.py — Maritime Navigation AI System
Unified model loader: MLflow Model Registry → .pkl fallback.

Load order for every model:
  1. MLflow Model Registry at `stage` (default: "Production")
  2. Local .pkl file at MODELS_PATH (same paths as scorer.py uses)
  3. None — caller must handle absence gracefully

This module is intentionally NOT imported by scorer.py or live_scorer.py.
Those files are read-only and load .pkl files directly.  This module is used
by:
  - mlops/model_registry/evaluate_models.py
  - src/dashboard/ml_monitoring.py
  - Any new scoring path that prefers registry-managed models

The .pkl fallback ensures the existing scorer.py continues to work unchanged
regardless of whether MLflow is installed or the registry has any entries.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

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
    is_mlflow_available, get_tracking_uri,
    ANOMALY_MODEL_NAME, CONGESTION_MODEL_NAME, PREDICTOR_MODEL_NAME,
)

try:
    from config import MODELS_PATH
except ImportError:
    MODELS_PATH = os.getenv("MODELS_PATH", "/app/models")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_pkl(path: Path):
    """Load a joblib pickle. Returns None on any error."""
    try:
        import joblib
        obj = joblib.load(path)
        logger.debug("Loaded pkl: %s", path)
        return obj
    except Exception as exc:
        logger.debug("pkl not available at %s: %s", path, exc)
        return None


def _load_from_registry(model_name: str, stage: str = "Production"):
    """
    Try loading a model from MLflow Model Registry.
    Returns (model, source_label) or (None, None).
    """
    if not is_mlflow_available():
        return None, None
    try:
        import mlflow
        mlflow.set_tracking_uri(get_tracking_uri())
        model_uri = f"models:/{model_name}/{stage}"
        model     = mlflow.pyfunc.load_model(model_uri)
        logger.info("[MLflow] Loaded '%s' from registry (%s)", model_name, stage)
        return model, f"mlflow:{stage}"
    except Exception as exc:
        logger.debug("[MLflow] Registry miss for '%s'/%s: %s",
                     model_name, stage, exc)
        return None, None


def _load_sklearn_from_registry(model_name: str, stage: str = "Production"):
    """
    Load a sklearn/XGBoost model from MLflow as a native object (not pyfunc).
    Falls back to pyfunc if the flavour is not sklearn/xgboost.
    """
    if not is_mlflow_available():
        return None, None
    try:
        import mlflow.sklearn
        import mlflow
        mlflow.set_tracking_uri(get_tracking_uri())
        model_uri = f"models:/{model_name}/{stage}"
        try:
            model = mlflow.sklearn.load_model(model_uri)
        except Exception:
            model = mlflow.pyfunc.load_model(model_uri)
        logger.info("[MLflow] Loaded sklearn '%s' (%s)", model_name, stage)
        return model, f"mlflow:{stage}"
    except Exception as exc:
        logger.debug("[MLflow] sklearn registry miss for '%s': %s",
                     model_name, exc)
        return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_anomaly_model(stage: str = "Production") -> Tuple[object, object, list, str]:
    """
    Load anomaly detection models.

    Returns:
        (isolation_forest, scaler, feature_names, source)
        source is one of: "mlflow:Production", "mlflow:Staging", "pkl", "none"

    Falls back through:
      MLflow registry → local .pkl → (None, None, [], "none")
    """
    mp = Path(MODELS_PATH)

    # Try MLflow first
    model, source = _load_sklearn_from_registry(ANOMALY_MODEL_NAME, stage)
    if model is None and stage != "Staging":
        model, source = _load_sklearn_from_registry(ANOMALY_MODEL_NAME, "Staging")

    if model is None:
        # .pkl fallback (same paths scorer.py uses)
        model    = _load_pkl(mp / "isolation_forest.pkl")
        scaler   = _load_pkl(mp / "scaler_anomaly.pkl")
        features = _load_pkl(mp / "anomaly_features.pkl") or []
        source   = "pkl" if model is not None else "none"
        if model:
            logger.info("Anomaly model loaded from pkl (%s)", source)
        else:
            logger.warning("Anomaly model not found in registry or pkl")
        return model, scaler, features, source

    # MLflow model loaded — also load scaler and features from pkl
    scaler   = _load_pkl(mp / "scaler_anomaly.pkl")
    features = _load_pkl(mp / "anomaly_features.pkl") or []
    return model, scaler, features, source


def load_congestion_model(stage: str = "Production") -> Tuple[object, object, list, str]:
    """
    Load congestion classifier.

    Returns:
        (random_forest, label_encoder, feature_names, source)
    """
    mp = Path(MODELS_PATH)

    model, source = _load_sklearn_from_registry(CONGESTION_MODEL_NAME, stage)
    if model is None and stage != "Staging":
        model, source = _load_sklearn_from_registry(CONGESTION_MODEL_NAME, "Staging")

    if model is None:
        model    = _load_pkl(mp / "congestion_rf.pkl")
        encoder  = _load_pkl(mp / "congestion_encoder.pkl")
        features = _load_pkl(mp / "congestion_features.pkl") or []
        source   = "pkl" if model is not None else "none"
        if model:
            logger.info("Congestion model loaded from pkl")
        else:
            logger.warning("Congestion model not found")
        return model, encoder, features, source

    encoder  = _load_pkl(mp / "congestion_encoder.pkl")
    features = _load_pkl(mp / "congestion_features.pkl") or []
    return model, encoder, features, source


def load_predictor_models(stage: str = "Production",
                          horizons: list = None) -> dict:
    """
    Load XGBoost position predictor models for each horizon.

    Returns:
        {
          5:  {"lat": model_lat, "lon": model_lon, "source": "pkl"},
          10: {"lat": model_lat, "lon": model_lon, "source": "mlflow:Production"},
          15: {"lat": model_lat, "lon": model_lon, "source": "none"},
        }
    """
    if horizons is None:
        horizons = [5, 10, 15]

    mp      = Path(MODELS_PATH)
    results = {}

    for minutes in horizons:
        lat_name = f"{PREDICTOR_MODEL_NAME}-{minutes}min-lat"
        lon_name = f"{PREDICTOR_MODEL_NAME}-{minutes}min-lon"

        m_lat, src = _load_sklearn_from_registry(lat_name, stage)
        m_lon, _   = _load_sklearn_from_registry(lon_name, stage)

        if m_lat is None:
            m_lat = _load_pkl(mp / f"xgb_lat_{minutes}min.pkl")
            m_lon = _load_pkl(mp / f"xgb_lon_{minutes}min.pkl")
            src   = "pkl" if m_lat is not None else "none"

        if m_lat is not None:
            logger.info("%d-min predictor loaded (%s)", minutes, src)

        results[minutes] = {
            "lat":    m_lat,
            "lon":    m_lon,
            "source": src or "none",
        }

    return results


def load_all(stage: str = "Production") -> dict:
    """
    Convenience: load all models and return a summary dict.

    Returns:
        {
          "anomaly":    {"model": ..., "scaler": ..., "features": [...], "source": ...},
          "congestion": {"model": ..., "encoder": ..., "features": [...], "source": ...},
          "predictor":  {5: {...}, 10: {...}, 15: {...}},
        }
    """
    anomaly_model, anomaly_scaler, anomaly_features, anomaly_src = load_anomaly_model(stage)
    cong_model,    cong_encoder,   cong_features,    cong_src    = load_congestion_model(stage)
    predictor                                                     = load_predictor_models(stage)

    return {
        "anomaly": {
            "model":    anomaly_model,
            "scaler":   anomaly_scaler,
            "features": anomaly_features,
            "source":   anomaly_src,
        },
        "congestion": {
            "model":    cong_model,
            "encoder":  cong_encoder,
            "features": cong_features,
            "source":   cong_src,
        },
        "predictor": predictor,
    }


def summarise() -> dict:
    """Return a plain-dict summary of model availability (no models loaded)."""
    mp = Path(MODELS_PATH)
    registry_available = is_mlflow_available()

    def _pkl_exists(name):
        return (mp / name).exists()

    summary = {
        "registry_available": registry_available,
        "pkl_path":           str(mp),
        "models": {
            "anomaly_detector": {
                "pkl_isolation_forest": _pkl_exists("isolation_forest.pkl"),
                "pkl_scaler":           _pkl_exists("scaler_anomaly.pkl"),
                "pkl_features":         _pkl_exists("anomaly_features.pkl"),
            },
            "congestion_classifier": {
                "pkl_rf":               _pkl_exists("congestion_rf.pkl"),
                "pkl_encoder":          _pkl_exists("congestion_encoder.pkl"),
                "pkl_features":         _pkl_exists("congestion_features.pkl"),
            },
            "position_predictor": {
                f"pkl_xgb_{m}min": {
                    "lat": _pkl_exists(f"xgb_lat_{m}min.pkl"),
                    "lon": _pkl_exists(f"xgb_lon_{m}min.pkl"),
                }
                for m in [5, 10, 15]
            },
        },
    }
    return summary


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(summarise(), indent=2))
