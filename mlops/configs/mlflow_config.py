"""
mlflow_config.py — Maritime Navigation AI System
Central MLflow configuration for all MLOps experiment scripts.

Supports two modes:
  LOCAL   — file-based tracking under mlops/mlruns/ (zero infrastructure)
  REMOTE  — MLFLOW_TRACKING_URI points to a running tracking server
             (e.g. docker/docker-compose.mlflow.yml)

Safe when MLflow is not installed: every function either returns a sentinel
value or raises ImportError only when the caller explicitly requests a
client, not at import time.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolved paths
# ---------------------------------------------------------------------------

# Project root is two levels above this file (mlops/configs/mlflow_config.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default local tracking store — mlops/mlruns relative to project root.
# Using a sub-directory of mlops/ avoids polluting the src/ tree and keeps
# all MLflow state out of the Docker volume mounts.
_LOCAL_MLRUNS = _PROJECT_ROOT / "mlops" / "mlruns"

# Artifact directory for plots / evaluation reports written by scripts
ARTIFACTS_DIR = _PROJECT_ROOT / "mlops" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Tracking URI resolution
# ---------------------------------------------------------------------------

def get_tracking_uri() -> str:
    """
    Return the MLflow tracking URI.

    Priority:
      1. MLFLOW_TRACKING_URI environment variable (remote server or
         custom local path)
      2. Default local file store at mlops/mlruns/
    """
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if uri:
        return uri
    _LOCAL_MLRUNS.mkdir(parents=True, exist_ok=True)
    return str(_LOCAL_MLRUNS)


def is_remote() -> bool:
    """Return True when the tracking URI points to an HTTP server."""
    uri = get_tracking_uri()
    return uri.startswith("http://") or uri.startswith("https://")


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Single parent experiment for all maritime models.
# Override with MLFLOW_EXPERIMENT_NAME if you want per-model separation.
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "maritime-ais-models")

# Sub-experiment names used by each training script
ANOMALY_EXPERIMENT     = "maritime-anomaly-detector"
CONGESTION_EXPERIMENT  = "maritime-congestion-classifier"
PREDICTOR_EXPERIMENT   = "maritime-position-predictor"

# Model registry names — these are the names used for staging/production
# promotion and for model_loader.py to fetch the correct version.
ANOMALY_MODEL_NAME     = "maritime-anomaly-detector"
CONGESTION_MODEL_NAME  = "maritime-congestion-classifier"
PREDICTOR_MODEL_NAME   = "maritime-position-predictor"   # appended with "-Xmin"

# Common tags added to every run for filtering in the UI
BASE_TAGS = {
    "project":  "maritime-navigation-ai",
    "domain":   "us-coastal-waters",
    "pipeline": "ais-medallion",
}


# ---------------------------------------------------------------------------
# MLflow availability check
# ---------------------------------------------------------------------------

def is_mlflow_available() -> bool:
    """
    Return True if mlflow is importable AND the tracking URI is reachable.
    Never raises — used as a guard in every experiment script.
    """
    try:
        import mlflow
        uri = get_tracking_uri()
        if is_remote():
            # Quick connectivity check against the remote server
            import urllib.request
            try:
                urllib.request.urlopen(uri + "/api/2.0/mlflow/experiments/list",
                                       timeout=2)
            except Exception:
                logger.warning(
                    "[MLflow] Remote server at %s is not reachable. "
                    "Falling back to local file store.", uri
                )
                return False
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Convenience: get a configured MlflowClient
# ---------------------------------------------------------------------------

def get_client():
    """
    Return a configured MlflowClient.

    Raises ImportError  if mlflow is not installed.
    Raises RuntimeError if the tracking server is unreachable.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise ImportError(
            "mlflow is not installed. Run: pip install mlflow"
        ) from exc

    mlflow.set_tracking_uri(get_tracking_uri())
    return MlflowClient()


# ---------------------------------------------------------------------------
# Convenience: set up an experiment and return its ID
# ---------------------------------------------------------------------------

def setup_experiment(name: str) -> Optional[str]:
    """
    Create or retrieve an MLflow experiment by name.
    Returns the experiment ID string, or None if MLflow is unavailable.
    """
    if not is_mlflow_available():
        logger.warning(
            "[MLflow] Not available — experiment '%s' not created.", name
        )
        return None
    try:
        import mlflow
        mlflow.set_tracking_uri(get_tracking_uri())
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            exp_id = mlflow.create_experiment(name)
        else:
            exp_id = exp.experiment_id
        logger.info("[MLflow] Experiment '%s' (id=%s)", name, exp_id)
        return exp_id
    except Exception as exc:
        logger.error("[MLflow] setup_experiment failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Convenience: register a model version
# ---------------------------------------------------------------------------

def register_model(run_id: str, artifact_path: str,
                   model_name: str, stage: str = "Staging") -> Optional[str]:
    """
    Register a logged model artifact to the MLflow Model Registry.
    Promotes to `stage` (default: Staging).

    Returns the registered model version string, or None on failure.
    """
    if not is_mlflow_available():
        return None
    try:
        import mlflow
        client = get_client()
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mv = mlflow.register_model(model_uri, model_name)
        client.transition_model_version_stage(
            name=model_name, version=mv.version, stage=stage
        )
        logger.info("[MLflow] Registered %s v%s → %s", model_name, mv.version, stage)
        return mv.version
    except Exception as exc:
        logger.error("[MLflow] register_model failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Print status when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    print(f"Tracking URI    : {get_tracking_uri()}")
    print(f"Remote server   : {is_remote()}")
    print(f"MLflow available: {is_mlflow_available()}")
    print(f"Artifacts dir   : {ARTIFACTS_DIR}")
    print(f"Experiment names:")
    print(f"  anomaly    → {ANOMALY_EXPERIMENT}")
    print(f"  congestion → {CONGESTION_EXPERIMENT}")
    print(f"  predictor  → {PREDICTOR_EXPERIMENT}")
