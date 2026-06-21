"""
conftest.py — Maritime Navigation AI System
Shared pytest fixtures and helpers for CI/CD test suite.

All fixtures are designed to skip gracefully when external services
(API server, model files) are not available in the CI environment.
"""

import os
import socket
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_MODELS_PATH  = Path(os.getenv("MODELS_PATH", str(_PROJECT_ROOT / "models")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Base URL for the FastAPI service."""
    return os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def models_path() -> Path:
    """Path to the directory containing trained .pkl model files."""
    return _MODELS_PATH


@pytest.fixture(scope="session")
def api_available(api_base_url) -> bool:
    """True when the API server is reachable."""
    host = api_base_url.split("//")[-1].split(":")[0]
    try:
        port = int(api_base_url.split(":")[-1].rstrip("/"))
    except ValueError:
        port = 8000
    return _port_open(host, port)


@pytest.fixture(scope="session")
def require_api(api_available):
    """Skip the test if the API is not reachable."""
    if not api_available:
        pytest.skip("API server not reachable — skipping (service unavailable)")


@pytest.fixture(scope="session")
def require_models(models_path):
    """Skip the test if no model files are present under MODELS_PATH."""
    pkl_files = list(models_path.glob("*.pkl"))
    if not pkl_files:
        pytest.skip(
            f"No .pkl model files found at {models_path} — skipping (models not trained)"
        )


@pytest.fixture(scope="session")
def require_heavy_deps():
    """Skip the test if heavy ML dependencies (sklearn, xgboost) are not installed."""
    try:
        import sklearn  # noqa: F401
        import joblib   # noqa: F401
    except ImportError:
        pytest.skip("sklearn/joblib not installed — skipping ML tests")
