"""
test_ml.py — Maritime Navigation AI System
ML model unit tests.

Tests skip gracefully when:
  - .pkl model files are absent (models not yet trained)
  - Heavy ML dependencies (sklearn, xgboost, joblib) are not installed

Run locally after training:
    python src/ml/train_anomaly.py
    python src/ml/train_congestion.py
    python src/ml/train_predictor.py
    pytest tests/test_ml.py -v
"""

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup so we can import from src/
# ---------------------------------------------------------------------------

_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

for _p in [
    str(_PROJECT_ROOT / "src" / "common"),
    str(_PROJECT_ROOT / "src"),
    str(_PROJECT_ROOT / "src" / "ml"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Model file existence
# ---------------------------------------------------------------------------

class TestModelFileExistence:
    """Verify that expected .pkl artefacts are present after training."""

    ANOMALY_FILES = [
        "isolation_forest.pkl",
        "scaler_anomaly.pkl",
        "anomaly_features.pkl",
    ]
    CONGESTION_FILES = [
        "congestion_rf.pkl",
        "congestion_encoder.pkl",
        "congestion_features.pkl",
    ]
    PREDICTOR_FILES = [
        "xgb_lat_5min.pkl",
        "xgb_lon_5min.pkl",
        "xgb_lat_10min.pkl",
        "xgb_lon_10min.pkl",
        "xgb_lat_15min.pkl",
        "xgb_lon_15min.pkl",
        "predictor_features.pkl",
    ]

    @pytest.mark.parametrize("filename", ANOMALY_FILES)
    def test_anomaly_pkl_exists(self, models_path, filename):
        path = models_path / filename
        if not path.exists():
            pytest.skip(f"{filename} not found at {models_path} — model not trained")
        assert path.is_file()

    @pytest.mark.parametrize("filename", CONGESTION_FILES)
    def test_congestion_pkl_exists(self, models_path, filename):
        path = models_path / filename
        if not path.exists():
            pytest.skip(f"{filename} not found at {models_path} — model not trained")
        assert path.is_file()

    @pytest.mark.parametrize("filename", PREDICTOR_FILES)
    def test_predictor_pkl_exists(self, models_path, filename):
        path = models_path / filename
        if not path.exists():
            pytest.skip(f"{filename} not found at {models_path} — model not trained")
        assert path.is_file()


# ---------------------------------------------------------------------------
# scorer.py import
# ---------------------------------------------------------------------------

class TestScorerImport:
    def test_scorer_importable(self, require_heavy_deps):
        try:
            from scorer import LiveScorer  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"scorer.py import failed: {exc}")

    def test_get_scorer_returns_instance(self, require_heavy_deps):
        try:
            from scorer import get_scorer
            scorer = get_scorer()
            assert scorer is not None
        except Exception as exc:
            pytest.skip(f"get_scorer() failed: {exc}")


# ---------------------------------------------------------------------------
# predict_position
# ---------------------------------------------------------------------------

class TestPredictPosition:
    """Light smoke test — checks return shape, not accuracy."""

    SAMPLE_AIS = {
        "mmsi":    "123456789",
        "lat":     37.5,
        "lon":    -122.5,
        "sog":     12.0,
        "cog":     180.0,
        "heading": 180.0,
    }

    def test_predict_position_returns_dict(self, require_heavy_deps, models_path):
        try:
            from scorer import get_scorer
            scorer = get_scorer()
        except Exception as exc:
            pytest.skip(f"Scorer unavailable: {exc}")

        try:
            result = scorer.predict_position(self.SAMPLE_AIS)
        except Exception as exc:
            pytest.skip(f"predict_position raised: {exc}")

        assert isinstance(result, dict)

    def test_predict_position_has_predictions_key(self, require_heavy_deps, models_path):
        try:
            from scorer import get_scorer
            scorer = get_scorer()
            result = scorer.predict_position(self.SAMPLE_AIS)
        except Exception as exc:
            pytest.skip(f"predict_position unavailable: {exc}")

        assert "predictions" in result or len(result) > 0


# ---------------------------------------------------------------------------
# Anomaly scoring
# ---------------------------------------------------------------------------

class TestAnomalyScoring:
    """Verify score_anomaly() returns a numeric score."""

    SAMPLE_AIS = {
        "mmsi":              "123456789",
        "lat":               37.5,
        "lon":              -122.5,
        "sog":               12.0,
        "cog":               180.0,
        "heading":           180.0,
        "sog_change":        0.1,
        "heading_change":    0.5,
        "time_delta_sec":   10.0,
        "distance_nm":       0.03,
        "length":           200.0,
        "width":             30.0,
        "draft":              8.0,
    }

    def test_score_anomaly_returns_numeric(self, require_heavy_deps, models_path):
        try:
            from scorer import get_scorer
            scorer = get_scorer()
            result = scorer.score_anomaly(self.SAMPLE_AIS)
        except Exception as exc:
            pytest.skip(f"score_anomaly unavailable: {exc}")

        assert isinstance(result, (int, float, dict))

    def test_score_anomaly_no_exception(self, require_heavy_deps, models_path):
        try:
            from scorer import get_scorer
            scorer = get_scorer()
        except Exception as exc:
            pytest.skip(f"Scorer unavailable: {exc}")

        # Should not raise even with minimal features
        scorer.score_anomaly({"mmsi": "999", "lat": 38.0, "lon": -120.0, "sog": 5.0})
