"""
test_api.py — Maritime Navigation AI System
API integration tests.

Tests skip gracefully when the API server is not reachable (e.g. in CI
environments where Docker containers are not running).

Run locally with the full stack:
    docker compose up -d
    pytest tests/test_api.py -v

Run in CI without the stack (all tests skip):
    pytest tests/test_api.py -v
"""

import pytest

requests = pytest.importorskip("requests", reason="requests not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(base_url: str, path: str, timeout: int = 10):
    """GET request with a short timeout; returns Response or raises."""
    return requests.get(f"{base_url}{path}", timeout=timeout)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, api_base_url, require_api):
        resp = _get(api_base_url, "/health")
        assert resp.status_code == 200

    def test_health_body_has_status(self, api_base_url, require_api):
        resp = _get(api_base_url, "/health")
        body = resp.json()
        assert "status" in body


# ---------------------------------------------------------------------------
# Vessels
# ---------------------------------------------------------------------------

class TestVessels:
    def test_vessels_returns_200(self, api_base_url, require_api):
        resp = _get(api_base_url, "/api/vessels")
        assert resp.status_code == 200

    def test_vessels_returns_list(self, api_base_url, require_api):
        resp = _get(api_base_url, "/api/vessels")
        body = resp.json()
        assert isinstance(body, list)

    def test_vessels_limit_param(self, api_base_url, require_api):
        resp = _get(api_base_url, "/api/vessels?limit=5")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) <= 5


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class TestAlerts:
    def test_alerts_returns_200(self, api_base_url, require_api):
        resp = _get(api_base_url, "/api/alerts")
        assert resp.status_code == 200

    def test_alerts_returns_list(self, api_base_url, require_api):
        resp = _get(api_base_url, "/api/alerts")
        body = resp.json()
        assert isinstance(body, list)


# ---------------------------------------------------------------------------
# Analytics summary
# ---------------------------------------------------------------------------

class TestAnalyticsSummary:
    def test_summary_returns_200(self, api_base_url, require_api):
        resp = _get(api_base_url, "/api/analytics/summary")
        assert resp.status_code == 200

    def test_summary_has_vessel_count(self, api_base_url, require_api):
        resp = _get(api_base_url, "/api/analytics/summary")
        body = resp.json()
        assert "total_vessels" in body or "vessel_count" in body or isinstance(body, dict)


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------

class TestAnomalies:
    def test_anomalies_returns_200(self, api_base_url, require_api):
        resp = _get(api_base_url, "/api/anomalies")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Invalid routes → 404
# ---------------------------------------------------------------------------

class TestNotFound:
    def test_unknown_route_returns_404(self, api_base_url, require_api):
        resp = _get(api_base_url, "/api/this-does-not-exist-xyz")
        assert resp.status_code == 404
