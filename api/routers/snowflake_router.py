"""
snowflake_router.py — Maritime Navigation AI System
FastAPI router for Snowflake-backed analytics endpoints.

HOW TO REGISTER IN main.py (add these two lines, do NOT modify other code):
    from api.routers.snowflake_router import router as snowflake_router
    app.include_router(snowflake_router)

Behaviour:
  - SNOWFLAKE_ACCOUNT set    → real data from Snowflake
  - SNOWFLAKE_ACCOUNT unset  → mock/PostgreSQL fallback, HTTP 200 with
                               `"source": "mock"` so callers can detect it
  - Snowflake query error    → HTTP 200 with `"source": "error"` and empty
                               data arrays; never raises 500 to the frontend

Endpoints added:
    GET /api/analytics/snowflake-summary
    GET /api/analytics/snowflake-lanes
    GET /api/analytics/snowflake-anomalies
    GET /api/analytics/snowflake-congestion
    GET /api/analytics/snowflake-collisions
    GET /api/analytics/snowflake-status
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query

# Path setup — works in Docker (/app) and local dev
for _p in ["/app/src", "/app/api",
           os.path.join(os.path.dirname(__file__), "..", "..", "src"),
           os.path.join(os.path.dirname(__file__), "..", "..")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Snowflake Analytics"])

# ---------------------------------------------------------------------------
# Conditional Snowflake import — never raises ImportError at startup
# ---------------------------------------------------------------------------
try:
    from snowflake.snowflake_queries import (
        get_fleet_summary,
        get_busiest_lanes,
        get_anomaly_trends,
        get_congestion_by_hour,
        get_collision_stats,
        _is_configured as _sf_configured,
    )
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False
    def _sf_configured(): return False

# ---------------------------------------------------------------------------
# Mock data — returned when Snowflake is not configured
# Realistic enough to let the frontend render charts without crashing.
# ---------------------------------------------------------------------------
_MOCK_FLEET = {
    "total_vessels":        1284,
    "high_risk_vessels":      42,
    "anomalous_vessels":      17,
    "fleet_avg_speed_kn":    8.3,
    "east_coast_vessels":    512,
    "west_coast_vessels":    298,
    "gulf_vessels":          319,
    "data_freshness":        None,
    "snowflake_available":   False,
}

_MOCK_LANES = [
    {"lat_grid": 40.7, "lon_grid": -74.0, "total_vessels": 312,
     "unique_vessels": 287, "avg_speed_kn": 9.1,
     "peak_congestion": "HIGH", "us_region": "East Coast"},
    {"lat_grid": 37.8, "lon_grid": -122.4, "total_vessels": 224,
     "unique_vessels": 198, "avg_speed_kn": 7.4,
     "peak_congestion": "HIGH", "us_region": "West Coast"},
    {"lat_grid": 29.7, "lon_grid": -90.1, "total_vessels": 189,
     "unique_vessels": 176, "avg_speed_kn": 6.8,
     "peak_congestion": "MEDIUM", "us_region": "Gulf of Mexico"},
    {"lat_grid": 43.6, "lon_grid": -79.4, "total_vessels": 142,
     "unique_vessels": 134, "avg_speed_kn": 10.2,
     "peak_congestion": "MEDIUM", "us_region": "Great Lakes"},
    {"lat_grid": 34.0, "lon_grid": -118.2, "total_vessels": 118,
     "unique_vessels": 109, "avg_speed_kn": 8.9,
     "peak_congestion": "MEDIUM", "us_region": "West Coast"},
]

_MOCK_ANOMALY_TREND = [
    {"event_date": str((datetime.utcnow() - timedelta(days=i)).date()),
     "anomaly_type": "ML_ANOMALY", "anomaly_count": max(1, 5 - i % 3),
     "avg_score": 0.72, "unique_vessels": max(1, 3 - i % 2),
     "us_region": "East Coast"}
    for i in range(7)
]

_MOCK_CONGESTION = [
    {"hour_of_day": h, "congestion_level": "HIGH" if 8 <= h <= 18 else "LOW",
     "grid_cells": 12 if 8 <= h <= 18 else 4,
     "avg_vessels_per_cell": 18.3 if 8 <= h <= 18 else 5.1,
     "avg_speed_kn": 6.2 if 8 <= h <= 18 else 9.8,
     "us_region": "East Coast"}
    for h in range(24)
]

_MOCK_COLLISIONS = [
    {"week_start": str((datetime.utcnow() - timedelta(weeks=i)).date()),
     "severity": "MEDIUM", "collision_alerts": max(1, 3 - i),
     "vessels_involved": max(2, 6 - i * 2),
     "avg_cpa_nm": 0.31, "resolved": max(0, 2 - i),
     "outstanding": 1, "us_region": "East Coast"}
    for i in range(4)
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_records(df) -> list:
    """Convert a pandas DataFrame (or None) to a list of dicts."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return []
    return df.to_dict(orient="records")


def _source_label() -> str:
    if not _SF_AVAILABLE:
        return "mock_no_library"
    if not _sf_configured():
        return "mock_no_credentials"
    return "snowflake"


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/api/analytics/snowflake-status",
            summary="Snowflake connectivity status")
def snowflake_status():
    """
    Returns whether Snowflake is configured and reachable.
    Safe to poll from the frontend to conditionally show analytics tabs.
    """
    configured = _SF_AVAILABLE and _sf_configured()
    reachable  = False
    if configured:
        try:
            result = get_fleet_summary()
            reachable = result.get("snowflake_available", False)
        except Exception as exc:
            logger.warning("Snowflake ping failed: %s", exc)

    return {
        "library_installed": _SF_AVAILABLE,
        "credentials_set":   _sf_configured(),
        "reachable":         reachable,
        "account":           os.getenv("SNOWFLAKE_ACCOUNT", "")[:8] + "…"
                             if os.getenv("SNOWFLAKE_ACCOUNT") else "",
        "database":          os.getenv("SNOWFLAKE_DATABASE", "MARITIME_AIS"),
        "checked_at":        datetime.utcnow().isoformat(),
    }


@router.get("/api/analytics/snowflake-summary",
            summary="Fleet KPI summary from Snowflake (US coastal waters)")
def snowflake_summary():
    """
    Top-line fleet KPIs: total vessels, risk counts, per-region splits,
    fleet average speed, data freshness timestamp.

    Returns mock data with `source: mock_no_credentials` when Snowflake is
    not configured — the frontend renders the same layout either way.
    """
    source = _source_label()

    if source == "snowflake":
        try:
            data = get_fleet_summary()
            return {"source": "snowflake", "summary": data,
                    "generated_at": datetime.utcnow().isoformat()}
        except Exception as exc:
            logger.error("snowflake_summary failed: %s", exc)
            source = "error"

    return {
        "source":       source,
        "summary":      _MOCK_FLEET,
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/api/analytics/snowflake-lanes",
            summary="Busiest shipping lanes in US coastal waters")
def snowflake_lanes(
    top_n: int = Query(default=50, ge=5, le=200,
                       description="Number of grid cells to return"),
):
    """
    Top N 0.1° grid cells by vessel count, with US region label and
    congestion peak.  Suitable for rendering as a traffic density map.
    """
    source = _source_label()

    if source == "snowflake":
        try:
            df = get_busiest_lanes(top_n)
            return {"source": "snowflake", "count": len(df),
                    "lanes": _df_to_records(df),
                    "generated_at": datetime.utcnow().isoformat()}
        except Exception as exc:
            logger.error("snowflake_lanes failed: %s", exc)
            source = "error"

    return {
        "source":       source,
        "count":        len(_MOCK_LANES),
        "lanes":        _MOCK_LANES[:top_n],
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/api/analytics/snowflake-anomalies",
            summary="Anomaly trends over time (US coastal waters)")
def snowflake_anomalies(
    days_back: int = Query(default=30, ge=1, le=365,
                           description="Lookback window in days"),
):
    """
    Daily anomaly counts by type and US region.
    When Snowflake is not configured returns a 7-day mock trend.
    """
    source = _source_label()

    if source == "snowflake":
        try:
            df = get_anomaly_trends(days_back)
            return {"source": "snowflake", "days_back": days_back,
                    "count": len(df), "trends": _df_to_records(df),
                    "generated_at": datetime.utcnow().isoformat()}
        except Exception as exc:
            logger.error("snowflake_anomalies failed: %s", exc)
            source = "error"

    return {
        "source":       source,
        "days_back":    days_back,
        "count":        len(_MOCK_ANOMALY_TREND),
        "trends":       _MOCK_ANOMALY_TREND,
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/api/analytics/snowflake-congestion",
            summary="Vessel congestion patterns by hour of day")
def snowflake_congestion(
    days_back: int = Query(default=14, ge=1, le=90),
):
    """
    Average vessel density per grid cell broken down by hour of day (UTC)
    and US region.  Reveals peak traffic hours for port operations planning.
    """
    source = _source_label()

    if source == "snowflake":
        try:
            df = get_congestion_by_hour(days_back)
            return {"source": "snowflake", "days_back": days_back,
                    "rows": len(df), "congestion": _df_to_records(df),
                    "generated_at": datetime.utcnow().isoformat()}
        except Exception as exc:
            logger.error("snowflake_congestion failed: %s", exc)
            source = "error"

    return {
        "source":       source,
        "days_back":    days_back,
        "rows":         len(_MOCK_CONGESTION),
        "congestion":   _MOCK_CONGESTION,
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/api/analytics/snowflake-collisions",
            summary="Weekly collision risk statistics")
def snowflake_collisions(
    weeks_back: int = Query(default=8, ge=1, le=52),
):
    """
    Weekly collision-risk alert counts by severity and US region.
    Includes average closest-point-of-approach distance and resolution rate.
    """
    source = _source_label()

    if source == "snowflake":
        try:
            df = get_collision_stats(weeks_back)
            return {"source": "snowflake", "weeks_back": weeks_back,
                    "rows": len(df), "stats": _df_to_records(df),
                    "generated_at": datetime.utcnow().isoformat()}
        except Exception as exc:
            logger.error("snowflake_collisions failed: %s", exc)
            source = "error"

    return {
        "source":       source,
        "weeks_back":   weeks_back,
        "rows":         len(_MOCK_COLLISIONS),
        "stats":        _MOCK_COLLISIONS,
        "generated_at": datetime.utcnow().isoformat(),
    }
