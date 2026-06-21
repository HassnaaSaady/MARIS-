"""
main.py — Maritime Navigation AI System | FastAPI Backend
All REST API endpoints for the React frontend dashboard.

Auto-generated docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import json, sys, os
from utils.vessel_types import vessel_label

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/api")

from common.config import POSTGRES_URL
from ml.scorer import get_scorer

app = FastAPI(
    title="Maritime Navigation AI System",
    description="Real-time AIS vessel tracking, anomaly detection, collision risk",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DB dependency ──────────────────────────────────────────────────────────────
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker
from models.database import (
    Base, DimVessel, FactVesselLatest, FactAisTrack,
    FactTrafficDensity, FactAlert, FactDailyStats,
    init_db,
)

engine  = create_engine(POSTGRES_URL, pool_size=20, pool_pre_ping=True)
Session = sessionmaker(bind=engine)

def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
async def startup():
    init_db(engine)
    print("✅  Database ready")



# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 1 — Vessel Tracking Map
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/vessels", tags=["Vessel Tracking"])
def get_vessels(
    vessel_type: Optional[str] = None,
    risk_level:  Optional[str] = None,
    min_sog:     float = 0.0,
    max_sog:     float = 60.0,
    is_anomaly:  Optional[bool] = None,
    data_split:  Optional[str] = None,
    limit:       int = 2000,
    db = Depends(get_db),
):
    """Get latest position for all vessels. Powers the live map."""
    q = db.query(FactVesselLatest)
    if vessel_type: q = q.filter(FactVesselLatest.vessel_type == vessel_type)
    if risk_level:  q = q.filter(FactVesselLatest.risk_level  == risk_level)
    if is_anomaly is not None:
        q = q.filter(FactVesselLatest.is_anomaly == is_anomaly)
    if data_split:  q = q.filter(FactVesselLatest.data_split == data_split)
    q = q.filter(
        (FactVesselLatest.sog == None) |
        FactVesselLatest.sog.between(min_sog, max_sog)
    )

    vessels = q.limit(limit).all()
    return {
        "count": len(vessels),
        "vessels": [
            {
                "mmsi":          v.mmsi,
                "vessel_name":   v.vessel_name,
                "vessel_type":   v.vessel_type,
                "vessel_type_label": vessel_label(v.vessel_type),
                "lat":           v.lat,
                "lon":           v.lon,
                "sog":           v.sog,
                "cog":           v.cog,
                "heading":       v.heading or v.cog or 0,
                "risk_level":    v.risk_level,
                "is_anomaly":    v.is_anomaly,
                "anomaly_score": v.anomaly_score,
                "anomaly_type":  v.anomaly_type,
                "predicted_lat": v.predicted_lat,
                "predicted_lon": v.predicted_lon,
                "updated_at":    v.updated_at.isoformat() if v.updated_at else None,
            }
            for v in vessels
        ],
    }


@app.get("/api/vessels/{mmsi}", tags=["Vessel Tracking"])
def get_vessel_detail(mmsi: str, db = Depends(get_db)):
    """Full detail card for one vessel."""
    v = db.query(FactVesselLatest).filter(
        FactVesselLatest.mmsi == mmsi).first()
    if not v:
        raise HTTPException(404, f"Vessel {mmsi} not found")

    # Also get dimension data
    d = db.query(DimVessel).filter(DimVessel.mmsi == mmsi).first()

    return {
        "mmsi":          v.mmsi,
        "vessel_name":   v.vessel_name,
        "vessel_type":   v.vessel_type,
        "lat":           v.lat,
        "lon":           v.lon,
        "sog":           v.sog,
        "cog":           v.cog,
        "heading":       v.heading,
        "risk_level":    v.risk_level,
        "is_anomaly":    v.is_anomaly,
        "anomaly_score": v.anomaly_score,
        "anomaly_type":  v.anomaly_type,
        "predicted_lat": v.predicted_lat,
        "predicted_lon": v.predicted_lon,
        "updated_at":    v.updated_at.isoformat() if v.updated_at else None,
        # Dimension data
        "imo":              d.imo        if d else None,
        "call_sign":        d.call_sign  if d else None,
        "length":           d.length     if d else None,
        "width":            d.width      if d else None,
        "draft":            d.draft      if d else None,
        "cargo":            d.cargo      if d else None,
        "total_records":    d.total_records if d else None,
        "first_seen":       d.first_seen.isoformat() if d and d.first_seen else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 2 — Historical Replay
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/vessels/{mmsi}/track", tags=["Historical Replay"])
def get_vessel_track(
    mmsi:       str,
    start_time: Optional[str] = None,
    end_time:   Optional[str] = None,
    data_split: Optional[str] = None,
    limit:      int = 5000,
    db = Depends(get_db),
):
    """Full historical track for replay slider."""
    q = db.query(FactAisTrack).filter(FactAisTrack.mmsi == mmsi)
    if start_time:
        q = q.filter(FactAisTrack.base_datetime >=
                     datetime.fromisoformat(start_time))
    if end_time:
        q = q.filter(FactAisTrack.base_datetime <=
                     datetime.fromisoformat(end_time))
    if data_split:
        q = q.filter(FactAisTrack.data_split == data_split)

    tracks = q.order_by(FactAisTrack.base_datetime).limit(limit).all()
    return {
        "mmsi":   mmsi,
        "count":  len(tracks),
        "points": [
            {
                "lat":           t.lat,
                "lon":           t.lon,
                "sog":           t.sog,
                "cog":           t.cog,
                "heading":       t.heading,
                "risk_level":    t.risk_level,
                "is_anomaly":    t.is_anomaly,
                "anomaly_type":  t.anomaly_type,
                "base_datetime": t.base_datetime.isoformat(),
            }
            for t in tracks
        ],
    }


@app.get("/api/replay/snapshot", tags=["Historical Replay"])
def get_replay_snapshot(
    timestamp:   str,
    vessel_type: Optional[str] = None,
    limit:       int = 500,
    db = Depends(get_db),
):
    """All vessel positions at a specific timestamp for replay."""
    ts   = datetime.fromisoformat(timestamp)
    low  = ts - timedelta(minutes=5)
    high = ts + timedelta(minutes=5)

    q = db.query(FactAisTrack).filter(
        FactAisTrack.base_datetime.between(low, high)
    )
    if vessel_type:
        q = q.filter(FactAisTrack.vessel_type == vessel_type)

    return {
        "timestamp": timestamp,
        "vessels": [
            {"mmsi": t.mmsi, "lat": t.lat, "lon": t.lon,
             "sog": t.sog, "heading": t.heading or t.cog,
             "vessel_type": t.vessel_type,
             "risk_level": t.risk_level}
            for t in q.limit(limit).all()
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 3 — Traffic Density Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/density", tags=["Traffic Density"])
def get_density(
    hours_back:  int = 24,
    min_vessels: int = 1,
    db = Depends(get_db),
):
    """Grid cell density for heatmap rendering.

    Returns one row per unique (lat_bin, lon_bin) grid cell — the peak
    vessel_count row for that cell across all time buckets.
    Falls back to all available data when the time window is empty
    (e.g. the table holds only historical parquet-loaded data).
    """
    since = datetime.utcnow() - timedelta(hours=hours_back)

    # One row per unique (lat_bin, lon_bin) using its peak vessel_count hour.
    # Balanced across congestion levels: 700 HIGH + 200 MEDIUM + 100 LOW,
    # so the heatmap renders red/orange/green circles, not just one colour.
    def _balanced_sql(time_filter: str) -> text:
        return text(f"""
            WITH cell_peaks AS (
                SELECT DISTINCT ON (lat_bin, lon_bin)
                       lat_bin, lon_bin, vessel_count, avg_sog, congestion_level
                FROM fact_traffic_density
                WHERE vessel_count >= :min_vessels {time_filter}
                ORDER BY lat_bin, lon_bin, vessel_count DESC
            )
            (SELECT * FROM cell_peaks WHERE congestion_level = 'HIGH'
             ORDER BY vessel_count DESC LIMIT 700)
            UNION ALL
            (SELECT * FROM cell_peaks WHERE congestion_level = 'MEDIUM'
             ORDER BY vessel_count DESC LIMIT 200)
            UNION ALL
            (SELECT * FROM cell_peaks WHERE congestion_level = 'LOW'
             ORDER BY vessel_count DESC LIMIT 100)
        """)

    rows = db.execute(
        _balanced_sql("AND hour_bucket >= :since"),
        {"since": since, "min_vessels": min_vessels},
    ).mappings().all()
    is_fallback = False
    if not rows:
        rows = db.execute(
            _balanced_sql(""),
            {"min_vessels": min_vessels},
        ).mappings().all()
        is_fallback = True

    # Third fallback: aggregate from fact_ais_track when fact_traffic_density
    # has no data at all (table empty / gold_job not yet run).
    if not rows:
        ais_rows = db.execute(text("""
            SELECT
                ROUND(lat::numeric, 1) AS lat_bin,
                ROUND(lon::numeric, 1) AS lon_bin,
                COUNT(*)               AS vessel_count,
                AVG(sog)               AS avg_sog
            FROM fact_ais_track
            GROUP BY 1, 2
            ORDER BY vessel_count DESC
            LIMIT 1000
        """)).mappings().all()
        if ais_rows:
            _mx = max((r["vessel_count"] for r in ais_rows), default=1)
            def _cong(n):
                if n >= 10000: return "HIGH"
                if n >= 1000:  return "MEDIUM"
                return "LOW"
            return {
                "cells": [
                    {
                        "lat":              float(r["lat_bin"]),
                        "lon":              float(r["lon_bin"]),
                        "vessel_count":     int(r["vessel_count"]),
                        "avg_sog":          float(r["avg_sog"] or 0),
                        "congestion_level": _cong(int(r["vessel_count"])),
                        "weight":           min(int(r["vessel_count"]) / _mx, 1.0),
                    }
                    for r in ais_rows
                ],
                "is_historical_fallback": True,
            }

    max_count = max((r["vessel_count"] for r in rows), default=1)
    return {
        "cells": [
            {
                "lat":              r["lat_bin"],
                "lon":              r["lon_bin"],
                "vessel_count":     r["vessel_count"],
                "avg_sog":          r["avg_sog"],
                "congestion_level": r["congestion_level"],
                "weight":           min(r["vessel_count"] / max_count, 1.0),
            }
            for r in rows
        ],
        "is_historical_fallback": is_fallback,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 4 — Congestion Detection
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/congestion", tags=["Congestion"])
def get_congestion(
    level: Optional[str] = None,
    db = Depends(get_db),
):
    """Current congestion zones."""
    since = datetime.utcnow() - timedelta(hours=1)
    q = (
        db.query(FactTrafficDensity)
        .filter(FactTrafficDensity.hour_bucket >= since)
        .filter(FactTrafficDensity.vessel_count > 3)
    )
    if level:
        q = q.filter(FactTrafficDensity.congestion_level == level)

    return {
        "zones": [
            {
                "lat":   r.lat_bin,
                "lon":   r.lon_bin,
                "count": r.vessel_count,
                "level": r.congestion_level,
                "avg_sog": r.avg_sog,
            }
            for r in q.limit(500).all()
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 5 — Position Prediction
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/vessels/{mmsi}/predict", tags=["Position Prediction"])
def predict_position(
    mmsi:          str,
    minutes_ahead: int = 10,
    db = Depends(get_db),
):
    """Predict next position using XGBoost or dead reckoning."""
    v = db.query(FactVesselLatest).filter(
        FactVesselLatest.mmsi == mmsi).first()
    if not v:
        raise HTTPException(404, f"Vessel {mmsi} not found")

    scorer = get_scorer()
    result = scorer.predict_position(
        lat=v.lat, lon=v.lon,
        sog=v.sog or 0, cog=v.cog or 0,
        heading=v.heading or v.cog or 0,
        minutes=minutes_ahead,
    )
    return {
        "mmsi":          mmsi,
        "vessel_name":   v.vessel_name,
        "current_lat":   v.lat,
        "current_lon":   v.lon,
        "current_sog":   v.sog,
        **result,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 6 — Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/anomalies", tags=["Anomaly Detection"])
def get_anomalies(
    hours_back: int = 24,
    alert_type: Optional[str] = None,
    severity:   Optional[str] = None,
    limit:      int = 200,
    db = Depends(get_db),
):
    """Recent anomaly alerts."""
    since = datetime.utcnow() - timedelta(hours=hours_back)
    anomaly_types = [
        "SUDDEN_STOP", "SHARP_TURN", "UNUSUAL_SPEED",
        "STATIONARY_RISK", "ML_ANOMALY", "UNEXPECTED_DIRECTION"
    ]
    q = (
        db.query(FactAlert)
        .filter(FactAlert.created_at >= since)
        .filter(FactAlert.alert_type.in_(anomaly_types))
    )
    if alert_type: q = q.filter(FactAlert.alert_type == alert_type)
    if severity:   q = q.filter(FactAlert.severity   == severity)

    alerts = q.order_by(FactAlert.created_at.desc()).limit(limit).all()
    return {
        "count": len(alerts),
        "anomalies": [
            {
                "id":          a.id,
                "mmsi":        a.mmsi_1,
                "vessel_name": a.vessel_name_1,
                "type":        a.alert_type,
                "severity":    a.severity,
                "lat":         a.lat,
                "lon":         a.lon,
                "description": a.description,
                "score":       a.anomaly_score,
                "created_at":  a.created_at.isoformat(),
            }
            for a in alerts
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 7 — Collision Risk
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/collision-risks", tags=["Collision Risk"])
def get_collision_risks(
    hours_back: int = 6,
    severity:   Optional[str] = None,
    db = Depends(get_db),
):
    """Active collision risk alerts."""
    since = datetime.utcnow() - timedelta(hours=hours_back)
    q = (
        db.query(FactAlert)
        .filter(FactAlert.created_at    >= since)
        .filter(FactAlert.alert_type    == "COLLISION_RISK")
        .filter(FactAlert.is_resolved   == False)
    )
    if severity:
        q = q.filter(FactAlert.severity == severity)

    risks = q.order_by(
        FactAlert.severity.desc(),
        FactAlert.created_at.desc()
    ).all()
    return {
        "count": len(risks),
        "risks": [
            {
                "id":          r.id,
                "mmsi_1":      r.mmsi_1,
                "mmsi_2":      r.mmsi_2,
                "vessel_1":    r.vessel_name_1,
                "vessel_2":    r.vessel_name_2,
                "severity":    r.severity,
                "distance_nm": r.distance_nm,
                "lat":         r.lat,
                "lon":         r.lon,
                "description": r.description,
                "created_at":  r.created_at.isoformat(),
            }
            for r in risks
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 8 — Alerts Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/alerts", tags=["Alerts"])
def get_alerts(
    hours_back:  int = 24,
    alert_type:  Optional[str] = None,
    severity:    Optional[str] = None,
    is_resolved: Optional[bool] = None,
    limit:       int = 500,
    db = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(hours=hours_back)
    q = db.query(FactAlert).filter(FactAlert.created_at >= since)
    if alert_type:    q = q.filter(FactAlert.alert_type  == alert_type)
    if severity:      q = q.filter(FactAlert.severity     == severity)
    if is_resolved is not None:
        q = q.filter(FactAlert.is_resolved == is_resolved)

    alerts = q.order_by(FactAlert.created_at.desc()).limit(limit).all()
    return {
        "count": len(alerts),
        "alerts": [
            {
                "id":          a.id,
                "type":        a.alert_type,
                "severity":    a.severity,
                "mmsi":        a.mmsi_1,
                "vessel_name": a.vessel_name_1,
                "lat":         a.lat,
                "lon":         a.lon,
                "description": a.description,
                "is_resolved": a.is_resolved,
                "created_at":  a.created_at.isoformat(),
            }
            for a in alerts
        ]
    }


@app.patch("/api/alerts/{alert_id}/resolve", tags=["Alerts"])
def resolve_alert(alert_id: int, db = Depends(get_db)):
    a = db.query(FactAlert).filter(FactAlert.id == alert_id).first()
    if not a:
        raise HTTPException(404, "Alert not found")
    a.is_resolved = True
    a.resolved_at = datetime.utcnow()
    db.commit()
    return {"success": True, "alert_id": alert_id}


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 9 — Analytics Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/analytics/summary", tags=["Analytics"])
def get_summary(
    days_back: int = 14,
    db = Depends(get_db),
):
    """KPI summary for analytics charts."""
    since = datetime.utcnow() - timedelta(days=days_back)

    total_vessels = db.query(FactVesselLatest).count()
    total_alerts  = db.query(FactAlert).filter(
        FactAlert.created_at >= since).count()
    high_risk     = db.query(FactVesselLatest).filter(
        FactVesselLatest.risk_level == "HIGH").count()
    anomalies     = db.query(FactVesselLatest).filter(
        FactVesselLatest.is_anomaly == True).count()

    sog_rows = db.query(FactVesselLatest.sog).all()
    avg_sog  = (sum(r.sog for r in sog_rows if r.sog)
                / max(len(sog_rows), 1))

    # Vessel type distribution
    type_dist = (
        db.query(
            FactVesselLatest.vessel_type,
            func.count(FactVesselLatest.mmsi).label("count")
        )
        .group_by(FactVesselLatest.vessel_type)
        .order_by(func.count(FactVesselLatest.mmsi).desc())
        .limit(10).all()
    )

    # Alert type distribution
    alert_dist = (
        db.query(
            FactAlert.alert_type,
            func.count(FactAlert.id).label("count")
        )
        .filter(FactAlert.created_at >= since)
        .group_by(FactAlert.alert_type)
        .all()
    )

    # Daily stats from Gold table
    daily = (
        db.query(FactDailyStats)
        .filter(FactDailyStats.stat_date >= since.date())
        .order_by(FactDailyStats.stat_date)
        .all()
    )

    return {
        "total_vessels":   total_vessels,
        "total_alerts":    total_alerts,
        "high_risk_count": high_risk,
        "anomaly_count":   anomalies,
        "avg_speed_kn":    round(avg_sog, 2),
        "vessel_types":    [{"type": t or "Unknown", "count": c}
                            for t, c in type_dist],
        "alert_types":     [{"type": t, "count": c}
                            for t, c in alert_dist],
        "daily_stats":     [
            {
                "date":          str(d.stat_date),
                "total_vessels": d.total_vessels,
                "avg_sog":       d.avg_sog,
                "high_risk":     d.high_risk_count,
                "anomalies":     d.anomaly_count,
            }
            for d in daily
        ],
        "generated_at": datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 10 — Search and Filtering
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/search", tags=["Search"])
def search_vessels(
    q:           str = Query(..., min_length=2),
    vessel_type: Optional[str] = None,
    risk_level:  Optional[str] = None,
    min_speed:   float = 0.0,
    max_speed:   float = 60.0,
    limit:       int = 100,
    db = Depends(get_db),
):
    """Search by MMSI, name, IMO, or call sign."""
    query = db.query(FactVesselLatest).filter(
        (FactVesselLatest.mmsi.ilike(f"%{q}%")) |
        (FactVesselLatest.vessel_name.ilike(f"%{q}%"))
    ).filter(FactVesselLatest.sog.between(min_speed, max_speed))

    if vessel_type: query = query.filter(
        FactVesselLatest.vessel_type == vessel_type)
    if risk_level: query = query.filter(
        FactVesselLatest.risk_level == risk_level)

    results = query.limit(limit).all()
    return {
        "query":   q,
        "count":   len(results),
        "results": [
            {
                "mmsi":        v.mmsi,
                "vessel_name": v.vessel_name,
                "vessel_type": v.vessel_type,
                "lat":         v.lat,
                "lon":         v.lon,
                "sog":         v.sog,
                "risk_level":  v.risk_level,
                "is_anomaly":  v.is_anomaly,
            }
            for v in results
        ]
    }



@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
