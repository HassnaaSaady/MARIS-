"""
live_scorer.py — Maritime Navigation AI System
Kafka consumer: reads ais_raw, scores with ML, writes to PostgreSQL.
Run as a daemon: python -m ml.live_scorer
"""
import os, sys, time, json, logging
from datetime import datetime

import pandas as pd
from kafka import KafkaConsumer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/api")

from common.config import (
    KAFKA_BOOTSTRAP_SERVERS, AIS_TOPIC,
    POSTGRES_URL,
)
from common.schema_utils import classify_risk
from ml.scorer import get_scorer, LiveScorer
from models.database import FactVesselLatest, FactAlert, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [live_scorer] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

BATCH_SIZE   = int(os.getenv("SCORER_BATCH_SIZE",    "200"))
BATCH_WINDOW = float(os.getenv("SCORER_BATCH_WINDOW", "5.0"))  # seconds


def _parse_dt(val) -> datetime | None:
    if not val:
        return None
    try:
        return datetime.fromisoformat(str(val).replace("Z", "+00:00").split("+")[0])
    except Exception:
        return None


def _upsert_vessels(session, records: list[dict]) -> None:
    """One-row-per-MMSI UPSERT into fact_vessel_latest."""
    for r in records:
        mmsi = str(r.get("mmsi", "")).strip()
        if not mmsi:
            continue
        row = session.query(FactVesselLatest).filter_by(mmsi=mmsi).first()
        ts  = _parse_dt(r.get("base_datetime")) or datetime.utcnow()
        if row:
            row.lat           = r.get("lat")
            row.lon           = r.get("lon")
            row.sog           = r.get("sog")
            row.cog           = r.get("cog")
            row.heading       = r.get("heading")
            row.risk_level    = r.get("risk_level", "LOW")
            row.is_anomaly    = bool(r.get("is_anomaly", False))
            row.anomaly_score = float(r.get("anomaly_score", 0.0))
            row.anomaly_type  = r.get("anomaly_type") or ""
            row.predicted_lat = r.get("predicted_lat")
            row.predicted_lon = r.get("predicted_lon")
            row.base_datetime = ts
            row.updated_at    = datetime.utcnow()
            row.data_split    = "live"
        else:
            session.add(FactVesselLatest(
                mmsi          = mmsi,
                vessel_name   = str(r.get("vessel_name", ""))[:100],
                vessel_type   = str(r.get("vessel_type", ""))[:50],
                lat           = r.get("lat"),
                lon           = r.get("lon"),
                sog           = r.get("sog"),
                cog           = r.get("cog"),
                heading       = r.get("heading"),
                risk_level    = r.get("risk_level", "LOW"),
                is_anomaly    = bool(r.get("is_anomaly", False)),
                anomaly_score = float(r.get("anomaly_score", 0.0)),
                anomaly_type  = r.get("anomaly_type") or "",
                predicted_lat = r.get("predicted_lat"),
                predicted_lon = r.get("predicted_lon"),
                base_datetime = ts,
                data_split    = "live",
                updated_at    = datetime.utcnow(),
            ))


def _write_alerts(session, alerts: list[dict]) -> None:
    for a in alerts:
        session.add(FactAlert(
            alert_type    = a.get("alert_type", "ML_ANOMALY"),
            severity      = a.get("severity", "MEDIUM"),
            mmsi_1        = a.get("mmsi_1") or a.get("mmsi"),
            mmsi_2        = a.get("mmsi_2"),
            vessel_name_1 = a.get("vessel_name_1") or a.get("vessel_name", ""),
            vessel_name_2 = a.get("vessel_name_2"),
            lat           = a.get("lat"),
            lon           = a.get("lon"),
            description   = a.get("description", ""),
            anomaly_score = a.get("anomaly_score"),
            distance_nm   = a.get("distance_nm"),
            is_resolved   = False,
            created_at    = datetime.utcnow(),
            data_split    = "live",
        ))


def _process_batch(scorer: LiveScorer, session, messages: list[dict]) -> None:
    if not messages:
        return

    df = pd.DataFrame(messages)

    for col in ["sog", "cog", "heading", "lat", "lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Anomaly scoring
    df = scorer.score_anomaly(df)

    # Risk classification
    df["risk_level"] = df.apply(
        lambda row: classify_risk(
            float(row.get("sog") or 0),
            float(row.get("lat") or 0),
            float(row.get("lon") or 0),
        ),
        axis=1,
    )

    records = df.to_dict("records")

    # Position prediction for moving vessels (5-min ahead)
    for r in records:
        if (r.get("sog") or 0) >= 0.1:
            try:
                pred = scorer.predict_position(
                    lat=float(r.get("lat") or 0),
                    lon=float(r.get("lon") or 0),
                    sog=float(r.get("sog") or 0),
                    cog=float(r.get("cog") or 0),
                    heading=float(r.get("heading") or r.get("cog") or 0),
                    minutes=5,
                )
                r["predicted_lat"] = pred.get("predicted_lat")
                r["predicted_lon"] = pred.get("predicted_lon")
            except Exception:
                pass

    # UPSERT vessel positions
    try:
        _upsert_vessels(session, records)
        session.commit()
    except Exception as exc:
        session.rollback()
        log.error("UPSERT failed: %s", exc)

    # Build anomaly alert list
    anomaly_alerts = [
        {
            "alert_type":   r.get("anomaly_type") or "ML_ANOMALY",
            "severity":     "HIGH" if (r.get("anomaly_score") or 0) >= 0.7 else "MEDIUM" if (r.get("anomaly_score") or 0) >= 0.5 else "LOW",
            "mmsi":         r.get("mmsi"),
            "vessel_name":  r.get("vessel_name", ""),
            "lat":          r.get("lat"),
            "lon":          r.get("lon"),
            "anomaly_score":r.get("anomaly_score"),
            "description":  (
                f"{r.get('anomaly_type', 'Anomaly')} detected for "
                f"vessel {r.get('mmsi')} — score {r.get('anomaly_score', 0):.2f}"
            ),
        }
        for r in records if r.get("is_anomaly")
    ]

    # Collision detection: one row per vessel, moving only, capped for performance
    latest = df.drop_duplicates("mmsi", keep="last")
    moving = latest[latest["sog"] >= 1.0]
    if len(moving) > 300:
        moving = moving.sample(300, random_state=42)
    collision_alerts = scorer.detect_collisions(moving) if len(moving) > 1 else []

    all_alerts = anomaly_alerts + collision_alerts
    if all_alerts:
        try:
            _write_alerts(session, all_alerts)
            session.commit()
            log.info(
                "Alerts: %d anomaly, %d collision",
                len(anomaly_alerts), len(collision_alerts),
            )
        except Exception as exc:
            session.rollback()
            log.error("Alert write failed: %s", exc)


def main() -> None:
    log.info("Live scorer starting …")

    engine  = create_engine(POSTGRES_URL, pool_size=5, pool_pre_ping=True)
    init_db(engine)
    Session = sessionmaker(bind=engine)
    scorer  = get_scorer()

    consumer = KafkaConsumer(
        AIS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="live_scorer_v1",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        max_poll_records=BATCH_SIZE,
    )

    log.info("Connected to Kafka — consuming %s", AIS_TOPIC)

    batch      = []
    last_flush = time.time()

    while True:
        try:
            poll_result = consumer.poll(timeout_ms=1000, max_records=BATCH_SIZE)
            for _, msgs in poll_result.items():
                for msg in msgs:
                    if isinstance(msg.value, dict):
                        batch.append(msg.value)

            elapsed = time.time() - last_flush
            if len(batch) >= BATCH_SIZE or (batch and elapsed >= BATCH_WINDOW):
                session = Session()
                try:
                    log.info("Processing batch of %d records …", len(batch))
                    _process_batch(scorer, session, batch)
                finally:
                    session.close()
                batch      = []
                last_flush = time.time()

        except KeyboardInterrupt:
            log.info("Shutting down …")
            break
        except Exception as exc:
            log.error("Consumer loop error: %s", exc, exc_info=True)
            time.sleep(5)

    consumer.close()


if __name__ == "__main__":
    main()
