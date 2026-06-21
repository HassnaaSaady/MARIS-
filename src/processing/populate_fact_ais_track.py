"""
populate_fact_ais_track.py — Load rows from parquet into fact_ais_track.
Processes ONE file at a time — no memory crash.

Run inside producer container:
  docker exec ais-producer python /app/src/processing/populate_fact_ais_track.py
"""
import sys, math, logging, os, random, glob

_BASE = "/opt/spark/app" if os.path.exists("/opt/spark/app/src") else "/app"
sys.path.insert(0, f"{_BASE}/src/common")

import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text

from config import POSTGRES_URL

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PARQUET_DIR  = f"{_BASE}/data/parquet"
BATCH_SIZE   = 2000
SAMPLE_FRAC  = 0.10
TARGET_ROWS  = 2_000_000

VESSEL_TYPE_LABELS = {
    "0": "Not Available",  "30": "Fishing",         "31": "Towing",
    "32": "Towing Large",  "35": "Military",        "36": "Sailing",
    "37": "Pleasure Craft","40": "High Speed",      "50": "Pilot Vessel",
    "51": "SAR Vessel",    "52": "Tug",             "53": "Port Tender",
    "55": "Law Enforcement","60": "Passenger",      "70": "Cargo",
    "71": "Cargo Hazardous","80": "Tanker",         "81": "Tanker Hazardous",
    "89": "Tanker",        "90": "Other",
}


def clean_vessel_type(val) -> str:
    if not val or str(val).strip() in ("", "nan", "None"):
        return ""
    try:
        code = str(int(float(str(val))))
        return VESSEL_TYPE_LABELS.get(code, code)
    except (ValueError, TypeError):
        return str(val).strip()


INSERT_SQL = text("""
    INSERT INTO fact_ais_track
      (mmsi, vessel_name, vessel_type, lat, lon, lat_bin, lon_bin,
       sog, cog, heading, status, risk_level, is_anomaly,
       anomaly_score, anomaly_type, base_datetime, ingested_at, data_split)
    VALUES
      (:mmsi, :vessel_name, :vessel_type, :lat, :lon, :lat_bin, :lon_bin,
       :sog, :cog, :heading, :status, :risk_level, :is_anomaly,
       :anomaly_score, :anomaly_type, :base_datetime, :ingested_at, :data_split)
""")

COLS = [
    "mmsi", "vessel_name", "vessel_type", "lat", "lon",
    "lat_bin", "lon_bin", "sog", "cog", "heading", "status",
    "risk_level", "is_anomaly", "anomaly_score", "anomaly_type",
    "base_datetime", "ingested_at", "data_split",
]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only train split
    if "data_split" in df.columns:
        df = df[df["data_split"] == "train"].copy()
    else:
        df = df.copy()

    if df.empty:
        return df

    # 10% random sample from this file
    df = df.sample(frac=SAMPLE_FRAC, random_state=42)

    # Ensure all required columns exist
    for col in ["vessel_name", "status", "anomaly_type"]:
        if col not in df.columns:
            df[col] = ""
    if "risk_level"    not in df.columns: df["risk_level"]    = "LOW"
    if "is_anomaly"    not in df.columns: df["is_anomaly"]    = False
    if "anomaly_score" not in df.columns: df["anomaly_score"] = 0.0

    df["vessel_type"]  = df["vessel_type"].apply(clean_vessel_type)
    df["is_anomaly"]   = df["is_anomaly"].fillna(False).astype(bool)
    df["anomaly_score"]= pd.to_numeric(df["anomaly_score"], errors="coerce").fillna(0.0)
    df["sog"]          = pd.to_numeric(df["sog"],     errors="coerce").fillna(0.0)
    df["cog"]          = pd.to_numeric(df["cog"],     errors="coerce").fillna(0.0)
    df["heading"]      = pd.to_numeric(df["heading"],  errors="coerce").fillna(0.0)
    df["lat_bin"]      = df["lat"].apply(
        lambda x: round(float(x) / 0.1) * 0.1 if not pd.isna(x) else None)
    df["lon_bin"]      = df["lon"].apply(
        lambda x: round(float(x) / 0.1) * 0.1 if not pd.isna(x) else None)
    df["data_split"]   = "train"
    df["ingested_at"]  = datetime.utcnow()

    df["vessel_name"]  = df["vessel_name"].fillna("").astype(str).str[:100]
    df["anomaly_type"] = df["anomaly_type"].fillna("").astype(str).str[:100]
    df["status"]       = df["status"].fillna("").astype(str).str[:20]
    df["risk_level"]   = df["risk_level"].fillna("LOW").astype(str).str[:10]
    df["vessel_type"]  = df["vessel_type"].fillna("").astype(str).str[:50]
    df["mmsi"]         = df["mmsi"].astype(str).str.strip()

    df = df.dropna(subset=["mmsi", "base_datetime", "lat", "lon"])
    df = df[df["mmsi"] != ""]
    return df


def insert_df(df: pd.DataFrame, conn) -> int:
    if df.empty:
        return 0
    records   = df[COLS].to_dict("records")
    n_batches = math.ceil(len(records) / BATCH_SIZE)
    total     = 0

    for i in range(n_batches):
        batch = records[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        cleaned = []
        for r in batch:
            cleaned.append({
                k: (None  if isinstance(v, float) and math.isnan(v)
                    else bool(v)  if isinstance(v, np.bool_)
                    else int(v)   if isinstance(v, np.integer)
                    else float(v) if isinstance(v, np.floating)
                    else v)
                for k, v in r.items()
            })
        try:
            conn.execute(INSERT_SQL, cleaned)
            conn.commit()
            total += len(cleaned)
        except Exception as exc:
            conn.rollback()
            log.warning("Batch skipped: %s", exc)

    return total


def main():
    log.info("=== Populate fact_ais_track (file-by-file streaming) ===")

    all_files = sorted(glob.glob(f"{PARQUET_DIR}/**/*.parquet", recursive=True))
    log.info("Found %d parquet files", len(all_files))

    if not all_files:
        log.error("No parquet files found at %s", PARQUET_DIR)
        return

    random.seed(42)
    random.shuffle(all_files)

    engine = create_engine(POSTGRES_URL, pool_pre_ping=True)

    with engine.connect() as conn:
        before = conn.execute(text("SELECT COUNT(*) FROM fact_ais_track")).scalar()
    log.info("Current fact_ais_track rows: %d", before)

    total = 0

    with engine.connect() as conn:
        for idx, fpath in enumerate(all_files, 1):
            if total >= TARGET_ROWS:
                log.info("Reached target %d rows — stopping early.", TARGET_ROWS)
                break
            try:
                df = pd.read_parquet(fpath)
                df = prepare(df)
                if df.empty:
                    continue
                n      = insert_df(df, conn)
                total += n
                log.info("File %d/%d  +%d rows  total=%d  (%s)",
                         idx, len(all_files), n, total,
                         os.path.basename(fpath))
            except Exception as exc:
                log.warning("File %s failed: %s", fpath, exc)

    with engine.connect() as conn:
        final = conn.execute(text("SELECT COUNT(*) FROM fact_ais_track")).scalar()
    log.info("=== Done! fact_ais_track now has %d rows ===", final)


if __name__ == "__main__":
    main()
