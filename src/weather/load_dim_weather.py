"""
load_dim_weather.py — Maritime Weather Loader
Reads the bronze weather parquet produced by fetch_weather.py,
computes weather_severity, and UPSERTs into PostgreSQL dim_weather.

Idempotent: runs repeatedly without creating duplicates.

DDL (applied automatically on first run):
    CREATE TABLE IF NOT EXISTS dim_weather (
        lat_bin          FLOAT     NOT NULL,
        lon_bin          FLOAT     NOT NULL,
        hour_bucket      TIMESTAMP NOT NULL,
        wind_speed_10m   FLOAT,
        wave_height      FLOAT,
        weather_severity FLOAT     NOT NULL,
        updated_at       TIMESTAMP DEFAULT NOW(),
        PRIMARY KEY (lat_bin, lon_bin, hour_bucket)
    );
    CREATE INDEX IF NOT EXISTS ix_weather_grid_time
        ON dim_weather (hour_bucket, lat_bin, lon_bin);

weather_severity definition (canonical — do not alter):
    wind_norm = clip(wind_speed_10m / 24.5, 0, 1)   # 24.5 m/s ≈ Beaufort 9
    wave_norm = clip(wave_height   / 6.0,  0, 1)    # 6 m ≈ very rough sea
    if wave_height is null (inland): severity = wind_norm
    else:                            severity = 0.6 * wind_norm + 0.4 * wave_norm

Usage:
    docker compose exec producer python src/weather/load_dim_weather.py
"""
from __future__ import annotations
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, "/app/src/common")
from config import PARQUET_DATA_PATH, POSTGRES_URL

_DATA_ROOT     = Path(PARQUET_DATA_PATH).parent
BRONZE_PARQUET = _DATA_ROOT / "weather" / "bronze" / "weather_bronze.parquet"

BATCH_SIZE = 5_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [load_dim_weather] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── DDL ───────────────────────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS dim_weather (
    lat_bin          FLOAT     NOT NULL,
    lon_bin          FLOAT     NOT NULL,
    hour_bucket      TIMESTAMP NOT NULL,
    wind_speed_10m   FLOAT,
    wave_height      FLOAT,
    weather_severity FLOAT     NOT NULL,
    updated_at       TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (lat_bin, lon_bin, hour_bucket)
);
CREATE INDEX IF NOT EXISTS ix_weather_grid_time
    ON dim_weather (hour_bucket, lat_bin, lon_bin);
"""

# ── UPSERT ────────────────────────────────────────────────────────────────────
_UPSERT = text("""
    INSERT INTO dim_weather
        (lat_bin, lon_bin, hour_bucket, wind_speed_10m, wave_height,
         weather_severity, updated_at)
    VALUES
        (:lat_bin, :lon_bin, :hour_bucket, :wind_speed_10m, :wave_height,
         :weather_severity, NOW())
    ON CONFLICT (lat_bin, lon_bin, hour_bucket) DO UPDATE SET
        wind_speed_10m   = EXCLUDED.wind_speed_10m,
        wave_height      = EXCLUDED.wave_height,
        weather_severity = EXCLUDED.weather_severity,
        updated_at       = NOW()
""")


def _compute_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather_severity column in-place.

    Canonical formula:
        wind_norm = clip(wind_speed_10m / 24.5, 0, 1)
        wave_norm = clip(wave_height   / 6.0,  0, 1)
        severity  = wind_norm              if wave_height is null
                  = 0.6*wind_norm + 0.4*wave_norm  otherwise
    """
    wind_norm = (df["wind_speed_10m"] / 24.5).clip(0.0, 1.0)
    wave_norm = (df["wave_height"]    / 6.0 ).clip(0.0, 1.0)

    # Wave_height NaN ≡ inland cell: severity comes from wind only.
    df["weather_severity"] = np.where(
        df["wave_height"].isna(),
        wind_norm,
        0.6 * wind_norm + 0.4 * wave_norm,
    )
    # Guard: if wind is also missing, default to 0.
    df["weather_severity"] = df["weather_severity"].fillna(0.0)
    return df


def main() -> None:
    if not BRONZE_PARQUET.exists():
        log.error(
            "Bronze parquet not found at %s — run fetch_weather.py first.",
            BRONZE_PARQUET,
        )
        sys.exit(1)

    log.info("Reading bronze parquet: %s", BRONZE_PARQUET)
    df = pd.read_parquet(BRONZE_PARQUET)
    log.info("  Rows read: %d", len(df))

    # Floor-truncate to hour → join key used by weather_features.py / eval_weather.py.
    df["hour_bucket"] = df["time"].dt.floor("h")

    # Aggregate multiple API rows to single hour (should already be hourly, but guard).
    df = (
        df.groupby(["lat_bin", "lon_bin", "hour_bucket"], as_index=False)
        .agg(
            wind_speed_10m=("wind_speed_10m", "mean"),
            wave_height   =("wave_height",    "mean"),
        )
    )
    log.info("  Rows after hour-level groupby: %d", len(df))

    df = _compute_severity(df)

    sev_min = df["weather_severity"].min()
    sev_max = df["weather_severity"].max()
    sev_mean = df["weather_severity"].mean()
    log.info(
        "  weather_severity  min=%.3f  mean=%.3f  max=%.3f",
        sev_min, sev_mean, sev_max,
    )

    engine = create_engine(POSTGRES_URL, pool_pre_ping=True)

    with engine.connect() as conn:
        # Apply DDL idempotently.
        for stmt in _DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()
        log.info("DDL applied (dim_weather table and index ensured).")

        # UPSERT in batches.
        records = df[[
            "lat_bin", "lon_bin", "hour_bucket",
            "wind_speed_10m", "wave_height", "weather_severity",
        ]].to_dict("records")

        # Coerce numpy types → Python natives for psycopg2.
        def _clean(r: dict) -> dict:
            out: dict = {}
            for k, v in r.items():
                if isinstance(v, float) and np.isnan(v):
                    out[k] = None
                elif isinstance(v, (np.integer,)):
                    out[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    out[k] = float(v)
                elif isinstance(v, pd.Timestamp):
                    out[k] = v.to_pydatetime()
                else:
                    out[k] = v
            return out

        total = 0
        for i in range(0, len(records), BATCH_SIZE):
            batch = [_clean(r) for r in records[i : i + BATCH_SIZE]]
            conn.execute(_UPSERT, batch)
            conn.commit()
            total += len(batch)
            log.info("  UPSERTed %d / %d rows …", total, len(records))

    log.info("Done. %d rows in dim_weather.", total)


if __name__ == "__main__":
    main()
