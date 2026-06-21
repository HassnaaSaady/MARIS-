"""
fetch_weather.py — Maritime Weather Fetch
Fetches hourly wind_speed_10m + wave_height from the Open-Meteo ERA5
archive for ONLY the occupied 0.5° grid cells found in the AIS Silver
data, and ONLY for the date range that data actually spans.

Per-cell parquet cache ensures idempotency (safe to re-run).
One combined bronze parquet written for downstream loaders.

Writes to: data/weather/bronze/          (sibling of data/parquet/, D:-backed)
           data/weather/bronze/cache/    (per-cell cache files)
           data/weather/bronze/weather_bronze.parquet

Usage (inside producer container):
    docker compose exec producer python src/weather/fetch_weather.py

Usage (host, with env vars pointing at host paths):
    PARQUET_DATA_PATH=/mnt/d/.../data/parquet \
    DELTA_SILVER_PATH=/path/to/delta/silver/ais_clean \
    python src/weather/fetch_weather.py
"""
from __future__ import annotations
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "common"))
sys.path.insert(0, "/app/src/common")
from config import DELTA_SILVER_PATH, PARQUET_DATA_PATH

# ── Output paths (D:-backed; derived from the configured data root) ────────────
_DATA_ROOT     = Path(PARQUET_DATA_PATH).parent   # e.g. /app/data
WEATHER_BRONZE = _DATA_ROOT / "weather" / "bronze"
CACHE_DIR      = WEATHER_BRONZE / "cache"
BRONZE_PARQUET = WEATHER_BRONZE / "weather_bronze.parquet"

# ── Open-Meteo ERA5 archive ────────────────────────────────────────────────────
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# DATA MINIMALISM: fetch ONLY the two variables needed by weather_severity.
_HOURLY_VARS = "wind_speed_10m,wave_height"

# ── Concurrency / rate limiting ───────────────────────────────────────────────
# 5 workers × 0.5 s inter-request delay ≈ 10 req/s total — within free-tier limit.
_MAX_WORKERS   = 5
_INTER_REQ_SEC = 0.5

# ── Grid resolution (must match train_congestion.py BIN_DEG) ──────────────────
_BIN_DEG = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fetch_weather] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _get_ais_cells(
    silver_path: str, parquet_dir: str
) -> tuple[list[tuple[float, float]], date, date]:
    """
    Derive the occupied 0.5° grid cells and date range from the AIS data.

    Reading strategy (Fix 2 — Delta staleness):
      1. If `deltalake` is installed: use DeltaTable.to_pandas() so the
         transaction log is respected and tombstoned files are excluded.
      2. Otherwise: fall back to pd.read_parquet with a WARNING.
         Stale/tombstoned files add at most a few extra grid cells — harmless.

    Reads only the three columns needed: lat, lon, base_datetime.
    """
    # Prefer Silver Delta; fall back to raw parquet if volume not mounted.
    use_path = silver_path if Path(silver_path).exists() else parquet_dir
    if use_path == parquet_dir:
        log.warning(
            "Silver Delta not found at %s — reading raw parquet from %s",
            silver_path, parquet_dir,
        )

    try:
        from deltalake import DeltaTable
        df = DeltaTable(use_path).to_pandas(
            columns=["lat", "lon", "base_datetime"]
        )
        log.info(
            "Read AIS positions via DeltaTable (transaction-log safe) from %s",
            use_path,
        )
    except ImportError:
        log.warning(
            "deltalake not installed — using pd.read_parquet; may include "
            "tombstoned/stale Delta files (harmless: adds at most a few extra cells)"
        )
        df = pd.read_parquet(use_path, columns=["lat", "lon", "base_datetime"])

    df["base_datetime"] = pd.to_datetime(df["base_datetime"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "base_datetime"])

    # 0.5° floor binning — must match train_congestion.py BIN_DEG logic.
    lat_bins = np.floor(df["lat"].values / _BIN_DEG) * _BIN_DEG
    lon_bins = np.floor(df["lon"].values / _BIN_DEG) * _BIN_DEG

    cells: list[tuple[float, float]] = sorted({
        (round(float(la), 1), round(float(lo), 1))
        for la, lo in zip(lat_bins, lon_bins)
    })

    start = df["base_datetime"].min().date()
    end   = df["base_datetime"].max().date()

    log.info("AIS cells   : %d unique 0.5° cells", len(cells))
    log.info("Date range  : %s → %s", start, end)
    return cells, start, end


def _json_to_df(data: dict, lat: float, lon: float) -> pd.DataFrame:
    """
    Parse one Open-Meteo JSON response → tidy DataFrame.

    Columns produced:
        time            datetime64[us]  — hourly timestamps (UTC)
        wind_speed_10m  float64         — wind speed at 10 m, m/s
        wave_height     float64         — significant wave height, m (NaN inland)
        lat_bin         float64         — 0.5° cell lat (floor-binned)
        lon_bin         float64         — 0.5° cell lon (floor-binned)

    Fix 1 — TIMESTAMP_NTZ cast:
        df["time"] is cast to datetime64[us] so the bronze parquet is
        Spark 3.4 TIMESTAMP_NTZ-safe and can be read directly into Delta Lake
        without the datetime64[us, UTC] → TIMESTAMP_LTZ promotion that causes
        schema conflicts in silver_job.py.
    """
    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    n = len(times)
    df = pd.DataFrame({
        "time": pd.to_datetime(times, utc=False),
        "wind_speed_10m": pd.to_numeric(
            pd.Series(hourly.get("wind_speed_10m") or ([None] * n)),
            errors="coerce",
        ),
        "wave_height": pd.to_numeric(
            pd.Series(hourly.get("wave_height") or ([None] * n)),
            errors="coerce",
        ),
    })

    # Fix 1: cast to datetime64[us] — Spark 3.4 TIMESTAMP_NTZ-safe.
    df["time"] = df["time"].astype("datetime64[us]")

    df["lat_bin"] = round(lat, 1)
    df["lon_bin"] = round(lon, 1)
    return df


def _fetch_cell(
    lat: float,
    lon: float,
    start: date,
    end: date,
    session: requests.Session,
) -> pd.DataFrame:
    """
    Fetch one 0.5° grid cell from Open-Meteo.
    Returns the cached parquet if the cell was already fetched (idempotent).
    """
    cache_file = CACHE_DIR / f"lat{lat:+.1f}_lon{lon:+.1f}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    # Inter-request delay — keeps throughput within free-tier rate limits.
    time.sleep(_INTER_REQ_SEC)

    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      start.isoformat(),
        "end_date":        end.isoformat(),
        "hourly":          _HOURLY_VARS,
        # m/s so normalization constant 24.5 m/s (Beaufort 9) is in native units.
        "wind_speed_unit": "ms",
        "timezone":        "UTC",
    }
    resp = session.get(_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    df = _json_to_df(resp.json(), lat, lon)
    if not df.empty:
        df.to_parquet(cache_file, index=False)
    return df


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    WEATHER_BRONZE.mkdir(parents=True, exist_ok=True)

    cells, start, end = _get_ais_cells(DELTA_SILVER_PATH, PARQUET_DATA_PATH)

    log.info(
        "Fetching %d cells for %s → %s (cache at %s) …",
        len(cells), start, end, CACHE_DIR,
    )

    frames: list[pd.DataFrame] = []
    session = _make_session()
    failed  = 0

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_cell, lat, lon, start, end, session): (lat, lon)
            for lat, lon in cells
        }
        done = 0
        for fut in as_completed(futures):
            lat, lon = futures[fut]
            done += 1
            try:
                df = fut.result()
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                failed += 1
                log.warning("Cell (%.1f, %.1f) failed: %s", lat, lon, exc)

            if done % 200 == 0 or done == len(cells):
                log.info(
                    "  Progress: %d / %d cells  (%d failed)", done, len(cells), failed
                )

    if not frames:
        log.error(
            "No weather data fetched — check network connectivity and date range."
        )
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(BRONZE_PARQUET, index=False)
    log.info(
        "Bronze parquet written: %s  (%d rows, %d cells, %d failed)",
        BRONZE_PARQUET, len(combined), len(frames), failed,
    )


if __name__ == "__main__":
    main()
