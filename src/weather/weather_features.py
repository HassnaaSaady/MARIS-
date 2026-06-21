"""
weather_features.py — Weather Feature Builder
Additive module: adds weather_severity to congestion density DataFrames.

Does NOT modify train_congestion.py or any existing feature generator.

Functions
---------
build_congestion_with_weather(density, pg_url=None) -> pd.DataFrame
    LEFT JOIN weather_severity from PostgreSQL dim_weather onto the density
    DataFrame produced by load_and_aggregate() (train_congestion.py).
    All density rows are preserved; unmatched rows get weather_severity=NaN.
"""
from __future__ import annotations
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, "/app/src/common")
from config import POSTGRES_URL

log = logging.getLogger(__name__)


def build_congestion_with_weather(
    density: pd.DataFrame,
    pg_url: str | None = None,
) -> pd.DataFrame:
    """
    Augment the density grid DataFrame with weather_severity via a LEFT JOIN
    against PostgreSQL dim_weather on (lat_bin, lon_bin, hour_bucket).

    Parameters
    ----------
    density : pd.DataFrame
        Output of load_and_aggregate() from train_congestion.py.
        Must have columns: lat_bin, lon_bin, hour_bucket.
    pg_url : str, optional
        SQLAlchemy connection URL.  Defaults to POSTGRES_URL from config.

    Returns
    -------
    pd.DataFrame
        density with 'weather_severity' column appended.
        Rows without a matching dim_weather record have weather_severity=NaN.
        The caller decides how to handle NaN (eval_weather.py fills with 0.0).

    Notes
    -----
    JOIN keys are rounded to 1 decimal place before matching to avoid IEEE-754
    float equality mismatches between parquet-loaded and DB-returned values.
    For example 29.500000000001 and 29.5 both become 29.5 after rounding.
    """
    from sqlalchemy import create_engine

    url    = pg_url or POSTGRES_URL
    engine = create_engine(url, pool_pre_ping=True)

    try:
        weather = pd.read_sql(
            "SELECT lat_bin, lon_bin, hour_bucket, weather_severity FROM dim_weather",
            engine,
        )
    except Exception as exc:
        log.warning(
            "Could not query dim_weather (%s) — weather_severity will be all NaN. "
            "Run load_dim_weather.py first.",
            exc,
        )
        density = density.copy()
        density["weather_severity"] = float("nan")
        return density

    if weather.empty:
        log.warning(
            "dim_weather is empty — weather_severity will be all NaN. "
            "Run fetch_weather.py then load_dim_weather.py first."
        )
        density = density.copy()
        density["weather_severity"] = float("nan")
        return density

    # Normalise types before joining.
    weather["hour_bucket"] = pd.to_datetime(weather["hour_bucket"])
    weather["lat_bin"]     = weather["lat_bin"].round(1)
    weather["lon_bin"]     = weather["lon_bin"].round(1)

    density = density.copy()
    density["lat_bin"]     = density["lat_bin"].round(1)
    density["lon_bin"]     = density["lon_bin"].round(1)
    density["hour_bucket"] = pd.to_datetime(density["hour_bucket"])

    merged = density.merge(
        weather[["lat_bin", "lon_bin", "hour_bucket", "weather_severity"]],
        on=["lat_bin", "lon_bin", "hour_bucket"],
        how="left",
    )

    n_matched = int(merged["weather_severity"].notna().sum())
    n_total   = len(merged)
    log.info(
        "Weather join: %d / %d density rows matched (%.1f%%)",
        n_matched, n_total, 100.0 * n_matched / max(n_total, 1),
    )
    return merged
