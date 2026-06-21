"""
snowflake_loader.py — Maritime Navigation AI System
ETL: Gold Delta tables → Snowflake analytics warehouse.

Design principles
-----------------
* Optional/future-ready: if SNOWFLAKE_ACCOUNT is not set the module imports
  cleanly and every public function returns early with a warning. The system
  runs fully without Snowflake credentials.
* Incremental: a watermark record per table prevents re-loading the full
  history on every run. Only rows newer than the last load are sent.
* US coastal waters focus: geographic filters cover the East Coast, West
  Coast, Gulf of Mexico, and Great Lakes — not the Suez Canal corridor.
* PostgreSQL stays primary OLTP: this loader writes to Snowflake for
  analytics queries that would be expensive on transactional PostgreSQL.
  Snowflake receives a copy, not a replacement.

Dependencies (install only if using Snowflake):
    pip install snowflake-connector-python[pandas] pyarrow>=14.0.0

Run:
    python src/snowflake/snowflake_loader.py
  or import and call load_all_gold_tables() from an Airflow / Databricks job.
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── US coastal waters bounding boxes ──────────────────────────────────────────
US_ZONES = {
    "east_coast": {
        "lat_min": 24.0, "lat_max": 47.0,
        "lon_min": -82.0, "lon_max": -66.0,
        "label": "Atlantic / East Coast",
    },
    "west_coast": {
        "lat_min": 32.0, "lat_max": 49.0,
        "lon_min": -125.0, "lon_max": -117.0,
        "label": "Pacific / West Coast",
    },
    "gulf_of_mexico": {
        "lat_min": 24.0, "lat_max": 31.0,
        "lon_min": -97.0, "lon_max": -80.0,
        "label": "Gulf of Mexico",
    },
    "great_lakes": {
        "lat_min": 41.0, "lat_max": 49.0,
        "lon_min": -93.0, "lon_max": -76.0,
        "label": "Great Lakes",
    },
}

# ── Snowflake configuration ────────────────────────────────────────────────────
# All values come from environment variables — never hard-coded credentials.
# Set SNOWFLAKE_ACCOUNT='' (empty) to disable Snowflake without errors.

SF_ACCOUNT   = os.getenv("SNOWFLAKE_ACCOUNT",   "")
SF_USER      = os.getenv("SNOWFLAKE_USER",       "")
SF_PASSWORD  = os.getenv("SNOWFLAKE_PASSWORD",   "")
SF_DATABASE  = os.getenv("SNOWFLAKE_DATABASE",   "MARITIME_AIS")
SF_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE",  "MARITIME_WH")
SF_SCHEMA    = os.getenv("SNOWFLAKE_SCHEMA",     "PUBLIC")
SF_ROLE      = os.getenv("SNOWFLAKE_ROLE",       "")

# Delta / Gold table paths — read from environment to support both Docker and
# Databricks DBFS. Defaults match the docker-compose volume layout.
DELTA_ROOT       = os.getenv("DELTA_ROOT", "/delta")
GOLD_VESSEL      = os.getenv("DELTA_GOLD_VESSEL_PATH",  f"{DELTA_ROOT}/gold/vessel_latest")
GOLD_DENSITY     = os.getenv("DELTA_GOLD_DENSITY_PATH", f"{DELTA_ROOT}/gold/traffic_density")
GOLD_STATS       = os.getenv("DELTA_GOLD_STATS_PATH",   f"{DELTA_ROOT}/gold/daily_stats")

# Snowflake target table names (created by snowflake_schema.sql)
TABLE_VESSEL_LATEST   = "FACT_VESSEL_LATEST"
TABLE_TRAFFIC_DENSITY = "FACT_TRAFFIC_DENSITY"
TABLE_DAILY_STATS     = "FACT_DAILY_STATS"
TABLE_WATERMARKS      = "ETL_WATERMARKS"

BATCH_SIZE = int(os.getenv("SNOWFLAKE_BATCH_SIZE", "50000"))


def is_configured() -> bool:
    """Return True only when all required Snowflake credentials are present."""
    return bool(SF_ACCOUNT and SF_USER and SF_PASSWORD)


def _get_connection():
    """
    Open a Snowflake connection.
    Raises ImportError if snowflake-connector-python is not installed.
    Raises RuntimeError if credentials are missing.
    """
    if not is_configured():
        raise RuntimeError(
            "Snowflake credentials not configured. "
            "Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD."
        )
    try:
        import snowflake.connector
    except ImportError as exc:
        raise ImportError(
            "snowflake-connector-python is not installed. "
            "Run: pip install snowflake-connector-python[pandas]"
        ) from exc

    kwargs = dict(
        account   = SF_ACCOUNT,
        user      = SF_USER,
        password  = SF_PASSWORD,
        database  = SF_DATABASE,
        warehouse = SF_WAREHOUSE,
        schema    = SF_SCHEMA,
    )
    if SF_ROLE:
        kwargs["role"] = SF_ROLE

    return snowflake.connector.connect(**kwargs)


def _ensure_watermark_table(cur) -> None:
    """Create the ETL watermark table if it does not exist."""
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_WATERMARKS} (
            table_name  VARCHAR(100) NOT NULL PRIMARY KEY,
            last_loaded TIMESTAMP_NTZ,
            rows_loaded BIGINT       DEFAULT 0,
            updated_at  TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)


def _get_watermark(cur, table_name: str) -> Optional[datetime]:
    """Return the last successfully loaded timestamp for a given table."""
    cur.execute(
        f"SELECT last_loaded FROM {TABLE_WATERMARKS} WHERE table_name = %s",
        (table_name,)
    )
    row = cur.fetchone()
    return row[0] if row else None


def _set_watermark(cur, table_name: str, ts: datetime, rows: int) -> None:
    cur.execute(f"""
        MERGE INTO {TABLE_WATERMARKS} AS t
        USING (SELECT %s AS table_name, %s AS last_loaded, %s AS rows_loaded) AS s
        ON t.table_name = s.table_name
        WHEN MATCHED THEN UPDATE SET
            last_loaded = s.last_loaded,
            rows_loaded = t.rows_loaded + s.rows_loaded,
            updated_at  = CURRENT_TIMESTAMP()
        WHEN NOT MATCHED THEN INSERT
            (table_name, last_loaded, rows_loaded)
            VALUES (s.table_name, s.last_loaded, s.rows_loaded)
    """, (table_name, ts, rows))


def _filter_us_waters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows within US coastal waters (any zone).
    Rows with null lat/lon are kept to avoid silently dropping data.
    """
    if df.empty or "lat" not in df.columns or "lon" not in df.columns:
        return df

    null_mask = df["lat"].isna() | df["lon"].isna()
    in_any_zone = null_mask.copy()
    for zone in US_ZONES.values():
        in_any_zone = in_any_zone | (
            df["lat"].between(zone["lat_min"], zone["lat_max"])
            & df["lon"].between(zone["lon_min"], zone["lon_max"])
        )
    return df[in_any_zone].copy()


def _read_delta_as_pandas(path: str, watermark: Optional[datetime] = None,
                          ts_col: str = "updated_at") -> pd.DataFrame:
    """
    Read a Gold Delta table into a pandas DataFrame.

    Attempts delta-rs (pip install deltalake) for pure-Python Delta reads.
    Falls back to reading Parquet part files directly if delta-rs is absent.
    The watermark filter keeps only rows newer than the last load.
    """
    try:
        from deltalake import DeltaTable
        dt = DeltaTable(path)
        df = dt.to_pandas()
    except ImportError:
        # Fallback: read Parquet part files (works for Gold tables which are
        # fully overwritten each run — no need for Delta transaction log).
        import glob as _glob
        files = _glob.glob(f"{path}/**/*.parquet", recursive=True)
        if not files:
            logger.warning("No Parquet files found at %s — returning empty DataFrame", path)
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    except Exception as exc:
        logger.error("Failed to read Delta table at %s: %s", path, exc)
        return pd.DataFrame()

    if watermark and ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df[df[ts_col] > pd.Timestamp(watermark, tz="UTC")]

    return df


def _write_batch(cur, df: pd.DataFrame, table: str) -> int:
    """
    Write a DataFrame to Snowflake in batches using write_pandas.
    Returns the number of rows actually written.
    """
    if df.empty:
        return 0

    try:
        from snowflake.connector.pandas_tools import write_pandas
    except ImportError as exc:
        raise ImportError("Install snowflake-connector-python[pandas]") from exc

    # Snowflake column names must be upper-case strings
    df.columns = [c.upper() for c in df.columns]

    total = 0
    for start in range(0, len(df), BATCH_SIZE):
        chunk = df.iloc[start: start + BATCH_SIZE]
        success, nchunks, nrows, output = write_pandas(
            cur.connection, chunk, table,
            auto_create_table=False,
            overwrite=False,
            quote_identifiers=False,
        )
        if not success:
            logger.error("write_pandas failed for %s: %s", table, output)
        else:
            total += nrows
            logger.info("  wrote chunk %d rows → %s", nrows, table)
    return total


# ── Public API ─────────────────────────────────────────────────────────────────

def load_vessel_latest(conn=None) -> int:
    """
    Load latest vessel positions (Gold 1) into Snowflake FACT_VESSEL_LATEST.
    Applies US waters geographic filter before upload.
    Returns number of rows loaded (0 if Snowflake not configured).
    """
    if not is_configured():
        logger.warning("[Snowflake] Not configured — skipping vessel_latest load.")
        return 0

    close_after = conn is None
    conn = conn or _get_connection()
    cur  = conn.cursor()

    try:
        _ensure_watermark_table(cur)
        wm = _get_watermark(cur, TABLE_VESSEL_LATEST)

        df = _read_delta_as_pandas(GOLD_VESSEL, watermark=wm, ts_col="updated_at")
        df = _filter_us_waters(df)

        if df.empty:
            logger.info("[Snowflake] vessel_latest: no new rows since %s", wm)
            return 0

        rows = _write_batch(cur, df, TABLE_VESSEL_LATEST)
        _set_watermark(cur, TABLE_VESSEL_LATEST, datetime.now(timezone.utc), rows)
        conn.commit()
        logger.info("[Snowflake] vessel_latest: loaded %d rows", rows)
        return rows

    finally:
        cur.close()
        if close_after:
            conn.close()


def load_traffic_density(conn=None) -> int:
    """
    Load traffic density grid (Gold 2) into Snowflake FACT_TRAFFIC_DENSITY.
    Filters to US coastal zones (East Coast, West Coast, Gulf, Great Lakes).
    Returns number of rows loaded.
    """
    if not is_configured():
        logger.warning("[Snowflake] Not configured — skipping traffic_density load.")
        return 0

    close_after = conn is None
    conn = conn or _get_connection()
    cur  = conn.cursor()

    try:
        _ensure_watermark_table(cur)
        wm = _get_watermark(cur, TABLE_TRAFFIC_DENSITY)

        df = _read_delta_as_pandas(GOLD_DENSITY, watermark=wm, ts_col="hour_bucket")
        # Rename Gold columns to match Snowflake schema
        if "lat_grid" in df.columns:
            df = df.rename(columns={"lat_grid": "lat_bin", "lon_grid": "lon_bin"})
        df = _filter_us_waters(df)

        if df.empty:
            logger.info("[Snowflake] traffic_density: no new rows since %s", wm)
            return 0

        rows = _write_batch(cur, df, TABLE_TRAFFIC_DENSITY)
        _set_watermark(cur, TABLE_TRAFFIC_DENSITY, datetime.now(timezone.utc), rows)
        conn.commit()
        logger.info("[Snowflake] traffic_density: loaded %d rows", rows)
        return rows

    finally:
        cur.close()
        if close_after:
            conn.close()


def load_daily_stats(conn=None) -> int:
    """
    Load daily fleet statistics (Gold 3) into Snowflake FACT_DAILY_STATS.
    Returns number of rows loaded.
    """
    if not is_configured():
        logger.warning("[Snowflake] Not configured — skipping daily_stats load.")
        return 0

    close_after = conn is None
    conn = conn or _get_connection()
    cur  = conn.cursor()

    try:
        _ensure_watermark_table(cur)
        wm = _get_watermark(cur, TABLE_DAILY_STATS)

        df = _read_delta_as_pandas(GOLD_STATS, watermark=wm, ts_col="stat_date")

        if df.empty:
            logger.info("[Snowflake] daily_stats: no new rows since %s", wm)
            return 0

        rows = _write_batch(cur, df, TABLE_DAILY_STATS)
        _set_watermark(cur, TABLE_DAILY_STATS, datetime.now(timezone.utc), rows)
        conn.commit()
        logger.info("[Snowflake] daily_stats: loaded %d rows", rows)
        return rows

    finally:
        cur.close()
        if close_after:
            conn.close()


def load_all_gold_tables() -> dict:
    """
    Run all three incremental Gold → Snowflake loads in a single connection.
    Safe to call on a schedule (cron / Databricks job / Airflow DAG).

    Returns a dict with row counts per table, or empty dict if not configured.
    """
    if not is_configured():
        logger.warning(
            "[Snowflake] Credentials not set — skipping all loads. "
            "Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD to enable."
        )
        return {}

    conn = _get_connection()
    results = {}
    try:
        results["vessel_latest"]   = load_vessel_latest(conn)
        results["traffic_density"] = load_traffic_density(conn)
        results["daily_stats"]     = load_daily_stats(conn)
    finally:
        conn.close()

    total = sum(results.values())
    logger.info("[Snowflake] Load complete — %d total rows across %d tables",
                total, len(results))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if not is_configured():
        print("[Snowflake] Credentials not configured. "
              "Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD.")
        sys.exit(0)

    counts = load_all_gold_tables()
    for tbl, n in counts.items():
        print(f"  {tbl:25s}: {n:>8,} rows loaded")
