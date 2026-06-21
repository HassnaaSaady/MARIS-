"""
snowflake_queries.py — Maritime Navigation AI System
Named analytics queries against the Snowflake warehouse.

All queries target US coastal waters monitoring:
  - East Coast (Atlantic): lon -82 to -66
  - West Coast (Pacific):  lon -125 to -117
  - Gulf of Mexico:        lon -97 to -80
  - Great Lakes:           lon -93 to -76

Every public function:
  - Returns a pandas DataFrame on success
  - Returns an empty DataFrame (with expected columns) if Snowflake is not
    configured — allowing the dashboard to render gracefully with no data
    rather than crashing.
  - Accepts an optional `conn` parameter for connection reuse across queries.

Dependencies: snowflake-connector-python[pandas]
"""

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SF_DATABASE  = os.getenv("SNOWFLAKE_DATABASE", "MARITIME_AIS")
SF_SCHEMA    = os.getenv("SNOWFLAKE_SCHEMA",   "PUBLIC")


def _is_configured() -> bool:
    return bool(
        os.getenv("SNOWFLAKE_ACCOUNT")
        and os.getenv("SNOWFLAKE_USER")
        and os.getenv("SNOWFLAKE_PASSWORD")
    )


def _get_conn():
    """Return a Snowflake connection using environment credentials."""
    try:
        import snowflake.connector
    except ImportError as exc:
        raise ImportError(
            "snowflake-connector-python is not installed. "
            "Run: pip install snowflake-connector-python[pandas]"
        ) from exc
    return snowflake.connector.connect(
        account   = os.environ["SNOWFLAKE_ACCOUNT"],
        user      = os.environ["SNOWFLAKE_USER"],
        password  = os.environ["SNOWFLAKE_PASSWORD"],
        database  = SF_DATABASE,
        warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "MARITIME_WH"),
        schema    = SF_SCHEMA,
        role      = os.getenv("SNOWFLAKE_ROLE", ""),
    )


def _run(sql: str, conn=None, params: tuple = ()) -> pd.DataFrame:
    """Execute SQL and return results as a DataFrame. Handles connection lifecycle."""
    close_after = conn is None
    conn = conn or _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        cols = [d[0].lower() for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        if close_after:
            conn.close()


# ── Query 1: Busiest shipping lanes ───────────────────────────────────────────

BUSIEST_LANES_SQL = """
SELECT
    ROUND(LAT_BIN, 1)                        AS lat_grid,
    ROUND(LON_BIN, 1)                        AS lon_grid,
    SUM(VESSEL_COUNT)                        AS total_vessels,
    SUM(UNIQUE_VESSELS)                      AS unique_vessels,
    ROUND(AVG(AVG_SOG), 2)                   AS avg_speed_kn,
    MAX(CONGESTION_LEVEL)                    AS peak_congestion,
    CASE
        WHEN LON_BIN BETWEEN -82  AND -66  THEN 'East Coast'
        WHEN LON_BIN BETWEEN -125 AND -117 THEN 'West Coast'
        WHEN LON_BIN BETWEEN -97  AND -80
         AND LAT_BIN BETWEEN 24   AND 31   THEN 'Gulf of Mexico'
        WHEN LON_BIN BETWEEN -93  AND -76
         AND LAT_BIN BETWEEN 41   AND 49   THEN 'Great Lakes'
        ELSE 'Other US Waters'
    END                                      AS us_region,
    COUNT(DISTINCT DATE_TRUNC('day', HOUR_BUCKET)) AS active_days
FROM FACT_TRAFFIC_DENSITY
WHERE
    -- US coastal waters bounding box (config.py US_WATERS)
    LAT_BIN BETWEEN 24.0 AND 49.0
    AND LON_BIN BETWEEN -125.0 AND -66.0
GROUP BY 1, 2, 7
ORDER BY total_vessels DESC
LIMIT %(limit)s
"""


def get_busiest_lanes(top_n: int = 50, conn=None) -> pd.DataFrame:
    """
    Return the busiest 0.1° grid cells in US coastal waters by vessel count.
    Suitable for rendering as a traffic heat-map overlay.
    """
    if not _is_configured():
        logger.warning("[Snowflake] Not configured — busiest_lanes returns empty.")
        return pd.DataFrame(columns=[
            "lat_grid", "lon_grid", "total_vessels", "unique_vessels",
            "avg_speed_kn", "peak_congestion", "us_region", "active_days",
        ])
    sql = BUSIEST_LANES_SQL.replace("%(limit)s", str(int(top_n)))
    return _run(sql, conn)


# ── Query 2: Anomaly trends over time ─────────────────────────────────────────

ANOMALY_TRENDS_SQL = """
SELECT
    DATE_TRUNC('day', BASE_DATETIME)         AS event_date,
    ANOMALY_TYPE,
    COUNT(*)                                 AS anomaly_count,
    ROUND(AVG(ANOMALY_SCORE), 4)             AS avg_score,
    COUNT(DISTINCT MMSI)                     AS unique_vessels,
    CASE
        WHEN LON BETWEEN -82  AND -66  THEN 'East Coast'
        WHEN LON BETWEEN -125 AND -117 THEN 'West Coast'
        WHEN LON BETWEEN -97  AND -80
         AND LAT BETWEEN 24   AND 31   THEN 'Gulf of Mexico'
        WHEN LON BETWEEN -93  AND -76
         AND LAT BETWEEN 41   AND 49   THEN 'Great Lakes'
        ELSE 'Other US Waters'
    END                                      AS us_region
FROM FACT_AIS_TRACK
WHERE
    IS_ANOMALY = TRUE
    AND LAT BETWEEN 24.0 AND 49.0
    AND LON BETWEEN -125.0 AND -66.0
    AND BASE_DATETIME >= DATEADD(day, %(days_back)s, CURRENT_DATE())
GROUP BY 1, 2, 6
ORDER BY 1 DESC, anomaly_count DESC
"""


def get_anomaly_trends(days_back: int = 30, conn=None) -> pd.DataFrame:
    """
    Return daily anomaly counts by type and US region for the last N days.
    Useful for time-series charts showing anomaly spikes.
    """
    if not _is_configured():
        logger.warning("[Snowflake] Not configured — anomaly_trends returns empty.")
        return pd.DataFrame(columns=[
            "event_date", "anomaly_type", "anomaly_count",
            "avg_score", "unique_vessels", "us_region",
        ])
    sql = ANOMALY_TRENDS_SQL.replace("%(days_back)s", str(-abs(days_back)))
    return _run(sql, conn)


# ── Query 3: Congestion by hour of day ────────────────────────────────────────

CONGESTION_BY_HOUR_SQL = """
SELECT
    HOUR(HOUR_BUCKET)                        AS hour_of_day,
    CONGESTION_LEVEL,
    COUNT(*)                                 AS grid_cells,
    ROUND(AVG(VESSEL_COUNT), 1)              AS avg_vessels_per_cell,
    ROUND(AVG(AVG_SOG), 2)                   AS avg_speed_kn,
    SUM(STOPPED_COUNT)                       AS total_stopped_vessels,
    CASE
        WHEN LON_BIN BETWEEN -82  AND -66  THEN 'East Coast'
        WHEN LON_BIN BETWEEN -125 AND -117 THEN 'West Coast'
        WHEN LON_BIN BETWEEN -97  AND -80
         AND LAT_BIN BETWEEN 24   AND 31   THEN 'Gulf of Mexico'
        WHEN LON_BIN BETWEEN -93  AND -76
         AND LAT_BIN BETWEEN 41   AND 49   THEN 'Great Lakes'
        ELSE 'Other US Waters'
    END                                      AS us_region
FROM FACT_TRAFFIC_DENSITY
WHERE
    LAT_BIN BETWEEN 24.0 AND 49.0
    AND LON_BIN BETWEEN -125.0 AND -66.0
    AND HOUR_BUCKET >= DATEADD(day, -%(days_back)s, CURRENT_DATE())
GROUP BY 1, 2, 7
ORDER BY 7, 1, 2
"""


def get_congestion_by_hour(days_back: int = 14, conn=None) -> pd.DataFrame:
    """
    Return congestion patterns broken down by hour of day and US region.
    Reveals peak traffic hours for port operations planning.
    """
    if not _is_configured():
        logger.warning("[Snowflake] Not configured — congestion_by_hour returns empty.")
        return pd.DataFrame(columns=[
            "hour_of_day", "congestion_level", "grid_cells",
            "avg_vessels_per_cell", "avg_speed_kn",
            "total_stopped_vessels", "us_region",
        ])
    sql = CONGESTION_BY_HOUR_SQL.replace("%(days_back)s", str(abs(days_back)))
    return _run(sql, conn)


# ── Query 4: Top high-risk vessels ────────────────────────────────────────────

TOP_RISKY_VESSELS_SQL = """
SELECT
    v.MMSI,
    v.VESSEL_NAME,
    v.VESSEL_TYPE_LABEL,
    v.LENGTH,
    v.DRAFT,
    t.RISK_COUNT,
    t.AVG_RISK_SOG,
    t.LAST_RISK_EVENT,
    t.PREDOMINANT_RISK_ZONE,
    f.LAT         AS last_lat,
    f.LON         AS last_lon,
    f.SOG         AS last_sog
FROM (
    SELECT
        MMSI,
        COUNT(*)                                    AS risk_count,
        ROUND(AVG(SOG), 2)                          AS avg_risk_sog,
        MAX(BASE_DATETIME)                          AS last_risk_event,
        -- Dominant zone for this vessel's risk events
        MODE(
            CASE
                WHEN LON BETWEEN -82  AND -66  THEN 'East Coast'
                WHEN LON BETWEEN -125 AND -117 THEN 'West Coast'
                WHEN LON BETWEEN -97  AND -80
                 AND LAT BETWEEN 24   AND 31   THEN 'Gulf of Mexico'
                WHEN LON BETWEEN -93  AND -76
                 AND LAT BETWEEN 41   AND 49   THEN 'Great Lakes'
                ELSE 'Other US Waters'
            END
        )                                           AS predominant_risk_zone
    FROM FACT_AIS_TRACK
    WHERE
        RISK_LEVEL = 'HIGH'
        AND LAT BETWEEN 24.0 AND 49.0
        AND LON BETWEEN -125.0 AND -66.0
        AND BASE_DATETIME >= DATEADD(day, -%(days_back)s, CURRENT_DATE())
    GROUP BY MMSI
    ORDER BY risk_count DESC
    LIMIT %(limit)s
) t
JOIN DIM_VESSEL v ON v.MMSI = t.MMSI
JOIN FACT_VESSEL_LATEST f ON f.MMSI = t.MMSI
ORDER BY t.risk_count DESC
"""


def get_top_risky_vessels(top_n: int = 25, days_back: int = 30,
                          conn=None) -> pd.DataFrame:
    """
    Return the vessels with the highest count of HIGH-risk AIS records
    in US coastal waters over the last N days.
    """
    if not _is_configured():
        logger.warning("[Snowflake] Not configured — top_risky_vessels returns empty.")
        return pd.DataFrame(columns=[
            "mmsi", "vessel_name", "vessel_type_label", "length", "draft",
            "risk_count", "avg_risk_sog", "last_risk_event",
            "predominant_risk_zone", "last_lat", "last_lon", "last_sog",
        ])
    sql = (TOP_RISKY_VESSELS_SQL
           .replace("%(days_back)s", str(abs(days_back)))
           .replace("%(limit)s",    str(int(top_n))))
    return _run(sql, conn)


# ── Query 5: Collision risk statistics ────────────────────────────────────────

COLLISION_STATS_SQL = """
SELECT
    DATE_TRUNC('week', CREATED_AT)           AS week_start,
    SEVERITY,
    COUNT(*)                                 AS collision_alerts,
    COUNT(DISTINCT MMSI_1)                   AS vessels_involved,
    ROUND(AVG(DISTANCE_NM), 3)               AS avg_cpa_nm,
    SUM(CASE WHEN IS_RESOLVED THEN 1 ELSE 0 END)   AS resolved,
    SUM(CASE WHEN NOT IS_RESOLVED THEN 1 ELSE 0 END) AS outstanding,
    CASE
        WHEN LON BETWEEN -82  AND -66  THEN 'East Coast'
        WHEN LON BETWEEN -125 AND -117 THEN 'West Coast'
        WHEN LON BETWEEN -97  AND -80
         AND LAT BETWEEN 24   AND 31   THEN 'Gulf of Mexico'
        WHEN LON BETWEEN -93  AND -76
         AND LAT BETWEEN 41   AND 49   THEN 'Great Lakes'
        ELSE 'Other US Waters'
    END                                      AS us_region
FROM FACT_ALERTS
WHERE
    ALERT_TYPE = 'COLLISION_RISK'
    AND LAT BETWEEN 24.0 AND 49.0
    AND LON BETWEEN -125.0 AND -66.0
    AND CREATED_AT >= DATEADD(week, -%(weeks_back)s, CURRENT_DATE())
GROUP BY 1, 2, 8
ORDER BY 1 DESC, collision_alerts DESC
"""


def get_collision_stats(weeks_back: int = 8, conn=None) -> pd.DataFrame:
    """
    Return weekly collision-risk alert statistics by severity and US region.
    Shows trends in near-miss incidents useful for coast guard reporting.
    """
    if not _is_configured():
        logger.warning("[Snowflake] Not configured — collision_stats returns empty.")
        return pd.DataFrame(columns=[
            "week_start", "severity", "collision_alerts", "vessels_involved",
            "avg_cpa_nm", "resolved", "outstanding", "us_region",
        ])
    sql = COLLISION_STATS_SQL.replace("%(weeks_back)s", str(abs(weeks_back)))
    return _run(sql, conn)


# ── Query 6: Fleet summary KPIs (single-row snapshot) ─────────────────────────

FLEET_SUMMARY_SQL = """
SELECT
    COUNT(DISTINCT MMSI)                                     AS total_vessels,
    COUNT(DISTINCT CASE WHEN RISK_LEVEL='HIGH'  THEN MMSI END) AS high_risk_vessels,
    COUNT(DISTINCT CASE WHEN IS_ANOMALY         THEN MMSI END) AS anomalous_vessels,
    ROUND(AVG(SOG), 2)                                       AS fleet_avg_speed_kn,
    COUNT(DISTINCT VESSEL_TYPE)                              AS vessel_type_diversity,
    -- East Coast vs West Coast split
    COUNT(DISTINCT CASE WHEN LON BETWEEN -82 AND -66  THEN MMSI END) AS east_coast_vessels,
    COUNT(DISTINCT CASE WHEN LON BETWEEN -125 AND -117 THEN MMSI END) AS west_coast_vessels,
    COUNT(DISTINCT CASE WHEN LON BETWEEN -97 AND -80
                          AND LAT BETWEEN 24  AND 31  THEN MMSI END) AS gulf_vessels,
    MAX(UPDATED_AT)                                          AS data_freshness
FROM FACT_VESSEL_LATEST
WHERE
    LAT BETWEEN 24.0 AND 49.0
    AND LON BETWEEN -125.0 AND -66.0
"""


def get_fleet_summary(conn=None) -> dict:
    """
    Return a single-row KPI snapshot of the current US coastal fleet.
    Used as the top-line metrics strip in the Snowflake analytics page.
    Returns a dict (not DataFrame) for easy key-access.
    """
    empty = {
        "total_vessels": 0, "high_risk_vessels": 0, "anomalous_vessels": 0,
        "fleet_avg_speed_kn": 0.0, "vessel_type_diversity": 0,
        "east_coast_vessels": 0, "west_coast_vessels": 0,
        "gulf_vessels": 0, "data_freshness": None,
        "snowflake_available": False,
    }
    if not _is_configured():
        return empty
    try:
        df = _run(FLEET_SUMMARY_SQL, conn)
        if df.empty:
            return empty
        row = df.iloc[0].to_dict()
        row["snowflake_available"] = True
        return row
    except Exception as exc:
        logger.error("[Snowflake] fleet_summary failed: %s", exc)
        return empty
