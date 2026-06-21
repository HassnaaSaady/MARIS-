-- =============================================================================
-- snowflake_schema.sql — Maritime Navigation AI System
-- Snowflake analytics warehouse schema for US coastal waters monitoring.
--
-- Data flow: PostgreSQL (live OLTP) → Snowflake (analytics OLAP)
--   The ETL is performed by src/snowflake/snowflake_loader.py.
--   These tables are append/merge targets — never the system of record.
--   PostgreSQL remains the authoritative live database; Snowflake receives
--   a copy optimised for analytical queries (window functions, aggregations,
--   large scans) that would degrade transactional PostgreSQL performance.
--
-- Run order:
--   1. Create database and warehouse manually in the Snowflake UI or CLI
--   2. Execute this script once:
--        snowsql -a <account> -u <user> -f src/snowflake/snowflake_schema.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Database / schema / warehouse
-- Idempotent — safe to re-run.
-- ---------------------------------------------------------------------------

CREATE DATABASE IF NOT EXISTS MARITIME_AIS;
USE DATABASE MARITIME_AIS;

CREATE SCHEMA IF NOT EXISTS PUBLIC;
USE SCHEMA PUBLIC;

CREATE WAREHOUSE IF NOT EXISTS MARITIME_WH
    WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND   = 60         -- suspend after 60 s idle (cost control)
    AUTO_RESUME    = TRUE
    INITIALLY_SUSPENDED = TRUE;

USE WAREHOUSE MARITIME_WH;


-- ---------------------------------------------------------------------------
-- ETL watermark table
-- Tracks the last successfully loaded timestamp per target table so the
-- loader only ships incremental rows on each run.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ETL_WATERMARKS (
    TABLE_NAME   VARCHAR(100)   NOT NULL PRIMARY KEY,
    LAST_LOADED  TIMESTAMP_NTZ,
    ROWS_LOADED  NUMBER(18, 0)  DEFAULT 0,
    UPDATED_AT   TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP()
);


-- =============================================================================
-- DIMENSION TABLES
-- =============================================================================

-- ---------------------------------------------------------------------------
-- DIM_VESSEL
-- One row per unique MMSI.  Slowly-changing vessel reference data.
-- Source: PostgreSQL dim_vessel (populated by gold_job.py).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS DIM_VESSEL (
    MMSI               VARCHAR(20)   NOT NULL PRIMARY KEY,
    VESSEL_NAME        VARCHAR(100),
    IMO                VARCHAR(20),
    CALL_SIGN          VARCHAR(20),
    VESSEL_TYPE        VARCHAR(50),
    VESSEL_TYPE_LABEL  VARCHAR(100),
    LENGTH             FLOAT,
    WIDTH              FLOAT,
    DRAFT              FLOAT,
    CARGO              VARCHAR(50),
    TRANSCEIVER_CLASS  VARCHAR(10),
    DATA_SPLIT         VARCHAR(20),
    FIRST_SEEN         TIMESTAMP_NTZ,
    LAST_SEEN          TIMESTAMP_NTZ,
    TOTAL_RECORDS      NUMBER(18, 0) DEFAULT 0,
    CREATED_AT         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    UPDATED_AT         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
CLUSTER BY (VESSEL_TYPE);

COMMENT ON TABLE  DIM_VESSEL IS 'Vessel reference dimension. One row per MMSI.';
COMMENT ON COLUMN DIM_VESSEL.DATA_SPLIT IS 'TRAIN / VALID / TEST / LIVE — ML partition label.';


-- =============================================================================
-- FACT TABLES
-- =============================================================================

-- ---------------------------------------------------------------------------
-- FACT_AIS_TRACK
-- Full position history for every vessel in US coastal waters.
-- Source: PostgreSQL fact_ais_track.
-- Clustered by (MMSI, BASE_DATETIME) for vessel-timeline queries.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS FACT_AIS_TRACK (
    ID             NUMBER(18, 0)  AUTOINCREMENT PRIMARY KEY,
    MMSI           VARCHAR(20)    NOT NULL,
    VESSEL_NAME    VARCHAR(100),
    VESSEL_TYPE    VARCHAR(50),
    LAT            FLOAT          NOT NULL,
    LON            FLOAT          NOT NULL,
    LAT_BIN        FLOAT,                      -- 0.1° grid cell
    LON_BIN        FLOAT,
    SOG            FLOAT,                      -- speed over ground (knots)
    COG            FLOAT,                      -- course over ground (degrees)
    HEADING        FLOAT,
    STATUS         VARCHAR(20),
    RISK_LEVEL     VARCHAR(10),                -- HIGH / MEDIUM / LOW
    IS_ANOMALY     BOOLEAN        DEFAULT FALSE,
    ANOMALY_SCORE  FLOAT          DEFAULT 0.0,
    ANOMALY_TYPE   VARCHAR(100),
    BASE_DATETIME  TIMESTAMP_NTZ  NOT NULL,
    INGESTED_AT    TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP(),
    DATA_SPLIT     VARCHAR(20)    NOT NULL DEFAULT 'train',
    -- US region label derived at load time
    US_REGION      VARCHAR(50)
        AS (
            CASE
                WHEN LON BETWEEN -82  AND -66  THEN 'East Coast'
                WHEN LON BETWEEN -125 AND -117 THEN 'West Coast'
                WHEN LON BETWEEN -97  AND -80
                 AND LAT BETWEEN 24   AND 31   THEN 'Gulf of Mexico'
                WHEN LON BETWEEN -93  AND -76
                 AND LAT BETWEEN 41   AND 49   THEN 'Great Lakes'
                ELSE 'Other US Waters'
            END
        ) VIRTUAL
)
CLUSTER BY (MMSI, DATE_TRUNC('day', BASE_DATETIME));

COMMENT ON TABLE FACT_AIS_TRACK IS
    'Full AIS position history. US coastal waters only. '
    'Supports vessel replay, anomaly analysis, collision risk queries.';


-- ---------------------------------------------------------------------------
-- FACT_VESSEL_LATEST
-- Most recent known position per MMSI.
-- Source: PostgreSQL fact_vessel_latest.
-- Refreshed in full on each ETL run (small table — ~tens of thousands of rows).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS FACT_VESSEL_LATEST (
    MMSI           VARCHAR(20)    NOT NULL PRIMARY KEY,
    VESSEL_NAME    VARCHAR(100),
    VESSEL_TYPE    VARCHAR(50),
    LAT            FLOAT          NOT NULL,
    LON            FLOAT          NOT NULL,
    SOG            FLOAT,
    COG            FLOAT,
    HEADING        FLOAT,
    RISK_LEVEL     VARCHAR(10),
    IS_ANOMALY     BOOLEAN        DEFAULT FALSE,
    ANOMALY_SCORE  FLOAT          DEFAULT 0.0,
    ANOMALY_TYPE   VARCHAR(100),
    PREDICTED_LAT  FLOAT,
    PREDICTED_LON  FLOAT,
    BASE_DATETIME  TIMESTAMP_NTZ,
    UPDATED_AT     TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP(),
    DATA_SPLIT     VARCHAR(20),
    US_REGION      VARCHAR(50)
        AS (
            CASE
                WHEN LON BETWEEN -82  AND -66  THEN 'East Coast'
                WHEN LON BETWEEN -125 AND -117 THEN 'West Coast'
                WHEN LON BETWEEN -97  AND -80
                 AND LAT BETWEEN 24   AND 31   THEN 'Gulf of Mexico'
                WHEN LON BETWEEN -93  AND -76
                 AND LAT BETWEEN 41   AND 49   THEN 'Great Lakes'
                ELSE 'Other US Waters'
            END
        ) VIRTUAL
);

COMMENT ON TABLE FACT_VESSEL_LATEST IS
    'Current position per vessel. One row per MMSI. '
    'Powers fleet KPI strip in the analytics dashboard.';


-- ---------------------------------------------------------------------------
-- FACT_TRAFFIC_DENSITY
-- Vessel counts per 0.1° grid cell per hour.
-- Source: PostgreSQL fact_traffic_density.
-- Clustered by time for rolling-window congestion queries.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS FACT_TRAFFIC_DENSITY (
    ID               NUMBER(18, 0)  AUTOINCREMENT PRIMARY KEY,
    LAT_BIN          FLOAT          NOT NULL,
    LON_BIN          FLOAT          NOT NULL,
    HOUR_BUCKET      TIMESTAMP_NTZ  NOT NULL,
    VESSEL_COUNT     NUMBER(9, 0)   NOT NULL,
    UNIQUE_VESSELS   NUMBER(9, 0),
    AVG_SOG          FLOAT,
    STOPPED_COUNT    NUMBER(9, 0)   DEFAULT 0,
    CARGO_COUNT      NUMBER(9, 0)   DEFAULT 0,
    TANKER_COUNT     NUMBER(9, 0)   DEFAULT 0,
    CONGESTION_LEVEL VARCHAR(10),               -- HIGH / MEDIUM / LOW
    DATA_SPLIT       VARCHAR(20),
    CREATED_AT       TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP(),
    US_REGION        VARCHAR(50)
        AS (
            CASE
                WHEN LON_BIN BETWEEN -82  AND -66  THEN 'East Coast'
                WHEN LON_BIN BETWEEN -125 AND -117 THEN 'West Coast'
                WHEN LON_BIN BETWEEN -97  AND -80
                 AND LAT_BIN BETWEEN 24   AND 31   THEN 'Gulf of Mexico'
                WHEN LON_BIN BETWEEN -93  AND -76
                 AND LAT_BIN BETWEEN 41   AND 49   THEN 'Great Lakes'
                ELSE 'Other US Waters'
            END
        ) VIRTUAL,
    CONSTRAINT uq_density_grid_hour UNIQUE (LAT_BIN, LON_BIN, HOUR_BUCKET)
)
CLUSTER BY (DATE_TRUNC('day', HOUR_BUCKET), LAT_BIN, LON_BIN);

COMMENT ON TABLE FACT_TRAFFIC_DENSITY IS
    'Pre-aggregated traffic density per 0.1° grid cell per hour. '
    'Powers heat-map and congestion analysis queries.';


-- ---------------------------------------------------------------------------
-- FACT_ALERTS
-- All system-generated maritime alerts (anomaly, collision, congestion).
-- Source: PostgreSQL fact_alerts.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS FACT_ALERTS (
    ID             NUMBER(18, 0)  AUTOINCREMENT PRIMARY KEY,
    ALERT_TYPE     VARCHAR(50)    NOT NULL,     -- COLLISION_RISK, ML_ANOMALY, etc.
    SEVERITY       VARCHAR(20)    NOT NULL,     -- HIGH / MEDIUM / LOW
    MMSI_1         VARCHAR(20),
    MMSI_2         VARCHAR(20),                 -- second vessel for collision alerts
    VESSEL_NAME_1  VARCHAR(100),
    VESSEL_NAME_2  VARCHAR(100),
    LAT            FLOAT,
    LON            FLOAT,
    DESCRIPTION    VARCHAR(2000),
    EXTRA_DATA     VARIANT,                     -- flexible JSON payload
    ANOMALY_SCORE  FLOAT,
    DISTANCE_NM    FLOAT,                       -- closest point of approach (nm)
    IS_RESOLVED    BOOLEAN        DEFAULT FALSE,
    CREATED_AT     TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP(),
    RESOLVED_AT    TIMESTAMP_NTZ,
    DATA_SPLIT     VARCHAR(20),
    US_REGION      VARCHAR(50)
        AS (
            CASE
                WHEN LON BETWEEN -82  AND -66  THEN 'East Coast'
                WHEN LON BETWEEN -125 AND -117 THEN 'West Coast'
                WHEN LON BETWEEN -97  AND -80
                 AND LAT BETWEEN 24   AND 31   THEN 'Gulf of Mexico'
                WHEN LON BETWEEN -93  AND -76
                 AND LAT BETWEEN 41   AND 49   THEN 'Great Lakes'
                ELSE 'Other US Waters'
            END
        ) VIRTUAL
)
CLUSTER BY (ALERT_TYPE, DATE_TRUNC('day', CREATED_AT));

COMMENT ON TABLE FACT_ALERTS IS
    'All maritime alerts. VARIANT column EXTRA_DATA holds alert-type-specific '
    'JSON payload without requiring schema changes.';


-- ---------------------------------------------------------------------------
-- FACT_DAILY_STATS
-- One row per calendar day — fleet-wide KPIs for trend charts.
-- Source: PostgreSQL fact_daily_stats.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS FACT_DAILY_STATS (
    ID                  NUMBER(9, 0)   AUTOINCREMENT PRIMARY KEY,
    STAT_DATE           DATE           NOT NULL UNIQUE,
    TOTAL_VESSELS       NUMBER(9, 0)   DEFAULT 0,
    TOTAL_RECORDS       NUMBER(18, 0)  DEFAULT 0,
    AVG_SOG             FLOAT,
    MAX_SOG             FLOAT,
    STOPPED_VESSELS     NUMBER(9, 0)   DEFAULT 0,
    HIGH_RISK_COUNT     NUMBER(9, 0)   DEFAULT 0,
    MEDIUM_RISK_COUNT   NUMBER(9, 0)   DEFAULT 0,
    ANOMALY_COUNT       NUMBER(9, 0)   DEFAULT 0,
    COLLISION_ALERTS    NUMBER(9, 0)   DEFAULT 0,
    CONGESTION_HOURS    NUMBER(9, 0)   DEFAULT 0,
    TOP_VESSEL_TYPE     VARCHAR(50),
    DATA_SPLIT          VARCHAR(20),
    CREATED_AT          TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP()
)
CLUSTER BY (STAT_DATE);

COMMENT ON TABLE FACT_DAILY_STATS IS
    'Daily fleet-wide KPI snapshot. One row per day. '
    'Drives trend line charts in the analytics dashboard.';


-- =============================================================================
-- VIEWS — pre-canned queries for common analytics patterns
-- =============================================================================

CREATE OR REPLACE VIEW V_US_FLEET_SUMMARY AS
SELECT
    US_REGION,
    COUNT(DISTINCT MMSI)                                          AS total_vessels,
    COUNT(DISTINCT CASE WHEN RISK_LEVEL = 'HIGH'  THEN MMSI END) AS high_risk_vessels,
    COUNT(DISTINCT CASE WHEN IS_ANOMALY           THEN MMSI END) AS anomalous_vessels,
    ROUND(AVG(SOG), 2)                                           AS avg_speed_kn,
    MAX(UPDATED_AT)                                              AS data_freshness
FROM FACT_VESSEL_LATEST
GROUP BY 1
ORDER BY total_vessels DESC;

COMMENT ON VIEW V_US_FLEET_SUMMARY IS
    'Current fleet snapshot grouped by US coastal region. '
    'Refreshes automatically as FACT_VESSEL_LATEST is updated.';


CREATE OR REPLACE VIEW V_DAILY_ANOMALY_TREND AS
SELECT
    DATE_TRUNC('week', STAT_DATE) AS week_start,
    SUM(TOTAL_VESSELS)            AS total_vessels,
    SUM(ANOMALY_COUNT)            AS total_anomalies,
    SUM(HIGH_RISK_COUNT)          AS total_high_risk,
    SUM(COLLISION_ALERTS)         AS total_collisions,
    ROUND(AVG(AVG_SOG), 2)        AS avg_fleet_speed
FROM FACT_DAILY_STATS
GROUP BY 1
ORDER BY 1 DESC;

COMMENT ON VIEW V_DAILY_ANOMALY_TREND IS
    'Weekly aggregation of daily stats for trend-line charts.';
