# Snowflake Integration Architecture

## System overview

The Maritime Navigation AI System uses three data stores, each chosen for
what it does best.  No single database is right for every access pattern.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE OVERVIEW                                  │
│                                                                             │
│  AIS Radio Feed                                                             │
│       │                                                                     │
│       ▼                                                                     │
│  Kafka (ais_raw topic)                                                      │
│       │                                                                     │
│       ├──► Spark Structured Streaming                                       │
│       │         └─► Bronze Delta  (/delta/bronze/ais)                      │
│       │                   └─► Silver Delta  (/delta/silver/ais_clean)      │
│       │                             └─► Gold Delta  (/delta/gold/*)        │
│       │                                       │                            │
│       │                              gold_job.py                           │
│       │                                       │  JDBC                      │
│       │                                       ▼                            │
│       └──────────────────────────────────────►│                            │
│                                               │                            │
│                              ┌────────────────▼───────────────────────┐   │
│                              │  PostgreSQL 15 — LIVE OPERATIONS       │   │
│                              │  (docker service: postgres:5432)       │   │
│                              │                                        │   │
│                              │  fact_vessel_latest   ← live API map  │   │
│                              │  fact_ais_track       ← replay slider │   │
│                              │  fact_alerts          ← alert stream  │   │
│                              │  fact_traffic_density ← heat-map      │   │
│                              │  fact_daily_stats     ← daily roll-up │   │
│                              └──────────────┬─────────────────────────┘   │
│                                             │                              │
│                              snowflake_loader.py                           │
│                              (incremental, ~hourly, via ETL_WATERMARKS)    │
│                                             │                              │
│                              ┌──────────────▼─────────────────────────┐   │
│                              │  Snowflake — ANALYTICS WAREHOUSE       │   │
│                              │  (MARITIME_AIS database)               │   │
│                              │                                        │   │
│                              │  DIM_VESSEL                           │   │
│                              │  FACT_AIS_TRACK      ← full history   │   │
│                              │  FACT_ALERTS         ← all alerts     │   │
│                              │  FACT_TRAFFIC_DENSITY                 │   │
│                              │  FACT_DAILY_STATS                     │   │
│                              │                                        │   │
│                              │  Views:                                │   │
│                              │    V_US_FLEET_SUMMARY                 │   │
│                              │    V_DAILY_ANOMALY_TREND              │   │
│                              └────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why three stores, not one

### Delta Lake (Bronze / Silver / Gold)

Delta is the **pipeline intermediate format** — not a query target for
end users.  Its job is durability, schema evolution, and time-travel
during Spark batch and streaming processing.

| Strength | How it's used here |
|---|---|
| ACID on object storage | Bronze append is idempotent — re-runs don't duplicate rows |
| Schema evolution | `mergeSchema=true` absorbs new AIS fields without downtime |
| Time travel | Gold tables can be queried at any historical version |
| Spark-native | Silver feature engineering and Gold aggregations run as Spark jobs |

### PostgreSQL (live operations)

PostgreSQL is the **system of record for real-time state**.  Every live
AIS message updates a row; the API and dashboard read from it at sub-second
latency.

| Strength | How it's used here |
|---|---|
| Row-level indexes | `ix_latest_lat_lon`, `ix_latest_risk` answer map bbox queries instantly |
| UPSERT | `ON CONFLICT DO UPDATE` keeps one row per MMSI without duplicates |
| Transactional alerts | No partial alert writes; a collision pair is committed atomically |
| Docker portability | Zero cloud dependency — runs on the vessel or in an offline lab |
| SQLAlchemy ORM | `database.py` maps tables to Python objects used by FastAPI and Streamlit |

### Snowflake (analytics warehouse)

Snowflake is the **analytical query engine** for queries that would
degrade transactional PostgreSQL performance.

| Strength | How it's used here |
|---|---|
| Columnar storage | Scanning 36 M AIS positions by lat/lon range is a single partition scan |
| Distributed window functions | Per-vessel timelines for anomaly trend queries |
| Automatic clustering | `CLUSTER BY (mmsi, base_datetime)` prunes 99 % of micro-partitions |
| VARIANT column | `EXTRA_DATA` in `FACT_ALERTS` stores any JSON payload without schema migration |
| Auto-suspend | Warehouse pauses after 60 s idle — near-zero cost between ETL runs |
| Separation of concerns | Analyst queries never compete with live AIS INSERT traffic |

---

## Databricks as the ETL bridge

When running in Databricks mode (production scale), the pipeline becomes:

```
Cloud Storage (Parquet / Auto Loader)
        │
        ▼
01_bronze_notebook.py  →  Bronze Delta  (dbfs:/mnt/maritime/delta/bronze)
        │
        ▼
02_silver_notebook.py  →  Silver Delta  (dbfs:/mnt/maritime/delta/silver)
        │
        ▼
03_gold_notebook.py    →  Gold Delta    (dbfs:/mnt/maritime/delta/gold)
        │
        │  Option A: JDBC (if PostgreSQL is network-accessible from Databricks)
        ▼
    PostgreSQL  →  (unchanged path)  →  Snowflake via snowflake_loader.py
        │
        │  Option B: Snowflake Spark connector (direct from Databricks)
        ▼
    Snowflake   ←  spark.write.format("snowflake") ...
```

### Option B: Direct Databricks → Snowflake write

Add this to `03_gold_notebook.py` (or a new Task 4 in the Workflow):

```python
# Install on cluster: net.snowflake:spark-snowflake_2.12:2.12.0-spark_3.4
SF_OPTIONS = {
    "sfURL":       f"{dbutils.secrets.get('maritime','snowflake-account')}.snowflakecomputing.com",
    "sfUser":      dbutils.secrets.get("maritime", "snowflake-user"),
    "sfPassword":  dbutils.secrets.get("maritime", "snowflake-password"),
    "sfDatabase":  "MARITIME_AIS",
    "sfSchema":    "PUBLIC",
    "sfWarehouse": "MARITIME_WH",
}

vessel_latest_df.write \
    .format("snowflake") \
    .options(**SF_OPTIONS) \
    .option("dbtable", "FACT_VESSEL_LATEST") \
    .mode("overwrite") \
    .save()
```

This bypasses PostgreSQL entirely for the analytics copy and is preferred
when Databricks can reach Snowflake but cannot reach PostgreSQL.

---

## Snowflake credentials in Databricks Secrets

The same `maritime` secret scope used for PostgreSQL holds Snowflake creds:

```bash
databricks secrets put --scope maritime --key snowflake-account
databricks secrets put --scope maritime --key snowflake-user
databricks secrets put --scope maritime --key snowflake-password
```

The `environment.py` config (`databricks/configs/environment.py`) can be
extended to read these:

```python
SF_ACCOUNT = _resolve_secret("maritime", "snowflake-account",
                              os.getenv("SNOWFLAKE_ACCOUNT", ""))
```

---

## ETL watermark mechanism

`src/snowflake/snowflake_loader.py` uses an `ETL_WATERMARKS` table in
Snowflake to track the last successfully loaded timestamp per table.

```
Run 1 (initial)      → loads all rows                 → watermark = NOW()
Run 2 (1 hour later) → loads only rows > watermark    → watermark = NOW()
Run 3 (next hour)    → loads only rows > watermark    → watermark = NOW()
```

This means the ETL is safe to run on a cron without manual coordination.
A failed run leaves the watermark at the last successful point — the next
run picks up from there without gaps or duplicates.

---

## Graceful degradation

All Snowflake-facing code checks `is_configured()` before attempting a
connection.  The dependency chain:

```
SNOWFLAKE_ACCOUNT env var
        │ unset
        ▼
snowflake_loader.py   → logs warning, returns {}
snowflake_queries.py  → returns empty DataFrame with correct columns
snowflake_router.py   → returns mock data, source="mock_no_credentials"
snowflake_analytics.py → falls back to PostgreSQL for all charts
```

No code path raises an unhandled exception when Snowflake is absent.
The Docker stack continues to operate identically with or without
Snowflake credentials present.

---

## Deployment checklist

- [ ] Create Snowflake trial or enterprise account
- [ ] Run `src/snowflake/snowflake_schema.sql` to provision tables and views
- [ ] Set credentials in `.env` (copy from `.env.snowflake.example`)
- [ ] Run `python src/snowflake/snowflake_loader.py` for initial load
- [ ] Verify via `streamlit run src/dashboard/snowflake_analytics.py`
- [ ] Add FastAPI router: `app.include_router(snowflake_router)` in `api/main.py`
- [ ] Schedule `snowflake_loader.py` hourly (cron / Databricks Workflow)
- [ ] For Databricks: add Snowflake secrets to the `maritime` scope
- [ ] Optional: enable Databricks → Snowflake direct write via Spark connector
