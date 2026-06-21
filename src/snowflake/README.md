# Snowflake Analytics Integration

## Why PostgreSQL AND Snowflake — not one or the other

This system uses both databases deliberately. They solve different problems.

```
                        ┌─────────────────────────────────────────┐
  AIS feed (Kafka)      │         OPERATIONAL LAYER               │
  Live scoring          │   PostgreSQL 15 (docker: postgres:5432) │
  FastAPI REST API  ──► │                                         │
  Streamlit live map    │   fact_vessel_latest   ← live map       │
  Collision alerts      │   fact_ais_track       ← replay         │
                        │   fact_alerts          ← alert feed     │
                        │   fact_traffic_density ← heatmap        │
                        │   fact_daily_stats     ← daily roll-up  │
                        └──────────────┬──────────────────────────┘
                                       │  ETL (snowflake_loader.py)
                                       │  incremental, ~hourly
                                       ▼
                        ┌─────────────────────────────────────────┐
                        │         ANALYTICS LAYER                 │
                        │   Snowflake (MARITIME_AIS database)     │
                        │                                         │
  Business intelligence │   DIM_VESSEL                           │
  Historical queries    │   FACT_AIS_TRACK      ← full history   │
  Trend analysis    ──► │   FACT_ALERTS         ← all alerts     │
  Coast guard reports   │   FACT_TRAFFIC_DENSITY                 │
  ML feature stores     │   FACT_DAILY_STATS                     │
  Long-term retention   │                                         │
                        │   Views: V_US_FLEET_SUMMARY             │
                        │          V_DAILY_ANOMALY_TREND          │
                        └─────────────────────────────────────────┘
```

### PostgreSQL is the right choice for live operations

| Requirement | Why PostgreSQL wins |
|---|---|
| Sub-second API response | Row-level indexes on `mmsi`, `lat/lon`, `risk_level` |
| UPSERT on every AIS ping | `ON CONFLICT DO UPDATE` with microsecond latency |
| Live anomaly alert writes | Transactional consistency — no alert is ever half-written |
| Docker portability | Ships as a container, zero cloud dependency |
| Connection pooling | SQLAlchemy `pool_size=20` handles the API concurrency |

### Snowflake is the right choice for analytics

| Requirement | Why Snowflake wins |
|---|---|
| Scan 36 M AIS records by region | Columnar storage + automatic micro-partition pruning |
| Window functions over vessel timelines | Distributed execution across virtual warehouse nodes |
| Joining dim_vessel × fact_ais_track | Broadcast joins with automatic statistics |
| Concurrent BI tool queries | Multi-cluster warehouse; analysts don't block each other |
| 90-day retention without degradation | Zero-maintenance storage; no VACUUM/REINDEX needed |
| `VARIANT` for alert JSON payloads | Schema-on-read; no ALTER TABLE for new alert fields |

### The mistake to avoid

Running complex analytical queries directly against PostgreSQL under live
AIS load causes lock contention and degrades API response times.  The
`fact_ais_track` table already has tens of millions of rows after 14 days
of data; a full-table scan for a 30-day anomaly trend report would compete
with the live scorer's INSERT stream.

Snowflake's auto-suspend (60 s idle) means it costs nothing when not in use
and scales to handle any analyst query load without touching PostgreSQL.

---

## Files in this directory

| File | Purpose |
|---|---|
| `snowflake_schema.sql` | `CREATE TABLE` DDL — run once to provision the warehouse |
| `snowflake_loader.py` | Incremental ETL: reads Gold Delta/PostgreSQL, writes to Snowflake |
| `snowflake_queries.py` | Named analytics queries returning pandas DataFrames |
| `__init__.py` | Makes `src/snowflake` a Python package |

---

## Quick start

### 1 — Create a Snowflake trial account
https://signup.snowflake.com (free 30-day trial, no credit card)

### 2 — Provision schema
```bash
snowsql -a <your-account> -u <your-user> \
        -f src/snowflake/snowflake_schema.sql
```

### 3 — Set credentials (copy from `.env.snowflake.example`)
```bash
export SNOWFLAKE_ACCOUNT=myorg-us-east-1
export SNOWFLAKE_USER=maritime_etl
export SNOWFLAKE_PASSWORD=...
export SNOWFLAKE_DATABASE=MARITIME_AIS
export SNOWFLAKE_WAREHOUSE=MARITIME_WH
```

### 4 — Run initial load
```bash
python src/snowflake/snowflake_loader.py
```

### 5 — Open the analytics dashboard
```bash
streamlit run src/dashboard/snowflake_analytics.py
```

The system continues to run fully if `SNOWFLAKE_ACCOUNT` is unset.
Every Snowflake function returns an empty DataFrame; the dashboard falls back
to PostgreSQL for the same metrics.

---

## Geographic scope: US coastal waters

All Snowflake queries filter to US coastal zones.  The Suez Canal corridor
served by the main Docker pipeline is excluded from Snowflake analytics to
keep warehouse storage and query costs focused on the US monitoring mission.

| Region | Lat | Lon |
|---|---|---|
| East Coast (Atlantic) | 24 – 47 °N | 82 – 66 °W |
| West Coast (Pacific) | 32 – 49 °N | 125 – 117 °W |
| Gulf of Mexico | 24 – 31 °N | 97 – 80 °W |
| Great Lakes | 41 – 49 °N | 93 – 76 °W |

---

## Operational notes

- **ETL frequency**: Run `snowflake_loader.py` hourly via cron or Databricks
  Workflows.  The watermark table `ETL_WATERMARKS` ensures each run is
  incremental — only rows newer than the last load are transferred.
- **Warehouse size**: `X-SMALL` handles all dashboard queries.  Promote to
  `SMALL` only if you add concurrent BI tool connections.
- **Cost guard**: `AUTO_SUSPEND=60` pauses the warehouse after 60 s idle.
  At X-SMALL pricing this keeps analytics costs under $5/day for normal use.
- **No Snowflake = no breakage**: `is_configured()` in `snowflake_loader.py`
  and `snowflake_queries.py` returns `False` when credentials are absent.
  The FastAPI router returns mock data; the Streamlit page falls back to
  PostgreSQL.
