# Architecture: Local Docker vs Databricks

## Why two architectures exist

The Maritime Navigation AI system was designed with a **local-first** philosophy:
the entire pipeline runs inside Docker on a single laptop or server without any
cloud account.  This makes it portable (vessels with intermittent connectivity,
air-gapped labs, offline demos) and cost-free during development.

Databricks is the **cloud scale-out target**: the same medallion pipeline
runs in a managed Spark environment with auto-scaling, scheduled jobs,
Unity Catalog governance, and MLflow tracking.  No business logic changes
are required — only infrastructure paths and credential sources differ, and
those are abstracted by `databricks/configs/environment.py`.

---

## Local Docker Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    docker-compose.yml stack                              ║
║                                                                          ║
║  ┌─────────────┐    JSON     ┌────────────────────────────────────────┐ ║
║  │  producer   │───────────►│             kafka:9092                  │ ║
║  │(kafka_       │  ais_raw   │    (Confluent 7.4.0 / Zookeeper)       │ ║
║  │ producer.py)│            └──────────────────┬─────────────────────┘ ║
║  └─────────────┘                               │ consume                ║
║                                                ▼                        ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │                  Spark Cluster (3 nodes)                         │   ║
║  │                                                                  │   ║
║  │  spark-master (7077)                                             │   ║
║  │  spark-worker-1  (4 cores, 4 GB)                                 │   ║
║  │  spark-worker-2  (4 cores, 4 GB)                                 │   ║
║  │                                                                  │   ║
║  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │   ║
║  │  │spark-stream  │  │  bronze_job  │  │  silver_job           │  │   ║
║  │  │(continuous   │  │  (batch,     │  │  (batch, dedup+       │  │   ║
║  │  │ Kafka →      │  │   Parquet→   │  │   features, writes    │  │   ║
║  │  │ Bronze Δ)    │  │   Bronze Δ)  │  │   Silver Δ)           │  │   ║
║  │  └──────┬───────┘  └──────┬───────┘  └──────────┬────────────┘  │   ║
║  │         │                 │                      │               │   ║
║  │         └────────┬────────┘                      │               │   ║
║  │                  ▼                               ▼               │   ║
║  │         ┌────────────────┐           ┌──────────────────────┐    │   ║
║  │         │  Bronze Δ      │           │  Silver Δ            │    │   ║
║  │         │  /delta/bronze │           │  /delta/silver       │    │   ║
║  │         │  /ais          │           │  /ais_clean          │    │   ║
║  │         └────────────────┘           └──────────┬───────────┘    │   ║
║  │                                                 │               │   ║
║  │                                      ┌──────────▼───────────┐    │   ║
║  │                                      │   gold_job.py        │    │   ║
║  │                                      │   (aggregations +    │    │   ║
║  │                                      │    JDBC sync)        │    │   ║
║  │                                      └──────────┬───────────┘    │   ║
║  └──────────────────────────────────────────────── │ ───────────────┘   ║
║                                                    │                    ║
║       ┌────────────────────────────────────────────▼──────────────┐     ║
║       │               Gold Delta Tables (/delta/gold/)            │     ║
║       │   vessel_latest  │  traffic_density  │  daily_stats        │     ║
║       └──────────────────────────────────────┬────────────────────┘     ║
║                                              │  JDBC (postgres:5432)    ║
║                                              ▼                          ║
║       ┌──────────────────────────────────────────────────────────┐      ║
║       │         PostgreSQL 15 — Maritime Star Schema             │      ║
║       │  fact_vessel_latest  │  fact_traffic_density             │      ║
║       │  fact_daily_stats    │  dim_vessel                       │      ║
║       └──────┬────────────────────────────────────────┬──────────┘      ║
║              │                                        │                 ║
║              ▼                                        ▼                 ║
║  ┌───────────────────────┐              ┌─────────────────────────┐     ║
║  │  FastAPI (port 8000)  │              │  Streamlit (port 8501)  │     ║
║  │  + live-scorer        │              │  + React (port 3000)    │     ║
║  └───────────────────────┘              └─────────────────────────┘     ║
║                                                                          ║
║  Volumes:  delta_data (Delta tables)  │  postgres_data  │  ivy_cache    ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### Data flow (local)

```
Raw CSV (.zst)
    │  convert_csv.py
    ▼
Parquet files (/app/data/parquet)
    │  bronze_job.py  OR  spark_streaming_consumer.py (live)
    ▼
Bronze Delta  (/delta/bronze/ais)
    │  silver_job.py
    ▼
Silver Delta  (/delta/silver/ais_clean)
    │  gold_job.py
    ├─► Gold Delta tables (/delta/gold/*)
    └─► PostgreSQL star schema (via JDBC)
```

### Service ports (local)

| Service        | Port  | URL                         |
|----------------|-------|-----------------------------|
| Spark Master   | 9090  | http://localhost:9090        |
| Kafka UI       | 8083  | http://localhost:8083        |
| FastAPI        | 8000  | http://localhost:8000/docs  |
| Streamlit      | 8501  | http://localhost:8501        |
| React frontend | 3000  | http://localhost:3000        |
| PostgreSQL     | 5432  | localhost:5432/maritime      |

---

## Databricks Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                      Databricks Workspace                                ║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │                   Databricks Workflows / Jobs                    │   ║
║  │                                                                  │   ║
║  │   [Task 1: Bronze]──►[Task 2: Silver]──►[Task 3: Gold]          │   ║
║  │    01_bronze_notebook   02_silver_notebook  03_gold_notebook     │   ║
║  │    (scheduled daily,    (depends on         (depends on          │   ║
║  │     or on file arrival)  bronze complete)    silver complete)    │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │              Databricks Cluster (auto-scaling)                   │   ║
║  │              DBR 13.3 LTS / Spark 3.4 / Delta 2.4               │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │                Databricks Secret Scope: maritime                  │   ║
║  │   postgres-host  │  postgres-password  │  kafka-bootstrap-servers │  ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
         │  DBFS reads/writes                     │  live stream
         ▼                                        ▼
╔════════════════════════════════╗   ╔════════════════════════════════════╗
║   Cloud Object Storage         ║   ║   Confluent Cloud / Event Hubs     ║
║   (S3 / ADLS / GCS)            ║   ║   (Kafka-compatible API)           ║
║                                ║   ║                                    ║
║  /mnt/maritime/                ║   ║   kafka_streaming_notebook.py      ║
║  ├── data/parquet/             ║   ║   → Bronze Delta (streaming)       ║
║  ├── delta/                    ║   ╚════════════════════════════════════╝
║  │   ├── bronze/ais/           ║
║  │   ├── silver/ais_clean/     ║
║  │   └── gold/                 ║
║  │       ├── vessel_latest/    ║
║  │       ├── traffic_density/  ║
║  │       └── daily_stats/      ║
║  ├── models/                   ║
║  └── checkpoints/              ║
╚════════════════════════════════╝
         │  optional JDBC (if network-accessible)
         ▼
╔════════════════════════════════╗
║  PostgreSQL (RDS / Cloud SQL)  ║
║  fact_vessel_latest            ║
║  fact_traffic_density          ║
║  fact_daily_stats  dim_vessel  ║
╚════════════════════════════════╝
         │
         ▼
╔════════════════════════════════╗
║  Databricks SQL Warehouse  OR  ║
║  External BI (Tableau/Power BI)║
╚════════════════════════════════╝
```

### Data flow (Databricks)

```
Cloud Storage (Parquet / Auto Loader)
    │  01_bronze_notebook.py
    ▼
Bronze Delta  (dbfs:/mnt/maritime/delta/bronze/ais)
    │  02_silver_notebook.py
    ▼
Silver Delta  (dbfs:/mnt/maritime/delta/silver/ais_clean)
    │  03_gold_notebook.py
    ├─► Gold Delta (dbfs:/mnt/maritime/delta/gold/*)
    └─► PostgreSQL (optional, requires network config)

Confluent Kafka / Event Hubs
    │  kafka_streaming_notebook.py (always-on Streaming Job)
    └─► Bronze Delta (append, continuous)
```

---

## Side-by-side comparison

```
┌─────────────────────────┬───────────────────────────┬─────────────────────────────┐
│ Concern                 │ Local Docker              │ Databricks                  │
├─────────────────────────┼───────────────────────────┼─────────────────────────────┤
│ Spark                   │ 3-node standalone cluster │ Managed cluster, auto-scale │
│ Delta storage           │ /delta (named volume)     │ dbfs:/mnt/maritime/delta    │
│ Kafka                   │ Confluent 7.4.0 container │ Confluent Cloud / Event Hubs│
│ PostgreSQL              │ postgres:5432 container   │ RDS / Cloud SQL (optional)  │
│ Credentials             │ docker-compose env vars   │ Databricks Secrets          │
│ Job scheduling          │ manual / cron in compose  │ Databricks Workflows        │
│ ML tracking             │ local file /app/models    │ MLflow on Databricks        │
│ Dashboard               │ Streamlit :8501           │ Databricks SQL / BI tool    │
│ Cold start              │ ~30s (images cached)      │ 2–5 min (cluster provision) │
│ Cost                    │ electricity only          │ DBU + cloud storage         │
│ Offline capability      │ Full                      │ None                        │
│ Max data scale          │ ~8 workers × 4 cores      │ Unlimited (auto-scale)      │
└─────────────────────────┴───────────────────────────┴─────────────────────────────┘
```

---

## Medallion layer mapping

```
                    RAW DATA
                      │
         ┌────────────▼────────────┐
         │     BRONZE LAYER        │
         │  Append-only, enriched  │
         │  Speed flags, zone flags│
         │  Rule-based risk level  │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │     SILVER LAYER        │
         │  Deduped (1/min/vessel) │
         │  Forward-filled fields  │
         │  ML features computed   │
         │  Partitioned year/mo/day│
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │     GOLD LAYER          │
         ├─────────────────────────┤
         │ vessel_latest           │  ← Dashboard live map
         │ traffic_density         │  ← Heat-map grid
         │ daily_stats             │  ← Trend charts
         │ (dim_vessel via JDBC)   │  ← Star schema dimension
         └─────────────────────────┘
```

---

## ML pipeline placement

```
Silver Delta
    │
    ├── train_anomaly.py    → Isolation Forest model  → /models/anomaly/
    ├── train_congestion.py → LightGBM model          → /models/congestion/
    └── train_predictor.py  → Dead-reckoning model    → /models/predictor/

Kafka (live feed)
    │
    └── live_scorer.py  (reads models, writes anomaly scores back to Gold)
```

On Databricks, models live in `dbfs:/mnt/maritime/models/` and are tracked
via MLflow experiments.  The scoring logic in `src/ml/scorer.py` and
`src/ml/live_scorer.py` is unchanged — only the model path is resolved
through `environment.py`.

---

## File map: local job → Databricks notebook

| Local job                              | Databricks notebook                              |
|----------------------------------------|--------------------------------------------------|
| `src/processing/bronze_job.py`         | `databricks/notebooks/01_bronze_notebook.py`     |
| `src/processing/silver_job.py`         | `databricks/notebooks/02_silver_notebook.py`     |
| `src/processing/gold_job.py`           | `databricks/notebooks/03_gold_notebook.py`       |
| `src/processing/spark_streaming_consumer.py` | `databricks/examples/kafka_streaming_notebook.py` |
| `src/common/config.py`                 | `databricks/configs/environment.py`              |
