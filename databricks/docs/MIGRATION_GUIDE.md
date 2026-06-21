# Migration Guide: Local Docker → Databricks

## Overview

This guide walks through moving the Maritime Navigation AI pipeline from its
primary local Docker runtime to Databricks.  The local stack is **not
deprecated** — it remains the development default and the offline/vessel-side
runtime.  Databricks is the horizontal scale-out target for large datasets and
production SLAs.

---

## Why local Docker is retained as primary runtime

The `docker-compose.yml` stack ships everything — Spark (3-node), Kafka,
PostgreSQL, Streamlit dashboard, FastAPI, and React frontend — in one command:

```bash
docker compose up
```

That zero-friction workflow is irreplaceable during iterative development.
The Databricks path adds:
- IAM credentials and workspace provisioning
- Cluster cold-start time (2–5 min)
- DBFS / Unity Catalog path management
- Secret scope setup for PostgreSQL and Kafka credentials

None of that overhead is justified until the dataset or throughput exceeds
what two Docker Spark workers can handle.  Both runtimes share the same
business logic — changes to `src/processing/*.py` propagate to the Databricks
notebooks by design.

---

## Prerequisites

### Local (already in place)
- Docker Desktop ≥ 24.x
- 16 GB RAM recommended (Spark cluster uses ~12 GB)
- `docker compose up` builds and starts all services

### Databricks
| Item | Requirement |
|------|-------------|
| Workspace | Databricks on AWS, Azure, or GCP |
| DBR version | 13.3 LTS or later (Spark 3.4 + Delta 2.4) |
| Cluster | Standard or Job Cluster, 4+ cores, 16 GB RAM |
| Libraries | `delta-spark==2.4.0`, `org.postgresql:postgresql:42.7.3` (Maven) |
| DBFS mount | `/mnt/maritime` mounted to your cloud storage bucket |
| Secrets scope | `maritime` (created via Databricks CLI or UI) |

---

## Step 1 — Create the Databricks Secret Scope

Secrets replace the hard-coded credentials in `docker-compose.yml`.

```bash
# Install Databricks CLI
pip install databricks-cli

# Authenticate
databricks configure --token

# Create scope
databricks secrets create-scope --scope maritime

# Add PostgreSQL credentials
databricks secrets put --scope maritime --key postgres-host
databricks secrets put --scope maritime --key postgres-port
databricks secrets put --scope maritime --key postgres-db
databricks secrets put --scope maritime --key postgres-user
databricks secrets put --scope maritime --key postgres-password

# Add Kafka credentials (skip if using local broker for testing)
databricks secrets put --scope maritime --key kafka-bootstrap-servers
databricks secrets put --scope maritime --key kafka-sasl-mechanism
databricks secrets put --scope maritime --key kafka-sasl-username
databricks secrets put --scope maritime --key kafka-sasl-password
```

`databricks/configs/environment.py` reads all of these automatically when
`DATABRICKS_RUNTIME_VERSION` is set in the cluster environment.

---

## Step 2 — Mount cloud storage to DBFS

The Delta Lake tables that live at `/delta` in Docker must move to a cloud
storage path mounted at `/mnt/maritime`.

### AWS S3
```python
# Run once in a Databricks notebook
dbutils.fs.mount(
    source      = "s3a://your-bucket/maritime",
    mount_point = "/mnt/maritime",
    extra_configs = {
        "fs.s3a.access.key": dbutils.secrets.get("maritime", "s3-access-key"),
        "fs.s3a.secret.key": dbutils.secrets.get("maritime", "s3-secret-key"),
    }
)
```

### Azure ADLS Gen2
```python
dbutils.fs.mount(
    source      = "abfss://maritime@youraccount.dfs.core.windows.net/",
    mount_point = "/mnt/maritime",
    extra_configs = {
        "fs.azure.account.key.youraccount.dfs.core.windows.net":
            dbutils.secrets.get("maritime", "adls-account-key")
    }
)
```

After mounting, the path layout mirrors the local structure:

| Local (Docker)             | Databricks (DBFS)                         |
|----------------------------|-------------------------------------------|
| `/delta/bronze/ais`        | `dbfs:/mnt/maritime/delta/bronze/ais`     |
| `/delta/silver/ais_clean`  | `dbfs:/mnt/maritime/delta/silver/ais_clean` |
| `/delta/gold/vessel_latest`| `dbfs:/mnt/maritime/delta/gold/vessel_latest` |
| `/app/data/parquet`        | `dbfs:/mnt/maritime/data/parquet`         |
| `/app/models`              | `dbfs:/mnt/maritime/models`               |

---

## Step 3 — Upload source data

### Option A: Databricks CLI
```bash
databricks fs cp -r /local/path/data/parquet  dbfs:/mnt/maritime/data/parquet
```

### Option B: Auto Loader (for ongoing ingestion)
Replace the static `spark.read.parquet(SOURCE_PATH)` call in
`01_bronze_notebook.py` with:

```python
# Auto Loader detects new Parquet files as they land in cloud storage
raw_df = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "parquet")
    .option("cloudFiles.schemaLocation", f"{BRONZE_PATH}/_schema")
    .load(SOURCE_PATH)
)
```

Auto Loader handles schema evolution and exactly-once semantics
automatically — no manual Parquet conversion step needed.

---

## Step 4 — Install cluster libraries

On the cluster **Libraries** tab (or in `cluster_config.json`):

**PyPI**
```
delta-spark==2.4.0
psycopg2-binary==2.9.9
```

**Maven**
```
org.postgresql:postgresql:42.7.3
io.delta:delta-core_2.12:2.4.0
org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.4
```

The `spark-sql-kafka` package is only needed for
`kafka_streaming_notebook.py`.

---

## Step 5 — Import notebooks into Databricks

```bash
# Using Databricks CLI
databricks workspace import_dir databricks/notebooks \
    /Shared/maritime/notebooks

databricks workspace import_dir databricks/examples \
    /Shared/maritime/examples

databricks workspace import \
    databricks/configs/environment.py \
    /Shared/maritime/configs/environment
```

> **Note**: Databricks strips the `# COMMAND ----------` markers and renders
> the file as a notebook automatically.  The `# MAGIC %md` and `# MAGIC %run`
> lines are interpreted as cell-type magic commands.

---

## Step 6 — Create the Databricks Job / Workflow

```json
{
  "name": "maritime-ais-pipeline",
  "tasks": [
    {
      "task_key": "bronze",
      "notebook_task": {
        "notebook_path": "/Shared/maritime/notebooks/01_bronze_notebook",
        "base_parameters": {
          "source_path":  "dbfs:/mnt/maritime/data/parquet",
          "bronze_path":  "dbfs:/mnt/maritime/delta/bronze/ais",
          "write_mode":   "append"
        }
      },
      "existing_cluster_id": "YOUR_CLUSTER_ID"
    },
    {
      "task_key": "silver",
      "depends_on": [{"task_key": "bronze"}],
      "notebook_task": {
        "notebook_path": "/Shared/maritime/notebooks/02_silver_notebook",
        "base_parameters": {
          "bronze_path": "dbfs:/mnt/maritime/delta/bronze/ais",
          "silver_path": "dbfs:/mnt/maritime/delta/silver/ais_clean",
          "splits":      "TRAIN,TEST"
        }
      },
      "existing_cluster_id": "YOUR_CLUSTER_ID"
    },
    {
      "task_key": "gold",
      "depends_on": [{"task_key": "silver"}],
      "notebook_task": {
        "notebook_path": "/Shared/maritime/notebooks/03_gold_notebook",
        "base_parameters": {
          "silver_path":      "dbfs:/mnt/maritime/delta/silver/ais_clean",
          "gold_vessel_path": "dbfs:/mnt/maritime/delta/gold/vessel_latest",
          "gold_density_path":"dbfs:/mnt/maritime/delta/gold/traffic_density",
          "gold_stats_path":  "dbfs:/mnt/maritime/delta/gold/daily_stats",
          "write_pg":         "false"
        }
      },
      "existing_cluster_id": "YOUR_CLUSTER_ID"
    }
  ],
  "schedule": {
    "quartz_cron_expression": "0 0 2 * * ?",
    "timezone_id": "UTC"
  }
}
```

Set `write_pg=true` only after completing Step 7.

---

## Step 7 — PostgreSQL connectivity (optional)

The `fact_*` tables are currently written to PostgreSQL by `gold_job.py`
on Docker.  To replicate this on Databricks:

1. Expose your PostgreSQL instance to the Databricks cluster CIDR range
   (security group / firewall rule)
2. Set `write_pg=true` in the gold notebook widget
3. Verify with:
   ```sql
   -- Run in the gold notebook after the write
   SELECT COUNT(*) FROM fact_vessel_latest;
   ```

If PostgreSQL is not reachable from Databricks, the API and dashboard can
query the Gold Delta tables directly via JDBC-over-Delta or via the
Databricks SQL Warehouse endpoint.

---

## Rollback procedure

Because the local Docker stack is never modified by this migration, rollback
is simply:

```bash
docker compose up
```

All local Delta tables, Kafka topics, and PostgreSQL data remain on their
respective Docker volumes.  The Databricks DBFS tables are a separate copy
and do not affect local state.

---

## Environment variable reference

| Variable                    | Local default             | Databricks override path    |
|-----------------------------|---------------------------|-----------------------------|
| `DELTA_ROOT`                | `/delta`                  | `DATABRICKS_DELTA_ROOT` env |
| `PARQUET_DATA_PATH`         | `/app/data/parquet`       | `DATABRICKS_DATA_ROOT` env  |
| `MODELS_PATH`               | `/app/models`             | `DATABRICKS_MODELS_ROOT` env|
| `POSTGRES_HOST`             | `postgres`                | Secret `postgres-host`      |
| `POSTGRES_PASSWORD`         | `maritime123`             | Secret `postgres-password`  |
| `KAFKA_BOOTSTRAP_SERVERS`   | `kafka:9092`              | Secret `kafka-bootstrap-servers` |
| `SPARK_MASTER`              | `spark://spark-master:7077`| Databricks runtime (auto)  |

All variables are resolved in `databricks/configs/environment.py`.
