# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 01 — Bronze Layer: Raw AIS Parquet → Delta Lake
# MAGIC
# MAGIC **Medallion stage**: Raw → Bronze (append-only, enriched)
# MAGIC
# MAGIC **Local counterpart**: `src/processing/bronze_job.py`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Why local Docker is retained as primary runtime
# MAGIC
# MAGIC `src/processing/bronze_job.py` runs directly inside the `spark-master` /
# MAGIC `spark-worker` containers defined in `docker-compose.yml`.  That setup is
# MAGIC the canonical development target because it:
# MAGIC
# MAGIC - Requires no cloud account or cluster provisioning
# MAGIC - Writes Delta tables to a Docker named volume (`delta_data`) that every
# MAGIC   Spark container mounts — no path translation needed
# MAGIC - Lets the whole pipeline (Kafka → Bronze → Silver → Gold → Postgres) run
# MAGIC   with a single `docker compose up`
# MAGIC
# MAGIC This notebook is the **Databricks scaling path**: use it when the AIS dataset
# MAGIC outgrows the two 4-core Docker workers, or when the job needs Databricks Job
# MAGIC scheduler / SLA monitoring.  The business logic is intentionally identical to
# MAGIC the local job so both can be maintained in parallel without divergence.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup — attach environment config
# MAGIC
# MAGIC `%run` executes the sibling config notebook in the same session, making
# MAGIC `PATHS`, `get_spark_session()`, `PG`, and `KAFKA` available here.
# MAGIC Locally (outside Databricks) import the module directly instead.

# COMMAND ----------

# Detect whether we are running inside Databricks to choose the import path.
import os

_IN_DATABRICKS = bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))

if _IN_DATABRICKS:
    # MAGIC %run ../configs/environment
    pass  # `%run` magic is executed by the Databricks runtime above
else:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from configs.environment import PATHS, PG, KAFKA, RUNTIME, RuntimeMode, get_spark_session

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters
# MAGIC
# MAGIC Databricks Jobs can override these widgets at runtime.
# MAGIC Locally they fall back to environment / config defaults.

# COMMAND ----------

if _IN_DATABRICKS:
    dbutils.widgets.text("source_path",  PATHS.data_root, "Parquet source directory")
    dbutils.widgets.text("bronze_path",  PATHS.bronze,    "Bronze Delta output path")
    dbutils.widgets.text("write_mode",   "append",        "Write mode (append | overwrite)")

    SOURCE_PATH = dbutils.widgets.get("source_path")
    BRONZE_PATH = dbutils.widgets.get("bronze_path")
    WRITE_MODE  = dbutils.widgets.get("write_mode")
else:
    SOURCE_PATH = PATHS.data_root
    BRONZE_PATH = PATHS.bronze
    WRITE_MODE  = "append"

print(f"Source      : {SOURCE_PATH}")
print(f"Bronze path : {BRONZE_PATH}")
print(f"Write mode  : {WRITE_MODE}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. SparkSession

# COMMAND ----------

spark = get_spark_session("MaritimeAIS-Bronze")

# Databricks uses TIMESTAMP_NTZ by default since DBR 11; set it explicitly
# so the notebook behaves identically to bronze_job.py which also enables it.
spark.conf.set("spark.sql.timestampType", "TIMESTAMP_NTZ")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Read Parquet source files
# MAGIC
# MAGIC Identical to the local job: read all Parquet files in the source directory.
# MAGIC On Databricks the path is a DBFS mount; locally it is a Docker volume path.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, TimestampNTZType

# ---------------------------------------------------------------------------
# Schema note: Parquet files were written by convert_csv.py with canonical
# column names (mmsi, base_datetime, lat, lon, sog, …).  We rely on schema
# inference here — same approach as bronze_job.py — because the Parquet
# schema is already normalised by schema_utils.resolve_columns().
# ---------------------------------------------------------------------------

raw_df = spark.read.parquet(SOURCE_PATH)

print(f"Raw records loaded: {raw_df.count():,}")
raw_df.printSchema()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Quality filters
# MAGIC
# MAGIC Mirrors the filter block in `bronze_job.py` (lines 68-76).
# MAGIC Rejects: null MMSI, out-of-bounds lat/lon, implausible SOG > 60 knots.

# COMMAND ----------

filtered_df = raw_df.filter(
    F.col("mmsi").isNotNull()
    & F.col("lat").cast(DoubleType()).between(-90.0, 90.0)
    & F.col("lon").cast(DoubleType()).between(-180.0, 180.0)
    & (F.col("sog").cast(DoubleType()) >= 0)
    & (F.col("sog").cast(DoubleType()) <= 60)
)

print(f"Records after quality filter: {filtered_df.count():,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Enrichment columns
# MAGIC
# MAGIC Adds operational flags and zone membership — mirrors `bronze_job.py` lines 79-111.
# MAGIC
# MAGIC ### Speed flags
# MAGIC | Column        | Condition     | Meaning                       |
# MAGIC |---------------|---------------|-------------------------------|
# MAGIC | `is_stopped`  | SOG < 0.5 kt  | Vessel stationary             |
# MAGIC | `is_slow`     | SOG < 2.0 kt  | Very slow / drifting          |
# MAGIC | `is_speeding` | SOG > 30.0 kt | Unusually fast — review needed|
# MAGIC
# MAGIC ### Zone flags
# MAGIC | Column         | Region                              |
# MAGIC |----------------|-------------------------------------|
# MAGIC | `in_suez_zone` | Suez Canal: lat 29.5–31.5, lon 31–33.5 |
# MAGIC
# MAGIC ### Risk classification (rule-based, pre-ML)
# MAGIC | Level  | Condition                                    |
# MAGIC |--------|----------------------------------------------|
# MAGIC | HIGH   | SOG < 1.0 AND in Suez zone                   |
# MAGIC | MEDIUM | SOG < 2.0 (slow anywhere)                    |
# MAGIC | LOW    | All other vessels                            |

# COMMAND ----------

enriched_df = (
    filtered_df
    # ---- timestamps -------------------------------------------------------
    .withColumn("event_time",
        F.col("base_datetime").cast(TimestampNTZType()))
    .withColumn("ingestion_time", F.current_timestamp())

    # ---- speed flags -------------------------------------------------------
    .withColumn("is_stopped",   F.col("sog").cast(DoubleType()) < 0.5)
    .withColumn("is_slow",      F.col("sog").cast(DoubleType()) < 2.0)
    .withColumn("is_speeding",  F.col("sog").cast(DoubleType()) > 30.0)

    # ---- Suez Canal zone ---------------------------------------------------
    # Bounding box from config.py: SUEZ_LAT_MIN/MAX, SUEZ_LON_MIN/MAX
    .withColumn("in_suez_zone",
        (F.col("lat").cast(DoubleType()).between(29.5, 31.5))
        & (F.col("lon").cast(DoubleType()).between(31.0, 33.5)))

    # ---- rule-based risk level (overwritten by ML scorer in gold layer) ---
    .withColumn("risk_level",
        F.when(
            (F.col("sog").cast(DoubleType()) < 1.0) & F.col("in_suez_zone"),
            "HIGH"
        ).when(
            F.col("sog").cast(DoubleType()) < 2.0,
            "MEDIUM"
        ).otherwise("LOW"))
)

enriched_df.select(
    "mmsi", "base_datetime", "lat", "lon", "sog",
    "event_time", "ingestion_time",
    "is_stopped", "is_slow", "is_speeding", "in_suez_zone", "risk_level"
).show(5, truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Write to Bronze Delta table
# MAGIC
# MAGIC `mergeSchema=true` lets new Parquet source columns flow in without
# MAGIC requiring a schema migration — matches the local job behaviour.

# COMMAND ----------

(
    enriched_df
    .write
    .format("delta")
    .mode(WRITE_MODE)
    .option("mergeSchema", "true")
    # Databricks auto-optimises file sizes; locally Delta manages this itself
    .option("delta.autoOptimize.optimizeWrite", "true")
    .save(BRONZE_PATH)
)

print(f"Bronze write complete → {BRONZE_PATH}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Verification

# COMMAND ----------

from delta.tables import DeltaTable

bronze_tbl = DeltaTable.forPath(spark, BRONZE_PATH)
bronze_hist = bronze_tbl.history(5)
bronze_hist.select("version", "timestamp", "operation", "operationMetrics").show(truncate=False)

final_count = spark.read.format("delta").load(BRONZE_PATH).count()
print(f"Bronze table total records: {final_count:,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Register in Unity Catalog (Databricks only)
# MAGIC
# MAGIC Registering makes the table visible in the Databricks Data Explorer and
# MAGIC queryable via SQL.  Skipped locally — the local job does not register
# MAGIC tables in any metastore.

# COMMAND ----------

if _IN_DATABRICKS:
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS maritime.bronze.ais
        USING DELTA
        LOCATION '{BRONZE_PATH}'
    """)
    print("Registered: maritime.bronze.ais in Unity Catalog")
else:
    print("[LOCAL] Skipping Unity Catalog registration — not applicable in Docker mode.")
