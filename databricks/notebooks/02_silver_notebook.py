# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 02 — Silver Layer: Bronze → Cleaned & Feature-Engineered AIS
# MAGIC
# MAGIC **Medallion stage**: Bronze → Silver (deduplicated, filled, ML features)
# MAGIC
# MAGIC **Local counterpart**: `src/processing/silver_job.py`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Why local Docker is retained as primary runtime
# MAGIC
# MAGIC `src/processing/silver_job.py` runs on the Docker Spark cluster and writes
# MAGIC Silver Delta tables to the shared `delta_data` volume.  The local runtime
# MAGIC is the daily development loop — fast iteration, no IAM, no cluster cold-start.
# MAGIC
# MAGIC This notebook targets Databricks for two scenarios:
# MAGIC 1. **Scale**: The silver job uses 200 shuffle partitions and 3 GB executor
# MAGIC    memory; if vessel count grows the Docker workers become the bottleneck.
# MAGIC 2. **Scheduling**: Databricks Jobs + Workflows let this run on a cron after
# MAGIC    the bronze job completes, with retry logic and alerting.
# MAGIC
# MAGIC The transformation logic (dedup, forward-fill, Haversine features) is kept
# MAGIC byte-for-byte identical to the local job so both paths can be tested against
# MAGIC the same expected output.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import os

_IN_DATABRICKS = bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))

if _IN_DATABRICKS:
    # MAGIC %run ../configs/environment
    pass
else:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from configs.environment import PATHS, get_spark_session

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

if _IN_DATABRICKS:
    dbutils.widgets.text("bronze_path", PATHS.bronze, "Bronze Delta input path")
    dbutils.widgets.text("silver_path", PATHS.silver, "Silver Delta output path")
    # Filter to TRAIN + TEST splits to exclude LIVE data from feature engineering
    # (mirrors the data_split filter in silver_job.py lines 53-56)
    dbutils.widgets.text("splits", "TRAIN,TEST", "Comma-separated data splits to process")

    BRONZE_PATH = dbutils.widgets.get("bronze_path")
    SILVER_PATH = dbutils.widgets.get("silver_path")
    SPLITS      = dbutils.widgets.get("splits").split(",")
else:
    BRONZE_PATH = PATHS.bronze
    SILVER_PATH = PATHS.silver
    SPLITS      = ["TRAIN", "TEST"]

print(f"Bronze path : {BRONZE_PATH}")
print(f"Silver path : {SILVER_PATH}")
print(f"Splits      : {SPLITS}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. SparkSession

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.types import DoubleType, IntegerType, LongType

spark = get_spark_session("MaritimeAIS-Silver")

# Silver job uses more shuffle partitions than bronze because of the
# per-vessel window operations (dedup + forward-fill + feature calc).
# Match the docker-compose SPARK_SHUFFLE_PARTS=200 setting.
spark.conf.set("spark.sql.shuffle.partitions", "200")
spark.conf.set("spark.sql.timestampType", "TIMESTAMP_NTZ")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Read from Bronze Delta

# COMMAND ----------

bronze_df = (
    spark.read
    .format("delta")
    .load(BRONZE_PATH)
    .filter(F.col("data_split").isin(SPLITS))
)

print(f"Bronze records (filtered splits {SPLITS}): {bronze_df.count():,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Deduplication
# MAGIC
# MAGIC Keep at most one record per vessel per minute.
# MAGIC Mirrors `silver_job.py` lines 62-75: rank by (mmsi, base_datetime),
# MAGIC keep rank == 1 (first occurrence within the minute).

# COMMAND ----------

dedup_window = (
    Window
    .partitionBy("mmsi", F.date_trunc("minute", F.col("base_datetime")))
    .orderBy(F.col("base_datetime").asc())
)

deduped_df = (
    bronze_df
    .withColumn("_rank", F.rank().over(dedup_window))
    .filter(F.col("_rank") == 1)
    .drop("_rank")
)

print(f"After deduplication: {deduped_df.count():,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Forward-fill vessel metadata
# MAGIC
# MAGIC AIS messages often omit static vessel fields (name, IMO, type) for
# MAGIC position-only reports.  A per-vessel unbounded window fills these
# MAGIC forward in time — mirrors `silver_job.py` lines 80-105.

# COMMAND ----------

fill_window = (
    Window
    .partitionBy("mmsi")
    .orderBy("base_datetime")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)

filled_df = (
    deduped_df
    .withColumn("vessel_name",  F.last("vessel_name",  ignorenulls=True).over(fill_window))
    .withColumn("vessel_type",  F.last("vessel_type",  ignorenulls=True).over(fill_window))
    .withColumn("imo",          F.last("imo",          ignorenulls=True).over(fill_window))
    .withColumn("call_sign",    F.last("call_sign",    ignorenulls=True).over(fill_window))
    .withColumn("length",       F.last("length",       ignorenulls=True).over(fill_window))
    .withColumn("width",        F.last("width",        ignorenulls=True).over(fill_window))
    .withColumn("draft",        F.last("draft",        ignorenulls=True).over(fill_window))
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Human-readable labels
# MAGIC
# MAGIC Maps integer codes to readable strings.
# MAGIC Matches the lookup tables in `silver_job.py` lines 108-131 and
# MAGIC `schema_utils.get_vessel_type_label()`.

# COMMAND ----------

vessel_type_map = {
    30: "Fishing",   31: "Towing",       32: "Towing (Large)",
    33: "Dredging",  34: "Diving",        35: "Military",
    36: "Sailing",   37: "Pleasure",      50: "Pilot Vessel",
    51: "SAR",       52: "Tug",           53: "Port Tender",
    60: "Passenger", 70: "Cargo",         80: "Tanker",
    90: "Other",
}

status_map = {
    0: "Underway/Engine", 1: "At Anchor",   2: "Not Under Command",
    3: "Restricted",      5: "Moored",       6: "Aground",
    7: "Fishing",         8: "Underway/Sail",
}

vessel_type_expr = F.create_map(
    *[item for pair in [(F.lit(k), F.lit(v)) for k, v in vessel_type_map.items()] for item in pair]
)
status_expr = F.create_map(
    *[item for pair in [(F.lit(k), F.lit(v)) for k, v in status_map.items()] for item in pair]
)

labelled_df = (
    filled_df
    .withColumn("vessel_type_label",
        F.coalesce(vessel_type_expr[F.col("vessel_type").cast(IntegerType())],
                   F.lit("Unknown")))
    .withColumn("status_label",
        F.coalesce(status_expr[F.col("status").cast(IntegerType())],
                   F.lit("Unknown")))
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. ML Feature Engineering
# MAGIC
# MAGIC Per-vessel, ordered by time window.  All features mirror `silver_job.py`
# MAGIC lines 134-185.
# MAGIC
# MAGIC | Feature           | Description                                          |
# MAGIC |-------------------|------------------------------------------------------|
# MAGIC | `sog_change`      | Δ speed from previous AIS report (knots)             |
# MAGIC | `heading_change`  | Circular Δ heading (accounts for 350°→10° = 20°)     |
# MAGIC | `time_delta_sec`  | Seconds between consecutive AIS pings                |
# MAGIC | `distance_nm`     | Haversine distance in nautical miles                 |
# MAGIC | `lat_bin_fine`    | 0.01° latitude grid cell                             |
# MAGIC | `lon_bin_fine`    | 0.01° longitude grid cell                            |

# COMMAND ----------

vessel_time_window = (
    Window
    .partitionBy("mmsi")
    .orderBy("base_datetime")
)

import math

featured_df = (
    labelled_df

    # ---- speed change -------------------------------------------------------
    .withColumn("sog_change",
        F.col("sog").cast(DoubleType())
        - F.lag("sog", 1).over(vessel_time_window).cast(DoubleType()))

    # ---- heading change (circular arithmetic) --------------------------------
    # Δh = h_now - h_prev; normalise to (-180, 180] to handle 350→10 = +20°
    .withColumn("_hdg_raw",
        F.col("heading").cast(DoubleType())
        - F.lag("heading", 1).over(vessel_time_window).cast(DoubleType()))
    .withColumn("heading_change",
        F.when(F.col("_hdg_raw") > 180,  F.col("_hdg_raw") - 360)
        .when(F.col("_hdg_raw") < -180,  F.col("_hdg_raw") + 360)
        .otherwise(F.col("_hdg_raw")))
    .drop("_hdg_raw")

    # ---- time delta ---------------------------------------------------------
    .withColumn("time_delta_sec",
        (F.unix_timestamp("base_datetime")
         - F.lag(F.unix_timestamp("base_datetime"), 1).over(vessel_time_window))
        .cast(LongType()))

    # ---- Haversine distance in nautical miles --------------------------------
    # Formula: 2R * arcsin(sqrt(sin²(Δlat/2) + cos(lat1)*cos(lat2)*sin²(Δlon/2)))
    # R = 3440.065 nm (Earth radius in nautical miles)
    .withColumn("_lat1",  F.lag("lat",  1).over(vessel_time_window).cast(DoubleType()))
    .withColumn("_lon1",  F.lag("lon",  1).over(vessel_time_window).cast(DoubleType()))
    .withColumn("_lat2",  F.col("lat").cast(DoubleType()))
    .withColumn("_lon2",  F.col("lon").cast(DoubleType()))
    .withColumn("_dlat",  F.radians(F.col("_lat2") - F.col("_lat1")))
    .withColumn("_dlon",  F.radians(F.col("_lon2") - F.col("_lon1")))
    .withColumn("_a",
        F.pow(F.sin(F.col("_dlat") / 2), 2)
        + F.cos(F.radians(F.col("_lat1")))
        * F.cos(F.radians(F.col("_lat2")))
        * F.pow(F.sin(F.col("_dlon") / 2), 2))
    .withColumn("distance_nm",
        F.lit(2 * 3440.065) * F.asin(F.sqrt(F.col("_a"))))
    .drop("_lat1", "_lon1", "_lat2", "_lon2", "_dlat", "_dlon", "_a")

    # ---- 0.01° grid bins (for density heat-maps and ML locality features) ---
    .withColumn("lat_bin_fine", (F.col("lat").cast(DoubleType()) / 0.01).cast(IntegerType()) * 0.01)
    .withColumn("lon_bin_fine", (F.col("lon").cast(DoubleType()) / 0.01).cast(IntegerType()) * 0.01)
)

featured_df.select(
    "mmsi", "base_datetime", "sog_change", "heading_change",
    "time_delta_sec", "distance_nm", "lat_bin_fine", "lon_bin_fine"
).show(5, truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Partition columns
# MAGIC
# MAGIC Add year/month/day so Databricks (and the local Spark job) can prune
# MAGIC efficiently when the gold layer reads recent windows.

# COMMAND ----------

partitioned_df = (
    featured_df
    .withColumn("year",  F.year("base_datetime"))
    .withColumn("month", F.month("base_datetime"))
    .withColumn("day",   F.dayofmonth("base_datetime"))
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Write Silver Delta table

# COMMAND ----------

(
    partitioned_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.autoOptimize.optimizeWrite", "true")
    .partitionBy("year", "month", "day")
    .save(SILVER_PATH)
)

print(f"Silver write complete → {SILVER_PATH}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Verification + OPTIMIZE
# MAGIC
# MAGIC `OPTIMIZE` and `ZORDER BY` are Databricks-native commands that compact
# MAGIC small files and cluster data for faster downstream reads.
# MAGIC On the local Docker cluster these are also supported by delta-core 2.4.0
# MAGIC but are optional — the Docker volume already has good file locality.

# COMMAND ----------

from delta.tables import DeltaTable

silver_tbl = DeltaTable.forPath(spark, SILVER_PATH)
silver_tbl.history(3).select("version", "timestamp", "operation").show(truncate=False)

final_count = spark.read.format("delta").load(SILVER_PATH).count()
print(f"Silver table total records: {final_count:,}")

if _IN_DATABRICKS:
    # Compact files and cluster by mmsi + base_datetime for fast gold reads
    spark.sql(f"OPTIMIZE delta.`{SILVER_PATH}` ZORDER BY (mmsi, base_datetime)")
    print("OPTIMIZE + ZORDER complete.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Register in Unity Catalog (Databricks only)

# COMMAND ----------

if _IN_DATABRICKS:
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS maritime.silver.ais_clean
        USING DELTA
        LOCATION '{SILVER_PATH}'
    """)
    print("Registered: maritime.silver.ais_clean in Unity Catalog")
else:
    print("[LOCAL] Skipping Unity Catalog registration — not applicable in Docker mode.")
