# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # 03 — Gold Layer: Silver → Aggregations + PostgreSQL Star Schema
# MAGIC
# MAGIC **Medallion stage**: Silver → Gold (analytics-ready aggregations)
# MAGIC
# MAGIC **Local counterpart**: `src/processing/gold_job.py`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Why local Docker is retained as primary runtime
# MAGIC
# MAGIC `src/processing/gold_job.py` writes four Gold Delta tables **and**
# MAGIC synchronises them to a PostgreSQL star schema via Spark JDBC.  In the
# MAGIC Docker stack the `postgres` service is reachable at `postgres:5432` on
# MAGIC the Docker bridge network — no firewall rules or VPN required.
# MAGIC
# MAGIC On Databricks the PostgreSQL endpoint must be:
# MAGIC   - Reachable from the cluster's VPC (private endpoint, VPC peering, or
# MAGIC     a publicly accessible host)
# MAGIC   - Backed by Databricks Secrets for credentials (never hard-coded)
# MAGIC
# MAGIC Until the PostgreSQL target is network-accessible from the Databricks
# MAGIC cluster, the Docker path remains the only option for JDBC writes.
# MAGIC The Gold Delta tables (vessel_latest, traffic_density, daily_stats) are
# MAGIC written in both runtimes; only the PostgreSQL sync is conditional.

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
    from configs.environment import PATHS, PG, get_spark_session, ML

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

if _IN_DATABRICKS:
    dbutils.widgets.text("silver_path",      PATHS.silver,         "Silver Delta input path")
    dbutils.widgets.text("gold_vessel_path",  PATHS.gold_vessel,    "Gold vessel_latest path")
    dbutils.widgets.text("gold_density_path", PATHS.gold_density,   "Gold traffic_density path")
    dbutils.widgets.text("gold_stats_path",   PATHS.gold_stats,     "Gold daily_stats path")
    dbutils.widgets.dropdown("write_pg",      "false", ["true", "false"], "Write to PostgreSQL?")

    SILVER_PATH      = dbutils.widgets.get("silver_path")
    GOLD_VESSEL      = dbutils.widgets.get("gold_vessel_path")
    GOLD_DENSITY     = dbutils.widgets.get("gold_density_path")
    GOLD_STATS       = dbutils.widgets.get("gold_stats_path")
    WRITE_POSTGRES   = dbutils.widgets.get("write_pg") == "true"
else:
    SILVER_PATH      = PATHS.silver
    GOLD_VESSEL      = PATHS.gold_vessel
    GOLD_DENSITY     = PATHS.gold_density
    GOLD_STATS       = PATHS.gold_stats
    WRITE_POSTGRES   = True   # Always enabled in Docker (postgres container is available)

print(f"Silver       : {SILVER_PATH}")
print(f"Gold vessel  : {GOLD_VESSEL}")
print(f"Gold density : {GOLD_DENSITY}")
print(f"Gold stats   : {GOLD_STATS}")
print(f"Write PG     : {WRITE_POSTGRES}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. SparkSession

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType

spark = get_spark_session("MaritimeAIS-Gold")
spark.conf.set("spark.sql.shuffle.partitions", "200")
spark.conf.set("spark.sql.timestampType",      "TIMESTAMP_NTZ")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Read Silver Delta

# COMMAND ----------

silver_df = spark.read.format("delta").load(SILVER_PATH).cache()
print(f"Silver records loaded: {silver_df.count():,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Gold 1 — Latest Vessel Positions (`fact_vessel_latest`)
# MAGIC
# MAGIC Mirrors `gold_job.py` lines 72-108.
# MAGIC One row per MMSI: the most recent position + risk level + ML placeholders.
# MAGIC
# MAGIC ML placeholder columns (`is_anomaly`, `anomaly_score`, `predicted_lat`, …)
# MAGIC are populated by the live scorer (`src/ml/live_scorer.py`) after this job.
# MAGIC They are included here as NULL-filled columns so the PostgreSQL schema
# MAGIC never needs an ALTER TABLE when the scorer is deployed.

# COMMAND ----------

latest_window = (
    Window
    .partitionBy("mmsi")
    .orderBy(F.col("base_datetime").desc())
)

vessel_latest_df = (
    silver_df
    .withColumn("_rank", F.row_number().over(latest_window))
    .filter(F.col("_rank") == 1)
    .drop("_rank")
    .select(
        "mmsi", "vessel_name", "vessel_type", "vessel_type_label",
        "lat", "lon", "sog", "cog", "heading",
        "risk_level", "base_datetime", "data_split",
        # ML placeholder columns
        F.lit(None).cast("boolean").alias("is_anomaly"),
        F.lit(None).cast(DoubleType()).alias("anomaly_score"),
        F.lit(None).cast(StringType()).alias("anomaly_type"),
        F.lit(None).cast(DoubleType()).alias("predicted_lat"),
        F.lit(None).cast(DoubleType()).alias("predicted_lon"),
        F.current_timestamp().alias("updated_at"),
    )
)

print(f"Unique vessels: {vessel_latest_df.count():,}")
vessel_latest_df.show(3, truncate=False)

# COMMAND ----------

(
    vessel_latest_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(GOLD_VESSEL)
)
print(f"Gold vessel_latest written → {GOLD_VESSEL}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Gold 2 — Traffic Density Grid (`fact_traffic_density`)
# MAGIC
# MAGIC Mirrors `gold_job.py` lines 111-148.
# MAGIC Aggregates vessel counts per 0.1° grid cell per hour.
# MAGIC
# MAGIC Congestion levels (from `config.py`):
# MAGIC | Level  | Threshold                 |
# MAGIC |--------|---------------------------|
# MAGIC | HIGH   | ≥ 15 vessels in cell/hour |
# MAGIC | MEDIUM | ≥ 5 vessels in cell/hour  |
# MAGIC | LOW    | < 5 vessels in cell/hour  |

# COMMAND ----------

density_df = (
    silver_df
    # 0.1° grid cells — coarser than the 0.01° fine bins used for ML features
    .withColumn("lat_grid",  (F.col("lat") / 0.1).cast(IntegerType()).cast(DoubleType()) * 0.1)
    .withColumn("lon_grid",  (F.col("lon") / 0.1).cast(IntegerType()).cast(DoubleType()) * 0.1)
    .withColumn("hour_bucket", F.date_trunc("hour", F.col("base_datetime")))
    .groupBy("lat_grid", "lon_grid", "hour_bucket")
    .agg(
        F.count("mmsi").alias("vessel_count"),
        F.countDistinct("mmsi").alias("unique_vessels"),
        F.avg("sog").cast(DoubleType()).alias("avg_sog"),
        F.sum(F.col("is_stopped").cast(IntegerType())).alias("stopped_count"),
    )
    .withColumn("congestion_level",
        F.when(F.col("vessel_count") >= ML.congestion_high,   "HIGH")
        .when(F.col("vessel_count") >= ML.congestion_medium,  "MEDIUM")
        .otherwise("LOW"))
)

print(f"Traffic density grid cells: {density_df.count():,}")
density_df.show(3, truncate=False)

# COMMAND ----------

(
    density_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(GOLD_DENSITY)
)
print(f"Gold traffic_density written → {GOLD_DENSITY}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Gold 3 — Daily Statistics (`fact_daily_stats`)
# MAGIC
# MAGIC Mirrors `gold_job.py` lines 151-182.
# MAGIC One row per calendar day: fleet-wide KPIs for the dashboard trend line.

# COMMAND ----------

daily_stats_df = (
    silver_df
    .withColumn("stat_date", F.to_date("base_datetime"))
    .groupBy("stat_date")
    .agg(
        F.countDistinct("mmsi").alias("total_vessels"),
        F.count("*").alias("total_records"),
        F.avg("sog").cast(DoubleType()).alias("avg_sog"),
        F.max("sog").cast(DoubleType()).alias("max_sog"),
        F.sum(F.when(F.col("risk_level") == "HIGH",   1).otherwise(0)).alias("high_risk_count"),
        F.sum(F.col("is_stopped").cast(IntegerType())).alias("stopped_vessels"),
        # anomaly_count stays 0 until the ML scorer runs (same as local job)
        F.lit(0).alias("anomaly_count"),
    )
    .orderBy("stat_date")
)

print(f"Daily stats rows: {daily_stats_df.count():,}")
daily_stats_df.show(5, truncate=False)

# COMMAND ----------

(
    daily_stats_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(GOLD_STATS)
)
print(f"Gold daily_stats written → {GOLD_STATS}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Gold 4 — Vessel Dimension (`dim_vessel`)
# MAGIC
# MAGIC Mirrors `gold_job.py` lines 185-215.
# MAGIC One row per unique MMSI: static metadata + activity window.

# COMMAND ----------

dim_vessel_df = (
    silver_df
    .groupBy("mmsi")
    .agg(
        F.first("vessel_name",  ignorenulls=True).alias("vessel_name"),
        F.first("imo",          ignorenulls=True).alias("imo"),
        F.first("call_sign",    ignorenulls=True).alias("call_sign"),
        F.first("vessel_type",  ignorenulls=True).alias("vessel_type"),
        F.first("vessel_type_label", ignorenulls=True).alias("vessel_type_label"),
        F.first("length",       ignorenulls=True).alias("length"),
        F.first("width",        ignorenulls=True).alias("width"),
        F.first("draft",        ignorenulls=True).alias("draft"),
        F.first("cargo",        ignorenulls=True).alias("cargo"),
        F.first("transceiver_class", ignorenulls=True).alias("transceiver_class"),
        F.min("base_datetime").alias("first_seen"),
        F.max("base_datetime").alias("last_seen"),
        F.count("*").alias("total_records"),
    )
)

print(f"Dimension vessels: {dim_vessel_df.count():,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. PostgreSQL JDBC write
# MAGIC
# MAGIC Mirrors `gold_job.py` lines 218-243.
# MAGIC
# MAGIC **DATABRICKS NOTE**: The `postgres` hostname used in Docker resolves on
# MAGIC the Docker bridge network only.  Before enabling `write_pg=true` on
# MAGIC Databricks you must:
# MAGIC
# MAGIC 1. Configure network connectivity (VPC peering / private endpoint / public IP)
# MAGIC 2. Store credentials in Databricks Secrets (`scope=maritime`):
# MAGIC    - `postgres-host`, `postgres-port`, `postgres-db`
# MAGIC    - `postgres-user`, `postgres-password`
# MAGIC 3. Install the PostgreSQL JDBC driver on the cluster:
# MAGIC    `org.postgresql:postgresql:42.7.3` (Maven coordinate)
# MAGIC
# MAGIC The `environment.py` config already reads these secrets — no code change
# MAGIC is needed here, only infrastructure setup.

# COMMAND ----------

def _write_jdbc(df, table_name: str):
    """Write a Gold DataFrame to PostgreSQL via JDBC."""
    (
        df.write
        .format("jdbc")
        .option("url",      PG.jdbc_url)
        .option("dbtable",  table_name)
        .option("user",     PG.user)
        .option("password", PG.password)
        .option("driver",   "org.postgresql.Driver")
        .mode("overwrite")
        .save()
    )
    print(f"  → wrote {df.count():,} rows to {table_name}")


if WRITE_POSTGRES:
    print("Writing to PostgreSQL star schema …")
    _write_jdbc(vessel_latest_df,  "fact_vessel_latest")
    _write_jdbc(density_df,        "fact_traffic_density")
    _write_jdbc(daily_stats_df,    "fact_daily_stats")
    _write_jdbc(dim_vessel_df,     "dim_vessel")
    print("PostgreSQL sync complete.")
else:
    print("[SKIPPED] PostgreSQL write disabled — set write_pg=true and configure network access.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Register in Unity Catalog (Databricks only)

# COMMAND ----------

if _IN_DATABRICKS:
    for tbl, path in [
        ("maritime.gold.vessel_latest",   GOLD_VESSEL),
        ("maritime.gold.traffic_density", GOLD_DENSITY),
        ("maritime.gold.daily_stats",     GOLD_STATS),
    ]:
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {tbl}
            USING DELTA LOCATION '{path}'
        """)
        print(f"Registered: {tbl}")
else:
    print("[LOCAL] Skipping Unity Catalog registration.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Summary metrics

# COMMAND ----------

from delta.tables import DeltaTable

for label, path in [
    ("vessel_latest",   GOLD_VESSEL),
    ("traffic_density", GOLD_DENSITY),
    ("daily_stats",     GOLD_STATS),
]:
    cnt = spark.read.format("delta").load(path).count()
    print(f"{label:20s}: {cnt:>10,} rows  →  {path}")
