"""
gold_job.py — Maritime Navigation AI System
Silver → Gold aggregations + PostgreSQL Star Schema sync.

Builds:
  Gold 1: Latest vessel position  → fact_vessel_latest
  Gold 2: Traffic density/hour   → fact_traffic_density
  Gold 3: Daily statistics       → fact_daily_stats
  All synced to PostgreSQL Star Schema tables.

Run after silver_job.py:
    docker compose exec spark-master \\
      /opt/spark/bin/spark-submit \\
        --packages io.delta:delta-core_2.12:2.4.0 \\
        src/processing/gold_job.py
"""
import sys
sys.path.insert(0, "/opt/spark/app/src/common")

from config import (
    SPARK_MASTER, POSTGRES_URL,
    DELTA_SILVER_PATH,
    DELTA_GOLD_VESSEL_PATH,
    DELTA_GOLD_DENSITY_PATH,
    DELTA_GOLD_STATS_PATH,
    CONGESTION_HIGH, CONGESTION_MEDIUM,
)
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, last, max as spark_max, min as spark_min,
    avg, count, countDistinct,
    date_trunc, when, lit,
    round as spark_round, to_date,
    current_timestamp,
)
from pyspark.sql.window import Window

# Build clean JDBC URL without credentials embedded
JDBC_URL = "jdbc:postgresql://postgres:5432/maritime"

JDBC_PROPS = {
    "url":      JDBC_URL,
    "driver":   "org.postgresql.Driver",
    "user":     "maritime",
    "password": "maritime123",
}


def build_spark():
    return (
        SparkSession.builder
        .appName("Gold_AIS_Aggregator")
        .master(SPARK_MASTER)
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def build_gold_vessel_latest(silver_df):
    """
    Gold 1: Latest position per vessel.
    What the LIVE MAP shows.
    """
    w = Window.partitionBy("mmsi").orderBy(col("base_datetime").desc())

    return (
        silver_df
        .withColumn("rn", __import__(
            "pyspark.sql.functions", fromlist=["row_number"]
        ).row_number().over(w))
        .filter(col("rn") == 1)
        .drop("rn")
        .select(
            "mmsi", "vessel_name", "vessel_type",
            "lat", "lon", "sog", "cog", "heading",
            "risk_level", "base_datetime", "data_split",
        )
        .withColumn("is_anomaly",    lit(False))
        .withColumn("anomaly_score", lit(0.0))
        .withColumn("anomaly_type",  lit(""))
        .withColumn("predicted_lat", lit(None).cast("double"))
        .withColumn("predicted_lon", lit(None).cast("double"))
        .withColumn("updated_at",    current_timestamp())
    )


def build_gold_density(silver_df):
    """
    Gold 2: Vessel density per 0.1° grid cell per hour.
    Used for heatmap and congestion detection.
    """
    return (
        silver_df
        .withColumn("hour_bucket",
                    date_trunc("hour", col("base_datetime")))
        .groupBy("lat_bin", "lon_bin", "hour_bucket", "data_split")
        .agg(
            count("*").alias("vessel_count"),
            countDistinct("mmsi").alias("unique_vessels"),
            avg("sog").alias("avg_sog"),
            count(when(col("sog") < 0.5, True)).alias("stopped_count"),
        )
        .withColumn(
            "congestion_level",
            when(col("vessel_count") >= CONGESTION_HIGH,   "HIGH")
            .when(col("vessel_count") >= CONGESTION_MEDIUM, "MEDIUM")
            .otherwise("LOW")
        )
    )


def build_gold_stats(silver_df):
    """
    Gold 3: Daily statistics for analytics dashboard.
    """
    return (
        silver_df
        .withColumn("stat_date", to_date(col("base_datetime")))
        .groupBy("stat_date", "data_split")
        .agg(
            countDistinct("mmsi").alias("total_vessels"),
            count("*").alias("total_records"),
            spark_round(avg("sog"), 2).alias("avg_sog"),
            spark_round(spark_max("sog"), 2).alias("max_sog"),
            count(when(col("risk_level") == "HIGH",   True)).alias("high_risk_count"),
            count(when(col("sog") < 0.5,              True)).alias("stopped_vessels"),
        )
        .withColumn("anomaly_count", lit(0))
    )

def sync_to_postgres(df, table_name: str, mode: str = "overwrite"):
    (
        df.write
        .format("jdbc")
        .option("url",      JDBC_URL)
        .option("dbtable",  table_name)
        .option("user",     JDBC_PROPS["user"])
        .option("password", JDBC_PROPS["password"])
        .option("driver",   JDBC_PROPS["driver"])
        .mode(mode)
        .save()
    )
    print(f"  Synced -> PostgreSQL:{table_name}")


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"📂  Reading Silver Delta: {DELTA_SILVER_PATH}")
    silver = spark.read.format("delta").load(DELTA_SILVER_PATH)
    # Do NOT cache all 36M rows — filter first
    total = silver.count()
    print(f"    Silver rows: {total:,}")


    # ── Gold 1: Latest vessel positions (Delta only — NOT synced to PostgreSQL)
    # fact_vessel_latest is owned by live_scorer.py which maintains it in
    # real-time via UPSERT. Syncing here would truncate the table mid-run,
    # causing a brief live map outage. Delta Gold is kept for ML / analytics use.
    print("\n🔨  Building Gold: Vessel Latest (Delta only) ...")
    vessel_latest = build_gold_vessel_latest(silver)
    (
        vessel_latest.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(DELTA_GOLD_VESSEL_PATH)
    )
    print(f"✅  Gold Vessel Latest (Delta): "
          f"{vessel_latest.count():,} unique vessels")

    # ── Gold 2: Traffic density ───────────────────────────────────────────────
    print("\n🔨  Building Gold: Traffic Density ...")
    density = build_gold_density(silver)
    (
        density.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(DELTA_GOLD_DENSITY_PATH)
    )
    sync_to_postgres(density, "fact_traffic_density")
    high_zones = density.filter(
        col("congestion_level") == "HIGH"
    ).count()
    print(f"✅  Gold Traffic Density: {high_zones} HIGH congestion zones")

    # ── Gold 3: Daily stats ───────────────────────────────────────────────────
    print("\n🔨  Building Gold: Daily Stats ...")
    stats = build_gold_stats(silver)
    (
        stats.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(DELTA_GOLD_STATS_PATH)
    )
    sync_to_postgres(stats, "fact_daily_stats")
    print(f"✅  Gold Daily Stats: {stats.count()} days")

    # ── Populate dim_vessel from Silver ───────────────────────────────────────
    print("\n🔨  Building dim_vessel ...")
    dim_vessel = (
        silver
        .groupBy("mmsi")
        .agg(
            last("vessel_name",  True).alias("vessel_name"),
            last("imo",          True).alias("imo"),
            last("call_sign",    True).alias("call_sign"),
            last("vessel_type",  True).alias("vessel_type"),
            last("vessel_type_label", True).alias("vessel_type_label"),
            spark_max("length").alias("length"),
            spark_max("width").alias("width"),
            spark_max("draft").alias("draft"),
            last("cargo",        True).alias("cargo"),
            last("transceiver_class", True).alias("transceiver_class"),
            last("data_split",   True).alias("data_split"),
            spark_min("base_datetime").alias("first_seen"),
            spark_max("base_datetime").alias("last_seen"),
            count("*").alias("total_records"),
        )
    )
    sync_to_postgres(dim_vessel, "dim_vessel")
    print(f"✅  dim_vessel: {dim_vessel.count():,} unique vessels")

    silver.unpersist()
    spark.stop()

    print("\n" + "=" * 50)
    print("  Gold Layer Complete!")
    print("  PostgreSQL Star Schema populated.")
    print("\n  Next steps:")
    print("  1. Train ML models: python src/ml/train_anomaly.py")
    print("  2. Start Kafka producer")
    print("  3. Open dashboard: http://localhost:8501")


if __name__ == "__main__":
    main()
