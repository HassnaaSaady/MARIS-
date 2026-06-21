"""
bronze_job.py — Maritime Navigation AI System
Loads converted Parquet files into Bronze Delta Lake.
Adds enrichment: risk flags, US Coastal Zones / Port Channels, speed flags.

Run after convert_csv.py:
    docker compose exec spark-master \
      /opt/spark/bin/spark-submit \
        --packages io.delta:delta-core_2.12:2.4.0 \
        /opt/spark/app/src/processing/bronze_job.py
"""
import sys
sys.path.insert(0, "/opt/spark/app/src/common")

from config import (
    SPARK_MASTER, PARQUET_DATA_PATH,
    DELTA_BRONZE_PATH, US_PORT_ZONES,
    ANOMALY_SPEED_MAX,
)
from functools import reduce
from operator import or_ as _or
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, current_timestamp, when,
    round as spark_round, lit,
    year, month, dayofmonth, hour,
    to_timestamp,
)


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("Bronze_AIS_Loader")
        .master(SPARK_MASTER)
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.parquet.datetimeRebaseModeInRead",  "CORRECTED")
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        .config("spark.sql.parquet.int96RebaseModeInRead",     "CORRECTED")
        .config("spark.sql.parquet.int96RebaseModeInWrite",    "CORRECTED")
        .getOrCreate()
    )


def enrich(df):
    """Add Bronze enrichment columns to raw Parquet data."""
    return (
        df
        # Keep base_datetime as string in Bronze; Silver converts to timestamp
        .withColumn("base_datetime",
                    col("base_datetime").cast("string"))
        .withColumn("event_time",
                    to_timestamp(col("base_datetime")))
        .withColumn("ingestion_time",
                    current_timestamp())

        # Speed flags
        .withColumn("is_stopped",  col("sog") < 0.5)
        .withColumn("is_slow",     col("sog") < 2.0)
        .withColumn("is_speeding", col("sog") > lit(ANOMALY_SPEED_MAX))

        # US Coastal Zones / Port Channels flag
        .withColumn(
            "in_us_port_zone",
            reduce(_or, [
                col("lat").between(z["lat_min"], z["lat_max"]) &
                col("lon").between(z["lon_min"], z["lon_max"])
                for z in US_PORT_ZONES
            ])
        )

        # Risk level — rule-based baseline before ML models exist
        .withColumn(
            "risk_level",
            when(
                (col("sog") < 0.5) & col("in_us_port_zone"), "HIGH"
            ).when(
                col("sog") < 2.0, "MEDIUM"
            ).otherwise("LOW")
        )
    )


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"📂  Reading Parquet from: {PARQUET_DATA_PATH}")
    df = spark.read.parquet(PARQUET_DATA_PATH)

    total   = df.count()
    vessels = df.select("mmsi").distinct().count()
    print(f"    Total rows    : {total:,}")
    print(f"    Total vessels : {vessels:,}")
    print(f"    Columns       : {df.columns}")

    # ── Quality filters ────────────────────────────────────────────────────────
    df = (
        df
        .filter(col("mmsi").isNotNull() & (col("mmsi") != ""))
        .filter(col("lat").isNotNull() & col("lon").isNotNull())
        .filter((col("lat") != 0.0) & (col("lon") != 0.0))
        .filter(col("lat").between(-90, 90))
        .filter(col("lon").between(-180, 180))
        .filter(col("sog").between(0, 60))
    )

    # ── Enrich ─────────────────────────────────────────────────────────────────
    df_enriched = enrich(df)

    # ── Write to Bronze Delta ─────────────────────────────────────────────────
    # overwrite: rebuilds the batch snapshot cleanly.
    # spark_streaming_consumer.py uses mode("append") for the live path.
    print(f"\n💾  Writing to Bronze Delta: {DELTA_BRONZE_PATH}")
    (
        df_enriched.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(DELTA_BRONZE_PATH)
    )

    # ── Verify ─────────────────────────────────────────────────────────────────
    count = spark.read.format("delta").load(DELTA_BRONZE_PATH).count()
    print(f"✅  Bronze Delta now has {count:,} records")
    print(f"\n    Next: run silver_job.py")
    spark.stop()


if __name__ == "__main__":
    main()