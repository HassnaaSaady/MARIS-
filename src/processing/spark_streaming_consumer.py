"""
spark_streaming_consumer.py — Maritime Navigation AI System
Reads LIVE AIS messages from Kafka → Bronze Delta (append).
Also scores each record with loaded ML model.
Runs continuously during live demo.

Started automatically by docker-compose spark-stream service.
"""
import sys, time
sys.path.insert(0, "/opt/spark/app/src/common")

from config import (
    KAFKA_BOOTSTRAP_SERVERS, AIS_TOPIC,
    DELTA_BRONZE_PATH, CHECKPOINT_BRONZE,
    SPARK_MASTER, US_PORT_ZONES, ANOMALY_SPEED_MAX,
)
from functools import reduce
from operator import or_ as _or
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, current_timestamp, to_timestamp,
    when, round as spark_round, lit,
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
)

SCHEMA = StructType([
    StructField("mmsi",              StringType(), True),
    StructField("base_datetime",     StringType(), True),
    StructField("lat",               DoubleType(), True),
    StructField("lon",               DoubleType(), True),
    StructField("sog",               DoubleType(), True),
    StructField("cog",               DoubleType(), True),
    StructField("heading",           DoubleType(), True),
    StructField("vessel_name",       StringType(), True),
    StructField("imo",               StringType(), True),
    StructField("call_sign",         StringType(), True),
    StructField("vessel_type",       StringType(), True),
    StructField("status",            StringType(), True),
    StructField("length",            DoubleType(), True),
    StructField("width",             DoubleType(), True),
    StructField("draft",             DoubleType(), True),
    StructField("cargo",             StringType(), True),
    StructField("transceiver_class", StringType(), True),
    StructField("data_split",        StringType(), True),
])


def build_spark():
    return (
        SparkSession.builder
        .appName("Maritime_Live_Stream")
        .master(SPARK_MASTER)
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def enrich(df):
    return (
        df
        .withColumn("event_time",
                    to_timestamp(col("base_datetime")))
        .withColumn("ingestion_time", current_timestamp())
        .withColumn("lat_bin",  spark_round(col("lat"), 1))
        .withColumn("lon_bin",  spark_round(col("lon"), 1))
        .withColumn("is_stopped",  col("sog") < 0.5)
        .withColumn("is_slow",     col("sog") < 2.0)
        .withColumn("is_speeding", col("sog") > lit(ANOMALY_SPEED_MAX))
        .withColumn(
            "in_us_port_zone",
            reduce(_or, [
                col("lat").between(z["lat_min"], z["lat_max"]) &
                col("lon").between(z["lon_min"], z["lon_max"])
                for z in US_PORT_ZONES
            ])
        )
        .withColumn(
            "risk_level",
            when(
                (col("sog") < 0.5) & col("in_us_port_zone"), "HIGH"
            ).when(col("sog") < 2.0, "MEDIUM")
             .otherwise("LOW")
        )
    )


def write_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        return
    enriched = enrich(batch_df)
    (
        enriched.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .save(DELTA_BRONZE_PATH)
    )
    print(f"  Batch {batch_id}: {enriched.count()} records → Bronze")


def main():
    print("⏳  Waiting 15s for cluster ...")
    time.sleep(15)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", AIS_TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", "10000")
        .load()
    )

    parsed = (
        raw
        .select(from_json(col("value").cast("string"),
                          SCHEMA).alias("d"))
        .select("d.*")
        .filter(col("mmsi").isNotNull() & (col("mmsi") != ""))
        .filter(col("lat").isNotNull() & col("lon").isNotNull())
        .filter((col("lat") != 0.0) & (col("lon") != 0.0))
        .filter(col("lat").between(-90, 90))
        .filter(col("lon").between(-180, 180))
        .filter(col("sog").between(0, 60))
    )

    query = (
        parsed.writeStream
        .foreachBatch(write_batch)
        .option("checkpointLocation", CHECKPOINT_BRONZE)
        .outputMode("append")
        .trigger(processingTime="10 seconds")
        .start()
    )

    print(f"🚀  Streaming live AIS → Bronze Delta")
    query.awaitTermination()


if __name__ == "__main__":
    main()
