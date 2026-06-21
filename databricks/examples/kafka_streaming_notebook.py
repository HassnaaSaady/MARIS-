# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # Kafka → Bronze Delta: Structured Streaming Example
# MAGIC
# MAGIC **Local counterpart**: `src/processing/spark_streaming_consumer.py`
# MAGIC
# MAGIC This notebook demonstrates real-time AIS data ingestion via Kafka Structured
# MAGIC Streaming into the Bronze Delta table.  It can run as a Databricks Streaming
# MAGIC Job with continuous trigger, or in triggered mode for micro-batch processing.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Why local Docker is retained as primary runtime
# MAGIC
# MAGIC `src/processing/spark_streaming_consumer.py` runs as the `spark-stream`
# MAGIC service in `docker-compose.yml`.  That container:
# MAGIC
# MAGIC - Connects to `kafka:9092` on the Docker bridge network (zero config)
# MAGIC - Writes checkpoints to the `delta_data` volume at `/delta/checkpoints/bronze`
# MAGIC - Starts automatically with `docker compose up` and restarts on failure
# MAGIC
# MAGIC The Databricks streaming job requires additional infrastructure:
# MAGIC - A Kafka cluster reachable from Databricks (Confluent Cloud / Event Hubs)
# MAGIC - SASL credentials stored in Databricks Secrets
# MAGIC - A cluster with the `kafka-clients` / `spark-sql-kafka` libraries
# MAGIC - DBFS or Unity Catalog Volume for checkpoint storage
# MAGIC
# MAGIC Use this notebook when the AIS live feed volume exceeds local throughput,
# MAGIC or when you need Databricks Jobs SLA and auto-scaling for the streaming layer.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Kafka message schema
# MAGIC
# MAGIC The `ais_raw` topic receives JSON messages produced by
# MAGIC `src/producer/kafka_producer.py`.  Each message carries the canonical
# MAGIC AIS fields defined in `src/common/schema_utils.py`:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "mmsi": "368000000",
# MAGIC   "base_datetime": "2025-05-07T14:23:11",
# MAGIC   "lat": 30.123,
# MAGIC   "lon": 32.456,
# MAGIC   "sog": 7.2,
# MAGIC   "cog": 183.4,
# MAGIC   "heading": 185,
# MAGIC   "vessel_name": "SUEZ CARRIER",
# MAGIC   "vessel_type": 70,
# MAGIC   "status": 0,
# MAGIC   "data_split": "LIVE"
# MAGIC }
# MAGIC ```

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
    from configs.environment import PATHS, KAFKA, get_spark_session

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

if _IN_DATABRICKS:
    dbutils.widgets.text("bronze_path",  PATHS.bronze,      "Bronze Delta output path")
    dbutils.widgets.text("checkpoint",   PATHS.ckpt_bronze, "Streaming checkpoint path")
    dbutils.widgets.dropdown("trigger",  "continuous",
                             ["continuous", "once", "processingTime"],
                             "Trigger type")
    dbutils.widgets.text("trigger_interval", "10 seconds",
                         "processingTime interval (ignored for other triggers)")

    BRONZE_PATH       = dbutils.widgets.get("bronze_path")
    CHECKPOINT_PATH   = dbutils.widgets.get("checkpoint")
    TRIGGER_TYPE      = dbutils.widgets.get("trigger")
    TRIGGER_INTERVAL  = dbutils.widgets.get("trigger_interval")
else:
    BRONZE_PATH       = PATHS.bronze
    CHECKPOINT_PATH   = PATHS.ckpt_bronze
    TRIGGER_TYPE      = "processingTime"
    TRIGGER_INTERVAL  = "10 seconds"

print(f"Bronze path  : {BRONZE_PATH}")
print(f"Checkpoint   : {CHECKPOINT_PATH}")
print(f"Kafka brokers: {KAFKA.bootstrap_servers}")
print(f"Topic        : {KAFKA.ais_topic}")
print(f"Trigger      : {TRIGGER_TYPE} / {TRIGGER_INTERVAL}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. SparkSession

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampNTZType, BooleanType
)

spark = get_spark_session("MaritimeAIS-KafkaStreaming")
spark.conf.set("spark.sql.shuffle.partitions",  "8")
spark.conf.set("spark.sql.timestampType",       "TIMESTAMP_NTZ")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. AIS JSON schema
# MAGIC
# MAGIC Matches the canonical field list in `src/common/schema_utils.py`.
# MAGIC Spark enforces this schema when parsing Kafka value bytes so corrupt or
# MAGIC unexpected messages produce NULL fields (and are filtered out below)
# MAGIC rather than crashing the stream.

# COMMAND ----------

AIS_SCHEMA = StructType([
    StructField("mmsi",              StringType(),       nullable=True),
    StructField("base_datetime",     StringType(),       nullable=True),
    StructField("lat",               DoubleType(),       nullable=True),
    StructField("lon",               DoubleType(),       nullable=True),
    StructField("sog",               DoubleType(),       nullable=True),
    StructField("cog",               DoubleType(),       nullable=True),
    StructField("heading",           DoubleType(),       nullable=True),
    StructField("vessel_name",       StringType(),       nullable=True),
    StructField("imo",               StringType(),       nullable=True),
    StructField("call_sign",         StringType(),       nullable=True),
    StructField("vessel_type",       IntegerType(),      nullable=True),
    StructField("status",            IntegerType(),      nullable=True),
    StructField("length",            DoubleType(),       nullable=True),
    StructField("width",             DoubleType(),       nullable=True),
    StructField("draft",             DoubleType(),       nullable=True),
    StructField("cargo",             StringType(),       nullable=True),
    StructField("transceiver_class", StringType(),       nullable=True),
    StructField("data_split",        StringType(),       nullable=True),
])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Read from Kafka
# MAGIC
# MAGIC ### Local Docker
# MAGIC `kafka:9092` with no authentication (PLAINTEXT listener).
# MAGIC
# MAGIC ### Databricks (Confluent Cloud / Azure Event Hubs)
# MAGIC The `KAFKA.spark_read_options` dict from `environment.py` automatically
# MAGIC includes SASL/SSL settings when secrets are present — no code change here.

# COMMAND ----------

kafka_stream_df = (
    spark.readStream
    .format("kafka")
    .options(**KAFKA.spark_read_options)
    .load()
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Parse JSON payloads + quality filters

# COMMAND ----------

parsed_df = (
    kafka_stream_df
    # Kafka value is binary — cast to string first
    .select(F.col("value").cast("string").alias("json_str"),
            F.col("timestamp").alias("kafka_timestamp"))
    # Parse JSON against AIS_SCHEMA
    .withColumn("data", F.from_json(F.col("json_str"), AIS_SCHEMA))
    .select("data.*", "kafka_timestamp")
    # Quality filters (mirrors bronze_job.py / 01_bronze_notebook.py)
    .filter(
        F.col("mmsi").isNotNull()
        & F.col("lat").between(-90.0, 90.0)
        & F.col("lon").between(-180.0, 180.0)
        & (F.col("sog") >= 0)
        & (F.col("sog") <= 60)
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Bronze enrichment
# MAGIC
# MAGIC Identical enrichment columns to `01_bronze_notebook.py` and
# MAGIC `bronze_job.py` — speed flags, Suez zone, rule-based risk.

# COMMAND ----------

enriched_stream_df = (
    parsed_df
    .withColumn("event_time",
        F.to_timestamp(F.col("base_datetime")).cast(TimestampNTZType()))
    .withColumn("ingestion_time", F.current_timestamp())
    .withColumn("is_stopped",    F.col("sog") < 0.5)
    .withColumn("is_slow",       F.col("sog") < 2.0)
    .withColumn("is_speeding",   F.col("sog") > 30.0)
    .withColumn("in_suez_zone",
        F.col("lat").between(29.5, 31.5) & F.col("lon").between(31.0, 33.5))
    .withColumn("risk_level",
        F.when((F.col("sog") < 1.0) & F.col("in_suez_zone"), "HIGH")
        .when(F.col("sog") < 2.0,                              "MEDIUM")
        .otherwise("LOW"))
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Write stream to Bronze Delta
# MAGIC
# MAGIC ### Trigger options
# MAGIC
# MAGIC | Trigger             | When to use                                             |
# MAGIC |---------------------|---------------------------------------------------------|
# MAGIC | `continuous`        | Sub-second latency; Databricks Streaming Jobs only      |
# MAGIC | `processingTime`    | Micro-batch; good default for 5–30s intervals           |
# MAGIC | `once`             | Run one batch and stop; useful for backfill             |
# MAGIC | `availableNow`      | Process all available data then stop (Spark 3.3+)       |

# COMMAND ----------

def _build_writer(df):
    writer = (
        df.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT_PATH)
        .option("mergeSchema",        "true")
        .option("delta.autoOptimize.optimizeWrite", "true")
        .queryName("ais_kafka_to_bronze")
    )
    if TRIGGER_TYPE == "continuous":
        writer = writer.trigger(continuous="1 second")
    elif TRIGGER_TYPE == "once":
        writer = writer.trigger(once=True)
    else:
        writer = writer.trigger(processingTime=TRIGGER_INTERVAL)
    return writer.start(BRONZE_PATH)


query = _build_writer(enriched_stream_df)

print(f"Stream started: {query.name}  (id={query.id})")
print(f"Status        : {query.status}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Monitoring
# MAGIC
# MAGIC ### Databricks
# MAGIC The Streaming tab in the cluster UI shows throughput and lag graphs.
# MAGIC You can also set up Ganglia alerts or use Databricks Lakehouse Monitoring.
# MAGIC
# MAGIC ### Local Docker
# MAGIC `spark_streaming_consumer.py` logs to stdout — view with:
# MAGIC ```
# MAGIC docker compose logs -f spark-stream
# MAGIC ```
# MAGIC
# MAGIC The cell below blocks until the stream ends (use `once` trigger for batch)
# MAGIC or until the cluster stops.  Comment it out for always-on streaming jobs.

# COMMAND ----------

import time

# For `once` / `availableNow` triggers, wait for completion.
# For continuous / processingTime, this cell runs indefinitely — interrupt
# the kernel manually when done.
if TRIGGER_TYPE in ("once", "availableNow"):
    query.awaitTermination()
    print("Stream complete.")
    print(f"Final progress: {query.lastProgress}")
else:
    # Show live progress for 60 seconds then detach
    for i in range(12):
        progress = query.lastProgress
        if progress:
            rows   = progress.get("numInputRows", 0)
            rate   = progress.get("inputRowsPerSecond", 0.0)
            offset = progress.get("sources", [{}])[0].get("endOffset", "–")
            print(f"[{i*5:3d}s] rows_this_batch={rows:6,}  rate={rate:7.1f} rows/s  offset={offset}")
        time.sleep(5)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Verify Bronze Delta

# COMMAND ----------

recent = (
    spark.read.format("delta").load(BRONZE_PATH)
    .orderBy(F.col("ingestion_time").desc())
    .limit(5)
    .select("mmsi", "base_datetime", "lat", "lon", "sog", "risk_level", "ingestion_time")
)
recent.show(truncate=False)
