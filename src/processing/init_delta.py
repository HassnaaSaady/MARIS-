"""
init_delta.py — Maritime Navigation AI System
Creates empty Bronze, Silver, and Gold Delta tables.
ALL timestamps use TIMESTAMP_NTZ to match pandas-written Parquet.
"""
import sys
sys.path.insert(0, "/opt/spark/app/src/common")

from config import (
    SPARK_MASTER,
    DELTA_BRONZE_PATH, DELTA_SILVER_PATH,
    DELTA_GOLD_VESSEL_PATH, DELTA_GOLD_DENSITY_PATH,
    DELTA_GOLD_ANOMALY_PATH, DELTA_GOLD_STATS_PATH,
)
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("InitDeltaLake")
    .master(SPARK_MASTER)
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.timestampType", "TIMESTAMP_NTZ")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

BRONZE_SCHEMA = """
    mmsi STRING, base_datetime TIMESTAMP_NTZ,
    lat DOUBLE, lon DOUBLE,
    sog DOUBLE, cog DOUBLE, heading DOUBLE,
    vessel_name STRING, imo STRING, call_sign STRING,
    vessel_type STRING, status STRING,
    length DOUBLE, width DOUBLE, draft DOUBLE,
    cargo STRING, transceiver_class STRING,
    risk_level STRING, lat_bin DOUBLE, lon_bin DOUBLE,
    is_stopped BOOLEAN, is_slow BOOLEAN, is_speeding BOOLEAN,
    in_us_port_zone BOOLEAN, data_split STRING,
    event_time TIMESTAMP_NTZ, ingestion_time TIMESTAMP_NTZ,
    year INT, month INT, day INT, hour INT
"""

SILVER_SCHEMA = """
    mmsi STRING, base_datetime TIMESTAMP_NTZ,
    lat DOUBLE, lon DOUBLE,
    sog DOUBLE, cog DOUBLE, heading DOUBLE,
    vessel_name STRING, imo STRING, call_sign STRING,
    vessel_type STRING, vessel_type_label STRING,
    status STRING, status_label STRING,
    length DOUBLE, width DOUBLE, draft DOUBLE,
    cargo STRING, transceiver_class STRING,
    risk_level STRING, lat_bin DOUBLE, lon_bin DOUBLE,
    lat_bin_fine DOUBLE, lon_bin_fine DOUBLE,
    is_stopped BOOLEAN, is_slow BOOLEAN, is_speeding BOOLEAN,
    in_us_port_zone BOOLEAN,
    sog_change DOUBLE, heading_change DOUBLE,
    time_delta_sec DOUBLE, distance_nm DOUBLE,
    data_split STRING,
    year INT, month INT, day INT, hour INT
"""

GOLD_VESSEL_SCHEMA = """
    mmsi STRING, vessel_name STRING, vessel_type STRING,
    lat DOUBLE, lon DOUBLE,
    sog DOUBLE, cog DOUBLE, heading DOUBLE,
    risk_level STRING, is_anomaly BOOLEAN,
    anomaly_score DOUBLE, anomaly_type STRING,
    predicted_lat DOUBLE, predicted_lon DOUBLE,
    base_datetime TIMESTAMP_NTZ, updated_at TIMESTAMP_NTZ,
    data_split STRING
"""

GOLD_DENSITY_SCHEMA = """
    lat_bin DOUBLE, lon_bin DOUBLE,
    hour_bucket TIMESTAMP_NTZ,
    vessel_count INT, unique_vessels INT,
    avg_sog DOUBLE, stopped_count INT,
    congestion_level STRING, data_split STRING
"""

GOLD_ANOMALY_SCHEMA = """
    mmsi STRING, vessel_name STRING,
    lat DOUBLE, lon DOUBLE,
    anomaly_type STRING, anomaly_score DOUBLE,
    sog DOUBLE, heading_change DOUBLE,
    risk_level STRING, base_datetime TIMESTAMP_NTZ,
    data_split STRING
"""

GOLD_STATS_SCHEMA = """
    stat_date DATE,
    total_vessels INT, total_records BIGINT,
    avg_sog DOUBLE, max_sog DOUBLE,
    high_risk_count INT, anomaly_count INT,
    data_split STRING
"""

tables = [
    (DELTA_BRONZE_PATH,       BRONZE_SCHEMA,       "Bronze AIS"),
    (DELTA_SILVER_PATH,       SILVER_SCHEMA,       "Silver AIS Clean"),
    (DELTA_GOLD_VESSEL_PATH,  GOLD_VESSEL_SCHEMA,  "Gold Vessel Latest"),
    (DELTA_GOLD_DENSITY_PATH, GOLD_DENSITY_SCHEMA, "Gold Traffic Density"),
    (DELTA_GOLD_ANOMALY_PATH, GOLD_ANOMALY_SCHEMA, "Gold Anomalies"),
    (DELTA_GOLD_STATS_PATH,   GOLD_STATS_SCHEMA,   "Gold Daily Stats"),
]

for path, schema, name in tables:
    (
        spark.createDataFrame([], schema)
             .write.format("delta")
             .mode("overwrite")
             .save(path)
    )
    print(f"  {name} -> {path}")

spark.stop()
print("\nAll Delta Lake tables initialised.")
print("Next: run bronze_job.py")