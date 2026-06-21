"""
config.py — Maritime Navigation AI System
Central configuration for ALL services.
Every env var lives here — no magic strings anywhere else.
"""
import os

# ── Kafka ──────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
AIS_TOPIC               = os.getenv("AIS_TOPIC",    "ais_raw")
ALERTS_TOPIC            = os.getenv("ALERTS_TOPIC", "ais_alerts")
STREAM_DELAY_SECONDS    = float(os.getenv("STREAM_DELAY_SECONDS", "0.01"))
LOOP                    = os.getenv("LOOP", "true").lower() == "true"

# ── Data paths ─────────────────────────────────────────────────────────────────
RAW_DATA_PATH     = os.getenv("RAW_DATA_PATH",    "/app/data/raw")
PARQUET_DATA_PATH = os.getenv("PARQUET_DATA_PATH", "/app/data/parquet")
AIS_INPUT_PATH    = os.getenv("AIS_INPUT_PATH",   "/app/data/parquet")
MODELS_PATH       = os.getenv("MODELS_PATH",      "/app/models")

# ── Delta Lake paths (Medallion Architecture) ──────────────────────────────────
DELTA_ROOT         = os.getenv("DELTA_ROOT", "/delta")

# Bronze — raw enriched records (append only)
DELTA_BRONZE_PATH  = f"{DELTA_ROOT}/bronze/ais"

# Silver — cleaned, deduplicated, feature-enriched
DELTA_SILVER_PATH  = f"{DELTA_ROOT}/silver/ais_clean"

# Gold — aggregated, ML-ready, dashboard-ready
DELTA_GOLD_VESSEL_PATH   = f"{DELTA_ROOT}/gold/vessel_latest"
DELTA_GOLD_DENSITY_PATH  = f"{DELTA_ROOT}/gold/traffic_density"
DELTA_GOLD_ANOMALY_PATH  = f"{DELTA_ROOT}/gold/anomalies"
DELTA_GOLD_STATS_PATH    = f"{DELTA_ROOT}/gold/daily_stats"

# Checkpoints
CHECKPOINT_BRONZE  = f"{DELTA_ROOT}/checkpoints/bronze"
CHECKPOINT_SILVER  = f"{DELTA_ROOT}/checkpoints/silver"

# ── PostgreSQL (Star Schema) ───────────────────────────────────────────────────
POSTGRES_HOST     = os.getenv("POSTGRES_HOST",     "postgres")
POSTGRES_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB       = os.getenv("POSTGRES_DB",       "maritime")
POSTGRES_USER     = os.getenv("POSTGRES_USER",     "maritime")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "maritime123")
POSTGRES_URL      = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# ── Spark ──────────────────────────────────────────────────────────────────────
SPARK_MASTER          = os.getenv("SPARK_MASTER", "spark://spark-master:7077")
SPARK_DRIVER_MEMORY   = os.getenv("SPARK_DRIVER_MEMORY",   "2g")
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
SPARK_EXECUTOR_CORES  = os.getenv("SPARK_EXECUTOR_CORES",  "4")
SPARK_SHUFFLE_PARTS   = os.getenv("SPARK_SHUFFLE_PARTS",   "8")

# ── ML Thresholds ──────────────────────────────────────────────────────────────
ANOMALY_CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", "0.01"))
ANOMALY_SPEED_MAX       = float(os.getenv("ANOMALY_SPEED_MAX",       "30.0"))
ANOMALY_HEADING_CHANGE  = float(os.getenv("ANOMALY_HEADING_CHANGE",  "45.0"))
COLLISION_DISTANCE_NM   = float(os.getenv("COLLISION_DISTANCE_NM",   "0.5"))
CONGESTION_HIGH         = int(os.getenv("CONGESTION_HIGH",           "15"))
CONGESTION_MEDIUM       = int(os.getenv("CONGESTION_MEDIUM",         "5"))

# ── Geographic zones ───────────────────────────────────────────────────────────
US_PORT_ZONES = [
    {"lat_min": 29.50, "lat_max": 29.85, "lon_min": -95.30, "lon_max": -94.80,
     "name": "Houston Ship Channel"},
    {"lat_min": 40.50, "lat_max": 40.75, "lon_min": -74.30, "lon_max": -73.90,
     "name": "New York Harbor"},
    {"lat_min": 33.70, "lat_max": 33.85, "lon_min": -118.35, "lon_max": -118.10,
     "name": "Port of Los Angeles / Long Beach"},
    {"lat_min": 29.00, "lat_max": 30.00, "lon_min": -90.50, "lon_max": -89.50,
     "name": "Port of New Orleans"},
    {"lat_min": 25.70, "lat_max": 25.85, "lon_min": -80.20, "lon_max": -80.05,
     "name": "Port of Miami"},
]

US_WATERS = {
    "lat_min": 24.0, "lat_max": 49.0,
    "lon_min": -125.0, "lon_max": -66.0,
    "name": "US Waters",
}

# ── Data split (14 days total) ─────────────────────────────────────────────────
# Days 1-8  → TRAIN  (57%)
# Days 9-10 → VALID  (14%)
# Days 11-12→ TEST   (14%)
# Days 13-14→ LIVE   (15%) ← what map shows
TRAIN_DAYS      = int(os.getenv("TRAIN_DAYS",      "8"))
VALIDATION_DAYS = int(os.getenv("VALIDATION_DAYS", "2"))
TEST_DAYS       = int(os.getenv("TEST_DAYS",        "2"))
LIVE_DAYS       = int(os.getenv("LIVE_DAYS",        "2"))

# ── Dashboard settings ─────────────────────────────────────────────────────────
DASHBOARD_REFRESH_SEC  = int(os.getenv("DASHBOARD_REFRESH_SEC", "3"))
MAP_DEFAULT_LAT        = float(os.getenv("MAP_DEFAULT_LAT",     "30.5"))
MAP_DEFAULT_LON        = float(os.getenv("MAP_DEFAULT_LON",     "32.3"))
MAP_DEFAULT_ZOOM       = int(os.getenv("MAP_DEFAULT_ZOOM",      "6"))
