# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # Maritime AIS — Environment Configuration
# MAGIC
# MAGIC Handles LOCAL (Docker / standalone Spark) vs DATABRICKS runtime transparently.
# MAGIC
# MAGIC ## Why local Docker is retained as primary runtime
# MAGIC
# MAGIC The docker-compose stack (`docker-compose.yml`) is the authoritative development
# MAGIC environment for this project.  It ships a full 3-node Spark cluster, Kafka broker,
# MAGIC PostgreSQL warehouse, and all application services in a single `docker compose up`.
# MAGIC Local Docker guarantees:
# MAGIC   - Zero cloud spend during iterative development
# MAGIC   - Deterministic data locality (Delta tables live alongside the code)
# MAGIC   - No IAM / secret rotation friction
# MAGIC   - Offline operation on the vessel or in restricted network environments
# MAGIC
# MAGIC Databricks is the *scale-out* target: it is used when the 14-day AIS dataset
# MAGIC grows beyond what two 4-core workers can process, or when the pipeline needs to
# MAGIC be scheduled and monitored in a production SLA context.

# COMMAND ----------

import os
from enum import Enum, auto
from typing import Optional
from dataclasses import dataclass, field


class RuntimeMode(Enum):
    LOCAL = auto()       # Docker Compose / local Spark standalone
    DATABRICKS = auto()  # Databricks Workspace (Serverless or Classic cluster)


def _detect_runtime() -> RuntimeMode:
    """
    Detect the runtime automatically.

    Databricks injects `DATABRICKS_RUNTIME_VERSION` into every cluster's
    environment.  If that variable is absent we assume local Docker mode,
    which is the default development target for this project.
    """
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return RuntimeMode.DATABRICKS
    # Fallback: caller can override with MARITIME_RUNTIME=DATABRICKS
    override = os.environ.get("MARITIME_RUNTIME", "").upper()
    if override == "DATABRICKS":
        return RuntimeMode.DATABRICKS
    return RuntimeMode.LOCAL


RUNTIME: RuntimeMode = _detect_runtime()

# COMMAND ----------

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# LOCAL: Delta tables are written to a Docker-managed named volume mounted at
#        /delta inside every Spark container (see docker-compose.yml volumes).
# DATABRICKS: Tables live in Unity Catalog Volumes or DBFS.  The DBFS path
#             /mnt/maritime mirrors the same medallion layout so notebooks
#             can be copied across environments without logic changes.
# ---------------------------------------------------------------------------

if RUNTIME == RuntimeMode.DATABRICKS:
    _DELTA_ROOT   = os.environ.get("DATABRICKS_DELTA_ROOT", "dbfs:/mnt/maritime/delta")
    _DATA_ROOT    = os.environ.get("DATABRICKS_DATA_ROOT",  "dbfs:/mnt/maritime/data")
    _MODELS_ROOT  = os.environ.get("DATABRICKS_MODELS_ROOT","dbfs:/mnt/maritime/models")
    _CHECKPOINT   = os.environ.get("DATABRICKS_CHECKPOINT", "dbfs:/mnt/maritime/checkpoints")
else:
    _DELTA_ROOT   = os.environ.get("DELTA_ROOT",       "/delta")
    _DATA_ROOT    = os.environ.get("PARQUET_DATA_PATH","/app/data/parquet")
    _MODELS_ROOT  = os.environ.get("MODELS_PATH",      "/app/models")
    _CHECKPOINT   = os.environ.get("CHECKPOINT_ROOT",  "/delta/checkpoints")


@dataclass(frozen=True)
class Paths:
    delta_root:     str
    data_root:      str
    models_root:    str
    checkpoint_root: str

    # Medallion layers
    bronze:         str = field(init=False)
    silver:         str = field(init=False)
    gold_vessel:    str = field(init=False)
    gold_density:   str = field(init=False)
    gold_stats:     str = field(init=False)
    gold_anomaly:   str = field(init=False)

    # Streaming checkpoints
    ckpt_bronze:    str = field(init=False)
    ckpt_silver:    str = field(init=False)

    def __post_init__(self):
        # frozen dataclass: use object.__setattr__ to initialise derived fields
        object.__setattr__(self, "bronze",      f"{self.delta_root}/bronze/ais")
        object.__setattr__(self, "silver",      f"{self.delta_root}/silver/ais_clean")
        object.__setattr__(self, "gold_vessel", f"{self.delta_root}/gold/vessel_latest")
        object.__setattr__(self, "gold_density",f"{self.delta_root}/gold/traffic_density")
        object.__setattr__(self, "gold_stats",  f"{self.delta_root}/gold/daily_stats")
        object.__setattr__(self, "gold_anomaly",f"{self.delta_root}/gold/anomalies")
        object.__setattr__(self, "ckpt_bronze", f"{self.checkpoint_root}/bronze")
        object.__setattr__(self, "ckpt_silver", f"{self.checkpoint_root}/silver")


PATHS = Paths(
    delta_root      = _DELTA_ROOT,
    data_root       = _DATA_ROOT,
    models_root     = _MODELS_ROOT,
    checkpoint_root = _CHECKPOINT,
)

# COMMAND ----------

# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------
# LOCAL: We create a local SparkSession with the same packages and settings
#        used in docker-compose (delta-core 2.4.0, spark-sql-kafka).
# DATABRICKS: `spark` is pre-injected by the runtime — we just retrieve it.
#             Creating a new SparkSession on Databricks silently returns the
#             existing one, but being explicit avoids unexpected config merges.
# ---------------------------------------------------------------------------

def get_spark_session(app_name: str = "MaritimeAIS"):
    """Return a SparkSession appropriate for the current runtime."""
    if RUNTIME == RuntimeMode.DATABRICKS:
        # Databricks always has a live SparkSession bound to `spark`
        try:
            from pyspark.sql import SparkSession
            return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        except ImportError:
            raise RuntimeError("PySpark not available — check cluster libraries.")

    # LOCAL: build a session that matches the docker-compose Spark config
    from pyspark.sql import SparkSession

    driver_mem = os.environ.get("SPARK_DRIVER_MEMORY",   "2g")
    exec_mem   = os.environ.get("SPARK_EXECUTOR_MEMORY", "4g")
    shuffle    = os.environ.get("SPARK_SHUFFLE_PARTS",   "8")

    return (
        SparkSession.builder
        .appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory",                    driver_mem)
        .config("spark.executor.memory",                  exec_mem)
        .config("spark.sql.shuffle.partitions",           shuffle)
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        .config("spark.sql.parquet.int96RebaseModeInWrite",    "CORRECTED")
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        .getOrCreate()
    )

# COMMAND ----------

# ---------------------------------------------------------------------------
# PostgreSQL / JDBC configuration
# ---------------------------------------------------------------------------
# LOCAL: Direct JDBC to the `postgres` container (hostname resolves inside
#        the Docker bridge network defined in docker-compose.yml).
# DATABRICKS: Credentials are stored in Databricks Secrets (scope=maritime).
#             Never hard-code passwords in notebook source — they end up in
#             git history and Databricks notebook revision logs.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PostgresConfig:
    host:     str
    port:     int
    database: str
    user:     str
    password: str
    jdbc_url: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self, "jdbc_url",
            f"jdbc:postgresql://{self.host}:{self.port}/{self.database}"
        )

    @property
    def jdbc_properties(self) -> dict:
        return {
            "user":   self.user,
            "password": self.password,
            "driver": "org.postgresql.Driver",
        }


def _resolve_secret(scope: str, key: str, default: str) -> str:
    """
    Read a Databricks secret when running in cloud mode; fall back to
    environment variables or the provided default for local Docker.
    """
    if RUNTIME == RuntimeMode.DATABRICKS:
        try:
            # dbutils is injected by Databricks — not available locally
            from pyspark.dbutils import DBUtils  # type: ignore
            from pyspark.sql import SparkSession
            dbutils = DBUtils(SparkSession.getActiveSession())
            return dbutils.secrets.get(scope=scope, key=key)
        except Exception:
            pass  # fall through to env-var / default
    return os.environ.get(key.upper().replace("-", "_"), default)


PG = PostgresConfig(
    host     = _resolve_secret("maritime", "postgres-host",     os.environ.get("POSTGRES_HOST", "postgres")),
    port     = int(_resolve_secret("maritime", "postgres-port", os.environ.get("POSTGRES_PORT", "5432"))),
    database = _resolve_secret("maritime", "postgres-db",       os.environ.get("POSTGRES_DB",   "maritime")),
    user     = _resolve_secret("maritime", "postgres-user",     os.environ.get("POSTGRES_USER", "maritime")),
    password = _resolve_secret("maritime", "postgres-password", os.environ.get("POSTGRES_PASSWORD", "maritime123")),
)

# COMMAND ----------

# ---------------------------------------------------------------------------
# Kafka configuration
# ---------------------------------------------------------------------------
# LOCAL:      kafka:9092 (Docker bridge network, see docker-compose.yml)
# DATABRICKS: Confluent Cloud or Azure Event Hubs with SASL/SSL.
#             Bootstrap server and credentials come from Databricks Secrets.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str
    ais_topic:         str
    alerts_topic:      str
    # SASL settings — empty strings mean no auth (local plaintext)
    sasl_mechanism:    str
    sasl_username:     str
    sasl_password:     str

    @property
    def spark_read_options(self) -> dict:
        """Return options dict for spark.readStream.format('kafka')."""
        opts = {
            "kafka.bootstrap.servers": self.bootstrap_servers,
            "subscribe":               self.ais_topic,
            "startingOffsets":         "latest",
            "failOnDataLoss":          "false",
        }
        if self.sasl_mechanism:
            opts.update({
                "kafka.security.protocol":           "SASL_SSL",
                "kafka.sasl.mechanism":              self.sasl_mechanism,
                "kafka.sasl.jaas.config": (
                    f"org.apache.kafka.common.security.plain.PlainLoginModule required "
                    f"username=\"{self.sasl_username}\" password=\"{self.sasl_password}\";"
                ),
            })
        return opts


KAFKA = KafkaConfig(
    bootstrap_servers = _resolve_secret("maritime", "kafka-bootstrap-servers",
                                        os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")),
    ais_topic         = os.environ.get("AIS_TOPIC",    "ais_raw"),
    alerts_topic      = os.environ.get("ALERTS_TOPIC", "ais_alerts"),
    sasl_mechanism    = _resolve_secret("maritime", "kafka-sasl-mechanism",    ""),
    sasl_username     = _resolve_secret("maritime", "kafka-sasl-username",     ""),
    sasl_password     = _resolve_secret("maritime", "kafka-sasl-password",     ""),
)

# COMMAND ----------

# ---------------------------------------------------------------------------
# ML thresholds — identical in both runtimes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MLConfig:
    anomaly_contamination: float = 0.01
    anomaly_speed_max:     float = 30.0
    anomaly_heading_change:float = 45.0
    collision_distance_nm: float = 0.5
    congestion_high:       int   = 15
    congestion_medium:     int   = 5


ML = MLConfig()

# COMMAND ----------

# ---------------------------------------------------------------------------
# Quick sanity print — useful in both local logs and Databricks driver output
# ---------------------------------------------------------------------------

print(f"[environment.py] Runtime mode : {RUNTIME.name}")
print(f"[environment.py] Delta root   : {PATHS.delta_root}")
print(f"[environment.py] Bronze path  : {PATHS.bronze}")
print(f"[environment.py] Silver path  : {PATHS.silver}")
print(f"[environment.py] Kafka brokers: {KAFKA.bootstrap_servers}")
print(f"[environment.py] Postgres host: {PG.host}:{PG.port}/{PG.database}")
