"""
maritime_pipeline_dag.py — Maritime Navigation AI System
=========================================================
Orchestration-only DAG: every task calls an EXISTING script via
the Docker Python SDK (`container.exec_run`) against the running
main stack.  Nothing here rewrites, imports, or replaces job logic.

PIPELINE (dependency order)
───────────────────────────
    bronze_job
        └─► silver_job
                └─► gold_job
                        ├─► train_anomaly   ─┐
                        ├─► train_predictor  ├─► evaluate_models
                        └─► train_congestion ─┘

  • bronze / silver / gold  → spark-submit inside spark-master container
  • train_*                 → python3 inside ais-producer container
  • evaluate_models         → python3 inside ais-producer container

SCHEDULE : @daily, PAUSED on creation.
          Trigger manually via the UI or CLI for the first run.

RETRIES  : 2 retries, 5-minute delay between attempts.
TIMEOUT  : 2 hours per task (guards against a hung Spark job).

PREREQUISITES
─────────────
  1. Main stack running:  docker compose up -d
  2. Spark containers up: spark-master, spark-worker-1/2
  3. Producer container:  ais-producer (with mlops/ mounted or copied)
     See docs/AIRFLOW.md → "One-time environment setup" for step 3.
  4. docker-py SDK installed in Airflow container:
       _PIP_ADDITIONAL_REQUIREMENTS: "docker"  (set in docker-compose.airflow.yml)
"""

from datetime import timedelta

import docker
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Container names — must match container_name in docker-compose.yml ─────────
_SPARK_MASTER = "spark-master"
_PRODUCER     = "ais-producer"

# ── Spark paths inside the spark-master container ────────────────────────────
_SPARK_SUBMIT = "/opt/spark/bin/spark-submit"

# Delta Lake Maven coordinate (pre-fetched into the image's ivy2 cache)
_DELTA_PKG = "io.delta:delta-core_2.12:2.4.0"

# PostgreSQL JDBC jar — pre-cached in ivy2 by the spark image build step.
_PG_JAR = "/root/.ivy2/jars/org.postgresql_postgresql-42.7.3.jar"

# Spark conf flags applied to every ETL job (as a list for exec_run)
_DELTA_CONFS = [
    "--conf", "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension",
    "--conf", "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog",
    "--conf", "spark.sql.shuffle.partitions=8",
]

# Extra env vars for ML tasks (mirrors the -e flags from the old docker exec call)
_ML_ENV = {
    "PYTHONPATH":           "/app/src:/app/src/common:/app/mlops",
    "MLFLOW_TRACKING_URI":  "/app/mlops/mlruns",
}


# ── Shared helper ─────────────────────────────────────────────────────────────
def _docker_exec(container_name: str, cmd: list, env: dict | None = None) -> None:
    """Run cmd inside container_name via the Docker SDK.  Raises on non-zero exit."""
    client = docker.from_env()
    container = client.containers.get(container_name)
    result = container.exec_run(cmd, environment=env)
    if result.exit_code != 0:
        raise Exception(
            f"{container_name} exited {result.exit_code}:\n"
            + (result.output.decode(errors="replace") if result.output else "")
        )


# ── Per-task callables ────────────────────────────────────────────────────────
def _run_bronze():
    _docker_exec(_SPARK_MASTER, [
        _SPARK_SUBMIT,
        "--packages", _DELTA_PKG,
        *_DELTA_CONFS,
        "/opt/spark/app/src/processing/bronze_job.py",
    ])


def _run_silver():
    _docker_exec(_SPARK_MASTER, [
        _SPARK_SUBMIT,
        "--packages", _DELTA_PKG,
        *_DELTA_CONFS,
        "/opt/spark/app/src/processing/silver_job.py",
    ])


def _run_gold():
    _docker_exec(_SPARK_MASTER, [
        _SPARK_SUBMIT,
        "--packages", _DELTA_PKG,
        "--jars", _PG_JAR,
        *_DELTA_CONFS,
        "/opt/spark/app/src/processing/gold_job.py",
    ])


def _run_train_anomaly():
    _docker_exec(_PRODUCER, ["python3", "/app/src/ml/train_anomaly.py"], env=_ML_ENV)


def _run_train_predictor():
    _docker_exec(_PRODUCER, ["python3", "/app/src/ml/train_predictor.py"], env=_ML_ENV)


def _run_train_congestion():
    _docker_exec(_PRODUCER, ["python3", "/app/src/ml/train_congestion.py"], env=_ML_ENV)


def _run_evaluate():
    _docker_exec(
        _PRODUCER,
        ["bash", "-lc", "PYTHONPATH=/app/mlops:/app python3 /app/mlops/model_registry/evaluate_models.py"],
        env=_ML_ENV,
    )


# ── Default task arguments ────────────────────────────────────────────────────
default_args = {
    "owner":             "maritime-data-eng",
    "depends_on_past":   False,
    "email_on_failure":  False,
    "email_on_retry":    False,
    "retries":           2,
    "retry_delay":       timedelta(minutes=5),
    # Guard against a Spark job hanging indefinitely
    "execution_timeout": timedelta(hours=2),
}

# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="maritime_ais_pipeline",
    default_args=default_args,
    description=(
        "Daily AIS batch pipeline: Bronze→Silver→Gold ETL, "
        "parallel ML retraining, model evaluation."
    ),
    schedule="@daily",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    is_paused_upon_creation=True,
    max_active_runs=1,
    tags=["maritime", "ais", "spark", "delta-lake", "ml", "mlflow"],
) as dag:

    # ──────────────────────────────────────────────────────────────────────────
    # BRONZE — Load raw Parquet into Delta Bronze layer
    # ──────────────────────────────────────────────────────────────────────────
    bronze_job = PythonOperator(task_id="bronze_job", python_callable=_run_bronze)

    # ──────────────────────────────────────────────────────────────────────────
    # SILVER — Clean Bronze → Silver (dedup, physics bounds, feature eng.)
    # ──────────────────────────────────────────────────────────────────────────
    silver_job = PythonOperator(task_id="silver_job", python_callable=_run_silver)

    # ──────────────────────────────────────────────────────────────────────────
    # GOLD — Aggregate Silver → Gold + sync PostgreSQL Star Schema
    # ──────────────────────────────────────────────────────────────────────────
    gold_job = PythonOperator(task_id="gold_job", python_callable=_run_gold)

    # ──────────────────────────────────────────────────────────────────────────
    # ML TRAINING — three models trained in parallel after Gold is fresh
    # ──────────────────────────────────────────────────────────────────────────
    train_anomaly    = PythonOperator(task_id="train_anomaly",    python_callable=_run_train_anomaly)
    train_predictor  = PythonOperator(task_id="train_predictor",  python_callable=_run_train_predictor)
    train_congestion = PythonOperator(task_id="train_congestion", python_callable=_run_train_congestion)

    # ──────────────────────────────────────────────────────────────────────────
    # EVALUATE — score all three models, emit evaluation_report.json + MLflow
    # ──────────────────────────────────────────────────────────────────────────
    evaluate_models = PythonOperator(task_id="evaluate_models", python_callable=_run_evaluate)

    # ── Dependency chain ──────────────────────────────────────────────────────
    # bronze → silver → gold → [anomaly ‖ predictor ‖ congestion] → evaluate
    (
        bronze_job
        >> silver_job
        >> gold_job
        >> [train_anomaly, train_predictor, train_congestion]
        >> evaluate_models
    )
