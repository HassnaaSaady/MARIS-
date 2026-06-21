# Airflow Orchestration — Maritime Navigation AI System

Apache Airflow provides **batch orchestration only**.  It does not replace
the Kafka streaming path, which remains a continuously running process
outside Airflow's control.

---

## Architecture: Orchestration Layer vs. Streaming Layer

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CONTINUOUS (always-on, NOT managed by Airflow)                         │
│                                                                         │
│  AIS Feed ──► Kafka (ais_raw) ──► spark-stream ──► Delta Bronze         │
│                                                                         │
│  live-scorer ──► Kafka consumer ──► fact_alerts (PostgreSQL)            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  BATCH / DAILY (Airflow DAG: maritime_ais_pipeline)                     │
│                                                                         │
│  bronze_job  ──►  silver_job  ──►  gold_job                             │
│       (Spark)           (Spark)         (Spark + PostgreSQL JDBC)       │
│                                            │                            │
│                             ┌──────────────┼──────────────┐             │
│                             ▼              ▼              ▼             │
│                        train_anomaly  train_predictor  train_congestion │
│                             │              │              │             │
│                             └──────────────┼──────────────┘             │
│                                            ▼                            │
│                                    evaluate_models                      │
│                                 (writes evaluation_report.json          │
│                                  + logs metrics to MLflow)              │
└─────────────────────────────────────────────────────────────────────────┘
```

**Orchestration-only boundary**: Airflow calls `docker exec <container>
python3 <existing_script>` or `docker exec <container> spark-submit ...`.
It never imports, modifies, or replaces the scripts themselves.  All job
logic lives in `src/` and `mlops/` exactly as before.

---

## Container Count and Resource Constraint

> This machine **crashes Docker past ~12 containers**.

The Airflow setup adds **one** container (`maritime-airflow`).  Before
starting it, free up headroom by stopping the heaviest services:

```bash
# Stop heavy containers (Spark workers + live-scorer = 3 freed)
docker compose stop spark-worker-1 spark-worker-2 live-scorer

# Stop the legacy Celery-based Airflow artefacts if they are running
# (these were created by airflow/docker-compose.yaml, the old setup)
docker stop airflow-postgres-1 airflow-redis-1 2>/dev/null || true
```

After stopping 5 containers and adding 1, you stay under the limit.

Airflow itself uses ~512–768 MB RAM and 1 vCPU — well within budget.

---

## One-Time Environment Setup

The `evaluate_models` DAG task runs inside `ais-producer`, which needs
`mlops/` available at `/app/mlops`.  Because the main `docker-compose.yml`
does not mount that directory (by design), you must do one of the following:

### Option A — Persistent volume mount (recommended)

Recreate `ais-producer` with the extra mount using the provided override:

```bash
docker compose -f docker-compose.yml \
               -f docker-compose.airflow-prep.yml \
               up -d producer
```

Verify:

```bash
docker exec ais-producer ls /app/mlops/model_registry/
```

Re-run this command any time the producer container is recreated.

### Option B — One-shot copy (simpler, non-persistent)

```bash
docker cp ./mlops ais-producer:/app/mlops
```

The copy persists until the container is removed (`docker compose down`).

---

## Starting Airflow

```bash
# 1. Free up containers (see above)
docker compose stop spark-worker-1 spark-worker-2 live-scorer

# 2. Build the image and start Airflow in the background
docker compose -f docker-compose.airflow.yml up -d --build

# 3. Wait ~30 seconds for standalone to initialise, then check it is up
docker compose -f docker-compose.airflow.yml ps
docker logs maritime-airflow --tail 20
```

---

## Getting the Admin Password

On first start `airflow standalone` auto-creates an **admin** user and
writes the password to a file:

```bash
docker exec maritime-airflow \
    cat /opt/airflow/standalone_admin_password.txt
```

Alternatively, grep the startup logs:

```bash
docker logs maritime-airflow 2>&1 | grep -i "password"
```

---

## Opening the UI

Navigate to **http://localhost:8080** in a browser.

- Username: `admin`
- Password: from the command above

> Port 8080 is free on the host because `spark-master` maps its internal
> port 8080 to host port **9090** (see `docker-compose.yml` line ~139).

---

## Triggering the DAG

The DAG is **paused on creation**.  You must unpause it before it runs.

### Via the UI

1. Open http://localhost:8080
2. Find `maritime_ais_pipeline` in the DAG list
3. Toggle the **Paused** slider to Active
4. Click the **▶ Trigger DAG** button (top-right)
5. Monitor progress in **Graph** or **Grid** view

### Via the CLI

```bash
# Unpause
docker exec maritime-airflow airflow dags unpause maritime_ais_pipeline

# Trigger a manual run
docker exec maritime-airflow airflow dags trigger maritime_ais_pipeline

# Watch task states
docker exec maritime-airflow \
    airflow dags state maritime_ais_pipeline \
    "$(date -u +%Y-%m-%dT%H:%M:%S)+00:00"

# View recent task instance logs (example: bronze_job)
docker exec maritime-airflow \
    airflow tasks logs maritime_ais_pipeline bronze_job LATEST
```

---

## Monitoring a Running DAG

```bash
# Tail the Airflow scheduler/webserver logs
docker logs maritime-airflow -f

# List recent DAG runs
docker exec maritime-airflow \
    airflow dags list-runs -d maritime_ais_pipeline

# List task instances for the latest run
docker exec maritime-airflow \
    airflow tasks list maritime_ais_pipeline --tree
```

Task-level stdout/stderr (including Spark logs) appears in
`./airflow/logs/dag_id=maritime_ais_pipeline/`.

---

## Stopping Airflow

```bash
# Stop the Airflow container
docker compose -f docker-compose.airflow.yml down

# Restart the Spark workers and live-scorer
docker compose start spark-worker-1 spark-worker-2 live-scorer
```

---

## Task Reference

| Task ID | Container | Script | What it does |
|---|---|---|---|
| `bronze_job` | `spark-master` | `src/processing/bronze_job.py` | Load raw Parquet → Delta Bronze |
| `silver_job` | `spark-master` | `src/processing/silver_job.py` | Clean Bronze → Delta Silver |
| `gold_job` | `spark-master` | `src/processing/gold_job.py` | Aggregate Silver → Gold + PostgreSQL |
| `train_anomaly` | `ais-producer` | `src/ml/train_anomaly.py` | Retrain IsolationForest on Silver |
| `train_predictor` | `ais-producer` | `src/ml/train_predictor.py` | Retrain XGBoost position predictor |
| `train_congestion` | `ais-producer` | `src/ml/train_congestion.py` | Retrain RF congestion classifier (v2, no leakage) |
| `evaluate_models` | `ais-producer` | `mlops/model_registry/evaluate_models.py` | Score all models, write `evaluation_report.json`, log to MLflow |

**Parallel tasks**: `train_anomaly`, `train_predictor`, and `train_congestion`
run concurrently after `gold_job` completes.  Each writes a new `.pkl` to
`models/` (bind-mounted from the host), so `live-scorer` and `maritime-api`
pick up the fresh model on their next load cycle.

---

## Scaling to the Full 62M-Row Dataset

All three Spark jobs are already written to use the Spark cluster.  To scale:

1. **Restart spark workers** before running the full pipeline:
   ```bash
   docker compose start spark-worker-1 spark-worker-2
   ```

2. **Increase executor memory** by passing extra `--conf` flags via
   environment variables or by editing the BashOperator commands in
   `airflow/dags/maritime_pipeline_dag.py`:
   ```python
   "--conf spark.executor.memory=4g "
   "--conf spark.driver.memory=2g "
   "--num-executors 2 "
   ```

3. **Increase timeout** for the DAG's `execution_timeout` (currently 2 hours)
   if Spark jobs take longer on the full dataset.

4. **Schedule**: `@daily` is appropriate for a batch pipeline.  If data
   arrives faster, change `schedule="@daily"` to `schedule="@hourly"` or
   a cron expression.

---

## Future Live Feed Integration

The architecture is designed for clean extension:

```
Current:  AIS Parquet files ──► Airflow batch ──► Bronze/Silver/Gold
Future:   Live AIS TCP/REST ──► Kafka (ais_raw) ──► spark-stream (Bronze)
                                                          │
                                           Airflow @hourly Silver+Gold
                                                          │
                                               live-scorer (real-time scoring)
```

- **Kafka streaming** (`spark-stream` container) writes Bronze in
  near-real-time — this path is already wired and runs independently of
  Airflow.
- **Airflow** would shift to an hourly schedule to refresh Silver/Gold
  more frequently and retrain models nightly.
- **live-scorer** remains a continuously running Kafka consumer that scores
  each incoming message without waiting for batch cycles.
- No changes to existing scripts are needed for this migration — only the
  `schedule` parameter in the DAG changes.

---

## File Inventory

| File | Purpose |
|---|---|
| `docker/Dockerfile.airflow` | Airflow image + Docker CLI |
| `docker-compose.airflow.yml` | Single-container standalone Airflow |
| `docker-compose.airflow-prep.yml` | Override: mounts `mlops/` into `ais-producer` |
| `airflow/dags/maritime_pipeline_dag.py` | The DAG (orchestration only) |
| `docs/AIRFLOW.md` | This document |

None of these files modify any existing script, Spark job, ML model,
or `docker-compose.yml`.
