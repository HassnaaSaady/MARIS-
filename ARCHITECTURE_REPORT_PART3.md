# Maritime Navigation AI System — Architecture Report
### Internship Defense Presentation | Big Data Internship Team | June 2026

---

# PART 3: WEATHER INTEGRATION, ENTERPRISE LAYER, INFRASTRUCTURE & CONSTRAINTS

---

## SECTION 12 — WEATHER INTEGRATION MODULE

### 12.1 Overview

The weather integration is a **strictly additive** module that enriches the congestion classification pipeline with environmental data. No existing files are modified. The production model `models/congestion_rf.pkl` is never overwritten.

**Location:** `src/weather/`  
**New PostgreSQL table:** `dim_weather`  
**Documentation:** `docs/WEATHER_INTEGRATION.md`

**Data source:** Open-Meteo ERA5 reanalysis archive (free API, historical weather data)  
**Variables fetched:** `wind_speed_10m` (m/s) and `wave_height` (m) per 0.5° grid cell

---

### 12.2 Four-File Module Structure

| File | Role |
|---|---|
| `fetch_weather.py` | Fetches weather data from Open-Meteo ERA5 API |
| `load_dim_weather.py` | Loads fetched data into PostgreSQL `dim_weather` |
| `weather_features.py` | Feature engineering for the weather-augmented model |
| `eval_weather.py` | Evaluates weather-augmented model vs. production baseline |

---

### 12.3 Step-by-Step Execution

**Step 1 — Fetch weather data**
```bash
docker compose exec producer python src/weather/fetch_weather.py
```

Process:
1. Reads occupied 0.5° grid cells and date range from AIS Silver Delta (or raw Parquet if Silver is not mounted)
2. Fetches `wind_speed_10m` and `wave_height` from ERA5 archive for those cells and dates only
3. Caches each grid cell as an individual Parquet file under `data/weather/bronze/cache/` (idempotent — cached cells are not re-fetched)
4. Writes one combined bronze Parquet to `data/weather/bronze/weather_bronze.parquet`

**Step 2 — Load into PostgreSQL**
```bash
docker compose exec producer python src/weather/load_dim_weather.py
```

Process:
1. Reads `weather_bronze.parquet`
2. Computes `weather_severity` score (formula combines wind_speed and wave_height)
3. Applies DDL to create `dim_weather` if it does not exist
4. UPSERTs all rows — idempotent, no duplicates on re-run

**Step 3 — Evaluate weather as a feature**
```bash
docker compose exec producer python src/weather/eval_weather.py
```

Process:
1. Builds the same density grid used by `train_congestion.py`
2. Joins `dim_weather` to the density features on (lat_bin, lon_bin, date)
3. Trains a candidate Random Forest with weather features added
4. Compares accuracy and F1 score against the existing `congestion_rf.BACKUP.pkl` baseline
5. Prints comparison report — the production model is **never overwritten automatically**

---

### 12.4 Weather Severity Formula

```
weather_severity = (wind_speed_10m / max_wind) * 0.6
                 + (wave_height / max_wave) * 0.4
```

Values are min-max normalized per dataset. `weather_severity` ranges from 0.0 (calm) to 1.0 (extreme conditions).

**Thresholds:** LOW < 0.3, MEDIUM 0.3–0.7, HIGH > 0.7

---

### 12.5 Why This Design?

The additive-only constraint protects the production system:
- Retraining and evaluation can run in parallel with the live pipeline
- A weather-augmented model can be promoted to production manually after human review
- If the ERA5 API is unavailable, the entire module is skipped — no impact on existing functionality
- The backup `congestion_rf.BACKUP.pkl` ensures rollback is always possible

---

## SECTION 13 — ENTERPRISE / CLOUD LAYER

### 13.1 Databricks Migration Path (`databricks/`)

The Databricks layer provides a migration path from local Docker Spark to managed Databricks Runtime. **The Bronze/Silver/Gold medallion architecture remains identical** — only the compute environment changes.

| Local (current) | Databricks (future) |
|---|---|
| Spark Master + 2 Workers in Docker | Databricks cluster (auto-scaling) |
| Delta files on `/delta` volume | Delta tables on DBFS or Unity Catalog |
| Manual job submission | Databricks Jobs UI / REST API |
| 8 GB total worker memory | Configurable cluster size (TB scale) |

**Notebooks provided:**

| Notebook | Equivalent local job |
|---|---|
| `databricks/notebooks/01_bronze_notebook.py` | `bronze_job.py` + `spark_streaming_consumer.py` |
| `databricks/notebooks/02_silver_notebook.py` | `silver_job.py` |
| `databricks/notebooks/03_gold_notebook.py` | `gold_job.py` |
| `databricks/examples/kafka_streaming_notebook.py` | Real-time Kafka → Delta streaming example |

**Supporting files:**
- `databricks/configs/environment.py` — environment variable management
- `databricks/docs/ARCHITECTURE.md` — Databricks-specific architecture notes
- `databricks/docs/MIGRATION_GUIDE.md` — step-by-step local → Databricks migration
- `databricks/docs/SNOWFLAKE_INTEGRATION.md` — Databricks to Snowflake data sync

---

### 13.2 Snowflake Integration (`src/snowflake/`)

Snowflake provides long-term archival and cross-fleet BI analytics beyond what PostgreSQL handles efficiently at scale.

**Data flow:**
```
PostgreSQL → snowflake_loader.py → Snowflake Warehouse (external SaaS)
```

**Files:**

| File | Purpose |
|---|---|
| `src/snowflake/snowflake_loader.py` | Reads PostgreSQL tables, writes to Snowflake via connector |
| `src/snowflake/snowflake_queries.py` | Library of analytical SQL queries optimized for Snowflake |
| `src/snowflake/snowflake_schema.sql` | Warehouse DDL — creates Snowflake tables matching PostgreSQL star schema |
| `src/snowflake/README.md` | Setup and usage guide |

**API exposure:**  
`api/routers/snowflake_router.py` adds `/snowflake/*` endpoints to FastAPI, allowing the Streamlit and React dashboards to query the Snowflake warehouse directly when PostgreSQL data is insufficient for long-range analytics.

**Use cases Snowflake enables:**
- Multi-month trend analysis beyond PostgreSQL retention window
- Cross-fleet benchmarking across shipping companies
- BI tool integration (Tableau, Looker) via Snowflake ODBC/JDBC
- Regulatory reporting with immutable historical audit trail

---

### 13.3 MLOps with MLflow (`mlops/`)

**Purpose:** Track experiments, compare model versions, and manage the model registry — enabling reproducible ML in production.

**Structure:**

```
mlops/
├── configs/
│   └── mlflow_config.py        ← MLflow server config, experiment names
├── experiments/
│   ├── train_anomaly_mlflow.py     ← IsolationForest with MLflow logging
│   ├── train_predictor_mlflow.py   ← XGBoost with MLflow logging
│   ├── train_congestion_mlflow.py  ← Random Forest with MLflow logging
│   └── feature_importance.py       ← SHAP / permutation importance
├── model_registry/
│   ├── evaluate_models.py          ← Cross-model comparison, champion selection
│   └── model_loader.py             ← Load registered models from MLflow
├── mlruns/                         ← Local experiment tracking (file-based)
├── artifacts/                      ← Model artifacts
└── model_registry/                 ← Registered model snapshots
```

**What each training script logs to MLflow:**

| Script | Logged metrics | Logged params |
|---|---|---|
| `train_anomaly_mlflow.py` | anomaly_rate, precision, recall | contamination, n_estimators |
| `train_predictor_mlflow.py` | RMSE lat (5/10/15 min), RMSE lon | horizon, n_estimators, max_depth |
| `train_congestion_mlflow.py` | accuracy, F1 (macro), F1 per class | n_estimators, max_depth |
| `feature_importance.py` | Feature importance rankings | method (SHAP or permutation) |

**Docker compose MLflow:**  
`docker/docker-compose.mlflow.yml` adds a standalone MLflow tracking server at `:5000` with PostgreSQL as the metadata backend.

---

### 13.4 Kubernetes Deployment (`k8s/`)

**Namespace:** `maritime-nav`

Kubernetes manifests provide a production-grade deployment layout for cloud migration. All services from Docker Compose have corresponding K8s resources.

**Deployments and scaling:**

| Deployment | Replicas | HPA min–max |
|---|---|---|
| `fastapi-deployment.yaml` | 2 | 2–10 pods |
| `frontend-deployment.yaml` | 2 | 2–5 pods |
| `streamlit-deployment.yaml` | 1 | 1–3 pods |
| `live-scorer-deployment.yaml` | 1 | fixed (1) |
| `postgres-deployment.yaml` | 1 | StatefulSet recommended |
| `kafka-deployment.yaml` | 1 | StatefulSet recommended |

**Supporting K8s resources:**

| File | Purpose |
|---|---|
| `k8s/hpa/hpa.yaml` | Horizontal Pod Autoscaler config |
| `k8s/configmaps/app-config.yaml` | Non-secret environment variables |
| `k8s/secrets/secrets-template.yaml` | Secret template (must be populated before deploy) |
| `k8s/services/services.yaml` | ClusterIP (internal) + LoadBalancer (external) services |
| `k8s/namespace.yaml` | Creates `maritime-nav` namespace |
| `k8s/deploy.sh` | One-command deploy script |

**Documentation:**
- `k8s/docs/MIGRATION_GUIDE.md` — Docker Compose → Kubernetes migration steps
- `k8s/docs/PRODUCTION_RECOMMENDATIONS.md` — Production hardening checklist
- `k8s/docs/SCALING_STRATEGY.md` — Load-based scaling strategy

---

### 13.5 CI/CD Pipeline (`.github/workflows/`)

Three GitHub Actions workflows automate quality gates:

| Workflow file | Trigger | Purpose |
|---|---|---|
| `ci.yml` | Push / PR to any branch | Full test suite (pytest), coverage, integration checks |
| `code-quality.yml` | Push / PR | Linting (flake8/ruff), type checks, formatting |
| `docker-build.yml` | Push to main | Builds all Docker images, validates compose stack |

**Branch protection:** `.github/BRANCH_PROTECTION.md` documents required status checks before merge to main.

---

## SECTION 14 — DOCKER COMPOSE INFRASTRUCTURE

### 14.1 Services Summary

| Service | Container | Technology | Port(s) |
|---|---|---|---|
| `zookeeper` | Kafka coordination | Confluent cp-zookeeper 7.4.0 | internal :2181 |
| `kafka` | Message broker | Confluent cp-kafka 7.4.0 | :9092 / :29092 |
| `kafka-ui` | Topic browser | provectuslabs/kafka-ui | :8083 |
| `spark-master` | Cluster scheduler | Apache Spark 3.4 | :9090, :7077 |
| `spark-worker-1` | Executor | Apache Spark 3.4 | :9091 |
| `spark-worker-2` | Executor | Apache Spark 3.4 | :9093 |
| `spark-stream` | Kafka → Bronze stream | Spark Structured Streaming | — |
| `ais-producer` | AIS feed simulator | Python 3.10, kafka-python | — |
| `live-scorer` | Real-time ML scoring | Python 3.10, scikit-learn, XGBoost | — |
| `maritime-api` | REST API | FastAPI 0.111, Uvicorn, Python 3.11 | :8000 |
| `postgres` | Operational DB | PostgreSQL 15-alpine | :5432 |
| `streamlit-dashboard` | Python dashboards | Streamlit | :8501 |
| `maritime-frontend` | React UI | React 18, Node 18-slim | :3000 |

### 14.2 Dockerfiles

| File | Service |
|---|---|
| `docker/Dockerfile.api` | FastAPI service |
| `docker/Dockerfile.frontend` | React frontend |
| `docker/Dockerfile.producer` | AIS producer + live_scorer |
| `docker/Dockerfile.spark` | Spark master and workers |
| `docker/Dockerfile.streamlit` | Streamlit dashboard |
| `docker/Dockerfile.airflow` | Airflow scheduler and worker |
| `docker/docker-compose.mlflow.yml` | Optional MLflow tracking server |

### 14.3 Airflow (Pending Integration)

New Airflow files are present but not yet merged into the main `docker-compose.yml`:

| File | Purpose |
|---|---|
| `docker-compose.airflow.yml` | Full Airflow stack (scheduler, webserver, worker) |
| `docker-compose.airflow-prep.yml` | Database initialization and setup |
| `docker/Dockerfile.airflow` | Custom Airflow image with Spark and Delta dependencies |
| `docs/AIRFLOW.md` | Setup and DAG documentation |

These files exist on the current branch (`fatemabranch`) as untracked additions.

---

## SECTION 15 — PORT REFERENCE

| Service | Host Port | Protocol | Notes |
|---|---|---|---|
| React Frontend | :3000 | HTTP | Primary user interface |
| FastAPI | :8000 | HTTP | REST API, auto-docs at /docs |
| Streamlit | :8501 | HTTP | Python analytics dashboards |
| PostgreSQL | :5432 | TCP | Not exposed externally in K8s |
| Kafka (host) | :29092 | TCP | External access for local dev tools |
| Kafka (internal) | :9092 | TCP | Internal Docker network only |
| Kafka UI | :8083 | HTTP | Topic browser |
| Spark Master UI | :9090 | HTTP | Job monitoring |
| Spark Worker 1 UI | :9091 | HTTP | Worker monitoring |
| Spark Worker 2 UI | :9093 | HTTP | Worker monitoring |
| MLflow (optional) | :5000 | HTTP | Experiment tracking UI |

---

## SECTION 16 — ARCHITECTURAL CONSTRAINTS & RULES

### 16.1 Critical Ownership Rules

These rules protect data integrity in the shared PostgreSQL serving layer:

**Rule 1 — fact_vessel_latest is owned exclusively by live_scorer**

gold_job uses Spark JDBC `mode=overwrite` which issues a TRUNCATE followed by INSERT. If gold_job wrote `fact_vessel_latest` to PostgreSQL, it would destroy all live ML-scored vessel positions accumulated since midnight. gold_job writes only to Delta Lake (`/delta/gold/vessel_latest`) — never to PostgreSQL for this table.

```
# NEVER restore this line in gold_job.py:
# sync_to_postgres(vessel_latest, "fact_vessel_latest")
```

**Rule 2 — fact_traffic_density uses data_split to separate speed and batch writes**

Both live_scorer (speed layer) and gold_job (batch layer) write to `fact_traffic_density`. They coexist because:
- live_scorer writes `data_split='live'` via UPSERT (accumulates counts per hour bucket)
- gold_job writes historical splits via JDBC (does not touch `data_split='live'` rows)

The `data_split` column is the partition key that prevents conflicts.

**Rule 3 — weather module never overwrites production models**

`src/weather/eval_weather.py` trains a candidate model and compares it against `models/congestion_rf.BACKUP.pkl`. Promotion to production requires manual copy. The backup was created before any weather retraining to guarantee rollback is always possible.

---

### 16.2 Performance Constraints

| Constraint | Detail |
|---|---|
| Silver row cap | 5,000,000 rows per batch run — limits Silver Delta size to fit in 8 GB combined Spark worker RAM |
| live_scorer batch | 200 records / 5s window — sized to keep transaction latency under 1 second at peak throughput |
| React polling | 3-second interval — balanced between live feel and PostgreSQL read load; `fact_vessel_latest` has primary key index on mmsi for fast full-table scans |
| Kafka retention | 7 days (168 hours) — allows replay of one week of AIS data if a consumer falls behind |
| Teleport filter | Implied speed > 100 knots dropped — removes GPS glitches that would corrupt haversine distance features |

---

### 16.3 What Is Live vs. Template

| Component | Status |
|---|---|
| Docker Compose core stack | **Running** — producer, kafka, spark, live_scorer, api, postgres, frontend, streamlit |
| Airflow DAG | **Template** — files present on branch, not yet integrated into main compose |
| Databricks notebooks | **Template** — ready for cloud migration, not active locally |
| Snowflake integration | **Template** — requires Snowflake account credentials to activate |
| Kubernetes manifests | **Template** — ready for cloud deployment, not active locally |
| MLflow tracking | **Optional** — available via `docker/docker-compose.mlflow.yml` |
| Weather module | **Ready** — code and docs complete, `dim_weather` created on first run |
| CI/CD workflows | **Active** — GitHub Actions run on push/PR |

---

## SECTION 17 — TEST SUITE

**Location:** `tests/`

| File | Coverage |
|---|---|
| `tests/conftest.py` | Shared fixtures; graceful `pytest.skip()` if services are not running |
| `tests/test_api.py` | FastAPI endpoint tests (15 endpoints) |
| `tests/test_ml.py` | ML model load, predict shape, scoring pipeline tests |

Tests are designed to pass in CI without a running Docker stack — all service-dependent tests skip cleanly when PostgreSQL, Kafka, or Spark are unavailable, allowing code quality checks to run on every commit.

---

## SECTION 18 — DOCUMENTATION INDEX

All documentation lives in `docs/`:

| File | Content |
|---|---|
| `ARCHITECTURE.md` | Full system architecture with Mermaid diagrams, ER diagram, component table, medallion layer table, ML model table |
| `ARCHITECTURE_DIAGRAM.md` | ASCII + Mermaid visual diagrams for all layers |
| `ARCHITECTURE_REVIEW.md` | Design decision justifications, trade-offs, reviewer notes |
| `AIRFLOW.md` | Airflow setup, DAG documentation, task dependencies |
| `CICD.md` | GitHub Actions workflow documentation |
| `COST_OPTIMIZATION.md` | Cloud cost analysis and optimization strategies |
| `DATA_QUALITY_FRAMEWORK.md` | Data quality checks at each medallion layer |
| `ENGINEERING_MATURITY.md` | Engineering maturity assessment against production readiness criteria |
| `OBSERVABILITY_GUIDE.md` | Monitoring, alerting, and logging guidance |
| `PRODUCTION_READINESS.md` | Checklist for production deployment |
| `WEATHER_INTEGRATION.md` | Weather module setup and execution guide |

---

## SECTION 19 — COMMIT HISTORY

| Commit | Date | Description |
|---|---|---|
| `6bcc2b9` | Initial | Initial commit: Maritime Navigation AI System |
| `e71b852` | — | Remove data and model files from tracking |
| `b4bbe87` | — | Complete maritime AI system — real-time pipeline, ML models, dual dashboards |
| `fdd44a2` | — | Database backup (344 MB compressed to 87 MB) |
| `7fe0ced` | 2026-05-23 | Add cloud-ready architecture: Snowflake, MLOps, CI/CD, Kubernetes templates |
| `3cc7c22` | 2026-05-30 | Add enterprise architecture layer: Databricks, Snowflake, MLOps, K8s, CI/CD |

**Current branch:** `fatemabranch`  
**Staged for next commit:** `src/weather/` module (4 files) + `docs/WEATHER_INTEGRATION.md`

---

*End of Architecture Report — Parts 1, 2, and 3*
