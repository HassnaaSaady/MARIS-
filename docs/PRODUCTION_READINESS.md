# Production Readiness Assessment — Maritime Navigation AI System

> **Purpose:** Evaluate the system against standard production criteria — reliability, operability, security, and recovery — and document what is already production-grade versus what requires work before a live deployment.

---

## 1. Readiness Summary

| Category | Status | Score |
|----------|--------|-------|
| Core Functionality | Implemented | 8/10 |
| Configuration Management | Partial | 6/10 |
| Secret Management | Needs Work | 4/10 |
| Health Checks & Probes | Partial | 5/10 |
| Logging & Observability | Partial | 6/10 |
| Error Handling & Resilience | Partial | 5/10 |
| Database Reliability | Needs Work | 4/10 |
| Security Hardening | Needs Work | 4/10 |
| Deployment Automation | Implemented | 7/10 |
| Disaster Recovery | Not Implemented | 2/10 |

**Overall: Not production-ready without completing the items in Section 3.**

---

## 2. What Is Already Production-Grade

### 2.1 Docker & Compose Stack

- All services containerized with dedicated Dockerfiles (`docker/Dockerfile.*`)
- Separation of concerns: API, frontend, Spark, producer, Streamlit each have their own image
- MLflow isolated in a separate compose file (`docker/docker-compose.mlflow.yml`)
- `pool_pre_ping=True` on SQLAlchemy engine prevents stale connection errors

### 2.2 Kubernetes Templates

The `k8s/` directory includes ready-to-use manifests:

```
k8s/
├── namespace.yaml                  ✅ Namespace isolation
├── deployments/                    ✅ One Deployment per service
├── services/services.yaml          ✅ ClusterIP + LoadBalancer definitions
├── hpa/hpa.yaml                    ✅ Horizontal Pod Autoscaler configured
├── configmaps/app-config.yaml      ✅ Non-secret config externalized
└── secrets/secrets-template.yaml   ⚠️  Template only — not populated
```

### 2.3 CI/CD Pipelines

- GitHub Actions workflows present: `ci.yml`, `code-quality.yml`, `docker-build.yml`
- Branch protection rules documented in `.github/BRANCH_PROTECTION.md`

### 2.4 ML Experiment Tracking

- MLflow integration with PostgreSQL artifact backend (`mlops/configs/mlflow_config.py`)
- Separate experiment scripts per model: anomaly, congestion, predictor

### 2.5 API Layer

- FastAPI with versioned router structure (`api/routers/`)
- Snowflake integration router isolated (`api/routers/snowflake_router.py`)
- SQLAlchemy models with indexed columns on `mmsi` and `base_datetime`

---

## 3. Gaps — Blocking for Production

### 3.1 Secret Management (Critical)

**Current state:**

```python
# api/models/database.py and streamlit_app.py
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost/maritime")
```

Hardcoded fallback credentials appear in source. The `k8s/secrets/secrets-template.yaml` is a template only.

**Required before production:**

```yaml
# secrets-template.yaml must become a real Secret (never committed)
apiVersion: v1
kind: Secret
metadata:
  name: maritime-secrets
  namespace: maritime-nav
type: Opaque
stringData:
  POSTGRES_URL: "postgresql://prod_user:REPLACE@postgres-svc/maritime"
  KAFKA_BOOTSTRAP_SERVERS: "kafka-svc:9092"
  SNOWFLAKE_ACCOUNT: "REPLACE"
  SNOWFLAKE_PASSWORD: "REPLACE"
```

Use a secrets manager (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault) to inject at deploy time — never store real values in git.

---

### 3.2 Database Reliability

**Current state:** Single PostgreSQL container with no backup, no replica, no failover.

**Required before production:**

| Item | Why Required | How |
|------|-------------|-----|
| Automated backups | Data loss prevention | `pg_dump` via cron or managed DB service |
| WAL archiving | Point-in-time recovery | `archive_mode=on` + S3/GCS destination |
| Connection pooling | Handle concurrent dashboard + API load | PgBouncer sidecar or `pgbouncer` container |
| Liveness probe | Restart on crash | `pg_isready` in k8s probe |

---

### 3.3 Health Checks

**Current state:** No `/health` or `/ready` endpoints on the FastAPI app. K8s probes in deployment manifests reference paths that may not exist.

**Minimum required endpoint:**

```python
# api/main.py
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/ready")
async def ready(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        raise HTTPException(503, "Database unavailable")
```

**K8s probe alignment:**

```yaml
# k8s/deployments/fastapi-deployment.yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 15
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 10
```

---

### 3.4 Structured Logging

**Current state:** `read_pg()` silently swallows all exceptions and returns an empty DataFrame. Failures are invisible in production.

```python
# Current — production-hostile
except Exception as e:
    return pd.DataFrame()
```

**Required:**

```python
import logging
logger = logging.getLogger(__name__)

except Exception as e:
    logger.error("read_pg failed: %s | query=%s", e, query[:200], exc_info=True)
    return pd.DataFrame()
```

Log format should be JSON for aggregation by Loki/CloudWatch/Datadog:

```python
import json, sys
logging.basicConfig(
    stream=sys.stdout,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
```

---

### 3.5 Rate Limiting on API

**Current state:** No rate limiting on FastAPI endpoints. An AIS data flood or misconfigured producer could exhaust PostgreSQL connections.

**Minimum viable:**

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/vessels")
@limiter.limit("60/minute")
async def get_vessels(request: Request):
    ...
```

---

## 4. Gaps — Should Fix Before Production

### 4.1 Graceful Shutdown

Kafka consumers and Spark streaming jobs do not handle `SIGTERM` cleanly, which causes offset commit failures on pod restarts.

```python
import signal, sys

def handle_sigterm(*_):
    consumer.close()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)
```

### 4.2 Idempotent Writes

The `fact_ais_track` insert in `populate_fact_ais_track.py` does not handle duplicate messages from Kafka re-delivery. Add `ON CONFLICT DO NOTHING`:

```sql
INSERT INTO fact_ais_track (mmsi, base_datetime, lat, lon, ...)
VALUES (:mmsi, :base_datetime, :lat, :lon, ...)
ON CONFLICT (mmsi, base_datetime) DO NOTHING;
```

This requires a unique constraint on `(mmsi, base_datetime)`.

### 4.3 Frontend Environment Variables

```javascript
// frontend/src/App.jsx — hardcoded API URL
const API_URL = "http://localhost:8000";
```

Must be replaced with a build-time environment variable (`REACT_APP_API_URL`) injected via Docker build args or a ConfigMap.

---

## 5. Disaster Recovery Baseline

| Scenario | Current State | Required Action |
|----------|--------------|-----------------|
| Pod crash | K8s restarts automatically (if probes work) | Add working health probes |
| DB data loss | No backup — unrecoverable | Add `pg_dump` cron + S3 upload |
| Kafka topic loss | Messages gone — unrecoverable | Enable topic replication factor ≥ 2 |
| ML model corruption | MLflow artifacts in local volume | Mount persistent volume + backup artifacts |
| Config loss | Tracked in git | Already handled |
| Secret loss | Not backed up | Store secrets in vault with backup |

**RTO/RPO Targets (suggested for v1 production):**

| Metric | Target |
|--------|--------|
| Recovery Time Objective (RTO) | < 30 minutes |
| Recovery Point Objective (RPO) | < 1 hour (last DB backup) |

---

## 6. Pre-Launch Checklist

### Security
- [ ] No hardcoded credentials in any committed file
- [ ] `secrets-template.yaml` → real Secret via vault injection
- [ ] API rate limiting enabled
- [ ] HTTPS/TLS configured on ingress
- [ ] Network policies restrict inter-pod communication

### Reliability
- [ ] `/health` and `/ready` endpoints implemented and referenced in K8s probes
- [ ] PostgreSQL backup cron running and tested
- [ ] `ON CONFLICT DO NOTHING` on track inserts
- [ ] SIGTERM handlers on Kafka consumer and Spark job
- [ ] Connection pool (PgBouncer) in front of PostgreSQL

### Observability
- [ ] Structured JSON logging on all services
- [ ] Prometheus metrics endpoint on FastAPI (`/metrics`)
- [ ] Alerts configured for: DB connection failures, Kafka consumer lag, API error rate > 1%

### Deployment
- [ ] `docker-compose.yml` environment variables sourced from `.env` (never committed)
- [ ] K8s secrets populated via CI/CD secret injection (not stored in repo)
- [ ] Image tags pinned to SHA or semver (not `latest`)
- [ ] Rollback procedure documented and tested

---

*Last updated: 2026-05-24*
