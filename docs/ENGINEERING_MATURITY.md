# Engineering Maturity Assessment — Maritime Navigation AI System

> **Purpose:** Evaluate the system across five engineering maturity dimensions — Code Quality, Testing, DevOps, ML Operations, and Team Practices — using a 1–5 scale. Distinguishes what is already implemented from what needs investment.

---

## Maturity Scale

| Level | Label | Meaning |
|-------|-------|---------|
| 1 | Initial | Ad-hoc, no defined process |
| 2 | Developing | Some practices in place, inconsistent |
| 3 | Defined | Consistent practices, documented standards |
| 4 | Managed | Metrics-driven, automated enforcement |
| 5 | Optimizing | Continuous improvement, self-healing |

---

## 1. Code Quality — Level 2.5 / 5

### What Is Implemented

- Python services follow a consistent package structure (`api/`, `src/`, `mlops/`)
- SQLAlchemy models are typed with explicit column types (`String(20)`, `Float`, etc.)
- FastAPI router separation: `snowflake_router.py` isolated from core `main.py`
- Kafka and Spark config externalized to environment variables

### Gaps

| Issue | Location | Impact |
|-------|----------|--------|
| Silent exception swallowing | `streamlit_app.py:read_pg()` | Failures invisible |
| Inline SQL strings in dashboard | `streamlit_app.py` throughout | Hard to test, SQLi risk if ever interpolated |
| No type hints on dashboard functions | `streamlit_app.py` | IDE support degraded |
| `import re as _re` inside conditional blocks | Dashboard replay section | Non-standard; move to module top |
| Magic numbers without constants | `LIMIT 5000`, `zoom=9`, ports | Brittle |

### Recommendations

```python
# Replace magic numbers with named constants at module top
TRACK_QUERY_LIMIT = 5000
MAP_DEFAULT_ZOOM = 9
KAFKA_POLL_TIMEOUT_MS = 1500

# Add type hints to public functions
def get_track(mmsi: str) -> pd.DataFrame: ...
def read_pg(query: str, params: dict | None = None) -> pd.DataFrame: ...
```

---

## 2. Testing — Level 1.5 / 5

### What Is Implemented

- GitHub Actions CI pipeline (`ci.yml`) exists and is referenced
- `code-quality.yml` workflow suggests linting is run in CI

### Gaps

| Gap | Risk |
|----|------|
| No unit test files found (`test_*.py` / `*_test.py`) | Regressions ship undetected |
| No integration tests for API endpoints | `/vessels`, `/alerts` endpoints untested |
| No data pipeline tests | Silent bad data enters `fact_ais_track` |
| No ML model tests (accuracy thresholds, schema checks) | Model degradation goes unnoticed |
| No dashboard smoke tests | Streamlit page errors invisible until user sees them |

### Recommended Test Structure

```
tests/
├── unit/
│   ├── test_normalization.py       # MMSI cleaning, coordinate validation
│   ├── test_models.py              # SQLAlchemy model constraints
│   └── test_ml_features.py        # Feature engineering functions
├── integration/
│   ├── test_api_vessels.py         # FastAPI TestClient against real DB
│   ├── test_pipeline_insert.py     # End-to-end: produce → consume → DB
│   └── test_alert_generation.py    # Proximity alert logic
└── smoke/
    └── test_dashboard_pages.py     # Streamlit page load without error
```

**Minimum viable test for the MMSI bug class:**

```python
import re

def normalize_mmsi(raw: str) -> str:
    return re.sub(r"[^\d]", "", raw.strip())

def test_normalize_mmsi_strips_markdown():
    assert normalize_mmsi("*366772760 *") == "366772760"
    assert normalize_mmsi("**366772760**") == "366772760"
    assert normalize_mmsi(" 366772760 ") == "366772760"
    assert normalize_mmsi("`366772760`") == "366772760"
    assert normalize_mmsi("366772760") == "366772760"
```

### Target Coverage

| Layer | Current | Target |
|-------|---------|--------|
| Unit tests | 0% | ≥ 70% |
| API integration | 0% | ≥ 60% |
| Data pipeline | 0% | ≥ 50% |
| ML model validation | 0% | ≥ 80% (accuracy gates) |

---

## 3. DevOps & Deployment — Level 3 / 5

### What Is Implemented

- Full Docker Compose stack for local development
- Kubernetes manifests for all services (`k8s/deployments/`, `k8s/services/`)
- HPA configured (`k8s/hpa/hpa.yaml`)
- Three GitHub Actions workflows (CI, code quality, Docker build)
- Branch protection rules documented
- K8s deploy script (`k8s/deploy.sh`)
- Separate MLflow compose file for optional stack

### Gaps

| Gap | Impact |
|----|--------|
| No CD pipeline (deployment step absent from workflows) | Manual deploys required |
| Image tags use `latest` in compose files | No rollback capability |
| Secrets template not populated | K8s deploy would fail without manual step |
| No staging environment defined | Changes go straight to production |
| `deploy.sh` not idempotent | Re-running may cause partial state |

### Recommended CI/CD Additions

```yaml
# .github/workflows/cd.yml (to be created)
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and push image
        run: |
          IMAGE_TAG="${GITHUB_SHA::8}"
          docker build -t ghcr.io/org/maritime-api:$IMAGE_TAG -f docker/Dockerfile.api .
          docker push ghcr.io/org/maritime-api:$IMAGE_TAG
      - name: Deploy to staging
        run: |
          kubectl set image deployment/fastapi-deployment \
            fastapi=ghcr.io/org/maritime-api:$IMAGE_TAG \
            -n maritime-nav
```

---

## 4. ML Operations — Level 2.5 / 5

### What Is Implemented

- MLflow experiment tracking with PostgreSQL backend
- Three separate training scripts: anomaly, congestion, route predictor
- Feature importance logging (`mlops/experiments/feature_importance.py`)
- `mlops/configs/mlflow_config.py` centralizes tracking URI and experiment names
- K8s deployment for live scorer (`k8s/deployments/live-scorer-deployment.yaml`)

### Gaps

| Gap | Impact |
|----|--------|
| No model performance monitoring in production | Drift undetected |
| No A/B testing framework | Cannot compare model versions safely |
| Model artifacts in local volume | Not portable across environments |
| No automated retraining trigger | Models stale after data distribution shifts |
| No data validation before model input | Bad AIS data causes silent prediction errors |

### Recommended MLOps Additions

```python
# mlops/monitoring/drift_detector.py (to be created)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def check_feature_drift(reference_df, current_df):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    results = report.as_dict()
    drift_detected = results["metrics"][0]["result"]["dataset_drift"]
    return drift_detected
```

**Model Registry Workflow (already partially available via MLflow):**

```
Train → Validate (accuracy ≥ threshold) → Register in MLflow → 
Promote to Staging → Load test → Promote to Production → 
Monitor drift → Trigger retraining if drift > 0.15
```

---

## 5. Team Practices — Level 2 / 5

### What Is Implemented

- Git repository with branch structure (`main`, feature branches)
- Documentation present: `ARCHITECTURE_REVIEW.md`, `CICD.md`, `OBSERVABILITY_GUIDE.md`
- Databricks migration guides and Snowflake integration docs
- K8s operational runbooks (`k8s/docs/`)

### Gaps

| Gap | Impact |
|----|--------|
| No `CONTRIBUTING.md` | Onboarding new engineers is slow |
| No PR template | Reviews miss checklist items |
| No ADR (Architecture Decision Records) | Decisions lost over time |
| No on-call runbook | Incident response is undocumented |
| No SLA/SLO definitions | No agreement on what "working" means |

### Recommended Additions

**PR Template** (`.github/pull_request_template.md`):

```markdown
## Summary
<!-- What does this change do? -->

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Tested locally with docker-compose

## Checklist
- [ ] No hardcoded credentials
- [ ] No `latest` image tags
- [ ] MMSI inputs normalized before DB query
- [ ] Logging added for new error paths
```

**SLO Definitions (suggested for v1):**

| Service | SLO |
|---------|-----|
| API availability | 99.5% uptime (< 3.6 h/month downtime) |
| Dashboard load | p95 < 3 s |
| Alert latency | < 60 s from AIS message to alert |
| Track query | p99 < 2 s for 5,000-point track |

---

## 6. Maturity Radar Summary

```
Code Quality    ████████░░░░░░░░░░░░  2.5 / 5
Testing         ███░░░░░░░░░░░░░░░░░  1.5 / 5
DevOps          ██████████████░░░░░░  3.0 / 5
MLOps           ████████░░░░░░░░░░░░  2.5 / 5
Team Practices  ████████░░░░░░░░░░░░  2.0 / 5
─────────────────────────────────────
Overall         ████████░░░░░░░░░░░░  2.3 / 5
```

**Primary investment areas to reach Level 3 across the board:**

1. **Testing first** — even 50% unit coverage on business logic (MMSI normalization, alert generation, ML features) would catch the class of bugs currently being fixed manually.
2. **CD pipeline** — automate the deploy step that currently exists only as a shell script.
3. **Model monitoring** — basic drift detection prevents silent ML degradation.

---

*Last updated: 2026-05-24*
