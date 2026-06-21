# Observability Guide — Maritime Navigation AI System

**Status:** Recommended setup — not yet implemented in the current codebase.  
**Goal:** Full-stack observability covering metrics, traces, and logs for the
real-time AIS monitoring platform.

---

## 1. Observability Stack Recommendation

```
Application Layer
      │
      ├── Metrics  → Prometheus (scrape) → Grafana (visualise)
      ├── Traces   → OpenTelemetry Collector → Jaeger / Tempo
      └── Logs     → Fluent Bit → Loki → Grafana
```

All three signals feed into **Grafana** as the single pane of glass.
This avoids running separate UIs for metrics, traces, and logs.

---

## 2. Prometheus Setup

### 2.1 Add to docker-compose.yml (new services)

```yaml
prometheus:
  image: prom/prometheus:v2.51.0
  volumes:
    - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
  ports:
    - "9090:9090"
  networks:
    - maritime-network

grafana:
  image: grafana/grafana:10.4.0
  environment:
    GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
  volumes:
    - grafana-data:/var/lib/grafana
    - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
  ports:
    - "3001:3000"
  networks:
    - maritime-network
```

### 2.2 prometheus.yml

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: fastapi
    static_configs:
      - targets: ['fastapi:8000']
    metrics_path: /metrics

  - job_name: kafka
    static_configs:
      - targets: ['kafka-exporter:9308']

  - job_name: postgres
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: spark
    static_configs:
      - targets: ['spark-master:4040']
    metrics_path: /metrics/prometheus

  - job_name: node
    static_configs:
      - targets: ['node-exporter:9100']
```

### 2.3 Expose /metrics in FastAPI

Install: `pip install prometheus-fastapi-instrumentator`

```python
# api/main.py
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)
```

This auto-exposes:
- `http_requests_total` (by method, path, status)
- `http_request_duration_seconds` (latency histogram)
- `http_request_size_bytes`

---

## 3. Key Metrics to Monitor

### 3.1 Kafka Consumer Lag

**Why it matters:** Lag = how far behind the Spark consumer is from the
latest AIS message. If lag grows, real-time vessel positions shown on the
React dashboard become stale.

```promql
# Current consumer lag per partition
kafka_consumer_group_lag{group="maritime-spark-consumer"}

# Alert: lag > 10,000 messages for > 5 minutes
ALERT KafkaConsumerLagHigh
  IF kafka_consumer_group_lag > 10000
  FOR 5m
  LABELS { severity="warning" }
  ANNOTATIONS { summary="Kafka consumer lag is high — dashboard data may be stale" }
```

**Exporter:** `danielqsj/kafka-exporter` — add to docker-compose.yml.

### 3.2 Spark Job Duration

**Why it matters:** A Silver job that normally runs in 90 seconds taking
10 minutes signals data skew, memory pressure, or a bad partition.

```promql
# Spark stage duration
spark_stage_duration_seconds{stage="silver_enrichment"}

# Alert: any stage > 600 seconds
ALERT SparkJobSlow
  IF spark_stage_duration_seconds > 600
  FOR 1m
  LABELS { severity="warning" }
```

**How to expose:** Enable Spark's Prometheus sink in `spark-defaults.conf`:

```
spark.metrics.conf.*.sink.prometheussink.class=org.apache.spark.metrics.sink.PrometheusServlet
spark.metrics.conf.*.sink.prometheussink.path=/metrics/prometheus
spark.ui.prometheus.enabled=true
```

### 3.3 API Response Time

**Why it matters:** The React dashboard polls FastAPI every few seconds for
vessel positions. Slow responses degrade the real-time experience.

```promql
# 95th percentile response time
histogram_quantile(0.95,
  rate(http_request_duration_seconds_bucket{path="/vessels"}[5m])
)

# Alert: p95 > 500ms
ALERT APIResponseSlow
  IF histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
  FOR 2m
  LABELS { severity="warning" }
```

Target: p50 < 50ms, p95 < 200ms, p99 < 500ms for vessel position queries.

### 3.4 ML Model Drift

**Why it matters:** AIS traffic patterns change seasonally (hurricane
season, port congestion). A model trained on May data will degrade by
November without retraining detection.

```python
# mlops/monitoring/drift_monitor.py
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

def check_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    share_drifted = result["metrics"][0]["result"]["share_of_drifted_columns"]
    return {"drift_detected": drift_detected, "share_drifted": share_drifted}
```

Expose drift score as a custom Prometheus gauge:

```python
from prometheus_client import Gauge
MODEL_DRIFT = Gauge("ml_model_drift_score", "Feature drift score", ["model_name"])
MODEL_DRIFT.labels(model_name="anomaly_detector").set(drift_score)
```

Alert threshold: drift score > 0.3 triggers model retraining job.

### 3.5 Alert Processing Rate

**Why it matters:** The system generated 24,792 alerts in 7 days (~3,542/day).
A sudden spike (>5,000/hour) may indicate a data quality issue (false
positives from bad GPS coordinates) rather than a real maritime incident.

```promql
# Alerts generated per minute
rate(maritime_alerts_total[1m]) * 60

# Alert: >200 alerts/minute for >3 minutes (likely false positive storm)
ALERT AlertRateAnomaly
  IF rate(maritime_alerts_total[1m]) * 60 > 200
  FOR 3m
  LABELS { severity="critical" }
  ANNOTATIONS { summary="Alert rate anomaly — possible data quality issue" }
```

---

## 4. OpenTelemetry Integration

### 4.1 Why OpenTelemetry

OpenTelemetry provides vendor-neutral distributed tracing. A single AIS
message flows through: Kafka producer → Kafka topic → Spark consumer →
PostgreSQL write → FastAPI read → React render. Without tracing, debugging
latency spikes requires correlating logs across 5 services manually.

### 4.2 Installation

```bash
pip install opentelemetry-sdk \
            opentelemetry-instrumentation-fastapi \
            opentelemetry-instrumentation-sqlalchemy \
            opentelemetry-exporter-otlp
```

### 4.3 FastAPI Instrumentation

```python
# api/main.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317"))
)
trace.set_tracer_provider(provider)

FastAPIInstrumentor.instrument_app(app)
SQLAlchemyInstrumentor().instrument()
```

### 4.4 Custom Span for AIS Processing

```python
tracer = trace.get_tracer("maritime.pipeline")

def process_vessel_message(msg: dict):
    with tracer.start_as_current_span("process_ais_message") as span:
        span.set_attribute("vessel.mmsi", msg.get("mmsi"))
        span.set_attribute("vessel.sog", msg.get("sog"))
        span.set_attribute("pipeline.stage", "silver")
        # ... processing logic
```

### 4.5 OTel Collector docker-compose service

```yaml
otel-collector:
  image: otel/opentelemetry-collector-contrib:0.98.0
  command: ["--config=/etc/otel-collector-config.yml"]
  volumes:
    - ./monitoring/otel-collector-config.yml:/etc/otel-collector-config.yml
  ports:
    - "4317:4317"   # OTLP gRPC
    - "4318:4318"   # OTLP HTTP
  networks:
    - maritime-network
```

---

## 5. Structured Logging

### 5.1 Current State (Problem)

The current codebase uses Python's default `logging` with a simple string
format. Log output looks like:

```
14:32:01  INFO     Chunk   10 | processed:  1,000,000 | kept:    999,996
```

This is human-readable but machine-unreadable. Log aggregation tools
(Loki, Elasticsearch) cannot query fields like `chunk_num` or `rows_kept`
from this format.

### 5.2 Recommended: structlog

```bash
pip install structlog
```

```python
# src/common/logging_config.py
import structlog
import logging

def configure_logging(service_name: str):
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    return structlog.get_logger(service=service_name)
```

Output becomes queryable JSON:

```json
{
  "timestamp": "2026-05-24T14:32:01.423Z",
  "level": "info",
  "service": "spark-silver",
  "event": "chunk_processed",
  "chunk_num": 10,
  "rows_processed": 1000000,
  "rows_kept": 999996,
  "mmsi_invalid": 4
}
```

### 5.3 Log Aggregation with Loki + Fluent Bit

```yaml
# docker-compose addition
fluent-bit:
  image: fluent/fluent-bit:3.0
  volumes:
    - /var/lib/docker/containers:/var/lib/docker/containers:ro
    - ./monitoring/fluent-bit.conf:/fluent-bit/etc/fluent-bit.conf
  networks:
    - maritime-network

loki:
  image: grafana/loki:3.0.0
  ports:
    - "3100:3100"
  networks:
    - maritime-network
```

Fluent Bit tails Docker container stdout, parses JSON logs, and ships to
Loki. Grafana's Explore view then lets you query:

```logql
{service="spark-silver"} | json | rows_kept < 1000
```

---

## 6. Alerting Rules Summary

| Alert | Condition | Severity | Action |
|---|---|---|---|
| KafkaConsumerLagHigh | lag > 10,000 for 5m | warning | Check Spark consumer health |
| SparkJobSlow | stage duration > 600s | warning | Check executor memory |
| APIResponseSlow | p95 > 500ms | warning | Enable Redis caching |
| AlertRateAnomaly | >200 alerts/min for 3m | critical | Check data quality |
| MLModelDrift | drift score > 0.3 | warning | Trigger retraining |
| KafkaBrokerDown | broker unreachable | critical | Restart broker, check ISR |
| PostgreSQLConnections | connections > 80% of max | warning | Add connection pooling |

---

## 7. Implementation Priority

1. **Week 1:** Add `prometheus-fastapi-instrumentator` to FastAPI, deploy
   Prometheus + Grafana containers. Connect existing services.

2. **Week 2:** Add structlog to all Python services. Deploy Loki + Fluent Bit.
   Build the Grafana dashboard for Kafka lag and API response time.

3. **Week 3:** Add OpenTelemetry tracing to FastAPI and the Kafka producer.
   Deploy Jaeger or Grafana Tempo.

4. **Week 4:** Implement ML drift monitoring with Evidently. Wire alerts to
   PagerDuty or Slack webhook.
