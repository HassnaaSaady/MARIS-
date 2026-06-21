# Architecture Review — Maritime Navigation AI System

**Reviewer role:** Senior Big Data Architect  
**Date:** 2026-05-24  
**Scope:** Full-stack review of the real-time maritime AIS monitoring platform  
**Data scale:** 62M rows, ~6 GB, 7 days, 30,771 vessels, 24,792 alerts  

> **Honest baseline:** This is a well-structured portfolio/prototype system.
> The review below separates what is *currently implemented* from what is
> *recommended for production*. No production deployment exists yet.

---

## 1. Architecture Overview

```
AIS CSV Files → Kafka → Spark (Bronze/Silver/Gold) → PostgreSQL → FastAPI → React/Streamlit
                                                    → Snowflake (analytics)
                                                    → MLflow (model registry)
```

The medallion pattern is correct for this domain. The dual-dashboard approach
(React for real-time, Streamlit for analytics) is a sensible split.

---

## 2. Strengths

| Area | Observation |
|---|---|
| Data modeling | Bronze/Silver/Gold medallion is the right pattern for AIS streams |
| Schema design | `schema_utils.py` with canonical field aliases handles dirty MarineCadastre headers well |
| ML integration | MLflow experiment tracking is wired in from the start |
| Deployment intent | Docker Compose + Kubernetes manifests shows forward planning |
| Data volume | Chunked CSV→Parquet pipeline handles 800MB+ files without OOM (mostly) |
| CI/CD | GitHub Actions pipeline covers lint, test, and build stages |

---

## 3. Bottlenecks

### 3.1 Kafka — Single Broker, No Replication

**Current state:** Single Kafka broker defined in `docker-compose.yml`.  
**Problem:** One broker means zero fault tolerance. A restart drops all
in-flight AIS messages. With real-time vessel tracking, even 60 seconds of
lost data creates position gaps that break dead-reckoning predictions.

**Fix:** Minimum 3-broker cluster with `replication.factor=3` and
`min.insync.replicas=2`.

### 3.2 Spark — No Resource Isolation

**Current state:** Spark runs in local mode inside a Docker container with
no explicit CPU/memory limits.  
**Problem:** During the CSV→Parquet conversion, `ais-2025-05-02.csv` failed
with `[Errno 12] Cannot allocate memory`. This confirms Spark and the host
OS are competing for WSL2 memory. In production, unconstrained Spark will
starve API containers.

**Fix:** Set `spark.executor.memory`, `spark.driver.memory` explicitly.
Add Docker `mem_limit` and `cpus` to the Spark service.

### 3.3 PostgreSQL — OLTP Used for Analytical Queries

**Current state:** PostgreSQL stores processed AIS records and serves the
FastAPI layer.  
**Problem:** At 62M rows/week, analytical queries (vessel history, alert
aggregations) will degrade OLTP response times as data accumulates. There
are no indexes defined beyond primary keys in the current schema.

**Fix:** Add composite indexes on `(mmsi, base_datetime)` and `(lat_bin, lon_bin)`.
Route analytical queries to Snowflake or a read replica. Archive rows older
than 30 days to cold storage.

### 3.4 Streaming — Batch File Ingestion Masquerading as Streaming

**Current state:** `kafka_producer.py` reads CSV files row-by-row and
publishes to Kafka. This is replay, not real streaming.  
**Problem:** There is no live AIS feed connected. The system cannot currently
ingest from a real AIS socket (TCP/UDP NMEA or a commercial AIS API).

**Fix for production:** Connect to a live AIS source (e.g., MarineCadastre
live feed, Spire Maritime API, or a VHF AIS receiver with gpsd). The Kafka
producer needs a socket reader, not a CSV reader.

### 3.5 Delta Lake — No VACUUM or OPTIMIZE Jobs Scheduled

**Current state:** Delta tables are written but no maintenance jobs run.  
**Problem:** Delta Lake accumulates small Parquet part files with every
streaming micro-batch. Without `OPTIMIZE` (file compaction) and `VACUUM`
(tombstone cleanup), query performance degrades linearly and storage costs
grow faster than data volume.

**Fix:** Schedule `OPTIMIZE` daily and `VACUUM RETAIN 168 HOURS` weekly
via Airflow or Databricks Jobs.

---

## 4. Scalability Limits

### 4.1 Current Ceiling

| Component | Estimated Limit | Reason |
|---|---|---|
| Kafka (1 broker) | ~50K msg/sec | Single partition, no replication |
| Spark local mode | ~10M rows/min | Single JVM, WSL2 memory cap |
| PostgreSQL | ~5M active rows | No partitioning, no read replica |
| FastAPI | ~500 req/sec | No caching layer (Redis absent) |
| Streamlit | 1 concurrent user | Streamlit's threading model |

### 4.2 Real AIS Scale

The US Coast Guard AIS network processes approximately 25M messages/day from
~200,000 active vessels nationwide. The current architecture supports
roughly 10% of that volume before hitting bottlenecks.

---

## 5. Security Issues

### 5.1 Secrets in Environment Variables (Unencrypted)

**Current state:** `docker-compose.yml` and `k8s/secrets/secrets-template.yaml`
use plaintext environment variables for DB passwords, Snowflake credentials,
and Kafka SASL keys.  
**Risk:** Any process with access to the container environment can read
credentials. Docker inspect reveals env vars to any user with Docker socket
access.  
**Fix:** Use Docker Secrets, Kubernetes Secrets (base64 encoded, RBAC-gated),
or HashiCorp Vault. Never store real credentials in `docker-compose.yml`.

### 5.2 No TLS on Internal Services

**Current state:** Kafka, PostgreSQL, and FastAPI communicate over plaintext
within the Docker network.  
**Risk:** On a shared host or cloud VPC, lateral movement by an attacker
allows credential sniffing.  
**Fix:** Enable TLS on Kafka (`ssl.keystore`), PostgreSQL (`sslmode=require`),
and terminate TLS at an ingress controller (nginx/Traefik) for FastAPI.

### 5.3 FastAPI Has No Authentication

**Current state:** All API endpoints in `api/main.py` are unauthenticated.
Any client on the network can read vessel positions and trigger alerts.  
**Risk:** AIS data has dual-use sensitivity (vessel positions can reveal
military or law enforcement patterns).  
**Fix:** Add OAuth2/JWT authentication. Implement role-based access:
`viewer` (read-only), `operator` (alert management), `admin` (config).

### 5.4 No Network Policies

**Current state:** All Docker services share one bridge network. All
Kubernetes pods have no `NetworkPolicy` defined.  
**Risk:** A compromised Streamlit container can directly connect to
PostgreSQL without restriction.  
**Fix:** Apply Kubernetes `NetworkPolicy` to allow only intended
service-to-service communication paths.

### 5.5 No Input Validation on AIS Ingestion

**Current state:** `kafka_producer.py` publishes raw CSV rows to Kafka
without schema validation.  
**Risk:** Malformed AIS records (injected via spoofed vessel transponders —
a known real-world attack vector) propagate into Silver/Gold layers and
corrupt ML training data.  
**Fix:** Validate against the canonical schema at the Kafka producer level.
Reject records with impossible coordinates, future timestamps, or invalid
MMSI ranges.

---

## 6. Weak Engineering Decisions

### 6.1 `data_split` Column Baked into Bronze Layer

The `convert_csv.py` script adds a `data_split` column (`train`/`test`/`live`)
to the Bronze Parquet files based on file index. This couples data ingestion
to ML experiment design. If you re-run with different files, the split
changes silently.

**Better approach:** Determine train/test splits in the ML training pipeline,
not during ingestion. Bronze data should be immutable raw facts.

### 6.2 Risk Scoring Hardcoded to Suez Canal Coordinates

In `convert_csv.py`:
```python
chunk.loc[
    (chunk["sog"] < 1.0) &
    (chunk["lat"].between(29.5, 31.5)) &
    (chunk["lon"].between(31.0, 33.5)),
    "risk_level"
] = "HIGH"
```
The data is US coastal waters. The Suez Canal coordinates (lat 29.5–31.5,
lon 31.0–33.5) are in Egypt. This enrichment is geographically wrong for
this dataset and will produce no HIGH risk flags.

**Fix:** Use US port zone coordinates from `config.py` (`US_PORT_ZONES`),
which already defines correct bounding boxes.

### 6.3 `existing_data_behavior="overwrite_or_ignore"` in Parquet Output

The Bronze writer silently ignores write conflicts. If two Spark jobs run
simultaneously (e.g., a re-run overlapping with a scheduled job), partial
writes are silently dropped rather than raising an error. This makes data
completeness impossible to verify.

**Fix:** Use Delta Lake's ACID transactions for Bronze writes, which provide
proper conflict detection and idempotent upserts.

### 6.4 Streamlit as a Production Dashboard

Streamlit re-executes the entire Python script on every user interaction.
With 62M+ rows in PostgreSQL, any unfiltered query triggered by a slider
or dropdown will lock the connection for seconds. Streamlit also has a
single-threaded model — one slow query blocks all users.

**Fix for production:** Replace Streamlit with a pre-aggregated dashboard
(Apache Superset, Grafana, or a proper React component backed by cached
API endpoints).

### 6.5 No Idempotency in Kafka Consumer

The Spark Streaming consumer does not implement exactly-once semantics.
Kafka `auto.offset.reset=earliest` combined with consumer group restarts
will reprocess messages and create duplicate rows in Silver/Gold tables.

**Fix:** Use Delta Lake merge (`MERGE INTO`) with MMSI + timestamp as the
deduplication key, or enable Spark Structured Streaming's built-in
idempotent sink.

---

## 7. Missing Production Features

| Feature | Priority | Notes |
|---|---|---|
| Live AIS socket ingestion | Critical | Currently CSV replay only |
| Authentication / RBAC | Critical | All endpoints open |
| TLS everywhere | Critical | Plaintext internal traffic |
| Kafka replication (3 brokers) | High | Zero fault tolerance now |
| Delta VACUUM / OPTIMIZE jobs | High | Storage will bloat |
| Redis caching for API | High | No response caching |
| Airflow orchestration | High | Manual pipeline triggering |
| Prometheus + Grafana | High | No metrics visibility |
| Structured logging (JSON) | Medium | Unstructured stdout logs |
| Data quality checks (Great Expectations) | Medium | No validation layer |
| Model serving endpoint (MLflow) | Medium | Models trained but not served |
| Multi-region HA | Low | Single-AZ setup |
| Data retention policy | Medium | No archival or purge logic |

---

## 8. Verdict

The system demonstrates strong architectural intent with the right technology
choices for the domain. The medallion pipeline, MLflow integration, and
dual-dashboard design are all correct decisions. The primary gaps are in
**security** (no auth, plaintext secrets), **operational correctness**
(wrong geo coordinates in risk scoring, no idempotency), and **production
readiness** (no live data feed, no observability, no HA).

With 4–6 weeks of focused engineering, the security and correctness issues
can be resolved and the system could be deployed in a limited production
context (internal maritime ops tool, not public-facing).
