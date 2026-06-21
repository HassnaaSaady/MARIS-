# Cost Optimization Guide — Maritime Navigation AI System

> **Scope:** Analysis of current resource usage and actionable recommendations to reduce infrastructure spend without compromising operational reliability.

---

## 1. Current Architecture Cost Profile

### 1.1 Service Inventory

| Service | Technology | Deployment | Cost Driver |
|---------|-----------|-----------|-------------|
| API Server | FastAPI (Docker) | Single container | CPU + memory |
| Frontend | React (Nginx) | Single container | Minimal |
| Streamlit Dashboard | Python | Single container | CPU bursts on render |
| Message Broker | Kafka (KRaft) | Single container | Memory-resident logs |
| Database | PostgreSQL | Single container | Storage + IOPS |
| Stream Processor | Spark | Single container | Memory-heavy |
| ML Experiments | MLflow | Optional container | Storage for artifacts |

### 1.2 Observed Cost Inefficiencies

#### Database (PostgreSQL)

```
Current: Single always-on instance, no read replicas, no connection pooling
Issue:   Dashboard queries run full scans on fact_ais_track (can exceed 10M rows)
Impact:  High CPU during peak dashboard load; storage grows unbounded
```

- `fact_ais_track` has no TTL or partition pruning — old rows accumulate indefinitely
- No query result caching between dashboard refreshes
- `read_pg()` in `streamlit_app.py` creates a new SQLAlchemy engine on every call

#### Kafka

```
Current: In-process KRaft mode, default log retention 7 days
Issue:   AIS messages at ~1 msg/sec = ~600K msgs/day retained unnecessarily long
Impact:  Disk usage grows at ~50–100 MB/day depending on message size
```

#### Spark

```
Current: Full Spark session launched for every streaming batch
Issue:   Spark overhead (~1 GB JVM) is disproportionate for single-node AIS volume
Impact:  Memory waste; cold-start latency on each pipeline restart
```

---

## 2. Implemented Cost Controls (Already in Codebase)

| Control | Location | Effect |
|---------|----------|--------|
| `LIMIT 5000` on track queries | `streamlit_app.py:392` | Caps per-query row fetch |
| `consumer_timeout_ms=1500` | `streamlit_app.py:132` | Bounds Kafka poll duration |
| `pool_pre_ping=True` | `streamlit_app.py:117` | Reuses connections, avoids reconnect overhead |
| Docker resource constraints | `docker-compose.yml` | Prevents runaway container memory |
| MLflow optional stack | `docker/docker-compose.mlflow.yml` | Separated from core stack |

---

## 3. Recommendations by Priority

### 3.1 High Impact — Database

#### 3.1.1 Partition `fact_ais_track` by Month

```sql
-- Migrate to range-partitioned table
CREATE TABLE fact_ais_track_partitioned (
    LIKE fact_ais_track INCLUDING ALL
) PARTITION BY RANGE (base_datetime);

CREATE TABLE fact_ais_track_2025_01
    PARTITION OF fact_ais_track_partitioned
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
-- Add monthly partitions; drop old partitions instead of DELETE
```

**Benefit:** Query planner skips irrelevant partitions. Archival = `DROP PARTITION` (instant, no vacuum needed).

#### 3.1.2 Add a Retention Policy

```sql
-- Run nightly via pg_cron or a cron job
DELETE FROM fact_ais_track
WHERE base_datetime < NOW() - INTERVAL '90 days';

VACUUM ANALYZE fact_ais_track;
```

**Benefit:** Bounds table growth; keeps index sizes manageable.

#### 3.1.3 Shared Engine / Connection Pool in Streamlit

```python
# In streamlit_app.py — create engine once, not per query
@st.cache_resource
def get_engine():
    from sqlalchemy import create_engine
    return create_engine(
        POSTGRES_URL,
        pool_size=5,
        max_overflow=2,
        pool_pre_ping=True,
    )
```

**Benefit:** Eliminates connection-per-query overhead; reduces PostgreSQL `max_connections` pressure.

---

### 3.2 High Impact — Kafka

#### 3.2.1 Reduce Log Retention

```properties
# In docker-compose.yml Kafka environment
KAFKA_LOG_RETENTION_HOURS: 24        # was: 168 (7 days)
KAFKA_LOG_SEGMENT_BYTES: 52428800    # 50 MB segments for faster cleanup
```

**Benefit:** Disk reclaimed daily; 24 h is sufficient for replay/reprocessing needs.

#### 3.2.2 Enable Log Compaction for Vessel State Topic

```properties
KAFKA_LOG_CLEANUP_POLICY: compact    # for dim_vessel topic
```

**Benefit:** Retains only the latest record per MMSI key, not full history.

---

### 3.3 Medium Impact — Spark

#### 3.3.1 Replace Spark with Faust or pandas for Low-Volume Processing

For AIS volumes under ~50K msgs/hour, a pure-Python stream processor is sufficient and uses 90% less memory.

```python
# Candidate replacement: Faust (Kafka Streams for Python)
import faust

app = faust.App("ais-processor", broker="kafka://localhost:9092")
ais_topic = app.topic("ais_raw", value_type=bytes)

@app.agent(ais_topic)
async def process(stream):
    async for msg in stream:
        record = json.loads(msg)
        await upsert_vessel(record)
```

**Benefit:** ~100 MB vs ~1 GB RAM; sub-second startup vs 15–30 s for Spark.
**Risk:** Lose Spark MLlib integration if ML inference is embedded in the pipeline.

#### 3.3.2 If Keeping Spark — Use Structured Streaming with `trigger(processingTime)`

```python
query = (
    df.writeStream
    .trigger(processingTime="30 seconds")   # was: continuous
    .outputMode("update")
    .foreachBatch(write_batch)
    .start()
)
```

**Benefit:** Batches micro-batches; reduces Spark scheduling overhead by ~60%.

---

### 3.4 Low Impact — Dashboard

#### 3.4.1 Cache Heavy Queries

```python
@st.cache_data(ttl=300)   # 5-minute cache
def load_traffic_density():
    return read_pg("SELECT lat_bin, lon_bin, COUNT(*) ...")
```

**Benefit:** Heatmap and alert queries run once per 5 min, not on every user interaction.

#### 3.4.2 Reduce MLflow Artifact Storage

```bash
# Prune runs older than 30 days with no registered model
mlflow gc --backend-store-uri postgresql://... --older-than 30d
```

---

## 4. Cost Estimation (Self-Hosted / Cloud VM)

| Scenario | vCPU | RAM | Storage | Est. Monthly (AWS/GCP) |
|----------|------|-----|---------|------------------------|
| Current (all services) | 4 | 8 GB | 50 GB SSD | ~$70–90 |
| After Spark → Faust | 2 | 4 GB | 30 GB SSD | ~$30–40 |
| After partitioning + retention | 2 | 4 GB | 20 GB SSD | ~$25–35 |
| Kubernetes (3-node) | 6 | 12 GB | 100 GB SSD | ~$150–200 |

*Estimates based on general-purpose instances (e3-standard-2 / t3.medium class). Actual costs vary by region and reserved pricing.*

---

## 5. Quick Wins Checklist

- [ ] Add `@st.cache_resource` to engine creation in `streamlit_app.py`
- [ ] Set Kafka `LOG_RETENTION_HOURS=24` in `docker-compose.yml`
- [ ] Add monthly `DELETE` cron for `fact_ais_track`
- [ ] Add `@st.cache_data(ttl=300)` to density and alert queries
- [ ] Switch Spark trigger to `processingTime="30 seconds"`
- [ ] Separate MLflow stack (already done — maintain this separation)
- [ ] Add `pg_stat_statements` to PostgreSQL to identify slow/expensive queries

---

*Last updated: 2026-05-24*
