# Maritime Navigation AI System — Architecture Report
### Internship Defense Presentation | Big Data Internship Team | June 2026

---

# PART 1: SYSTEM OVERVIEW, ARCHITECTURE PATTERN & DATA PIPELINE

---

## SECTION 1 — ARCHITECTURE OVERVIEW

### 1.1 Architecture Pattern: Lambda + Medallion

The system combines two industry-standard architectural patterns:

| Pattern | Purpose |
|---|---|
| **Lambda Architecture** | Runs two parallel data processing paths (speed layer + batch layer) that write to the same serving layer, balancing real-time responsiveness with batch accuracy |
| **Medallion Architecture** | Organizes the batch layer into three quality tiers (Bronze → Silver → Gold), so each tier has a single well-defined responsibility |

Together they answer a fundamental challenge in data engineering: **how do you serve real-time ML predictions while also maintaining high-quality historical aggregations from the same data stream?**

The speed layer answers in milliseconds. The batch layer corrects and enriches overnight.

---

### 1.2 High-Level System Flow

```
AIS Data (CSV / Parquet files — 14-day dataset, ~36 million rows)
        │
        ▼
  kafka_producer.py   ──► Kafka topic: ais_raw
        │
        ├──────────────────────────────────────┐
        │                                      │
   SPEED LAYER                          BATCH LAYER
   live_scorer.py                       Spark 3.4 + Airflow
   (real-time, 5s batches)              (daily @midnight)
        │                                      │
        │                                      │
        ▼                                      ▼
        └──────────────┬───────────────────────┘
                       │
              PostgreSQL Star Schema
              (fact_vessel_latest, fact_alerts,
               fact_traffic_density, fact_ais_track,
               fact_daily_stats, dim_vessel)
                       │
              FastAPI  :8000
                       │
          ┌────────────┴──────────────┐
          │                           │
   React Frontend               Streamlit Dashboard
      :3000                        :8501
```

---

### 1.3 Geographic Scope

The system is scoped to **US coastal waters**:

- Latitude: 24°N – 49°N
- Longitude: 125°W – 66°W

**Five high-priority port zones monitored:**

| Port Zone | Lat Range | Lon Range |
|---|---|---|
| Houston Ship Channel | 29.50–29.85°N | 95.30–94.80°W |
| New York Harbor | 40.50–40.75°N | 74.30–73.90°W |
| Port of Los Angeles / Long Beach | 33.70–33.85°N | 118.35–118.10°W |
| Port of New Orleans | 29.00–30.00°N | 90.50–89.50°W |
| Port of Miami | 25.70–25.85°N | 80.20–80.05°W |

Dashboard maps default to center `[38.5, −75.5]` at zoom level 6 (US East Coast view).

---

## SECTION 2 — INGESTION LAYER

### 2.1 AIS Producer

**File:** `src/producer/kafka_producer.py`  
**Container:** `ais-producer`

The producer simulates a live AIS radio feed by replaying the `live` data split:

1. Reads rows from Parquet files in `data/parquet/`
2. Calls `normalize_ais_record()` to standardize field names and types
3. Serializes to JSON
4. Publishes each message to Kafka topic `ais_raw`
5. Loops continuously to simulate a perpetual live feed

**Throughput:** ~0.002 seconds per message (500 messages/second sustained)

**Key fields in each message:**
```json
{
  "mmsi": "123456789",
  "vessel_name": "ATLANTIC STAR",
  "lat": 40.6892,
  "lon": -74.0445,
  "sog": 12.3,
  "cog": 185.0,
  "heading": 184.0,
  "base_datetime": "2021-01-15T14:22:05",
  "vessel_type": "70",
  "status": "0",
  "data_split": "live"
}
```

---

## SECTION 3 — MESSAGE BUS (KAFKA)

### 3.1 Confluent Kafka 7.4

Three containers handle the message bus:

| Container | Role | Port |
|---|---|---|
| `zookeeper` | Broker coordination, leader election, metadata management | internal :2181 |
| `kafka` | Message broker, hosts topic `ais_raw` | :9092 (internal) / :29092 (host) |
| `kafka-ui` | GUI topic browser, consumer lag monitor | :8083 |

### 3.2 Why Kafka?

- **Decoupling:** Producer runs independently of all consumers. Speed layer and batch layer both read `ais_raw` without affecting each other.
- **Durability:** Kafka persists messages to disk. If the live_scorer crashes, it resumes from its last committed offset — no data loss.
- **Replay:** Any consumer can replay from offset 0 to reprocess historical messages.
- **Fan-out:** Multiple consumers (spark-stream, live_scorer) read the same topic simultaneously.

### 3.3 Topic Configuration

- Topic name: `ais_raw`
- Partitions: 1 (single broker, local development)
- Retention: default (168 hours / 7 days)

---

## SECTION 4 — SPEED LAYER (REAL-TIME PIPELINE)

### 4.1 live_scorer.py

**File:** `src/ml/live_scorer.py` (251 lines)  
**Container:** `live-scorer`  
**Technology:** Python 3.10, scikit-learn 1.5.1, XGBoost 2.0.3

The live scorer is the heart of the real-time path. It runs as a standalone Kafka consumer process, completely separate from the API, so ML scoring never blocks HTTP responses.

**Batch parameters:**
- `batch_size = 200` messages
- `window = 5 seconds` (processes whatever arrives within 5s, up to 200 records)

### 4.2 Processing Steps Per Batch

```
1. Poll Kafka for up to 200 messages within 5s window
        │
        ▼
2. fillna(0) on all numeric columns
        │
        ▼
3. score_anomaly(df)
   IsolationForest → anomaly_score, is_anomaly, anomaly_type
        │
        ▼
4. predict_position(lat, lon, sog, cog, 5min)
   XGBoost → predicted_lat, predicted_lon
        │
        ▼
5. classify_risk(sog, lat, lon)
   Rule-based → risk_level (HIGH / MEDIUM / LOW)
        │
        ▼
6. build_anomaly_alerts()
   if is_anomaly → create alert record
        │
        ▼
7. build_collision_alerts()
   haversine pairwise scan on all vessels in batch
   if distance < threshold → create collision alert
        │
        ▼
8. Single PostgreSQL transaction:
   ├── UPSERT fact_vessel_latest (per MMSI)
   ├── INSERT fact_alerts (anomaly + collision)
   ├── UPSERT fact_traffic_density (data_split='live')
   └── Bulk INSERT fact_ais_track (data_split='live')
```

### 4.3 PostgreSQL Writes (Speed Layer)

All four writes share one transaction per batch — either all succeed or all roll back:

| Table | Write mode | Key |
|---|---|---|
| `fact_vessel_latest` | UPSERT (ON CONFLICT mmsi) | One row per vessel, always current |
| `fact_alerts` | INSERT | Append-only alert log |
| `fact_traffic_density` | UPSERT (ON CONFLICT lat_bin, lon_bin, hour_bucket) | Accumulates vessel counts per grid cell per hour |
| `fact_ais_track` | Bulk INSERT | Full position history |

---

## SECTION 5 — BATCH LAYER (SPARK + AIRFLOW)

### 5.1 Spark Cluster

**Three containers:**

| Container | Role | Port |
|---|---|---|
| `spark-master` | Cluster scheduler, job submission | :9090 (Web UI) / :7077 (RPC) |
| `spark-worker-1` | Executor node | :9091 (Web UI) |
| `spark-worker-2` | Executor node | :9093 (Web UI) |

**Worker specification:** 4 cores / 4 GB RAM each → 8 total cores, 8 GB total memory  
**Sized for:** Intel i7-14700 / 32 GB host machine

**Technology:** Apache Spark 3.4 + Delta Core 2.4.0

### 5.2 Spark Streaming Consumer

**File:** `src/processing/spark_streaming_consumer.py`  
**Job:** Kafka `ais_raw` → Bronze Delta Lake (continuous append)

- Uses Spark Structured Streaming with `processingTime 10s` trigger
- Checkpoints at `/delta/checkpoints/bronze` → exactly-once Bronze writes
- Runs as `--master spark://spark-master:7077`

### 5.3 Bronze Job

**File:** `src/processing/bronze_job.py` (132 lines)  
**Job:** Parquet files → Bronze Delta Lake (batch append)

Transformations applied at Bronze:
- Non-null filter: mmsi, lat, lon must be present
- Range validation: lat ∈ [−90, 90], lon ∈ [−180, 180], sog ∈ [0, 60]
- Enrichment columns added:
  - `event_time` (parsed timestamp)
  - `ingestion_time` (processing timestamp)
  - `is_stopped` (sog < 0.5)
  - `is_slow` (sog < 3.0)
  - `is_speeding` (sog > 30.0)
  - `in_us_port_zone` (bounding-box check against 5 port zones)
  - `risk_level` (rule-based: HIGH / MEDIUM / LOW)

### 5.4 Silver Job

**File:** `src/processing/silver_job.py` (203 lines)  
**Job:** Bronze Delta → Silver Delta (clean, feature-engineered)

Transformations applied at Silver:

| Transformation | Detail |
|---|---|
| AIS sentinel nulling | `sog = 102.3` → null (AIS "not available"); `heading = 511` → null |
| Kinematic bounds filter | Drops physically impossible values |
| Teleport / GPS-glitch filter | Drops records where implied speed > 100 knots between pings |
| Deduplication | On `(mmsi, base_datetime)` — removes AIS re-broadcasts |
| Vessel info forward-fill | Fills null vessel_name, vessel_type forward per MMSI within partition |
| Label enrichment | Human-readable `vessel_type_label` and `status_label` |
| ML feature engineering | `sog_change`, `heading_change`, `distance_nm` (haversine), `time_delta_sec`, `lat_bin_fine`, `lon_bin_fine` |
| First-ping drop | Removes the first record per MMSI (no delta features available) |
| Partitioning | Written partitioned by `year / month / day` |

**Row cap:** Up to 5,000,000 rows per run (memory constraint for 8 GB worker pool)

### 5.5 Gold Job

**File:** `src/processing/gold_job.py` (242 lines)  
**Job:** Silver Delta → Gold Delta + PostgreSQL (serving layer)

Three Gold tables produced:

| Gold Table | Delta Path | PostgreSQL Target | Logic |
|---|---|---|---|
| `vessel_latest` | `/delta/gold/vessel_latest` | `fact_vessel_latest` (via JDBC) | `row_number()` window over mmsi ordered by `base_datetime DESC` → keep rank=1 only |
| `traffic_density` | `/delta/gold/traffic_density` | `fact_traffic_density` (via JDBC) | Group by (lat_bin, lon_bin, hour_bucket, data_split); agg vessel_count, unique_vessels, avg_sog, stopped_count; congestion_level = HIGH if ≥ 15, MEDIUM if ≥ 5, else LOW |
| `daily_stats` | `/delta/gold/daily_stats` | `fact_daily_stats` (via JDBC) | Group by (stat_date, data_split); agg total_vessels, total_records, avg_sog, max_sog, high_risk_count, stopped_vessels |

> **Critical ownership rule:** gold_job uses Spark JDBC `mode=overwrite` which issues a TRUNCATE before insert. It must **never** write `fact_vessel_latest` to PostgreSQL — doing so would destroy all live ML-scored vessel positions accumulated since midnight by live_scorer.

### 5.6 Airflow DAG

**Schedule:** `@daily` (runs at midnight)

```
bronze_job
    │
    ▼
silver_job
    │
    ▼
gold_job
    │
    ├── train_anomaly    ─┐
    ├── train_predictor  ─┼──► evaluate
    └── train_congestion ─┘
```

The three training jobs run in parallel after gold_job completes, then a single evaluate task validates the newly trained models.

---

## SECTION 6 — DELTA LAKE MEDALLION LAYERS

### 6.1 Layer Summary

| Layer | Path | PostgreSQL target | Live row count | Purpose |
|---|---|---|---|---|
| **Raw** | `data/parquet/` | — | ~36M rows across 14 days | Source of truth; columnar storage; `data_split` partitions into train/valid/test/live |
| **Bronze** | `/delta/bronze/ais` | — | Streaming (grows continuously) | Raw + quality-filtered + enriched; immutable; written by both spark-stream and bronze_job |
| **Silver** | `/delta/silver/ais_clean` | — | Up to 5M rows (batch cap) | Clean, deduplicated, ML-ready; partitioned year/month/day |
| **Gold vessel_latest** | `/delta/gold/vessel_latest` | `fact_vessel_latest` | 1 row per MMSI | Latest ping per vessel |
| **Gold traffic_density** | `/delta/gold/traffic_density` | `fact_traffic_density` | Aggregated grid cells | 0.1° spatial grid, hourly buckets |
| **Gold daily_stats** | `/delta/gold/daily_stats` | `fact_daily_stats` | 1 row per day | Fleet-wide daily summary |
| **Gold anomalies** | `/delta/gold/anomalies` | — | Reserved | Batch anomaly aggregations |

### 6.2 Why Delta Lake Over Plain Parquet?

| Feature | Plain Parquet | Delta Lake |
|---|---|---|
| ACID transactions | No | Yes |
| Schema enforcement | No | Yes (rejects wrong types at write) |
| Time travel | No | Yes (`versionAsOf`) |
| Upsert support | No | Yes (`MERGE INTO`) |
| Concurrent write safety | No | Yes |
| Streaming + batch unified | No | Yes |

Delta Lake's `mergeSchema=true` also allows new AIS fields to be added without breaking existing readers — important for evolving data sources.

---

*Continues in Part 2: ML Models, PostgreSQL Star Schema, API, Frontends*
