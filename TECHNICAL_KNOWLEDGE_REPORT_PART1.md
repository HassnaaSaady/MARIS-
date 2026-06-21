# Maritime Navigation AI System — Complete Technical Knowledge Report
### For Internship Defense Presentation | Prepared for Junior Interns
### Authors: Big Data Internship Team | Date: June 2026

---

# PART 1: EXECUTIVE SUMMARY, ARCHITECTURE & DATASET

---

## SECTION 1 — EXECUTIVE SUMMARY (BUSINESS LANGUAGE)

### 1.1 What Problem Does This System Solve?

Every day, tens of thousands of vessels — cargo ships, tankers, passenger ferries, fishing boats — navigate the world's oceans and coastal waters. These vessels broadcast their position, speed, and identity every few seconds using the **Automatic Identification System (AIS)**, a mandatory radio transponder system. The raw data is public, real-time, and enormous (millions of records per day globally).

The problem: **nobody is intelligently analyzing this data in real time.** Maritime authorities, port operators, shipping companies, and coast guards currently rely on simple visual tracking maps. They can see where vessels are — but they cannot:

- Predict where a vessel will be in 5, 10, or 15 minutes
- Detect when a vessel is behaving abnormally (suspicious route, dangerous speed change, wrong area)
- Forecast which sea lanes and port approaches will become congested in the next hour
- Automatically alert operators about collision risks before they become emergencies

### 1.2 Who Benefits?

| Stakeholder | Business Benefit |
|---|---|
| Port Authorities | Predict congestion, schedule berths, reduce waiting time (fuel cost savings) |
| Coast Guard / Maritime Safety | Detect anomalous vessels early — potential smuggling, distress, collision risk |
| Shipping Companies | Know predicted arrival, optimize fuel by avoiding congested routes |
| Insurance Companies | Identify high-risk vessel behaviors for premium adjustment |
| Environmental Agencies | Monitor vessels in protected zones, detect illegal stops |
| Traffic Management Centers | Anticipate bottlenecks, issue dynamic routing advisories |

### 1.3 What Does the System Do — In One Paragraph?

Our system ingests a continuous stream of AIS vessel positions via Apache Kafka, processes and cleans the data through a three-layer Medallion architecture (Bronze → Silver → Gold) using Apache Spark, stores clean analytical data in PostgreSQL using a star schema, and serves three machine learning models in real time: an **Isolation Forest** that detects vessels behaving anomalously, an **XGBoost regressor** that predicts vessel positions 5–15 minutes ahead, and a **Random Forest classifier** that forecasts port-area congestion levels. Results are visualized on a live map dashboard (Streamlit) and exposed through a REST API (FastAPI), with an optional React frontend.

### 1.4 Why Is AI Necessary Here?

Rule-based systems fail because:
- "Anomalous behavior" depends on context: a speed of 25 knots is normal in open ocean, dangerous near a port
- Vessel patterns are high-dimensional: 6+ features interact non-linearly
- The scale is too large for human inspection: 5M+ records per week
- Congestion is temporal and spatial simultaneously — simple thresholds miss this

AI solves this by learning the distribution of normal behavior from historical data and flagging deviations, predicting future states from learned kinematic patterns, and classifying congestion from multi-feature temporal patterns.

### 1.5 Key Numbers for the Presentation

| Metric | Value |
|---|---|
| Raw records processed | ~36M (7 days Bronze) |
| Clean records (Silver) | ~5M (after deduplication + cleaning) |
| Unique vessels tracked | 1,000+ |
| Position prediction MAE (5-min) | 4.24 nautical miles |
| Congestion classification accuracy | 90.4% |
| Anomaly contamination rate | 0.2% of all records |
| Real-time latency | < 5 seconds (Kafka → dashboard) |
| Data pipeline layers | 3 (Bronze / Silver / Gold) |

---

## SECTION 2 — END-TO-END ARCHITECTURE

### 2.1 Architecture Overview Diagram (Text)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                 │
│  MarineCadastre AIS Parquet Files (7 days, May 1-7 2025, US waters) │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   KAFKA PRODUCER                                    │
│  Reads LIVE split parquet → publishes to topic: ais_raw             │
│  Rate: ~500 msg/sec (STREAM_DELAY_SECONDS=0.002)                    │
└──────────┬──────────────────────────────────────────────────────────┘
           │  topic: ais_raw
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    APACHE KAFKA                                     │
│  Topic: ais_raw (vessel positions)                                  │
│  Topic: ais_alerts (collision alerts, anomalies)                    │
│  Retention: 7 days | Partitions: by MMSI key                        │
└──────────┬────────────────────────────────────────────────────────┬─┘
           │                                                        │
           ▼                                                        ▼
┌──────────────────────────┐                           ┌─────────────────────┐
│  SPARK STREAMING          │                           │   LIVE SCORER       │
│  CONSUMER                 │                           │   (live_scorer.py)  │
│  Kafka → Bronze Delta     │                           │   Kafka → ML →      │
│  Trigger: 10 sec          │                           │   PostgreSQL        │
└──────────┬───────────────┘                           └──────────┬──────────┘
           │                                                      │
           ▼                                                      │
┌─────────────────────────────────────────────────────┐           │
│              BRONZE LAYER (Delta Lake)              │           │
│  /delta/bronze/ais                                  │           │
│  Raw enriched data, append-only                     │           │
│  ~36M rows, 7 days                                  │           │
│  Added: is_stopped, in_us_port_zone, risk_level     │           │
└──────────┬──────────────────────────────────────────┘           │
           │  (Spark batch job)                                   │
           ▼                                                      │
┌─────────────────────────────────────────────────────┐           │
│              SILVER LAYER (Delta Lake)              │           │
│  /delta/silver/ais_clean                            │           │
│  Deduplicated, cleaned, feature-engineered          │           │
│  ~5M rows, partitioned by year/month/day            │           │
│  Added: prev_lat, prev_lon, distance_nm,            │           │
│         sog_change, heading_change, time_delta_sec  │           │
└──────────┬──────────────────────────────────────────┘           │
           │  (Spark batch job)                                   │
           ▼                                                      │
┌─────────────────────────────────────────────────────┐           │
│              GOLD LAYER (Delta Lake)                │           │
│  /delta/gold/vessel_latest                          │           │
│  /delta/gold/traffic_density                        │           │
│  /delta/gold/daily_stats                            │           │
│  Aggregated, analytics-ready                        │           │
└──────────┬──────────────────────────────────────────┘           │
           │  (JDBC write)                                        │
           ▼                                                      │
┌─────────────────────────────────────────────────────────────────┴──┐
│                     POSTGRESQL DATABASE                             │
│  Star Schema: dim_vessel, dim_status                               │
│  Facts: fact_vessel_latest, fact_ais_track,                        │
│         fact_traffic_density, fact_daily_stats, fact_alerts         │
└──────────┬────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   SERVING LAYER                                     │
│  FastAPI REST API (port 8000) — /api/vessels, /api/grid,            │
│                                 /api/alerts, /api/predict           │
│  Streamlit Dashboard (port 8501) — Live map, heatmap, alerts        │
│  React Frontend (port 3000) — Alternative web UI                    │
└─────────────────────────────────────────────────────────────────────┘

     ┌─────────────────────────────────────────────┐
     │              ML LAYER (Training)            │
     │  Isolation Forest — Anomaly Detection       │
     │  XGBoost — Position Prediction (5/10/15 min)│
     │  Random Forest — Congestion Classification  │
     │  Trained on Silver TRAIN split              │
     │  Models persisted to /models/*.pkl          │
     └─────────────────────────────────────────────┘
```

### 2.2 Component Deep Dive

#### 2.2.1 Apache Kafka

**What it is:** Kafka is a distributed event streaming platform. It acts as a durable, high-throughput message bus between data producers and consumers.

**Why it exists in this project:** AIS data is inherently a stream — vessels broadcast positions continuously. Kafka decouples the producer (parquet reader simulating live AIS feed) from multiple consumers (Spark Streaming, Live Scorer, Dashboard), allowing each to process at its own pace without data loss.

**Why Kafka specifically (not RabbitMQ, Redis Streams, etc.):**
- **Durability:** Messages persisted to disk for 7 days — consumers can replay from any offset
- **Fan-out:** Multiple consumer groups can read the same topic independently (Spark Streaming + Live Scorer simultaneously)
- **Throughput:** Handles 500+ msg/sec easily (Kafka is rated for millions/sec)
- **Ordering guarantee:** Per partition, messages arrive in order — critical for temporal AIS data
- **Exactly-once semantics:** When configured, prevents duplicate processing

**Alternatives considered:**
| Alternative | Why Not Chosen |
|---|---|
| RabbitMQ | No message replay; once consumed, message gone |
| Redis Pub/Sub | No persistence; if consumer disconnects, messages lost |
| AWS Kinesis | Cloud-only; adds vendor lock-in for a local demo project |
| Direct file polling | No real-time; introduces latency and complexity |

**Kafka configuration in this project:**
- Topic: `ais_raw` — all vessel position messages
- Topic: `ais_alerts` — collision and anomaly alerts from live_scorer
- Message key: MMSI — ensures all messages from the same vessel go to the same partition (preserves per-vessel ordering)
- Consumer group: `spark-bronze-consumer`, `live-scorer-group`

**Possible interview questions:**
- Q: Why use Kafka key=MMSI? A: Guarantees that all records for one vessel land in the same partition, preserving chronological order needed for lag/delta feature computation.
- Q: What happens if Kafka goes down? A: Spark Streaming uses checkpoints to resume from last committed offset; Live Scorer falls back to empty scoring; Streamlit switches to demo mode.

---

#### 2.2.2 Apache Spark

**What it is:** Distributed computation engine. Processes data in parallel across a cluster.

**Why Spark (not pandas, Dask, Flink):**
- **Scale:** 36M Bronze rows cannot fit in single-machine RAM comfortably; Spark distributes across 2 workers × 5GB each
- **Delta Lake native:** Spark has first-class Delta Lake support — ACID transactions, schema evolution, time travel
- **Structured Streaming:** Same API for batch (bronze/silver/gold jobs) and streaming (Kafka consumer)
- **SQL + DataFrame API:** Enables complex window functions needed for lag features

**Spark cluster configuration:**
```
Master:    2GB RAM, 2 CPU
Worker 1:  5GB RAM, 4 CPU
Worker 2:  5GB RAM, 4 CPU
Driver:    2GB RAM (per job)
Executor:  4GB RAM, 4 cores
Shuffle partitions: 8 (tuned for dataset size — default 200 would be wasteful)
```

**Why shuffle_partitions=8 (not default 200)?**
- Default 200 partitions works for TB-scale; our 5M Silver rows are ~2GB
- Too many partitions → task overhead dominates → slower
- 8 partitions → each ~250MB → optimal for 2 workers × 4 cores

**Alternatives:**
| Alternative | Tradeoff |
|---|---|
| Pandas | Single machine, no Delta Lake, no streaming |
| Dask | No Delta Lake; weaker SQL; smaller ecosystem |
| Apache Flink | Better streaming semantics, but no Delta Lake; steeper learning curve |
| DuckDB | Excellent for analytics but single-node |

---

#### 2.2.3 Delta Lake

**What it is:** Open-source storage format layered on Parquet that adds ACID transactions, schema enforcement, and time travel.

**Why Delta Lake (not plain Parquet, Iceberg, Hudi):**
- **ACID transactions:** Bronze appends and Silver overwrites are atomic — no partial writes seen by readers
- **Schema enforcement:** Rejects records with wrong column types at write time
- **Time travel:** Can query Bronze layer "as of" any past timestamp — useful for debugging and reprocessing
- **Z-ordering:** Can co-locate related data on disk for faster queries (not heavily used here but available)
- **Native Spark integration:** `spark.read.format("delta")` works out of the box

**Alternatives:**
| Format | Tradeoff |
|---|---|
| Plain Parquet | No ACID; concurrent writers corrupt data; no schema enforcement |
| Apache Iceberg | Also excellent; better multi-engine support; but less native Spark integration for Delta-specific features |
| Apache Hudi | Better for record-level upserts; more complex setup |
| ORC | Older; less ecosystem support; no ACID |

---

#### 2.2.4 PostgreSQL (Star Schema)

**What it is:** Relational database serving as the analytical serving layer for dashboards and API.

**Why PostgreSQL (not MySQL, MongoDB, Cassandra, Snowflake):**
- **Star schema support:** Relational model is ideal for dimension + fact table joins
- **JSONB support:** Useful for storing variable anomaly metadata
- **Full-text search:** Vessel name search
- **Mature:** Reliable, well-documented, battle-tested
- **Free and open-source:** No licensing cost for demo
- **SQLAlchemy support:** FastAPI ORM integration trivial

**Star schema design rationale:**
- **dim_vessel:** Vessel metadata changes rarely → dimension table (slowly changing)
- **dim_status:** Navigation status codes → lookup dimension (static)
- **fact_vessel_latest:** One row per vessel, updated on every position → optimized for live map queries (SELECT all, ORDER BY risk, LIMIT 500)
- **fact_ais_track:** Full history, append-only → optimized for time-range queries per vessel
- **fact_traffic_density:** Pre-aggregated grid → eliminates expensive real-time GROUP BY queries on the map heatmap
- **fact_daily_stats:** Pre-computed daily rollups → instant dashboard KPI cards
- **fact_alerts:** Alert log → sorted by detected_at for real-time alert feed

**Alternatives:**
| Alternative | Tradeoff |
|---|---|
| MongoDB | Good for schemaless; but joins are expensive; no ACID on multi-document |
| Cassandra | Best for time-series at massive scale; but SQL aggregations are very limited |
| ClickHouse | Excellent analytical performance; but overkill for this scale; heavier setup |
| Snowflake | Cloud-managed; great for BI; but adds cost and cloud dependency |

---

#### 2.2.5 FastAPI

**What it is:** Modern Python web framework for building REST APIs. Used to expose vessel data, predictions, and alerts to the frontend.

**Why FastAPI:**
- **Automatic OpenAPI docs:** `/docs` endpoint works out of the box — important for demo
- **Async support:** Non-blocking I/O for database queries
- **Pydantic validation:** Request parameter validation built in
- **Python ecosystem:** Shares ML dependencies (sklearn, xgboost) with the rest of the pipeline

**Key endpoints:**
- `GET /api/vessels` — paginated list of vessels with filters (type, risk, speed, anomaly)
- `GET /api/vessels/{mmsi}` — full vessel detail including ML scores
- `GET /api/grid` — traffic density heatmap data
- `GET /api/alerts` — maritime alerts with severity
- `GET /api/predict/{mmsi}` — position predictions for next 5/10/15 min

---

#### 2.2.6 Streamlit Dashboard

**What it is:** Python library for building data apps with minimal frontend code. Used as the primary visualization layer.

**Features implemented:**
- Live vessel map (Folium/Leaflet) with risk-colored markers
- Traffic heatmap overlay
- Historical replay with time slider
- Anomaly detection feed
- Maritime alerts panel
- Auto-refresh every 30 seconds

**Why Streamlit vs. Dash vs. Grafana:**
- Streamlit requires minimal JavaScript — faster development for Python data engineers
- Grafana is better for time-series metrics (DevOps) but poor for geographic maps
- Dash (Plotly) offers more control but more boilerplate

---

## SECTION 3 — DATASET DEEP DIVE

### 3.1 AIS — What Is It?

**AIS (Automatic Identification System)** is a VHF radio transponder system mandated by the International Maritime Organization (IMO) for vessels over 300 gross tons in international waters and all passenger ships. It broadcasts the vessel's identity, position, speed, and course every 2–10 seconds (underway) or every 3 minutes (at anchor).

The data we use comes from **MarineCadastre** (US government), which aggregates AIS broadcasts from US coastal receivers. Our dataset covers **May 1–7, 2025**, US waters (primarily East Coast, Gulf of Mexico, West Coast, Hawaii).

### 3.2 Every AIS Field — Definitions, Units, Importance

#### MMSI (Maritime Mobile Service Identity)

**Definition:** A 9-digit unique number that identifies a vessel or maritime mobile radio station globally.

**Format:** Always 9 digits. Country prefix: digits 1-3 identify the country (e.g., 338 = USA, 232 = UK).

**Why important:** MMSI is the primary vessel identifier — the JOIN key in our star schema and the Kafka message key. Without MMSI, we cannot track individual vessels over time.

**Project usage:** Primary key in `dim_vessel` and `fact_vessel_latest`; partition key in Kafka; `GROUP BY` key in all window functions.

**Data quality note:** Some vessels spoof or share MMSIs (intentionally or due to transponder misconfiguration). Our deduplication step (mmsi, base_datetime) handles duplicate broadcasts but cannot detect MMSI spoofing.

**Real-world maritime meaning:** If a vessel's MMSI is 366000001, the "366" prefix indicates USA. Military vessels and certain law enforcement often do not broadcast AIS (intentional blank spots on the map).

**Example:** MMSI = 368000001 → US vessel, 9 digits confirmed.

**MMSI from the data:** We see scientific notation in raw files (3.68E+08) fixed by our `fix_mmsi()` function → "368000000".

---

#### Latitude (lat)

**Definition:** North-South geographic position in decimal degrees. Range: -90.0 (South Pole) to +90.0 (North Pole). Positive = North.

**Units:** Decimal degrees (WGS84 coordinate system)

**Why important:** Half of the vessel's position. Without lat, no geographic mapping is possible.

**AIS broadcast format:** Originally in integer format (degrees × 10^4 for minutes). MarineCadastre provides pre-converted decimal degrees.

**Project usage:** Map rendering, haversine distance calculations, grid binning (lat_bin = ROUND(lat, 1) for density heatmap), geofence checks (US port zones defined as lat ranges).

**Data quality issue:** AIS devices occasionally broadcast lat=0.0, lon=0.0 (the "Null Island" — intersection of equator and prime meridian in the Atlantic, where nothing is). Our `is_valid_position()` function rejects lat=0, lon=0.

**Valid range check:** Bronze layer enforces lat ∈ [-90, 90]. Any record outside is dropped.

---

#### Longitude (lon)

**Definition:** East-West geographic position in decimal degrees. Range: -180.0 (West) to +180.0 (East). Negative = West (Americas).

**Units:** Decimal degrees (WGS84)

**Project usage:** Same as latitude — paired with lat for all geographic operations.

**Important note for haversine:** Longitude differences are NOT equal to East-West distances everywhere. A 1° longitude difference equals ~111 km at the equator but ~0 km at the poles. The haversine formula accounts for this correctly via `cos(lat)` in the longitude term.

**US waters longitude range:** Approximately -170° (Hawaii) to -65° (East Coast Maine). All negative (West).

---

#### SOG — Speed Over Ground

**Definition:** The actual speed of the vessel relative to the Earth's surface (not relative to water).

**Units:** **Knots** (nautical miles per hour). 1 knot = 1.852 km/h = 1.151 mph.

**AIS encoding:** Broadcast as integer tenths of a knot (e.g., 102 = 10.2 knots). Special value: **102.3 knots = "not available"** (AIS sentinel value). MarineCadastre may pre-decode this.

**Real-world maritime meaning:**
- 0.0 knots: vessel is stationary
- < 0.5 knots: operationally "stopped" (at anchor, moored, drifting)
- 0.5–5 knots: very slow (maneuvering, port approach, congested waterway)
- 5–15 knots: typical cargo/tanker cruising speed
- 15–25 knots: fast cargo or container ship
- > 25 knots: high-speed ferry or naval vessel
- > 30 knots: anomalous for most commercial vessels

**Our project thresholds:**
```
is_stopped:   SOG < 0.5 knots
is_slow:      SOG < 2.0 knots
is_speeding:  SOG > 30.0 knots (anomaly rule trigger)
Sentinel:     SOG = 102.3 → set to NULL
Valid range:  SOG ∈ [0, 60] (or NULL)
```

**Project usage in ML:**
- Anomaly detection feature
- Position predictor feature
- Congestion predictor feature (avg_sog per grid cell)
- sog_change = current_sog - prev_sog (sudden acceleration/deceleration detector)

**Why 60 knots as upper bound?** The fastest commercial vessels (hydrofoils, high-speed ferries) rarely exceed 50 knots. 60 is a conservative upper limit. Anything above is GPS error or data corruption.

---

#### COG — Course Over Ground

**Definition:** The actual direction the vessel is moving relative to true North, measured clockwise.

**Units:** Degrees, range 0.0° to 359.9°. 0° = North, 90° = East, 180° = South, 270° = West.

**Difference from Heading:** COG is where the vessel is actually going (resultant of engine direction + current + wind). Heading is where the bow is pointing. In crosscurrent, these differ significantly.

**Real-world importance:** COG is more useful for predicting future position than Heading because it accounts for environmental effects.

**AIS encoding:** Broadcast as tenths of a degree integer. Special value: 360° = "not available."

**Our project usage:**
- Feature in all three ML models
- Position predictor: COG combined with SOG determines the trajectory vector
- Grid binning: determines which grid cell the vessel is heading toward

**Data quality:** Some vessels broadcast 360.0 (sentinel) which we treat as invalid. Our Silver job enforces COG ∈ [0, 360) (exclusive upper bound).

---

#### Heading (True Heading)

**Definition:** The direction the vessel's bow is pointing relative to True North, measured clockwise.

**Units:** Degrees, range 0° to 359°. 

**Difference from COG:** A vessel heading North (0°) may have COG of 020° if a strong eastward current is pushing it right.

**AIS sentinel value:** **511 = "not available."** Very common — many vessels don't broadcast heading or their gyrocompass is not connected to the AIS transponder.

**Our treatment:** Silver job sets Heading = NULL when Heading = 511.

**Real-world importance:** Heading + SOG together define the vessel's kinematic state. Sharp changes in heading while underway indicate turns.

**Project usage:**
- Anomaly detection: heading_change feature
- Position predictor: heading and COG both included as features
- Dead reckoning fallback: `delta_lat = (sog * cos(heading_rad)) / 60`

**heading_change calculation (wrap-around aware):**
```python
raw_diff = current_heading - prev_heading
if raw_diff > 180:  raw_diff -= 360   # e.g., 350→10° = 10-350 = -340 → raw = -340+360 = 20°
if raw_diff < -180: raw_diff += 360
heading_change = abs(raw_diff)        # magnitude of turn
```
This correctly computes 350°→10° as a 20° turn (not 340°).

---

#### base_datetime (Timestamp)

**Definition:** UTC timestamp of the AIS position broadcast.

**Format:** ISO 8601 string → cast to TIMESTAMP_NTZ (no timezone, assumed UTC) in Bronze.

**Why important:** Enables time-ordered processing, lag feature computation, traffic density bucketing, daily statistics.

**Project usage:**
- Primary sort key within each vessel track (ORDER BY mmsi, base_datetime)
- Used in window functions for lag/delta features
- Bucketed to hours for density heatmap: `DATE_TRUNC('hour', base_datetime)`
- Extracted into year, month, day for partition pruning
- Extracted into hour, day_of_week for congestion model features
- time_delta_sec = Unix timestamp of current - Unix timestamp of previous record

**Data quality issue:** Timestamps may be out of order if AIS receivers receive delayed broadcasts. Our producer streams in chronological order to simulate this correctly.

---

#### vessel_name

**Definition:** The vessel's registered name, broadcast by the AIS transponder.

**Data quality:** Highly unreliable. Vessels can change names; names may contain special characters, truncations (AIS limits to 20 characters), or be left blank.

**Project treatment:** Forward-filled per MMSI using `FIRST_VALUE(vessel_name IGNORE NULLS) OVER (PARTITION BY mmsi)`. This propagates the first known name to all subsequent records for that vessel.

---

#### vessel_type (type code)

**Definition:** Integer code indicating the vessel category (from ITU-R M.1371 standard).

**Common codes and their labels (as implemented in Silver job):**
```
70-79:  Cargo (container ships, bulk carriers)
80-89:  Tanker (oil, chemical, LNG)
60-69:  Passenger (cruise ships, ferries)
30:     Fishing
36:     Sailing
37:     Pleasure craft
50-57:  Special (pilot boats, tugs, fire boats)
```

**Project usage:** Display label in dashboard; filter parameter in API; potential feature for future ML refinement (tankers behave differently from fishing vessels).

---

#### IMO Number

**Definition:** International Maritime Organization permanent vessel identification number (7 digits). Unlike MMSI, the IMO number stays with the hull even if ownership or flag changes.

**Why both MMSI and IMO?** MMSI is operational (can be reassigned), IMO is permanent. Together they allow cross-referencing vessel registry databases.

**Project usage:** Stored in dim_vessel; not used as ML feature (too sparse — many vessels don't broadcast IMO).

---

#### Navigation Status

**Definition:** Integer code (0-15) indicating the vessel's current navigational state.

**Key codes:**
```
0: Under way using engine
1: At anchor
2: Not under command
5: Moored
7: Engaged in fishing
15: Not defined / default
```

**Project treatment:** Mapped to human-readable labels in Silver job; drives `dim_status` lookup table; influences risk classification.

---

### 3.3 Data Split Strategy

The dataset spans 14 days of AIS data:

| Split | Days | Purpose | Approx % |
|---|---|---|---|
| TRAIN | Days 1-8 | Model training | 57% |
| VALIDATION | Days 9-10 | Hyperparameter tuning, early stopping | 14% |
| TEST | Days 11-12 | Final model evaluation (held-out) | 14% |
| LIVE | Days 13-14 | Simulated live stream via Kafka | 15% |

**Why time-based split (not random)?**

This is critical and commonly misunderstood. If we randomly shuffle rows and split 80/20, we commit **temporal leakage**: training data can contain future information (rows from Day 10 in training while Day 7 rows are in test). For time-series data, models must be trained on past and tested on strictly future data.

**Why this matters for our models:**
- Congestion predictor: trained to predict NEXT hour from CURRENT hour → must not see future hours in training
- Position predictor: predicts next N-minute position → future positions must not appear in training
- Anomaly detector: learns "normal" from historical patterns → distribution shift between training and live periods must be realistic

**Interview question:** "Why didn't you shuffle the data before splitting?"
**Answer:** "Because AIS data is temporal. Random shuffling would allow the model to see future vessel positions during training, artificially inflating performance metrics. A time-based split ensures our test metrics reflect true generalization to unseen future data — the real production scenario."

---

## SECTION 4 — DATA ENGINEERING PIPELINE IN DEPTH

### 4.1 Bronze Layer — Raw Ingestion

**File:** `src/processing/bronze_job.py`
**Input:** Parquet files from `/app/data/parquet` (raw MarineCadastre AIS data)
**Output:** Delta Lake table at `/delta/bronze/ais`
**Volume:** ~36M rows

#### What the Bronze layer does:

**Step 1: Schema Normalization**
```python
# Column aliases handled (from schema_utils.py):
"LAT" → "lat", "BaseDateTime" → "base_datetime", "SOG" → "sog", ...
# fix_mmsi(): 3.68E+08 → "368000000"
```

**Why schema normalization first?** MarineCadastre files have inconsistent column naming across years. Canonicalizing column names ensures the rest of the pipeline works regardless of source file vintage.

**Step 2: Basic Validation**
Rows dropped if:
- MMSI is NULL
- lat or lon is NULL
- lat ∉ [-90, 90] or lon ∉ [-180, 180]
- SOG ∉ [0, 60]

**Why not fix these values?** The Bronze layer philosophy: preserve raw data, only drop clearly impossible records. Null MMSI means we cannot track the vessel. Coordinates outside physical bounds are hardware errors. SOG > 60 is physically impossible for any ship.

**Step 3: Enrichment Columns Added**
```python
is_stopped     = SOG < 0.5
is_slow        = SOG < 2.0
is_speeding    = SOG > 30.0
in_us_port_zone = (within Houston Ship Channel) OR (within NY Harbor) OR ...
risk_level     = "HIGH" if stopped in port zone else "MEDIUM" if slow else "LOW"
ingestion_time = current_timestamp()
```

**Why add these in Bronze (not Silver)?**
- These enrichments require no cross-record context (no window functions needed)
- Pre-computing them here reduces Silver job computation
- Bronze enrichments serve as quality signals: if `in_us_port_zone=True` but vessel is unknown, it may warrant manual review

**Design philosophy of Bronze:** Append-only. Never delete or update. If reprocessing is needed, append a corrected version and manage versioning via Delta time travel. This creates an auditable raw history.

**Alternative approach:** Some architectures call Bronze "raw" without any enrichment. Our approach is a hybrid: we call it Bronze but add lightweight enrichments. This is valid — the key principle is append-only and no complex transformations.

#### US Port Zones (Hardcoded):
```python
PORT_ZONES = {
    "Houston Ship Channel": (29.50, 29.85, -95.30, -94.80),
    "New York Harbor":      (40.50, 40.75, -74.30, -73.90),
    "Port LA/Long Beach":   (33.70, 33.85, -118.35, -118.10),
    "Port of New Orleans":  (29.00, 30.00, -90.50, -89.50),
    "Port of Miami":        (25.70, 25.85, -80.20, -80.05),
}
```

**Why hardcoded (not a database lookup)?** For a demo system, hardcoding is acceptable and avoids adding a configuration database dependency. In production, these zones would be loaded from a geospatial database (PostGIS) and updated dynamically. Hardcoding is a known limitation we should acknowledge.

---

### 4.2 Silver Layer — Cleaning and Feature Engineering

**File:** `src/processing/silver_job.py`
**Input:** Bronze Delta layer
**Output:** Delta Lake table at `/delta/silver/ais_clean`
**Volume:** ~5M rows (from ~36M Bronze rows — ~86% reduction)

The Silver layer is the most technically complex component of the pipeline. Every transformation has a specific business and technical reason.

#### Step 1: Deduplication

**What:** `DROP DUPLICATES ON (mmsi, base_datetime)`

**Why:** AIS signals are broadcast over VHF radio and received by multiple shore stations simultaneously. The same position broadcast may be received and logged twice. Without deduplication, lag features would compute Δt=0 for duplicate pairs, producing infinite implied speed.

**Example:**
```
BEFORE deduplication:
MMSI=368000001, datetime=2025-05-01 12:00:00, lat=29.71, lon=-95.12 (received by station A)
MMSI=368000001, datetime=2025-05-01 12:00:00, lat=29.71, lon=-95.12 (received by station B)

AFTER:
MMSI=368000001, datetime=2025-05-01 12:00:00, lat=29.71, lon=-95.12 (one row)
```

---

#### Step 2: AIS Sentinel Value Handling

**What:**
```python
sog = NULLIF(sog, 102.3)      # 102.3 is the official "not available" code
heading = NULLIF(heading, 511) # 511 is the official "not available" code
```

**Why 102.3 and 511?** These are defined by the ITU-R M.1371 AIS standard as the reserved values indicating "information not available." If we treat 102.3 as a real speed, the Isolation Forest would flag every vessel with unavailable speed data as an anomaly — creating thousands of false positives.

**Impact if we missed this:** Training data would contain speed values of 102.3 knots. The scaler would normalize based on a range of [0, 102.3]. At inference time, real vessel speeds of 15–20 knots would appear near-zero after scaling, distorting anomaly detection entirely.

---

#### Step 3: Physical Range Validation

**What:**
```python
WHERE sog IS NULL OR (sog >= 0 AND sog <= 60)
  AND (cog >= 0 AND cog < 360)
  AND (heading IS NULL OR (heading >= 0 AND heading < 360))
  AND lat BETWEEN -90 AND 90
  AND lon BETWEEN -180 AND 180
```

**Why:** Even after sentinel handling, some AIS transponders broadcast physically impossible values due to hardware malfunction. COG=400° is impossible. A range filter is the last line of defense.

**Risk if thresholds too strict:** Legitimate vessel at COG=359.8° passes; at COG=359.999° it also passes — no practical issue at these boundaries.

---

#### Step 4: Lag Feature Computation (Window Functions)

**What:** For each vessel (MMSI), order records by base_datetime, then compute previous values:

```sql
WINDOW w = PARTITION BY mmsi ORDER BY base_datetime
prev_lat    = LAG(lat, 1) OVER w
prev_lon    = LAG(lon, 1) OVER w
prev_sog    = LAG(sog, 1) OVER w
prev_heading = LAG(heading, 1) OVER w
prev_time   = LAG(base_datetime, 1) OVER w
```

**Why partition by MMSI only (not MMSI + date)?** 

This is a critical design decision. If we partition by (MMSI, date), the first record of each day for each vessel has no previous record (NULL prev values) even though the vessel's last record from yesterday is in the dataset. This means every vessel would generate a "first ping" at midnight every day, wasting ~50% of the training data (each vessel generates ~2 first pings per day × 8 training days).

By partitioning only by MMSI, the window continues across day boundaries. A vessel's position at 00:00:01 correctly sees its last position at 23:59:45 the previous night as its previous record.

**Trade-off:** The Spark executor must hold all records for one MMSI in memory simultaneously. For vessels with 100,000+ records, this increases executor memory pressure. Mitigation: ensure executor memory = 4GB, and MMSI tracks rarely exceed 10,000 records.

---

#### Step 5: Time Delta Calculation

**What:**
```python
time_delta_sec = UNIX_TIMESTAMP(base_datetime) - UNIX_TIMESTAMP(prev_time)
```

**Why Unix timestamps (not DATEDIFF)?** DATEDIFF in Spark returns integer seconds. Unix timestamp subtraction preserves sub-second precision and works correctly across day/month/year boundaries.

**Typical values:** AIS broadcasts every 2–10 seconds when underway; every 3 minutes at anchor. Expected time_delta_sec: 2–180 seconds.

**First pings have time_delta_sec = NULL** (no previous record). These are dropped from Silver because:
1. They cannot have distance_nm computed
2. sog_change and heading_change are undefined
3. Including them would add NULL features that require imputation (introducing risk)

---

#### Step 6: Haversine Distance

**Formula:**
```
a = sin²(Δlat/2) + cos(lat₁) × cos(lat₂) × sin²(Δlon/2)
c = 2 × atan2(√a, √(1-a))
distance_nm = R × c      where R = 3440.065 nm (Earth radius in nautical miles)
```

**Why haversine (not Euclidean)?**

Euclidean distance treats lat/lon as a flat grid. This fails because:
1. 1° latitude ≈ 60 nm everywhere, but 1° longitude ≈ 60 nm only at the equator, ≈ 30 nm at 60°N
2. Euclidean error grows with latitude — at Houston (29°N), longitude error is ~13% vs equator

Haversine computes the great-circle distance on a sphere, which is accurate to within 0.5% for maritime distances (< 500 nm).

**Example:**
```
Position A: lat=29.71, lon=-95.12 (Houston Ship Channel)
Position B: lat=29.73, lon=-95.10
Euclidean: sqrt((0.02)² + (0.02)²) × 60 = 1.70 nm (WRONG — ignores cos(lat))
Haversine: 1.54 nm (CORRECT)
```

**Why not Vincenty (more accurate)?** Vincenty is accurate to millimeters but computationally expensive. Haversine is 100× faster and accurate to <1% for our use case — an acceptable trade-off.

---

#### Step 7: Teleport / GPS Glitch Detection

**What:**
```python
implied_speed_kn = (distance_nm / time_delta_sec) × 3600
DROP WHERE implied_speed_kn > 100
```

**Why 100 knots as threshold?** The fastest ships in the world (military hydrofoils) reach ~60 knots. 100 knots provides a generous buffer for legitimate fast vessels while eliminating clear GPS teleports (where a vessel "jumps" 50 nm in 2 seconds due to GPS error or timestamp corruption).

**Example of teleport:**
```
Record 1: lat=29.71, lon=-95.12, datetime=12:00:00
Record 2: lat=40.70, lon=-74.00, datetime=12:00:05  (New York Harbor!)
distance_nm ≈ 1,200 nm, time_delta_sec = 5
implied_speed = (1200/5) × 3600 = 864,000 knots → DROP
```

**Risk if threshold too strict (e.g., 50 knots):** We might drop legitimate fast vessels (military, racing sailboats). Risk if too lenient (e.g., 200 knots): GPS glitches pass through and train the position predictor on impossible trajectories.

---

#### Step 8: Speed Change and Heading Change Features

**sog_change:**
```python
sog_change = sog - prev_sog
```
Positive = acceleration, Negative = deceleration.

**Anomaly rule:**
```python
if sog_change < -5.0:  # sudden deceleration of >5 knots
    anomaly_type = "SUDDEN_STOP"
    anomaly_score = 0.8
```

**heading_change (wrap-around aware):**
```python
raw = heading - prev_heading
if raw > 180:  raw -= 360
if raw < -180: raw += 360
heading_change = abs(raw)
```

**Why wrap-around aware?** Naively: 350°→10° = 10-350 = -340°. But the actual turn was only 20° (right). The wrap-around correction: -340 + 360 = 20°. Without this, we would compute a 340° turn where only a 20° turn occurred — making every compass-crossing maneuver appear as a dramatic anomaly.

---

#### Step 9: Grid Binning

**Fine grid (Silver):**
```python
lat_bin = ROUND(lat, 2)   # 0.01° precision ≈ 0.6 nm
lon_bin = ROUND(lon, 2)
```

**Density grid (Gold/congestion):**
```python
lat_bin = ROUND(lat, 1)   # 0.1° precision ≈ 6 nm
lon_bin = ROUND(lon, 1)
```

**Congestion grid (model training):**
```python
lat_bin = ROUND(lat, 0) × 0.5   # 0.5° precision ≈ 30 nm
lon_bin = ROUND(lon, 0) × 0.5
```

**Why different resolutions?** Fine grid (0.01°) is for individual vessel proximity. Density heatmap (0.1°) balances visual resolution vs. cell sparsity. Congestion model (0.5°) uses coarser cells to ensure each cell has enough vessel observations for statistical stability (≥5 observations per hour).

---

### 4.3 Gold Layer — Aggregations and PostgreSQL Sync

**File:** `src/processing/gold_job.py`
**Input:** Silver Delta layer
**Output:** Gold Delta tables + PostgreSQL tables

#### Gold Table 1: fact_vessel_latest

**What:** One row per unique MMSI, showing the vessel's most recent position and ML scores.

```sql
SELECT mmsi, vessel_name, vessel_type, lat, lon, sog, cog, heading,
       risk_level, is_anomaly, anomaly_score, anomaly_type,
       predicted_lat, predicted_lon, base_datetime
FROM silver
QUALIFY ROW_NUMBER() OVER (PARTITION BY mmsi ORDER BY base_datetime DESC) = 1
```

**Why this table?** The live map needs to display 1,000+ vessels at once. A SELECT from the full 5M-row Silver table with a GROUP BY mmsi and MAX(datetime) would take 30+ seconds. Pre-computing one row per vessel reduces the map query to milliseconds.

**Used by:** Streamlit live map, FastAPI `/api/vessels`, collision detector.

---

#### Gold Table 2: fact_traffic_density

**What:** Count of unique vessels per 0.1° grid cell per 1-hour bucket.

```sql
SELECT
  ROUND(lat, 1) as lat_bin,
  ROUND(lon, 1) as lon_bin,
  DATE_TRUNC('hour', base_datetime) as hour_bucket,
  COUNT(DISTINCT mmsi) as vessel_count,
  AVG(sog) as avg_sog,
  SUM(CASE WHEN sog < 0.5 THEN 1 ELSE 0 END) as stopped_count,
  CASE WHEN COUNT(DISTINCT mmsi) >= 15 THEN 'HIGH'
       WHEN COUNT(DISTINCT mmsi) >= 5  THEN 'MEDIUM'
       ELSE 'LOW' END as congestion_level
FROM silver
GROUP BY 1, 2, 3
```

**Why pre-aggregate?** Heatmap rendering requires this computation for potentially 100,000+ grid-hour cells. Running it live would make the dashboard unusable.

---

#### Gold Table 3: fact_daily_stats

**What:** One row per (date, data_split), aggregating: total vessels, total records, avg SOG, max SOG, high-risk count, stopped vessel count, anomaly count.

**Used by:** KPI summary cards on dashboard. "Today: 1,247 vessels tracked, 12 anomalies detected."

---

#### Gold Table 4: dim_vessel

**What:** Slowly changing dimension — vessel metadata, one row per MMSI.

**Why "slowly changing"?** Vessel metadata (name, type, dimensions) rarely changes. We store first_seen and last_seen timestamps and total_records count. If we had SCD Type 2, we'd track history of name changes; for this project, Type 1 (overwrite) is sufficient.

---

## SECTION 5 — DATA QUALITY AND CLEANING DECISION TABLE

| Cleaning Rule | What It Removes | Business Reason | Risk if Removed | Risk if Too Strict |
|---|---|---|---|---|
| Null MMSI drop | Records with no vessel ID | Cannot track vessel; MMSI is the primary key | Join failures in star schema; ghost tracks | None — no legitimate use for null MMSI |
| lat/lon range check | Physically impossible coordinates | GPS hardware error | False positions on wrong continent | Very unlikely; standard range |
| SOG sentinel (102.3→NULL) | AIS "not available" speed codes | Prevents false anomaly on speed | 102.3 kn treated as ultra-fast → false anomaly spike | None — 102.3 is not a real speed |
| Heading sentinel (511→NULL) | AIS "not available" heading codes | Same as above | 511° heading creates circular error in features | None — 511 is not a real heading |
| Deduplication (mmsi, datetime) | Duplicate AIS broadcasts | Multi-receiver capture creates exact duplicates | Inflated vessel counts; Δt=0 → infinite implied speed | Removing genuinely different readings (unlikely) |
| Teleport filter (>100 kn) | GPS glitch jumps | Haversine distance > 200 nm in 2 sec | Trains position predictor on impossible teleports | Removes legitimate fast vessels if threshold too low |
| First ping drop | Records with no prior position | Cannot compute lag features; NULL features break models | NULL features require imputation or crash models | None — one ping per vessel per session |
| COG range (0-360) | Invalid course values | COG=400° is physically meaningless | Distorts position prediction; corrupts heading_change | Legitimate 359.9° safe, border: extremely unlikely |
| SOG range (0-60) | Extreme speed values | No commercial vessel exceeds 60 kn | Trains anomaly detector on impossible speeds | 50-60 kn range has legitimate military/racing vessels |

---

*End of Part 1. Continue reading TECHNICAL_KNOWLEDGE_REPORT_PART2.md for ML, Evaluation, Design Decisions, Limitations, 120 Q&A, and Presentation Guide.*
