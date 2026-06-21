# Maritime Navigation AI System — Architecture Report
### Internship Defense Presentation | Big Data Internship Team | June 2026

---

# PART 2: ML MODELS, POSTGRESQL STAR SCHEMA, API & FRONTENDS

---

## SECTION 7 — MACHINE LEARNING LAYER

### 7.1 Overview

Three ML models run in the live_scorer process, scoring every incoming AIS message in real time. All models are trained on the Silver Delta Lake `TRAIN` split and serialized to `.pkl` files in the `models/` volume, which is mounted into the live-scorer container.

| Model | File(s) | Algorithm | Task |
|---|---|---|---|
| Anomaly Detector | `isolation_forest_v2.pkl` + `scaler_anomaly_v2.pkl` | IsolationForest | Detects vessels with abnormal kinematics |
| Position Predictor | `xgb_lat/lon_{5,10,15}min.pkl` (6 files) | XGBoost Regressor | Predicts vessel position 5, 10, 15 min ahead |
| Congestion Classifier | `congestion_rf.pkl` + `congestion_encoder.pkl` | Random Forest | Classifies 0.1° grid cells as LOW/MEDIUM/HIGH congestion |

---

### 7.2 Anomaly Detection — IsolationForest

**Files:** `isolation_forest_v2.pkl`, `scaler_anomaly_v2.pkl`  
**Training:** `src/ml/train_anomaly.py`  
**Parameters:** `n_estimators=200`, `contamination=0.002`

**How it works:**

IsolationForest detects anomalies by randomly partitioning feature space and measuring how quickly a point is isolated. Points that are isolated quickly (few splits needed) are anomalies — they are different from the dense normal region.

`contamination=0.002` means 0.2% of training data is expected to be anomalous. This is deliberately low to minimize false positives — a coast guard alert system with 5% false positive rate is unusable.

**Input features (6):**

| Feature | What it captures |
|---|---|
| `sog` | Speed over ground (knots) |
| `cog` | Course over ground (degrees) |
| `heading` | True compass heading (degrees) |
| `sog_change` | Delta SOG between consecutive pings — sudden acceleration/deceleration |
| `heading_change` | Delta heading — sudden turns |
| `distance_nm` | Haversine distance since last ping (nautical miles) |

**Output → alert severity mapping:**

| anomaly_score | is_anomaly | Severity |
|---|---|---|
| ≥ 0.7 | True | HIGH |
| ≥ 0.5 | True | MEDIUM |
| < 0.5 | False | (no alert) |

**Why v2?** The v1 model used default `contamination=0.1`, generating too many false positives. v2 tightened contamination to 0.002 and added `scaler_anomaly_v2.pkl` (StandardScaler) to normalize features before training. Both files must be loaded together — the scaler must be applied to new data using the exact same fit as training.

---

### 7.3 Position Predictor — XGBoost

**Files:** `xgb_lat_5min.pkl`, `xgb_lon_5min.pkl`, `xgb_lat_10min.pkl`, `xgb_lon_10min.pkl`, `xgb_lat_15min.pkl`, `xgb_lon_15min.pkl`  
**Training:** `src/ml/train_predictor.py`

**Why 6 separate models?**

Each prediction horizon (5, 10, 15 minutes) requires a different model because the relationship between current kinematics and future position changes with time horizon. A 5-minute prediction can assume roughly constant heading; a 15-minute prediction must account for more heading variation. Training separate models lets each one learn the appropriate temporal uncertainty.

Latitude and longitude are modeled separately because their dynamics differ — vessels don't move in perfect compass bearings and the lat/lon projections have different curvature effects.

**Input features:** `sog`, `cog`, `heading`, `lat`, `lon`, `sog_change`, `heading_change`, `distance_nm`, `time_delta_sec`, `lat_bin_fine`, `lon_bin_fine`

**Output:** `predicted_lat`, `predicted_lon` (stored in `fact_vessel_latest`)

**Use in collision risk:** The predicted positions at 5/10/15 min are used to compute the Closest Point of Approach (CPA) between vessel pairs. If two vessels' predicted paths converge within a threshold distance, a collision alert is generated.

---

### 7.4 Congestion Classifier — Random Forest

**Files:** `congestion_rf.pkl`, `congestion_encoder.pkl`, `congestion_features.pkl`  
**Training:** `src/ml/train_congestion.py`

**Input:** Gold `traffic_density` aggregations — each row is a 0.1° grid cell at a specific hour

**Output classes:** `LOW`, `MEDIUM`, `HIGH`

**Backup:** `models/congestion_rf.BACKUP.pkl` — a snapshot saved before any retraining. The weather module evaluation runs against this backup to ensure the production model is never overwritten unintentionally.

**Rule-based baseline (used in gold_job):**

| vessel_count in cell | congestion_level |
|---|---|
| ≥ 15 | HIGH |
| ≥ 5 | MEDIUM |
| < 5 | LOW |

The ML model learns non-linear patterns beyond simple vessel counts — e.g., 8 slow vessels in a narrow channel may be HIGH congestion while 8 fast vessels in open water are MEDIUM.

---

### 7.5 MLOps — Experiment Tracking

**Location:** `mlops/`

All three models have MLflow-instrumented training scripts:

| Script | What it logs |
|---|---|
| `mlops/experiments/train_anomaly_mlflow.py` | contamination, n_estimators, anomaly rate, precision/recall |
| `mlops/experiments/train_predictor_mlflow.py` | horizon, RMSE lat, RMSE lon, feature importance |
| `mlops/experiments/train_congestion_mlflow.py` | n_estimators, accuracy, F1 per class |
| `mlops/model_registry/evaluate_models.py` | Cross-model comparison, champion selection |
| `mlops/experiments/feature_importance.py` | SHAP or permutation importance export |

**MLflow backend:** PostgreSQL (optional; uses file-based `mlruns/` by default in local setup)

---

## SECTION 8 — POSTGRESQL STAR SCHEMA

### 8.1 Schema Overview

PostgreSQL 15 is the operational serving store. All dashboards and the API read from it. The schema follows a **star model**: fact tables hold measurements and events; dimension tables hold descriptive attributes.

**Configuration:**
- Port: `:5432`
- `shared_buffers = 512 MB`
- `effective_cache_size = 2 GB`
- Image: `postgres:15-alpine`

**No foreign key constraints are declared** — the ORM uses logical MMSI references for join performance at dashboard polling rates (3-second React polling of 20,000+ rows).

---

### 8.2 Table Definitions

#### fact_vessel_latest

One row per MMSI — the current live position and ML score for every tracked vessel.

| Column | Type | Notes |
|---|---|---|
| `mmsi` | VARCHAR(20) PK | Vessel identifier |
| `vessel_name` | VARCHAR(100) | Nullable |
| `vessel_type` | VARCHAR(50) | Nullable |
| `lat` | FLOAT NOT NULL | Current latitude |
| `lon` | FLOAT NOT NULL | Current longitude |
| `sog` | FLOAT | Speed over ground (knots) |
| `cog` | FLOAT | Course over ground (degrees) |
| `heading` | FLOAT | True heading (degrees) |
| `risk_level` | VARCHAR(10) | HIGH / MEDIUM / LOW |
| `is_anomaly` | BOOLEAN | IsolationForest output |
| `anomaly_score` | FLOAT | 0.0 – 1.0 |
| `anomaly_type` | VARCHAR(100) | Category label |
| `predicted_lat` | FLOAT | XGBoost 5-min prediction |
| `predicted_lon` | FLOAT | XGBoost 5-min prediction |
| `base_datetime` | DATETIME | AIS message timestamp |
| `updated_at` | DATETIME | Last UPSERT time |
| `data_split` | VARCHAR(20) | 'live', 'train', etc. |

**Writer:** live_scorer.py only (UPSERT per MMSI)  
**Live row count:** 21,659

---

#### fact_alerts

Append-only log of all anomaly and collision alerts generated by live_scorer.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `alert_type` | VARCHAR(50) NOT NULL | 'anomaly', 'collision_risk', etc. |
| `severity` | VARCHAR(20) NOT NULL | HIGH / MEDIUM / LOW |
| `mmsi_1` | VARCHAR(20) | Primary vessel |
| `mmsi_2` | VARCHAR(20) | Second vessel (collision alerts only) |
| `vessel_name_1` | VARCHAR(100) | |
| `vessel_name_2` | VARCHAR(100) | Collision pair only |
| `lat` | FLOAT | Alert location |
| `lon` | FLOAT | Alert location |
| `description` | TEXT | Human-readable alert text |
| `extra_data` | JSON | Additional context |
| `anomaly_score` | FLOAT | IsolationForest score |
| `distance_nm` | FLOAT | CPA distance (collision alerts) |
| `is_resolved` | BOOLEAN | Manual resolution flag |
| `created_at` | DATETIME | Alert generation time |
| `resolved_at` | DATETIME | Resolution time |
| `data_split` | VARCHAR(20) | 'live' |

**Writer:** live_scorer.py only (INSERT)  
**Live row count:** 103,332

---

#### fact_ais_track

Full position history — every AIS ping ever processed.

| Column | Type | Notes |
|---|---|---|
| `id` | BIGINT PK | Auto-increment |
| `mmsi` | VARCHAR(20) NOT NULL | Indexed |
| `vessel_name` | VARCHAR(100) | |
| `vessel_type` | VARCHAR(50) | |
| `lat` | FLOAT NOT NULL | WGS84 decimal degrees |
| `lon` | FLOAT NOT NULL | WGS84 decimal degrees |
| `lat_bin` | FLOAT | Rounded to 0.1° for spatial bins |
| `lon_bin` | FLOAT | Rounded to 0.1° for spatial bins |
| `sog` | FLOAT | Knots |
| `cog` | FLOAT | Degrees |
| `heading` | FLOAT | Degrees |
| `status` | VARCHAR(20) | AIS navigation status code |
| `risk_level` | VARCHAR(10) | |
| `is_anomaly` | BOOLEAN | |
| `anomaly_score` | FLOAT | |
| `anomaly_type` | VARCHAR(100) | |
| `base_datetime` | DATETIME NOT NULL | AIS timestamp |
| `ingested_at` | DATETIME | Processing time |
| `data_split` | VARCHAR(20) NOT NULL | train/valid/test/live |

**Indexes:** `ix_track_mmsi_time (mmsi, base_datetime)`, `ix_track_lat_bin (lat_bin, lon_bin)`  
**Writers:** `populate_fact_ais_track.py` (batch) + live_scorer.py (`data_split='live'`)

---

#### fact_traffic_density

Aggregated vessel density grid — one row per (lat_bin, lon_bin, hour_bucket) combination.

| Column | Type | Notes |
|---|---|---|
| `lat_bin` | FLOAT PK | 0.1° grid cell latitude |
| `lon_bin` | FLOAT PK | 0.1° grid cell longitude |
| `hour_bucket` | DATETIME PK | Rounded to nearest hour |
| `vessel_count` | BIGINT NOT NULL | Total vessel pings in cell |
| `unique_vessels` | BIGINT | Distinct MMSIs |
| `avg_sog` | FLOAT | Average speed in cell |
| `stopped_count` | BIGINT | Vessels with sog < 0.5 |
| `congestion_level` | TEXT | HIGH / MEDIUM / LOW |
| `data_split` | TEXT | 'live' (speed layer) or 'train' etc. (batch) |

**Writers:** live_scorer.py (UPSERT, `data_split='live'`) + gold_job (historical splits via JDBC)

---

#### fact_daily_stats

One summary row per day per data_split.

| Column | Type | Notes |
|---|---|---|
| `stat_date` | DATE PK NOT NULL | |
| `data_split` | VARCHAR(20) | |
| `total_vessels` | INTEGER | Distinct MMSIs seen |
| `total_records` | BIGINT | Total AIS pings |
| `avg_sog` | FLOAT | Fleet average speed |
| `max_sog` | FLOAT | Fastest vessel observed |
| `high_risk_count` | INTEGER | Vessels flagged HIGH risk |
| `stopped_vessels` | INTEGER | Vessels with sog < 0.5 |
| `anomaly_count` | INTEGER | Anomalies detected |

**Writer:** gold_job only

---

#### dim_vessel

Vessel metadata dimension — one row per MMSI.

| Column | Type | Notes |
|---|---|---|
| `mmsi` | VARCHAR(20) PK | |
| `vessel_name` | VARCHAR(100) | |
| `imo` | VARCHAR(20) | IMO registration number |
| `call_sign` | VARCHAR(20) | Radio call sign |
| `vessel_type` | VARCHAR(50) | Numeric code |
| `vessel_type_label` | VARCHAR(100) | Human-readable label |
| `length` | FLOAT | Meters |
| `width` | FLOAT | Meters |
| `draft` | FLOAT | Meters |
| `cargo` | VARCHAR(50) | Cargo type code |
| `transceiver_class` | VARCHAR(10) | Class A or B |
| `data_split` | VARCHAR(20) | |
| `first_seen` | DATETIME | |
| `last_seen` | DATETIME | |
| `total_records` | BIGINT | Total pings from this vessel |
| `created_at` | DATETIME | |
| `updated_at` | DATETIME | |

**Writer:** gold_job only

---

#### dim_status

Static reference table mapping AIS navigation status codes to descriptions.

| Column | Type |
|---|---|
| `status_id` | INTEGER PK |
| `status_code` | VARCHAR(10) UNIQUE |
| `status_label` | VARCHAR(100) |
| `is_underway` | BOOLEAN |
| `is_stopped` | BOOLEAN |
| `is_anchored` | BOOLEAN |
| `is_moored` | BOOLEAN |
| `is_aground` | BOOLEAN |

---

#### dim_weather

Weather dimension table — added by the weather integration module.

| Column | Notes |
|---|---|
| `lat_bin` | 0.5° grid cell |
| `lon_bin` | 0.5° grid cell |
| `date` | Date of observation |
| `wind_speed_10m` | Wind speed at 10m height (m/s) |
| `wave_height` | Significant wave height (m) |
| `weather_severity` | Derived severity score |

**Writer:** `src/weather/load_dim_weather.py` (UPSERT — idempotent)

---

### 8.3 Table Ownership Summary

| Table | Speed Layer (live_scorer) | Batch Layer (gold_job) |
|---|---|---|
| `fact_vessel_latest` | **UPSERT per MMSI** | Delta Lake only — never PostgreSQL |
| `fact_alerts` | **INSERT** | Never |
| `fact_traffic_density` | UPSERT (`data_split='live'`) | JDBC (historical splits) |
| `fact_ais_track` | Bulk INSERT (`data_split='live'`) | populate_fact_ais_track.py (batch) |
| `fact_daily_stats` | Never | **JDBC overwrite** |
| `dim_vessel` | Never | **JDBC overwrite** |

---

## SECTION 9 — API LAYER (FastAPI)

### 9.1 FastAPI Service

**File:** `api/main.py`  
**Container:** `maritime-api`  
**Port:** `:8000`  
**Technology:** FastAPI 0.111.0, Uvicorn 0.30.1, SQLAlchemy 2.0.31, Python 3.11

**15 REST endpoints** covering all dashboard features:

| Endpoint category | Tables queried |
|---|---|
| `/api/vessels` | `fact_vessel_latest` |
| `/api/anomalies` | `fact_alerts` LEFT JOIN `fact_vessel_latest` (lat/lon fallback) |
| `/api/collisions` | `fact_alerts` (alert_type collision) |
| `/api/density` | `fact_traffic_density` (fallback to `fact_ais_track`) |
| `/api/replay` | `fact_ais_track` |
| `/api/stats` | `fact_daily_stats` |
| `/api/vessels/{mmsi}` | `dim_vessel` + `fact_vessel_latest` |
| `/snowflake/*` | Snowflake warehouse (via `api/routers/snowflake_router.py`) |

### 9.2 Key Implementation Details

- **Async framework** — FastAPI + Uvicorn handle concurrent React polling requests without blocking
- **SQLAlchemy ORM** — all tables mapped as Python classes in `api/models/database.py`; no raw SQL strings
- **Density fallback** — if `fact_traffic_density` has no live rows, the density endpoint falls back to aggregating `fact_ais_track` directly
- **OpenAPI docs** — auto-generated at `/docs` and `/redoc`
- **Vessel type lookup** — `api/utils/vessel_types.py` provides human-readable labels for AIS vessel type codes

---

## SECTION 10 — FRONTEND LAYER

### 10.1 React 18 Dashboard (`:3000`)

**File:** `frontend/src/App.jsx` (2560 lines)  
**Technology:** React 18, react-leaflet 4.x, Recharts, Node 18-slim

Polls FastAPI every **3 seconds** for live data. Six views:

| View | Technology | Data source |
|---|---|---|
| Vessel Map | Leaflet / react-leaflet, CircleMarker | `fact_vessel_latest` |
| Anomaly Detection | CircleMarkers colored by severity (HIGH=red, MEDIUM=orange, LOW=green) | `fact_alerts` |
| Collision Risk | Paired markers + Polyline connecting vessel pair | `fact_alerts` (collision type) |
| Traffic Heatmap | 0.1° grid cells shaded by congestion_level | `fact_traffic_density` |
| Historical Replay | Time-scrub slider over track | `fact_ais_track` |
| Analytics | Time-series charts, fleet statistics | `fact_daily_stats` |

**Performance:** CircleMarker pooling handles 20,000+ vessel markers efficiently without DOM thrashing.

**Map default:** center `[38.5, −75.5]`, zoom 6 (US East Coast).

---

### 10.2 Streamlit Dashboards (`:8501`)

Three Python-native analytics dashboards:

| File | Dashboard | Purpose |
|---|---|---|
| `src/dashboard/streamlit_app.py` | Operational Dashboard | Live vessel map, replay, heatmap, anomaly list |
| `src/dashboard/ml_monitoring.py` | ML Monitoring Dashboard | Model performance metrics, alert rate trends, anomaly score distribution |
| `src/dashboard/snowflake_analytics.py` | Snowflake Analytics Dashboard | Cloud warehouse analytics, cross-fleet queries |

Streamlit reads directly from PostgreSQL (via SQLAlchemy) and from Delta Lake (via Delta reader), giving it both live and historical views.

---

## SECTION 11 — END-TO-END MESSAGE FLOW

### 11.1 One AIS Message — Full Journey

```
1. AIS Producer reads row from live Parquet split
   normalize_ais_record() → JSON message
        │
        ▼
2. Kafka topic: ais_raw
        │
   ┌────┴──────────────────────┐
   │                           │
   ▼                           ▼
3a. Spark Structured Streaming    3b. live_scorer.py (batch ≤ 200)
    poll() batch                      fillna(0) on numerics
    parse JSON schema                 ↓
    enrich: speed flags,              score_anomaly()
    port zone, risk_level             → anomaly_score, is_anomaly
    ↓                                 predict_position()
    Bronze Delta (append)             → predicted_lat, predicted_lon
    10s trigger                       classify_risk()
                                      build alerts
                                      ↓
                                      PostgreSQL transaction:
                                      UPSERT fact_vessel_latest
                                      INSERT fact_alerts
                                      UPSERT fact_traffic_density
                                      Bulk INSERT fact_ais_track
        │
        ▼
4. React frontend polls FastAPI every 3s:
   GET /api/vessels
   → SELECT * FROM fact_vessel_latest WHERE data_split='live'
   → JSON array → render CircleMarkers on Leaflet map

   GET /api/anomalies?hours_back=24
   → SELECT FROM fact_alerts LEFT JOIN fact_vessel_latest
   → JSON array → render colored CircleMarkers by severity
```

---

*Continues in Part 3: Weather Integration, Enterprise/Cloud Layer, Port Reference, Architectural Constraints*
