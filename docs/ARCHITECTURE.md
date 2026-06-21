# Maritime Navigation AI System — Architecture

> All details verified against the live repo: `docker-compose.yml`, `api/models/database.py`,
> `src/common/config.py`, all processing jobs, and live PostgreSQL row counts (queried
> 2026-06-08).

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph DATA_SOURCE["Data Sources"]
        CSV["CSV / Parquet files\n(14-day AIS dataset\ntrain / valid / test / live splits)"]
    end

    subgraph INGESTION["Ingestion layer"]
        PROD["AIS Producer\nsrc/producer/kafka_producer.py\nPython 3.10 · kafka-python"]
        CSV --> PROD
    end

    subgraph KAFKA_LAYER["Message Bus · Confluent Kafka 7.4"]
        ZK["Zookeeper :2181"]
        KB["Kafka Broker\ntopic: ais_raw\n:9092 / :29092"]
        KUI["Kafka UI\n:8083"]
        ZK --> KB
        KB --- KUI
    end

    PROD -->|"JSON messages\n~0.002 s delay"| KB

    subgraph SPARK_CLUSTER["Spark 3.4 cluster"]
        SM["Spark Master\n:9090 web · :7077 RPC"]
        SW1["Worker-1\n4 cores · 4 GB"]
        SW2["Worker-2\n4 cores · 4 GB"]
        SS["spark-stream\nStructured Streaming\nbronze append"]
        SM --> SW1 & SW2
        SS -->|"--master spark://spark-master:7077"| SM
    end

    KB -->|"Structured Streaming\nSpark SQL Kafka connector"| SS

    subgraph DELTA["Delta Lake · Medallion · /delta volume"]
        BRZ["Bronze\n/delta/bronze/ais\nraw + enriched · append"]
        SLV["Silver\n/delta/silver/ais_clean\nclean · deduplicated · ML features\npartitioned year/month/day"]
        GV["Gold · vessel_latest\n/delta/gold/vessel_latest"]
        GD["Gold · traffic_density\n/delta/gold/traffic_density"]
        GS["Gold · daily_stats\n/delta/gold/daily_stats"]
        BRZ -->|"silver_job.py\nAIS sentinel nulling\nteleport filter\nhaversine features"| SLV
        SLV -->|"gold_job.py\nrow_number latest-per-MMSI"| GV
        SLV -->|"gold_job.py\n0.1° grid · hourly agg"| GD
        SLV -->|"gold_job.py\ndaily stats"| GS
    end

    SS --> BRZ

    subgraph ML["ML layer  (models/ volume)"]
        IF["IsolationForest\nisolation_forest_v2.pkl\nAnomaly detection"]
        XGB["XGBoost\nxgb_lat/lon_{5,10,15}min.pkl\nPosition prediction"]
        RF["RandomForest\ncongestion_rf.pkl\nCongestion classification"]
    end

    subgraph SCORER["Live Scorer · live_scorer.py · Python 3.10"]
        LS["Kafka consumer\nbatch_size=200 · window=5s\nScores · UPSERTs · writes alerts"]
    end

    KB -->|"ais_raw"| LS
    IF & XGB & RF --> LS

    subgraph PG["PostgreSQL 15 · :5432 · Star Schema"]
        FVL["fact_vessel_latest\n21 659 rows"]
        FA["fact_alerts\n103 332 rows"]
        FAT["fact_ais_track"]
        FTD["fact_traffic_density"]
        FDS["fact_daily_stats"]
        DV["dim_vessel"]
        DS["dim_status"]
    end

    LS -->|"UPSERT latest position"| FVL
    LS -->|"INSERT anomaly +\ncollision alerts"| FA
    GV & GD & GS -->|"JDBC overwrite"| FVL & FTD & FDS
    GV -->|"JDBC"| DV

    subgraph API["FastAPI 0.111 · Uvicorn · :8000"]
        FAPI["api/main.py\n15 REST endpoints\nSQLAlchemy ORM"]
    end

    FVL & FA & FAT & FTD & FDS --> FAPI

    subgraph FRONTEND["React 18 · :3000"]
        MAP["Vessel Map\nLeaflet / react-leaflet"]
        ANOM["Anomaly Detection\nCircleMarkers by severity"]
        COLL["Collision Risk\nPaired markers + Polyline"]
        HEAT["Traffic Heatmap"]
        REPLAY["Historical Replay"]
        ANAL["Analytics\nRecharts"]
    end

    FAPI -->|"REST polling 3 s"| MAP & ANOM & COLL & HEAT & REPLAY & ANAL

    subgraph STREAMLIT["Streamlit · :8501"]
        ST["src/dashboard/streamlit_app.py\nPython analytics dashboard"]
    end

    FAPI --> ST
    FVL --> ST
```

---

## 2. End-to-End Data Flow — One AIS Message

```mermaid
sequenceDiagram
    participant P  as AIS Producer<br/>(kafka_producer.py)
    participant K  as Kafka<br/>topic: ais_raw
    participant SS as Spark Streaming<br/>(spark_streaming_consumer.py)
    participant BD as Bronze Delta<br/>/delta/bronze/ais
    participant LS as Live Scorer<br/>(live_scorer.py)
    participant ML as ML Models<br/>(IsolationForest · XGBoost)
    participant PG as PostgreSQL<br/>(fact_vessel_latest · fact_alerts)
    participant FA as FastAPI<br/>(:8000)
    participant UI as React Frontend<br/>(:3000)

    P->>P: Read Parquet row from live split<br/>normalize_ais_record()
    P->>K: Produce JSON message<br/>{mmsi, lat, lon, sog, cog, heading, …}

    par Streaming path
        K->>SS: poll() — batch of records
        SS->>SS: parse JSON schema<br/>enrich: speed flags, port zone, risk_level
        SS->>BD: appendToFile (Delta append)<br/>trigger: processingTime 10s
    and Live scoring path
        K->>LS: poll() — batch ≤ 200 msgs<br/>within 5 s window
        LS->>LS: fillna(0) on numeric cols
        LS->>ML: score_anomaly(df)<br/>IsolationForest → anomaly_score, is_anomaly
        ML-->>LS: anomaly_score, anomaly_type
        LS->>ML: predict_position(lat, lon, sog, cog, 5 min)<br/>XGBoost → predicted_lat, predicted_lon
        ML-->>LS: predicted_lat, predicted_lon
        LS->>LS: classify_risk(sog, lat, lon)
        LS->>PG: UPSERT fact_vessel_latest (one row per MMSI)
        LS->>LS: build anomaly_alerts if is_anomaly<br/>build collision_alerts via haversine pairwise scan
        LS->>PG: INSERT INTO fact_alerts
    end

    UI->>FA: GET /api/vessels (poll 3 s)
    FA->>PG: SELECT * FROM fact_vessel_latest<br/>WHERE data_split='live' …
    PG-->>FA: rows with lat, lon, risk_level, is_anomaly
    FA-->>UI: JSON vessel array

    UI->>FA: GET /api/anomalies?hours_back=24
    FA->>PG: SELECT FROM fact_alerts<br/>WHERE alert_type IN (anomaly types)<br/>LEFT JOIN fact_vessel_latest for null lat/lon
    PG-->>FA: anomaly array with lat, lon, severity
    FA-->>UI: JSON anomaly array

    UI->>UI: render CircleMarker per anomaly<br/>color by severity (HIGH=red, MEDIUM=orange, LOW=green)
```

---

## 3. PostgreSQL Star Schema — ER Diagram

> Columns taken verbatim from `api/models/database.py`. Logical mmsi references
> are shown as relationships; no FK constraints are declared in the ORM.

```mermaid
erDiagram
    DIM_VESSEL {
        varchar_20  mmsi              PK
        varchar_100 vessel_name
        varchar_20  imo
        varchar_20  call_sign
        varchar_50  vessel_type
        varchar_100 vessel_type_label
        float       length
        float       width
        float       draft
        varchar_50  cargo
        varchar_10  transceiver_class
        varchar_20  data_split
        datetime    first_seen
        datetime    last_seen
        bigint      total_records
        datetime    created_at
        datetime    updated_at
    }

    DIM_STATUS {
        integer    status_id    PK
        varchar_10 status_code  UK
        varchar_100 status_label
        boolean    is_underway
        boolean    is_stopped
        boolean    is_anchored
        boolean    is_moored
        boolean    is_aground
    }

    FACT_VESSEL_LATEST {
        varchar_20 mmsi          PK
        varchar_100 vessel_name
        varchar_50 vessel_type
        float      lat           "NOT NULL"
        float      lon           "NOT NULL"
        float      sog
        float      cog
        float      heading
        varchar_10 risk_level
        boolean    is_anomaly
        float      anomaly_score
        varchar_100 anomaly_type
        float      predicted_lat
        float      predicted_lon
        datetime   base_datetime
        datetime   updated_at
        varchar_20 data_split
    }

    FACT_AIS_TRACK {
        bigint     id            PK
        varchar_20 mmsi          "NOT NULL"
        varchar_100 vessel_name
        varchar_50 vessel_type
        float      lat           "NOT NULL"
        float      lon           "NOT NULL"
        float      lat_bin
        float      lon_bin
        float      sog
        float      cog
        float      heading
        varchar_20 status
        varchar_10 risk_level
        boolean    is_anomaly
        float      anomaly_score
        varchar_100 anomaly_type
        datetime   base_datetime "NOT NULL"
        datetime   ingested_at
        varchar_20 data_split    "NOT NULL"
    }

    FACT_TRAFFIC_DENSITY {
        float    lat_bin        PK
        float    lon_bin        PK
        datetime hour_bucket    PK
        bigint   vessel_count   "NOT NULL"
        bigint   unique_vessels
        float    avg_sog
        bigint   stopped_count
        text     congestion_level
        text     data_split
    }

    FACT_ALERTS {
        integer  id             PK
        varchar_50 alert_type   "NOT NULL"
        varchar_20 severity     "NOT NULL"
        varchar_20 mmsi_1
        varchar_20 mmsi_2
        varchar_100 vessel_name_1
        varchar_100 vessel_name_2
        float    lat
        float    lon
        text     description
        json     extra_data
        float    anomaly_score
        float    distance_nm
        boolean  is_resolved
        datetime created_at
        datetime resolved_at
        varchar_20 data_split
    }

    FACT_DAILY_STATS {
        date    stat_date       PK "NOT NULL"
        varchar_20 data_split
        integer total_vessels
        bigint  total_records
        float   avg_sog
        float   max_sog
        integer high_risk_count
        integer stopped_vessels
        integer anomaly_count
    }

    DIM_VESSEL        ||--o{ FACT_VESSEL_LATEST : "mmsi"
    DIM_VESSEL        ||--o{ FACT_AIS_TRACK     : "mmsi"
    DIM_VESSEL        ||--o{ FACT_ALERTS        : "mmsi_1 / mmsi_2"
    DIM_STATUS        ||--o{ FACT_AIS_TRACK     : "status_code → status"
    FACT_VESSEL_LATEST ||--o{ FACT_ALERTS       : "mmsi_1 (lat/lon fallback)"
```

---

## 4. Component Table

| Service | Container | Purpose | Host port(s) | Technology | Why chosen |
|---|---|---|---|---|---|
| **Zookeeper** | `zookeeper` | Kafka broker coordination | — (internal :2181) | Confluent cp-zookeeper 7.4.0 | Required by Confluent Kafka 7.4; manages broker metadata and leader election |
| **Kafka** | `kafka` | Message bus for AIS stream | :9092 (internal), :29092 (host) | Confluent cp-kafka 7.4.0, topic `ais_raw` | Decouples producer from all consumers; durable log allows replay; exactly-once semantics for the streaming path |
| **Kafka UI** | `kafka-ui` | Topic browser and consumer lag monitor | :8083 | provectuslabs/kafka-ui | Zero-config GUI; observe `ais_raw` throughput without CLI |
| **Spark Master** | `spark-master` | Cluster scheduler | :9090 (Web UI), :7077 (RPC) | Apache Spark 3.4 + Delta Core 2.4.0 | Native Delta Lake integration; distributed compute for the 36 M-row AIS dataset |
| **Spark Worker 1** | `spark-worker-1` | Executor node | :9091 (Web UI) | Spark Worker, 4 cores / 4 GB | Parallel processing with Worker 2; sized for i7-14700 / 32 GB host |
| **Spark Worker 2** | `spark-worker-2` | Executor node | :9093 (Web UI) | Spark Worker, 4 cores / 4 GB | See above |
| **Spark Stream** | `spark-stream` | Kafka → Bronze live ingest | — | Spark Structured Streaming, `spark-sql-kafka-0-10_2.12:3.4.4` | Micro-batch streaming with Delta append; checkpoint at `/delta/checkpoints/bronze` ensures exactly-once Bronze writes |
| **AIS Producer** | `ais-producer` | Replays live-split Parquet to Kafka | — | Python 3.10, kafka-python 2.0.2 | Simulates live AIS feed at 0.002 s/message; loops continuously |
| **Live Scorer** | `live-scorer` | Real-time ML scoring + alert writing | — | Python 3.10, scikit-learn 1.5.1, XGBoost 2.0.3; batch_size=200, window=5 s | Separate process from the API so scoring never blocks HTTP; writes `fact_vessel_latest` (UPSERT) and `fact_alerts` (INSERT) directly to PostgreSQL |
| **FastAPI** | `maritime-api` | REST API for all dashboard features | :8000 | FastAPI 0.111.0, Uvicorn 0.30.1, SQLAlchemy 2.0.31, Python 3.11 | Async framework with auto-generated OpenAPI docs; SQLAlchemy ORM maps directly to the star schema; 15 endpoints covering all 10 dashboard features |
| **PostgreSQL** | `postgres` | Operational data store (star schema) | :5432 | PostgreSQL 15-alpine, shared_buffers=512 MB, effective_cache=2 GB | ACID guarantees for alert state; row-level locking for concurrent UPSERT from live_scorer; fast indexed reads for the React 3-second poll cycle |
| **Streamlit** | `streamlit-dashboard` | Python analytics dashboard | :8501 | Streamlit, pandas, Delta Lake reader | Rapid Python-native prototyping of ML monitoring views without a separate frontend build step |
| **React Frontend** | `maritime-frontend` | Primary user-facing dashboard | :3000 | React 18, react-leaflet 4.x, Recharts, Node 18-slim | Leaflet renders 20 000+ vessel markers efficiently with CircleMarker pooling; Recharts covers time-series analytics; hot-reload dev server via `npm start` |

---

## 5. Medallion Layer Table

| Layer | Delta path | PostgreSQL table(s) | Live row count | Transformations applied |
|---|---|---|---|---|
| **Raw** | `/app/data/parquet` | — | ~36 M rows across 14 days | Parquet files converted from CSV; columnar storage for efficient Spark reads; `data_split` column partitions into train / valid / test / live |
| **Bronze** | `/delta/bronze/ais` | — | Streaming append (grows continuously) | Quality filters: non-null mmsi/lat/lon, valid ranges (lat ∈ [−90,90], lon ∈ [−180,180], sog ∈ [0,60]); enrichment: `event_time`, `ingestion_time`, `is_stopped`, `is_slow`, `is_speeding`, `in_us_port_zone`, rule-based `risk_level`; written by both `spark_streaming_consumer.py` (live) and `bronze_job.py` (batch) |
| **Silver** | `/delta/silver/ais_clean` | — | Up to 5 000 000 rows (batch cap) | AIS sentinel nulling (sog = 102.3 → null, heading = 511 → null); kinematic bounds filter; teleport/GPS-glitch filter (implied speed > 100 kn dropped); deduplication on (mmsi, base_datetime); vessel info forward-fill per MMSI; human-readable `vessel_type_label` and `status_label`; ML features: `sog_change`, `heading_change`, `distance_nm` (haversine, R = 3 440.065 nm), `time_delta_sec`, `lat_bin_fine` (2 dp), `lon_bin_fine` (2 dp); first pings dropped; partitioned by year / month / day |
| **Gold — vessel_latest** | `/delta/gold/vessel_latest` | `fact_vessel_latest` (21 659 rows) | One row per MMSI | `row_number()` window over mmsi ordered by `base_datetime DESC`; keeps only latest ping; JDBC overwrite to PostgreSQL |
| **Gold — traffic_density** | `/delta/gold/traffic_density` | `fact_traffic_density` (0 rows — batch not yet run) | Aggregated grid cells | Group by (`lat_bin`, `lon_bin`, `hour_bucket`, `data_split`); agg: `vessel_count`, `unique_vessels`, `avg_sog`, `stopped_count`; `congestion_level` = HIGH if count ≥ 15, MEDIUM if ≥ 5, else LOW |
| **Gold — daily_stats** | `/delta/gold/daily_stats` | `fact_daily_stats` (0 rows — batch not yet run) | One row per day | Group by (`stat_date`, `data_split`); agg: `total_vessels`, `total_records`, `avg_sog`, `max_sog`, `high_risk_count`, `stopped_vessels`; `anomaly_count` placeholder (lit 0) |
| **Gold — anomalies** | `/delta/gold/anomalies` | — | Written by gold_job | Reserved path for batch anomaly aggregations (`DELTA_GOLD_ANOMALY_PATH`) |
| **Serving (live)** | — | `fact_alerts` (103 332 rows) | Continuously growing | Written exclusively by `live_scorer.py`: IsolationForest anomaly alerts + haversine pairwise collision detection; `fact_vessel_latest` enriched with `anomaly_score`, `is_anomaly`, `predicted_lat/lon` from XGBoost |

---

## ML Models

| Model | File | Algorithm | Training data | Features | Purpose |
|---|---|---|---|---|---|
| Anomaly detector | `isolation_forest_v2.pkl` + `scaler_anomaly_v2.pkl` | Isolation Forest (sklearn, n_estimators=200, contamination=0.002) | Silver TRAIN split | `sog`, `cog`, `heading`, `sog_change`, `heading_change`, `distance_nm` | Flags vessels with unusual kinematics; score ≥ 0.7 → HIGH, ≥ 0.5 → MEDIUM |
| Position predictor | `xgb_lat_{5,10,15}min.pkl` + `xgb_lon_{5,10,15}min.pkl` | XGBoost regressor (one model per horizon per axis) | Silver TRAIN split | Kinematic + position features | Predicts vessel position 5 / 10 / 15 minutes ahead for CPA calculation |
| Congestion classifier | `congestion_rf.pkl` + `congestion_encoder.pkl` | Random Forest (sklearn) | Gold traffic_density aggregations | Density grid features | Classifies grid cells as LOW / MEDIUM / HIGH congestion |

---

## Geographic Scope

The system is scoped to **US waters** (lat 24–49°N, lon 125–66°W) with five named high-priority port zones:

| Zone | Lat range | Lon range |
|---|---|---|
| Houston Ship Channel | 29.50–29.85°N | 95.30–94.80°W |
| New York Harbor | 40.50–40.75°N | 74.30–73.90°W |
| Port of Los Angeles / Long Beach | 33.70–33.85°N | 118.35–118.10°W |
| Port of New Orleans | 29.00–30.00°N | 90.50–89.50°W |
| Port of Miami | 25.70–25.85°N | 80.20–80.05°W |

Dashboard maps default to `[38.5, −75.5]` zoom 6 (US East Coast).
