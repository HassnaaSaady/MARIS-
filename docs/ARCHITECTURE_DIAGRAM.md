# Architecture Diagram — Maritime Navigation AI System

> **Purpose:** Visual and textual representation of the system's data flow, service boundaries, and integration points. All diagrams use ASCII/Mermaid notation for readability in any editor.

---

## 1. High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     EXTERNAL DATA SOURCES                               │
│                                                                         │
│   AIS Radio Feeds ──►  AIS Producer (Python)  ──►  Kafka Topic         │
│   (vessel transponders)  [src/producer/]             [ais_raw]          │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      KAFKA BROKER            │
                    │  (KRaft mode, single node)   │
                    │  Topics: ais_raw             │
                    └──────────────┬───────────────┘
                                   │
         ┌─────────────────────────┼──────────────────────────┐
         │                         │                          │
         ▼                         ▼                          ▼
┌────────────────┐      ┌──────────────────┐      ┌──────────────────────┐
│  SPARK STREAM  │      │  LIVE CONSUMER   │      │   API SERVER         │
│  PROCESSOR     │      │  (Streamlit      │      │   (FastAPI)          │
│                │      │   direct Kafka)  │      │                      │
│ Bronze → Silver│      │  [streamlit_app] │      │  /vessels            │
│ → Gold tables  │      │                  │      │  /alerts             │
│ [databricks/   │      └────────┬─────────┘      │  /snowflake/*        │
│  notebooks/]   │               │                │  [api/routers/]      │
└───────┬────────┘               │                └──────────┬───────────┘
        │                        │                           │
        └──────────┬─────────────┘                           │
                   ▼                                         │
        ┌──────────────────────┐                             │
        │    POSTGRESQL DB      │◄────────────────────────────┘
        │                      │
        │  fact_ais_track      │  ← full position history
        │  fact_vessel_latest  │  ← current state per MMSI
        │  fact_proximity_alert│  ← collision / anomaly alerts
        │  dim_vessel          │  ← vessel metadata
        │  dim_port            │  ← port reference data
        └──────────┬───────────┘
                   │
         ┌─────────┼──────────────┐
         ▼         ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────────────────┐
│  STREAMLIT   │ │  REACT   │ │  MLFLOW              │
│  DASHBOARD   │ │  FRONTEND│ │  EXPERIMENT TRACKER  │
│              │ │          │ │                      │
│  Live Map    │ │  Map UI  │ │  Anomaly model       │
│  Replay      │ │  [front  │ │  Congestion model    │
│  Heatmap     │ │  end/]   │ │  Route predictor     │
│  Anomalies   │ │          │ │  [mlops/experiments/]│
│  Alerts      │ └──────────┘ └──────────────────────┘
└──────────────┘
```

---

## 2. Data Flow — Real-Time Pipeline

```
AIS Transponder Signal
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  AIS PRODUCER  [src/producer/ais_producer.py]         │
│                                                       │
│  1. Read raw NMEA/AIS message                         │
│  2. Parse: MMSI, lat, lon, SOG, heading, timestamp    │
│  3. Serialize to JSON                                 │
│  4. Publish to Kafka topic "ais_raw"                  │
└───────────────────┬───────────────────────────────────┘
                    │ Kafka message
                    ▼
┌───────────────────────────────────────────────────────┐
│  SPARK STRUCTURED STREAMING                            │
│  [databricks/notebooks/]                              │
│                                                       │
│  Bronze (01): Raw ingest → Parquet/Delta              │
│      ↓                                                │
│  Silver (02): Clean, validate, deduplicate            │
│      ↓                                                │
│  Gold (03):   Aggregations, vessel state, alerts      │
│      ↓                                                │
│  Write to PostgreSQL via JDBC                         │
└───────────────────┬───────────────────────────────────┘
                    │ SQL INSERT / UPSERT
                    ▼
┌───────────────────────────────────────────────────────┐
│  POSTGRESQL  [api/models/database.py]                 │
│                                                       │
│  fact_ais_track        ← append-only track log        │
│  fact_vessel_latest    ← UPSERT on mmsi (current pos) │
│  fact_proximity_alert  ← INSERT on alert event        │
└───────────────────────────────────────────────────────┘
```

---

## 3. Data Flow — ML Inference

```
fact_ais_track (PostgreSQL)
        │
        ▼ batch read (pandas/SQLAlchemy)
┌───────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING  [mlops/experiments/]            │
│                                                       │
│  - Speed features: SOG delta, acceleration            │
│  - Position features: lat/lon bins, port proximity    │
│  - Temporal features: hour, day_of_week               │
│  - Vessel features: type, flag, length                │
└───────────────────┬───────────────────────────────────┘
                    │ feature matrix
          ┌─────────┼─────────────┐
          ▼         ▼             ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│  ANOMALY     │ │ CONGEST- │ │  ROUTE       │
│  DETECTOR    │ │ ION      │ │  PREDICTOR   │
│  (Isolation  │ │ PREDICTOR│ │  (LSTM /     │
│   Forest)    │ │ (XGBoost)│ │   Linear)    │
└──────┬───────┘ └────┬─────┘ └──────┬───────┘
       │              │              │
       └──────────────┼──────────────┘
                      │ predictions logged
                      ▼
              ┌───────────────┐
              │  MLFLOW       │
              │  (PostgreSQL  │
              │   backend)    │
              └───────────────┘
                      │ registered model
                      ▼
              ┌───────────────┐
              │  LIVE SCORER  │
              │  K8s Deployment│
              │  [k8s/deploy  │
              │   ments/live- │
              │   scorer-     │
              │   deployment] │
              └───────────────┘
```

---

## 4. Service Dependency Map

```
                       ┌──────────┐
                       │  Kafka   │
                       └────┬─────┘
            ┌───────────────┼────────────────┐
            ▼               ▼                ▼
     ┌──────────┐    ┌──────────────┐  ┌──────────┐
     │  Spark   │    │  Streamlit   │  │  API     │
     │ Processor│    │  Dashboard   │  │ (FastAPI)│
     └────┬─────┘    └──────┬───────┘  └────┬─────┘
          │                 │               │
          └─────────────────┼───────────────┘
                            ▼
                     ┌──────────────┐
                     │  PostgreSQL  │
                     └──────────────┘
                            │
                     ┌──────┴───────┐
                     ▼              ▼
              ┌──────────┐   ┌──────────────┐
              │  MLflow  │   │  Snowflake   │
              │ (opt.)   │   │  (opt.)      │
              └──────────┘   └──────────────┘
```

**Dependency direction (arrows = "depends on"):**

- Streamlit → PostgreSQL (read-heavy)
- Streamlit → Kafka (live feed consumer)
- API → PostgreSQL (read/write)
- Spark → Kafka (source), PostgreSQL (sink)
- MLflow → PostgreSQL (artifact/metadata backend)
- Snowflake → PostgreSQL (data sync via `snowflake_loader.py`)

---

## 5. Kubernetes Deployment Layout

```
Namespace: maritime-nav
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │  fastapi       │  │  frontend      │  │  streamlit   │  │
│  │  Deployment    │  │  Deployment    │  │  Deployment  │  │
│  │  HPA: 2-10     │  │  HPA: 2-5      │  │  HPA: 1-3    │  │
│  └────────┬───────┘  └────────────────┘  └──────────────┘  │
│           │                                                 │
│  ┌────────▼───────┐  ┌────────────────┐                     │
│  │  postgres      │  │  kafka         │                     │
│  │  Deployment    │  │  Deployment    │                     │
│  │  (StatefulSet  │  │  (StatefulSet  │                     │
│  │   recommended) │  │   recommended) │                     │
│  └────────────────┘  └────────────────┘                     │
│                                                             │
│  ┌────────────────┐                                         │
│  │  live-scorer   │                                         │
│  │  Deployment    │                                         │
│  └────────────────┘                                         │
│                                                             │
│  ConfigMaps: app-config.yaml                                │
│  Secrets:    secrets-template.yaml (populate before deploy) │
│  Services:   services.yaml (ClusterIP + LoadBalancer)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Network Ports Reference

| Service | Internal Port | External (NodePort/LB) | Protocol |
|---------|--------------|------------------------|----------|
| FastAPI | 8000 | 80 / 443 (via ingress) | HTTP/HTTPS |
| React Frontend | 3000 | 80 (via ingress) | HTTP |
| Streamlit | 8501 | 8501 | HTTP |
| PostgreSQL | 5432 | Not exposed | TCP |
| Kafka | 9092 | Not exposed | TCP |
| Kafka (controller) | 9093 | Not exposed | TCP |
| MLflow | 5000 | Optional | HTTP |
| Snowflake | N/A | External SaaS | HTTPS |

---

## 7. Data Schema — Core Tables

```
fact_ais_track
┌─────────────────┬──────────────┬──────────────────────────┐
│ Column          │ Type         │ Notes                    │
├─────────────────┼──────────────┼──────────────────────────┤
│ id              │ BIGSERIAL PK │ Auto-increment           │
│ mmsi            │ VARCHAR(20)  │ Indexed, digit string    │
│ vessel_name     │ VARCHAR(100) │ Nullable                 │
│ vessel_type     │ VARCHAR(50)  │ Nullable                 │
│ lat             │ FLOAT        │ WGS84 decimal degrees    │
│ lon             │ FLOAT        │ WGS84 decimal degrees    │
│ lat_bin         │ FLOAT        │ Rounded to 0.1° for bins │
│ lon_bin         │ FLOAT        │ Rounded to 0.1° for bins │
│ sog             │ FLOAT        │ Speed over ground (knots)│
│ cog             │ FLOAT        │ Course over ground (°)   │
│ heading         │ FLOAT        │ True heading (°)         │
│ base_datetime   │ TIMESTAMP    │ AIS message timestamp    │
│ created_at      │ TIMESTAMP    │ Insert time              │
└─────────────────┴──────────────┴──────────────────────────┘
Indexes: ix_track_mmsi_time (mmsi, base_datetime)
         ix_track_lat_bin   (lat_bin, lon_bin)

fact_vessel_latest
┌─────────────────┬──────────────┬──────────────────────────┐
│ mmsi            │ VARCHAR(20)  │ Primary key              │
│ vessel_name     │ VARCHAR(100) │                          │
│ lat             │ FLOAT        │ Latest position          │
│ lon             │ FLOAT        │ Latest position          │
│ sog             │ FLOAT        │                          │
│ heading         │ FLOAT        │                          │
│ last_seen       │ TIMESTAMP    │                          │
└─────────────────┴──────────────┴──────────────────────────┘
```

---

## 8. Optional Integrations

```
┌─────────────────────────────────────────────────────────────┐
│  SNOWFLAKE (optional, via api/routers/snowflake_router.py)  │
│                                                             │
│  PostgreSQL ──► snowflake_loader.py ──► Snowflake Warehouse │
│                 [src/snowflake/]         (external SaaS)    │
│                                                             │
│  Use case: long-term archival, BI tools, cross-fleet        │
│            analytics beyond single-vessel scope             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  DATABRICKS (optional migration path)                       │
│                                                             │
│  Notebooks in databricks/notebooks/ replace Spark Docker   │
│  container with managed Databricks cluster.                 │
│                                                             │
│  Bronze/Silver/Gold architecture remains the same;         │
│  compute moves from local Docker to Databricks Runtime.     │
└─────────────────────────────────────────────────────────────┘
```

---

*Last updated: 2026-05-24*
