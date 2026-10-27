# 🚢 MARIS – Maritime AI Real-Time & Batch Intelligence System

> An end-to-end Big Data platform that processes large-scale Automatic Identification System (AIS) vessel data through batch and real-time pipelines, enabling maritime monitoring, historical analytics, traffic congestion analysis, and AI-assisted risk assessment.

<p align="center">
  <img src="screenshots/maris-overview.png.png" alt="MARIS Project Overview" width="950">
</p>

<p align="center">
  <b>Big Data Engineering · Real-Time Streaming · Lakehouse Architecture · Maritime Analytics</b>
</p>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Project Objectives](#-project-objectives)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [End-to-End Data Pipeline](#-end-to-end-data-pipeline)
- [Real-Time Streaming Pipeline](#-real-time-streaming-pipeline)
- [Batch Processing and Lakehouse](#-batch-processing-and-lakehouse)
- [Data Quality and Transformation](#-data-quality-and-transformation)
- [Analytics and Machine Learning](#-analytics-and-machine-learning)
- [Data Serving and APIs](#-data-serving-and-apis)
- [Technology Stack](#-technology-stack)
- [Key Engineering Concepts](#-key-engineering-concepts)
- [Repository Structure](#-repository-structure)
- [Screenshots and Demo](#-screenshots-and-demo)
- [Future Enhancements](#-future-enhancements)
- [Contributors](#-contributors)

---

## 📌 Overview

MARIS is a maritime intelligence platform designed to process large volumes of vessel tracking data using modern Big Data technologies.

The system combines batch processing for historical analysis with real-time streaming for continuously arriving vessel data. Its architecture integrates Apache Kafka, Apache Spark, Delta Lake, Apache Airflow, PostgreSQL, and FastAPI to support data ingestion, transformation, storage, orchestration, and downstream analytics.

The project processes **62M+ AIS records** and demonstrates how large-scale maritime data can be organized into reliable analytical pipelines.

### Core Capabilities

- Real-time vessel data processing
- Historical maritime traffic analytics
- Large-scale ETL pipeline development
- Bronze, Silver, and Gold lakehouse layers
- Vessel traffic congestion analysis
- Anomaly detection and collision-risk assessment
- Structured analytical data storage
- API-based data access
- KPI reporting and visualization

---

## 🎯 Project Objectives

MARIS addresses key data engineering challenges associated with large-scale vessel tracking data:

1. **Scalable Processing:** Process millions of AIS records using distributed data processing technologies.
2. **Dual Processing Modes:** Support both historical batch workloads and real-time streaming workflows.
3. **Data Organization:** Structure data into multiple lakehouse layers to support progressive cleaning and transformation.
4. **Workflow Orchestration:** Coordinate batch processing tasks through Apache Airflow.
5. **Analytical Data Access:** Store selected processed data in PostgreSQL for downstream applications.
6. **Maritime Intelligence:** Make processed vessel data available for monitoring, analytics, and risk-related use cases.

---

## 🏗️ System Architecture

MARIS combines a Lambda-style processing design with the Delta Lake Medallion Architecture.

The batch and real-time pipelines serve different processing needs while supporting a shared maritime analytics platform.

### High-Level Architecture

```text
                   AIS Vessel Traffic Data
                              |
                              v
                    Data Ingestion Layer
                              |
                    CSV / Structured Data
                              |
                              v
                     Data Preparation
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
           Real-Time Pipeline       Batch Pipeline
                  |                       |
            Apache Kafka            Apache Spark
                  |                       |
       Spark Structured Streaming    Delta Lake
                  |                  Bronze Layer
                  |                       |
                  |                  Silver Layer
                  |                       |
                  |                   Gold Layer
                  |                       |
                  +-----------+-----------+
                              |
                              v
                       PostgreSQL
                      Serving Layer
                              |
                              v
                         FastAPI
                              |
                              v
                    Analytics & Applications
```

*This is a conceptual overview of the platform. Refer to the implementation and screenshots for the exact component connections.*

### Architecture Components

| Component | Responsibility |
|---|---|
| Data Ingestion | Read and prepare AIS vessel traffic data |
| Streaming Layer | Process incoming records through Kafka and Spark Structured Streaming |
| Batch Processing | Execute distributed historical data transformations with PySpark |
| Lakehouse | Organize processed data using Delta Lake and Medallion layers |
| Orchestration | Schedule and manage batch workflows with Apache Airflow |
| Serving Database | Store selected processed records and analytical outputs in PostgreSQL |
| API Layer | Expose supported data through FastAPI |
| Analytics | Support maritime monitoring, historical analysis, and KPIs |

---

## 📦 Dataset

The project uses Automatic Identification System (AIS) vessel traffic data.

AIS data provides information about vessel movements and positions, making it useful for maritime traffic analysis and monitoring.

| Item | Details |
|---|---|
| Dataset | AIS Vessel Traffic Data |
| Source | NOAA MarineCadastre |
| Records Processed | 62M+ AIS records |
| Input Data | CSV |
| Processing Formats | Structured and columnar data formats |
| Processing Modes | Batch and real-time |

**Dataset source:** [NOAA MarineCadastre – Vessel Traffic Data](https://hub.marinecadastre.gov/pages/vesseltraffic)

The dataset is prepared for processing and analytical workloads. Where applicable, columnar storage such as Parquet can improve scan efficiency and reduce unnecessary data reads.

---

## 🔄 End-to-End Data Pipeline

The platform organizes data processing into ingestion, transformation, storage, and serving stages.

```text
                AIS Vessel Traffic Data
                           |
                           v
                     Data Ingestion
                           |
                           v
                  Schema & Data Handling
                           |
                           v
                  Prepared Vessel Data
                           |
              +------------+------------+
              |                         |
              v                         v
       Real-Time Pipeline        Batch Pipeline
              |                         |
         Apache Kafka             Apache Spark
              |                         |
    Spark Structured Streaming     Delta Lake
              |                   Bronze → Silver → Gold
              |                         |
              +------------+------------+
                           |
                           v
                      PostgreSQL
                           |
                           v
                       FastAPI
                           |
                           v
                 Downstream Applications
```

### Processing Stages

1. **Ingestion:** Read AIS records from the available source data.
2. **Preparation:** Handle schemas, data types, and data quality requirements.
3. **Processing:** Execute real-time transformations and historical batch jobs.
4. **Storage:** Organize historical datasets using Delta Lake layers.
5. **Serving:** Make selected processed data available through PostgreSQL.
6. **Consumption:** Expose supported data through APIs for analytics and applications.

---

## ⚡ Real-Time Streaming Pipeline

The real-time pipeline is designed to process vessel data as it arrives, supporting low-latency data handling and timely maritime monitoring.

### Streaming Workflow

```text
              Prepared Vessel Data
                       |
                       v
                 Kafka Producer
                       |
                       v
                  Apache Kafka
                       |
                       v
           Spark Structured Streaming
                       |
                       v
              Real-Time Processing
                       |
                       v
              PostgreSQL / Serving
```

### Streaming Responsibilities

- Ingest streaming records through Apache Kafka.
- Process incoming data using Spark Structured Streaming.
- Apply supported transformations and processing logic.
- Make selected results available to downstream analytical components.
- Support real-time vessel monitoring and risk-related use cases.

The exact processing latency, delivery guarantees, and streaming checkpoint configuration depend on the implemented pipeline configuration.

---

## 🗄️ Batch Processing and Lakehouse

The batch pipeline processes historical AIS data for structured storage, data transformation, and analytical workloads.

### Batch Workflow

```text
             Historical AIS Data
                      |
                      v
                Apache Spark
                      |
                      v
                 Delta Lake
                      |
                      v
                Bronze Layer
                      |
                      v
                Silver Layer
                      |
                      v
                 Gold Layer
                      |
                      v
             Historical Analytics
```

### Medallion Architecture

| Layer | Purpose |
|---|---|
| **Bronze** | Preserve ingested data in the initial lakehouse layer |
| **Silver** | Store cleaned, validated, and transformed data |
| **Gold** | Provide curated datasets for reporting and analytical use cases |

The layered design separates ingestion from transformation and analytical preparation, making the data processing workflow easier to organize and maintain.

### Workflow Orchestration

Apache Airflow is used to orchestrate batch workflows.

Depending on the implemented DAGs, orchestration can coordinate task dependencies, execution order, scheduling, and failure handling.

The real-time streaming pipeline is treated separately from batch workflow scheduling.

---

## 🧹 Data Quality and Transformation

Data quality is an important part of processing large-scale AIS data.

The pipeline's data preparation and transformation stages can address the following requirements, where implemented:

- Schema consistency and data type handling
- Missing or invalid value identification
- Duplicate record handling
- Data validation
- Consistent storage formats
- Preparation of structured datasets for downstream analysis

These checks help improve the reliability of historical analytics and downstream data consumption.

---

## 🤖 Analytics and Machine Learning

Processed AIS data can support maritime intelligence use cases that combine historical analysis with operational monitoring.

### Analytical Use Cases

| Use Case | Objective |
|---|---|
| Vessel Monitoring | Analyze vessel position and movement data |
| Traffic Congestion | Identify and analyze maritime traffic density patterns |
| Anomaly Detection | Identify unusual vessel behavior |
| Collision-Risk Assessment | Support analysis of potential vessel interaction risks |
| Historical Analytics | Explore vessel traffic patterns over time |
| KPI Reporting | Summarize processed data for analytical consumption |

Machine learning models and real-time inference should be documented here according to the algorithms and model-serving components actually implemented in the repository.

---

## 🗃️ Data Serving and APIs

PostgreSQL serves as a relational storage layer for selected processed records and analytical outputs.

FastAPI provides a REST API layer for exposing supported data to downstream applications.

### Serving Responsibilities

- Store selected vessel and analytical data
- Support structured querying
- Provide data access to API consumers
- Make processed results available to dashboards and analytical tools

The actual tables, API routes, and response schemas depend on the current implementation.

---

## 🛠️ Technology Stack

### Data Engineering

- Apache Spark
- PySpark
- Apache Kafka
- Spark Structured Streaming
- Delta Lake
- Apache Airflow

### Databases and Storage

- PostgreSQL
- Snowflake
- Parquet
- Delta Lake

### Backend and APIs

- Python
- FastAPI

### Development and Deployment

- Docker

---

## 💡 Key Engineering Concepts

This project demonstrates practical application of:

- Batch and real-time data processing
- ETL pipeline design
- Distributed data processing
- Streaming data ingestion
- Lakehouse architecture
- Medallion data organization
- Data quality and transformation
- Workflow orchestration
- Relational data serving
- API integration
- Historical and real-time analytics

---

## 📁 Repository Structure

The following is an illustrative structure. Update it to match the actual files and directories in this repository.

```text
MARIS/
├── README.md
├── docker-compose.yml
├── docs/
│   └── architecture.png
├── screenshots/
│   ├── maris-overview.png
│   ├── streaming-pipeline.png
│   ├── batch-pipeline.png
│   └── analytics-dashboard.png
├── data/
├── streaming/
├── batch/
├── api/
├── airflow/
├── models/
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

The development environment may require the following components, depending on the services included in the repository:

- Python
- Apache Kafka
- Apache Spark
- Delta Lake
- PostgreSQL
- Apache Airflow
- Docker and Docker Compose

### Setup

1. Clone the repository.
2. Review the project configuration and environment variables.
3. Configure the required services and data source paths.
4. Install the dependencies specified by the project.
5. Start the infrastructure using the repository's documented setup.
6. Execute the relevant batch or streaming workflow.
7. Verify the output using the available database queries, APIs, or dashboards.

**Note:** Add the exact clone command, dependency installation commands, and service startup commands after confirming the current repository configuration and entry-point scripts.

---

## 📸 Screenshots and Demo

Screenshots and demonstration videos can be added to document the implementation.

Suggested assets include:

- System architecture diagram
- Kafka streaming workflow
- Spark processing results
- Delta Lake Medallion layers
- PostgreSQL serving tables
- API responses
- Analytics dashboards

Example of displaying an image stored in the repository:

```markdown
![MARIS Architecture](docs/architecture.png)
```

---

## 🔮 Future Enhancements

Potential future improvements include:

- Automated data quality reporting
- Improved pipeline observability and monitoring
- CI/CD integration
- Cloud-based data lake integration
- Distributed multi-node deployment
- Enhanced streaming recovery and checkpoint management
- MLflow model lifecycle management, if machine learning workflows are included
- Prometheus and Grafana monitoring

---

## 👩‍💻 Contributors

- Hassnaa Saady
- Fatema Samir
- Lamiaa Nasser

---

## 📄 Project Information

MARIS is a collaborative Big Data Engineering project focused on maritime data processing, streaming, lakehouse organization, and analytics.

See the repository's applicable license and contribution terms for information about reuse and redistribution.
