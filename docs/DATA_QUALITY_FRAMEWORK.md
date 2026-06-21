# Data Quality Framework — Maritime Navigation AI System

**Status:** Recommended design — not yet implemented in the current codebase.  
**Domain:** AIS vessel tracking data, US coastal waters  
**Scale:** ~9M rows/day, 17 columns, 30,771 unique vessels  

---

## 1. Why AIS Data Is Inherently Dirty

AIS (Automatic Identification System) data has well-documented quality
problems that differ from typical enterprise data:

| Issue | Cause | Frequency |
|---|---|---|
| MMSI collisions | Vessels sharing or spoofing MMSI numbers | ~0.3% of records |
| GPS drift at anchor | Vessel stationary but position jitters ±50m | Common |
| Speed of 102.3 knots | AIS encoding sentinel for "not available" | Frequent |
| Future timestamps | Incorrect vessel clock or replay injection | Rare but dangerous for ML |
| Lat/Lon = 0.0 | Missing position default (null island problem) | ~0.01% |
| Heading = 511 | AIS sentinel for "not available" | Common |
| Teleporting vessels | Two valid positions but physically impossible speed | ~0.1% |
| Duplicate messages | Multi-base-station reception of same transmission | ~2-5% |

The current Silver layer (`convert_csv.py`) filters some of these (SOG > 60,
lat/lon = 0, coordinate bounds). This document extends that to a systematic
framework.

---

## 2. Great Expectations Integration

### 2.1 Installation

```bash
pip install great_expectations
```

### 2.2 Initialise

```bash
cd /mnt/d/Maritime-navigation-AI-system
great_expectations init
```

This creates `great_expectations/` directory with:
```
great_expectations/
├── great_expectations.yml
├── expectations/
│   └── ais_bronze_suite.json
├── checkpoints/
│   └── bronze_checkpoint.yml
└── uncommitted/
    └── validations/
```

### 2.3 AIS Bronze Expectation Suite

```python
# src/quality/create_ais_expectations.py
import great_expectations as gx

context = gx.get_context()

suite = context.add_expectation_suite("ais_bronze_suite")

# MMSI: 9-digit numeric string
suite.add_expectation(gx.expectations.ExpectColumnValuesToMatchRegex(
    column="mmsi",
    regex=r"^\d{9}$",
    mostly=0.99,  # allow 1% invalid for spoofed/test vessels
    meta={"notes": "MMSI must be 9 digits per ITU-R M.585"}
))

# Latitude: valid range
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
    column="lat", min_value=-90, max_value=90, mostly=0.999
))

# Longitude: valid range
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
    column="lon", min_value=-180, max_value=180, mostly=0.999
))

# SOG: 0-102 knots (102.3 is the AIS "not available" sentinel)
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
    column="sog", min_value=0, max_value=102, mostly=0.999
))

# COG: 0-360 degrees (360.0 = not available)
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
    column="cog", min_value=0, max_value=360, mostly=0.999
))

# Heading: 0-359 or 511 (511 = not available)
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
    column="heading", min_value=0, max_value=511, mostly=0.999
))

# Timestamp: not null, not in the future
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(
    column="base_datetime", mostly=1.0
))

# No null island (lat=0, lon=0)
suite.add_expectation(gx.expectations.ExpectColumnPairValuesToNotBeEqual(
    column_A="lat", column_B="lon",
    # Custom: at least one must be non-zero
    meta={"notes": "Filter null island positions"}
))

# Row count: each day should have 8M-10M rows based on observed data
suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(
    min_value=7_000_000,
    max_value=11_000_000
))

# Unique vessels per day: should be 25K-35K for US coastal
suite.add_expectation(gx.expectations.ExpectColumnUniqueValueCountToBeBetween(
    column="mmsi",
    min_value=20_000,
    max_value=50_000
))

context.save_expectation_suite(suite)
print("Expectation suite saved.")
```

### 2.4 Checkpoint (run in pipeline)

```python
# src/quality/run_quality_check.py
import great_expectations as gx
from pathlib import Path

def validate_bronze_parquet(parquet_path: str) -> dict:
    context = gx.get_context()

    datasource = context.sources.add_or_update_pandas(name="bronze_parquet")
    asset = datasource.add_parquet_asset(
        name="ais_bronze",
        glob_patterns={"path": parquet_path}
    )
    batch_request = asset.build_batch_request()

    checkpoint = context.add_or_update_checkpoint(
        name="bronze_checkpoint",
        validations=[{
            "batch_request": batch_request,
            "expectation_suite_name": "ais_bronze_suite",
        }]
    )

    result = checkpoint.run()
    success = result["success"]

    if not success:
        failed = [
            k for k, v in result.run_results.items()
            if not v["validation_result"]["success"]
        ]
        print(f"QUALITY FAILED: {len(failed)} expectation(s) failed")

    return {"success": success, "result": result}
```

---

## 3. AIS-Specific Validation Rules

### 3.1 MMSI Validation

```python
# src/common/ais_validators.py

def validate_mmsi(mmsi: str) -> tuple[bool, str]:
    """Validate MMSI per ITU-R M.585 specification."""
    clean = str(mmsi).strip()

    if not clean.isdigit():
        return False, "non-numeric"
    if len(clean) != 9:
        return False, f"wrong_length_{len(clean)}"

    prefix = int(clean[:3])

    # MID (Maritime Identification Digits) range: 201-775
    # Special ranges:
    # 970-972: SAR transponders
    # 974: EPIRB
    # 98x: crafts associated with a parent ship
    if prefix < 201 or (prefix > 775 and prefix < 970):
        return False, "invalid_mid_prefix"

    return True, "valid"
```

### 3.2 Teleportation Detection

```python
def detect_teleportation(
    prev_lat: float, prev_lon: float, prev_time,
    curr_lat: float, curr_lon: float, curr_time,
    max_speed_knots: float = 50.0
) -> bool:
    """
    Return True if position change implies physically impossible speed.
    Uses haversine distance and elapsed time.
    """
    from src.common.schema_utils import haversine_nm
    import pandas as pd

    elapsed_hours = (curr_time - prev_time).total_seconds() / 3600.0
    if elapsed_hours <= 0:
        return True  # backwards time is also invalid

    distance_nm = haversine_nm(prev_lat, prev_lon, curr_lat, curr_lon)
    implied_speed = distance_nm / elapsed_hours

    return implied_speed > max_speed_knots
```

### 3.3 Sentinel Value Handling

AIS uses specific numeric sentinels for "not available" that must not be
treated as real measurements:

```python
AIS_SENTINELS = {
    "sog":     102.3,   # speed not available
    "cog":     360.0,   # course not available
    "heading": 511,     # heading not available
    "length":  0,       # dimension not available
    "width":   0,
    "draft":   0,
}

def replace_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    for col, sentinel in AIS_SENTINELS.items():
        if col in df.columns:
            df[col] = df[col].replace(sentinel, float("nan"))
    return df
```

---

## 4. Schema Evolution with Delta Lake

### 4.1 The Problem

AIS data providers occasionally add or rename columns:
- MarineCadastre renamed `BaseDateTime` to `base_date_time` in 2023
- The `TransceiverClass` column was added in 2022
- Future versions may add AIS Class B extended reports

If the Bronze writer uses `overwrite_or_ignore`, a schema change either
silently drops new columns or crashes the writer.

### 4.2 Delta Lake Schema Evolution Strategy

```python
# Use Delta Lake with schema evolution enabled
df.write.format("delta") \
    .option("mergeSchema", "true") \
    .option("overwriteSchema", "false") \
    .mode("append") \
    .save(delta_path)
```

`mergeSchema=true`: adds new columns automatically, fills old rows with null.  
`overwriteSchema=false`: prevents accidentally replacing all columns.

### 4.3 Schema Registry

Maintain a versioned schema registry alongside the Delta table:

```python
# src/quality/schema_registry.py
SCHEMA_VERSIONS = {
    "v1": {  # pre-2022
        "required": ["mmsi", "base_datetime", "lat", "lon", "sog", "cog"],
        "optional": [],
    },
    "v2": {  # 2022: added transceiver_class
        "required": ["mmsi", "base_datetime", "lat", "lon", "sog", "cog"],
        "optional": ["transceiver_class"],
    },
    "v3": {  # current (2023+)
        "required": ["mmsi", "base_datetime", "lat", "lon", "sog", "cog",
                     "heading", "vessel_name", "vessel_type", "status"],
        "optional": ["imo", "call_sign", "length", "width", "draft",
                     "cargo", "transceiver_class"],
    }
}

def detect_schema_version(columns: list[str]) -> str:
    cols = set(c.lower() for c in columns)
    if "transceiver_class" in cols or "transceiver" in cols:
        return "v3" if "heading" in cols else "v2"
    return "v1"
```

### 4.4 Breaking Change Protocol

When a column is removed or renamed:

1. Add the old column name as an alias in `schema_utils.py`
2. Write a Delta Lake migration notebook:

```python
# databricks/notebooks/schema_migration_v3_to_v4.py
from delta.tables import DeltaTable

dt = DeltaTable.forPath(spark, "/data/delta/silver")

# Rename column (Delta Lake 2.0+ only)
dt.toDF() \
  .withColumnRenamed("vessel_type", "ship_type") \
  .write.format("delta") \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .save("/data/delta/silver_v4")
```

3. Update the expectation suite with the new column name
4. Tag the Delta table: `ALTER TABLE silver SET TBLPROPERTIES ('schema_version' = 'v4')`

---

## 5. Data Quality Anomaly Detection

### 5.1 Statistical Anomalies in Quality Metrics

Beyond rule-based validation, monitor the quality metrics themselves for
anomalies. A sudden drop in row count or spike in null rates indicates an
upstream feed problem.

```python
# src/quality/quality_anomaly_detector.py
import pandas as pd
from scipy import stats

def detect_quality_anomaly(
    daily_metrics: list[dict],
    metric_name: str,
    zscore_threshold: float = 3.0
) -> bool:
    """
    Return True if today's metric is a statistical outlier vs
    the past 7 days (z-score method).
    """
    values = [d[metric_name] for d in daily_metrics[-8:-1]]  # last 7 days
    today = daily_metrics[-1][metric_name]

    if len(values) < 3:
        return False

    z = abs(stats.zscore(values + [today])[-1])
    return z > zscore_threshold
```

### 5.2 Quality Metrics to Track Daily

```python
DAILY_QUALITY_METRICS = {
    "row_count": "Total rows received",
    "null_mmsi_pct": "% records with missing MMSI",
    "invalid_coords_pct": "% records with lat/lon out of bounds",
    "null_island_pct": "% records at 0.0, 0.0",
    "sentinel_sog_pct": "% records with SOG = 102.3",
    "duplicate_pct": "% duplicate MMSI+timestamp combinations",
    "unique_vessel_count": "Count of distinct MMSIs",
    "future_timestamp_count": "Count of records with timestamp > now",
}
```

### 5.3 Integration with Pipeline

Embed quality checks as a gate in the Bronze→Silver transition:

```python
# src/processing/silver_job.py
from src.quality.run_quality_check import validate_bronze_parquet

def bronze_to_silver(date_partition: str):
    bronze_path = f"data/parquet/year=2025/month=5/day={date_partition}"

    quality_result = validate_bronze_parquet(bronze_path)

    if not quality_result["success"]:
        # Don't promote bad data to Silver
        raise ValueError(
            f"Bronze quality check failed for {date_partition}. "
            "Aborting Silver promotion."
        )

    # proceed with Silver transformation
    ...
```

---

## 6. Validation Rules Summary

| Column | Rule | Action on Failure |
|---|---|---|
| mmsi | 9-digit numeric, valid MID prefix | Quarantine to `bronze_rejected/` |
| lat | -90 to +90, not null, not 0.0 | Drop row |
| lon | -180 to +180, not null, not 0.0 | Drop row |
| sog | 0 to 60 (operational), < 102 (raw) | Replace 102.3 with NaN |
| cog | 0 to 360 | Replace 360.0 with NaN |
| heading | 0 to 359 or 511 | Replace 511 with NaN |
| base_datetime | not null, not future, format valid | Drop row |
| vessel implied speed | < 50 knots between consecutive positions | Flag as teleport, quarantine |
| daily row count | 7M – 11M rows | Alert pipeline team |
| unique vessel count | 20K – 50K per day | Alert — check feed |
