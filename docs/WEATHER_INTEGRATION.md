# Weather Integration — Maritime Navigation AI System

## Overview

Weather data (wind speed + wave height) is fetched from the Open-Meteo ERA5
archive, stored in a PostgreSQL dimension table (`dim_weather`), and used to
train an augmented congestion classifier that is evaluated against the existing
production baseline.

The integration is **strictly additive**: no existing file is modified.
The production model (`models/congestion_rf.pkl`) is never overwritten.

---

## How to Run

### Prerequisites

- Main stack running: `docker compose up -d`
- Parquet data present in `data/parquet/` (populated by `kafka_producer.py`)
- `requests` package available in the producer container

### Step 1 — Fetch weather data

```bash
docker compose exec producer python src/weather/fetch_weather.py
```

What it does:
- Reads occupied 0.5° grid cells and the date range from the AIS Silver Delta
  (or raw parquet if Silver is not mounted).
- Fetches **only** `wind_speed_10m` and `wave_height` from the Open-Meteo ERA5
  archive API for those cells and that date range.
- Caches each cell as an individual parquet file under
  `data/weather/bronze/cache/` for idempotency.
- Writes one combined bronze parquet to `data/weather/bronze/weather_bronze.parquet`.

Re-running is safe: cached cells are not re-fetched from the API.

### Step 2 — Load into PostgreSQL

```bash
docker compose exec producer python src/weather/load_dim_weather.py
```

What it does:
- Reads the bronze parquet.
- Computes `weather_severity` (see formula below).
- Applies DDL to create `dim_weather` if it does not exist.
- UPSERTs all rows — idempotent, no duplicates on re-run.

### Step 3 — Evaluate weather as a feature

```bash
docker compose exec producer \
  python src/weather/eval_weather.py
```

What it does:
- Builds the same density grid as `train_congestion.py` (same data, same
  aggregation, same time-based 80/20 split).
- Trains a **baseline** RandomForest with the 8 existing features.
- Trains an **augmented** RandomForest with those 8 features + `weather_severity`.
- Prints accuracy and macro-F1 for both, plus the delta.
- Saves the augmented model to `models/congestion_rf_weather.pkl`.
- **Never touches** `models/congestion_rf.pkl`.

---

## dim_weather Schema

| Column            | Type      | Description                                              |
|-------------------|-----------|----------------------------------------------------------|
| `lat_bin`         | FLOAT PK  | 0.5° floor-binned latitude (matches `train_congestion.py`) |
| `lon_bin`         | FLOAT PK  | 0.5° floor-binned longitude                              |
| `hour_bucket`     | TIMESTAMP PK | UTC hour (floored to the hour)                        |
| `wind_speed_10m`  | FLOAT     | Wind speed at 10 m in **m/s** (null if API error)       |
| `wave_height`     | FLOAT     | Significant wave height in **m** (null at inland cells) |
| `weather_severity`| FLOAT     | Normalised [0, 1] severity index (see formula)           |
| `updated_at`      | TIMESTAMP | Set by PostgreSQL `NOW()` on every upsert                |

**Primary key**: `(lat_bin, lon_bin, hour_bucket)`

**Index**: `ix_weather_grid_time ON (hour_bucket, lat_bin, lon_bin)` — covers
the lookup pattern used by `build_congestion_with_weather()`.

### DDL (informational — applied automatically)

```sql
CREATE TABLE IF NOT EXISTS dim_weather (
    lat_bin          FLOAT     NOT NULL,
    lon_bin          FLOAT     NOT NULL,
    hour_bucket      TIMESTAMP NOT NULL,
    wind_speed_10m   FLOAT,
    wave_height      FLOAT,
    weather_severity FLOAT     NOT NULL,
    updated_at       TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (lat_bin, lon_bin, hour_bucket)
);
CREATE INDEX IF NOT EXISTS ix_weather_grid_time
    ON dim_weather (hour_bucket, lat_bin, lon_bin);
```

---

## weather_severity Formula

This is the canonical definition. Do not alter it without updating both
`load_dim_weather.py` and this document.

```
wind_norm = clip(wind_speed_10m / 24.5, 0, 1)
               # 24.5 m/s ≈ Beaufort force 9 (severe gale)

wave_norm = clip(wave_height / 6.0, 0, 1)
               # 6 m ≈ "very rough" sea state (Douglas scale 6)

if wave_height is NULL (inland cell):
    weather_severity = wind_norm
else:
    weather_severity = 0.6 * wind_norm + 0.4 * wave_norm
```

**Why these constants?**

| Constant | Value | Meaning |
|----------|-------|---------|
| 24.5 m/s | Beaufort 9 | Wind that causes structural damage at sea |
| 6.0 m    | Douglas 6  | Very rough seas; limits vessel operations |
| 0.6 / 0.4 | —     | Wind dominates because it affects all vessels; waves are irrelevant at inland cells |

---

## Promoting the Augmented Challenger Model

`eval_weather.py` deliberately **does not** promote automatically. Promotion is
a manual, deliberate act to prevent an underperforming challenger from silently
replacing the production model.

### When to promote

Promote the challenger if:
1. `eval_weather.py` reports `Macro F1 Δ > 0` on the hold-out set.
2. You have confirmed the weather fetch covered ≥ 50% of test rows
   (check `weather_coverage_test_pct` in `evaluation_report_weather.json`).
3. `live_scorer.py` is updated to include `weather_severity` in its feature set.

### How to promote

```bash
# Inside the producer container:
cp models/congestion_rf_weather.pkl models/congestion_rf.pkl
cp models/congestion_encoder_weather.pkl models/congestion_encoder.pkl
cp models/congestion_features_weather.pkl models/congestion_features.pkl
```

Then restart the `live-scorer` service so it picks up the new model:

```bash
docker compose restart live-scorer
```

**Before promoting**, update `src/ml/scorer.py` so that `predict_congestion()`
includes `weather_severity` in `feat_vals`. Otherwise the live scorer will
pass a feature vector with the wrong length and the model will raise an error.

### Rollback

The original models are committed to git and also backed up as
`models/congestion_rf.BACKUP.pkl`. To roll back:

```bash
cp models/congestion_rf.BACKUP.pkl models/congestion_rf.pkl
docker compose restart live-scorer
```

---

## File Layout

```
src/weather/
├── __init__.py
├── fetch_weather.py        # Step 1 — fetch from Open-Meteo, write bronze parquet
├── load_dim_weather.py     # Step 2 — compute severity, UPSERT into dim_weather
├── weather_features.py     # Helper — build_congestion_with_weather()
└── eval_weather.py         # Step 3 — train baseline + augmented, compare

data/weather/
└── bronze/
    ├── cache/              # Per-cell parquet files (one per 0.5° cell)
    │   ├── lat+29.5_lon-95.0.parquet
    │   └── ...
    └── weather_bronze.parquet   # Combined file for load_dim_weather.py

models/
├── congestion_rf.pkl              # Production model (NEVER overwritten by weather eval)
├── congestion_rf_weather.pkl      # Challenger — augmented with weather_severity
├── congestion_encoder_weather.pkl # LabelEncoder for challenger
└── congestion_features_weather.pkl # Feature list for challenger

mlops/artifacts/
└── evaluation_report_weather.json  # Metrics + train/test date ranges
```

---

## Data Minimalism Rationale

| Decision | Reason |
|----------|--------|
| Only `wind_speed_10m` + `wave_height` fetched | `weather_severity` depends on exactly these two; fetching more wastes API quota and adds irrelevant columns to the model |
| Only occupied 0.5° cells | Fetching a full bounding box would add thousands of ocean/land cells with no AIS data |
| Only Silver date range | Historical weather outside the training window is unused data |
| Per-cell cache | Re-runs do not hit the API; adds < 1 KB/cell |
| One combined parquet | Downstream readers (`load_dim_weather.py`) need a single file; avoids glob complexity |
