"""
train_predictor.py — Maritime Navigation AI System
Trains XGBoost position predictor on Silver TRAIN split.
Predicts next lat/lon delta given current vessel state.

Run after train_anomaly.py:
    docker compose exec producer python src/ml/train_predictor.py
"""
import os, sys, joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/app/src/common")
from config import DELTA_SILVER_PATH, MODELS_PATH

FEATURES = [
    "lat", "lon",
    "sog", "cog", "heading",
    "sog_change", "heading_change",
    "time_delta_sec",
    "hour", "month",
]

PREDICT_MINUTES = [5, 10, 15]  # predict position N minutes ahead


def load_train_data() -> pd.DataFrame:
    """Load train split using chunked sampling to avoid memory crash."""
    parquet_path = Path("/app/data/parquet")
    silver_path  = Path(DELTA_SILVER_PATH)

    load_path = silver_path if silver_path.exists() else parquet_path

    print(f"📂  Loading from: {load_path}")

    all_files = sorted(load_path.rglob("*.parquet"))
    print(f"    Found {len(all_files)} parquet files")

    MAX_ROWS        = 2_000_000
    SAMPLE_PER_FILE = MAX_ROWS // max(len(all_files), 1)

    chunks = []
    total  = 0

    for f in all_files:
        try:
            chunk = pd.read_parquet(f)

            if "data_split" in chunk.columns:
                chunk = chunk[chunk["data_split"] == "train"]

            if len(chunk) == 0:
                continue

            if len(chunk) > SAMPLE_PER_FILE:
                chunk = chunk.sample(n=SAMPLE_PER_FILE, random_state=42)

            chunks.append(chunk)
            total += len(chunk)
            print(f"    Loaded {total:,} rows so far...")

            if total >= MAX_ROWS:
                break

        except Exception as e:
            print(f"    Skipping {f.name}: {e}")
            continue

    if not chunks:
        raise FileNotFoundError("No train data found")

    df = pd.concat(chunks, ignore_index=True)
    df["base_datetime"] = pd.to_datetime(df["base_datetime"], errors="coerce")
    df = df.sort_values(["mmsi", "base_datetime"])
    print(f"✅  Train data: {len(df):,} rows, {df['mmsi'].nunique():,} vessels")
    return df

def build_targets(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    Build prediction targets: delta_lat and delta_lon
    N minutes into the future for each vessel.
    """
    # Approximate rows to skip (1 row per ~10 seconds)
    # steps = max(1, int(minutes * 60 / 10))
    steps = max(1, int(minutes * 60 / 28))

    df = df.copy()
    w = df.groupby("mmsi")

    df["next_lat"] = w["lat"].shift(-steps)
    df["next_lon"] = w["lon"].shift(-steps)
    df["delta_lat"] = df["next_lat"] - df["lat"]
    df["delta_lon"] = df["next_lon"] - df["lon"]

    # Remove rows with no future position
    df = df.dropna(subset=["delta_lat", "delta_lon"])

    # Remove huge jumps (data errors)
    df = df[
        (df["delta_lat"].abs() < 1.0) &
        (df["delta_lon"].abs() < 1.0)
    ]
    return df


def train_xgboost(X, y_lat, y_lon, label: str):
    """Train two XGBoost regressors: one for lat, one for lon."""
    try:
        import xgboost as xgb
    except ImportError:
        print("⚠️  XGBoost not installed — pip install xgboost")
        return None, None

    print(f"\n🔨  Training XGBoost ({label})")
    print(f"    Samples : {len(X):,}")

    model_lat = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model_lon = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    # Train/val split (80/20 within train data)
    split = int(len(X) * 0.8)
    X_tr, X_val  = X[:split], X[split:]
    yl_tr, yl_val = y_lat[:split], y_lat[split:]
    yn_tr, yn_val = y_lon[:split], y_lon[split:]

    model_lat.fit(X_tr, yl_tr,
                  eval_set=[(X_val, yl_val)],
                  verbose=False)
    model_lon.fit(X_tr, yn_tr,
                  eval_set=[(X_val, yn_val)],
                  verbose=False)

    # Evaluate
    pred_lat = model_lat.predict(X_val)
    pred_lon = model_lon.predict(X_val)

    # Mean absolute error in nautical miles (~1° ≈ 60 nm)
    lat_err_nm = np.mean(np.abs(pred_lat - yl_val)) * 60
    lon_err_nm = np.mean(np.abs(pred_lon - yn_val)) * 60
    total_err  = np.sqrt(lat_err_nm**2 + lon_err_nm**2)

    print(f"    Lat error : {lat_err_nm:.3f} nm")
    print(f"    Lon error : {lon_err_nm:.3f} nm")
    print(f"    Total MAE : {total_err:.3f} nm")

    return model_lat, model_lon


def main():
    start = datetime.now()
    print("=" * 55)
    print("  Maritime AI — Position Predictor Training")
    print("=" * 55)

    df = load_train_data()

    feat_cols = [c for c in FEATURES if c in df.columns]
    Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)

    for minutes in PREDICT_MINUTES:
        print(f"\n── {minutes}-minute prediction ──")
        df_t = build_targets(df, minutes)

        if len(df_t) < 1000:
            print(f"  ⚠️  Not enough samples ({len(df_t)}) — skipping")
            continue

        X     = df_t[feat_cols].fillna(0).values
        y_lat = df_t["delta_lat"].values
        y_lon = df_t["delta_lon"].values

        model_lat, model_lon = train_xgboost(
            X, y_lat, y_lon, f"{minutes}min"
        )

        if model_lat is not None:
            joblib.dump(model_lat,
                f"{MODELS_PATH}/xgb_lat_{minutes}min.pkl")
            joblib.dump(model_lon,
                f"{MODELS_PATH}/xgb_lon_{minutes}min.pkl")
            print(f"  💾  Saved xgb_lat_{minutes}min.pkl")

    joblib.dump(feat_cols, f"{MODELS_PATH}/predictor_features.pkl")

    elapsed = (datetime.now() - start).seconds
    print(f"\n✅  Predictor models saved to {MODELS_PATH}/")
    print(f"    Training time: {elapsed}s")
    print(f"\n    Next: python src/ml/train_congestion.py")


if __name__ == "__main__":
    main()
