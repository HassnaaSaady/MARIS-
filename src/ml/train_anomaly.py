"""
train_anomaly.py — Maritime Navigation AI System
Trains Isolation Forest anomaly detector on Silver AIS clean data.
Saves model to models/isolation_forest_v2.pkl.

Run after silver_job.py:
    docker compose exec producer python src/ml/train_anomaly.py
"""
import sys, joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/app/src/common")
from config import MODELS_PATH

SILVER_PATH = Path("/delta/silver/ais_clean")

FEATURES = ["sog", "cog", "heading", "sog_change", "heading_change", "distance_nm"]

CONTAMINATION = 0.002
N_ESTIMATORS  = 200


def haversine_nm(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Return great-circle distance in nautical miles between consecutive points."""
    R_nm = 3440.065
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R_nm * 2 * np.arcsin(np.sqrt(a))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sog_change, heading_change, distance_nm per vessel track."""
    df = df.copy()

    ts_col = next((c for c in ("timestamp", "ts", "base_date_time") if c in df.columns), None)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.sort_values(["mmsi", ts_col])
    else:
        df = df.sort_values("mmsi")

    grp = df.groupby("mmsi", sort=False)

    df["sog_change"] = grp["sog"].diff().fillna(0)

    raw_hdg_diff = grp["heading"].diff().fillna(0)
    df["heading_change"] = ((raw_hdg_diff + 180) % 360 - 180).abs()

    if {"lat", "lon"}.issubset(df.columns):
        lat1 = grp["lat"].shift(1)
        lon1 = grp["lon"].shift(1)
        dist = haversine_nm(
            lat1.values, lon1.values,
            df["lat"].values, df["lon"].values,
        )
        dist = np.where(lat1.isna(), 0.0, dist)
        df["distance_nm"] = dist
    else:
        df["distance_nm"] = 0.0

    return df


def load_silver_data() -> pd.DataFrame:
    """Load all parquet files from /delta/silver/ais_clean."""
    if not SILVER_PATH.exists():
        raise FileNotFoundError(f"Silver path not found: {SILVER_PATH}")

    all_files = sorted(SILVER_PATH.rglob("*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No parquet files under {SILVER_PATH}")

    print(f"Loading from: {SILVER_PATH}  ({len(all_files)} files)")

    MAX_ROWS        = 2_000_000
    sample_per_file = MAX_ROWS // len(all_files)
    chunks, total   = [], 0

    for f in all_files:
        try:
            chunk = pd.read_parquet(f)
            if "data_split" in chunk.columns:
                chunk = chunk[chunk["data_split"] == "train"]
            if chunk.empty:
                continue
            if len(chunk) > sample_per_file:
                chunk = chunk.sample(n=sample_per_file, random_state=42)
            chunks.append(chunk)
            total += len(chunk)
            if total >= MAX_ROWS:
                break
        except Exception as e:
            print(f"  Skipping {f.name}: {e}")

    if not chunks:
        raise RuntimeError("No usable train rows found in silver data")

    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df):,} rows, {df['mmsi'].nunique():,} vessels")
    return df


def train_isolation_forest(df: pd.DataFrame):
    """Train Isolation Forest on the six engineered features."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    feat_cols = [c for c in FEATURES if c in df.columns]
    X = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)

    print(f"\nTraining Isolation Forest")
    print(f"  Samples      : {len(X):,}")
    print(f"  Features     : {feat_cols}")
    print(f"  Contamination: {CONTAMINATION}")
    print(f"  n_estimators : {N_ESTIMATORS}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=CONTAMINATION,
        n_estimators=N_ESTIMATORS,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    return model, scaler, feat_cols


def main():
    start = datetime.now()
    print("=" * 55)
    print("  Maritime AI — Anomaly Detector Training v2")
    print("=" * 55)

    df = load_silver_data()
    df = engineer_features(df)

    model, scaler, features = train_isolation_forest(df)

    out = Path(MODELS_PATH)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,    out / "isolation_forest_v2.pkl")
    joblib.dump(scaler,   out / "scaler_anomaly_v2.pkl")
    joblib.dump(features, out / "anomaly_features_v2.pkl")

    elapsed = (datetime.now() - start).seconds
    print(f"\nModels saved to {out}/")
    print(f"Training time  : {elapsed}s")
    print(f"\nNext: python src/ml/train_predictor.py")


if __name__ == "__main__":
    main()
