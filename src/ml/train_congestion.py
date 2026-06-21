"""
train_congestion.py — Maritime Navigation AI System
Trains Random Forest to predict congestion level (LOW/MEDIUM/HIGH)
in a grid cell at a future time.

Run after train_predictor.py:
    docker compose exec producer python src/ml/train_congestion.py
"""
from pathlib import Path
import os, sys, joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/app/src/common")
from config import DELTA_GOLD_DENSITY_PATH, MODELS_PATH

FEATURES = [
    "vessel_count", "avg_sog", "stopped_count",
    "hour", "day_of_week", "is_weekend",
    "lat_bin", "lon_bin",
]


def load_density_data() -> pd.DataFrame:
    """
    Build density features from Parquet data directly.
    Groups by grid cell and hour to create congestion training data.
    """
    parquet_path = Path("/app/data/parquet")
    gold_path    = Path(DELTA_GOLD_DENSITY_PATH)

    # Try Gold Delta first
    if gold_path.exists():
        print(f"📂  Loading from Gold Delta: {gold_path}")
        df = pd.read_parquet(gold_path)
        if "data_split" in df.columns:
            df = df[df["data_split"] == "train"].copy()
        print(f"✅  Density data: {len(df):,} rows")
        return df

    # Fall back to building from Parquet
    print(f"📂  Building density from Parquet: {parquet_path}")

    all_files = sorted(parquet_path.rglob("*.parquet"))
    MAX_ROWS        = 1_000_000
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
        raise FileNotFoundError("No data found")

    df = pd.concat(chunks, ignore_index=True)
    df["base_datetime"] = pd.to_datetime(df["base_datetime"], errors="coerce")

    # Build density grid from raw positions
    print(f"    Building density grid from {len(df):,} rows...")
    df["hour_bucket"] = df["base_datetime"].dt.floor("H")
    df["lat_bin"]     = (df["lat"] / 0.1).astype(int) * 0.1
    df["lon_bin"]     = (df["lon"] / 0.1).astype(int) * 0.1

    density = (
        df.groupby(["lat_bin", "lon_bin", "hour_bucket"])
        .agg(
            vessel_count  = ("mmsi", "nunique"),
            avg_sog       = ("sog",  "mean"),
            stopped_count = ("sog",  lambda x: (x < 0.5).sum()),
        )
        .reset_index()
    )

    density["hour"]        = density["hour_bucket"].dt.hour
    density["day_of_week"] = density["hour_bucket"].dt.dayofweek
    density["is_weekend"]  = (density["day_of_week"] >= 5).astype(int)

    density["congestion_level"] = density["vessel_count"].apply(
        lambda n: "HIGH" if n >= 15 else ("MEDIUM" if n >= 5 else "LOW")
    )

    print(f"✅  Density data: {len(density):,} grid-hour cells")
    print(f"    Congestion dist:\n{density['congestion_level'].value_counts()}")
    return density


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    start = datetime.now()
    print("=" * 55)
    print("  Maritime AI — Congestion Classifier Training")
    print("=" * 55)

    df = load_density_data()

    feat_cols = [c for c in FEATURES if c in df.columns]
    X = df[feat_cols].fillna(0).values

    le = LabelEncoder()
    y  = le.fit_transform(df["congestion_level"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n🔨  Training Random Forest Congestion Classifier")
    print(f"    Samples  : {len(X_tr):,} train / {len(X_te):,} test")
    print(f"    Features : {feat_cols}")
    print(f"    Classes  : {le.classes_}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    print(f"\n📊  Classification Report:")
    print(classification_report(y_te, preds,
                                target_names=le.classes_,
                                zero_division=0))

    # Feature importance
    importance = sorted(
        zip(feat_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("📊  Feature Importance:")
    for feat, imp in importance:
        bar = "█" * int(imp * 40)
        print(f"    {feat:<20} {bar} {imp:.3f}")

    Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)
    joblib.dump(model,     f"{MODELS_PATH}/congestion_rf.pkl")
    joblib.dump(le,        f"{MODELS_PATH}/congestion_encoder.pkl")
    joblib.dump(feat_cols, f"{MODELS_PATH}/congestion_features.pkl")

    elapsed = (datetime.now() - start).seconds
    print(f"\n✅  Congestion model saved to {MODELS_PATH}/")
    print(f"    Training time: {elapsed}s")
    print(f"\n    All models trained! Start the dashboard:")
    print(f"    docker compose exec producer "
          f"python src/producer/kafka_producer.py")


if __name__ == "__main__":
    main()
