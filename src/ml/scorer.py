"""
scorer.py — Maritime Navigation AI System
Loads trained ML models and scores live vessel records.
Used by the Streamlit dashboard and FastAPI to score
every new position received from Kafka in real time.
"""
import os, sys, joblib
import pandas as pd
import numpy as np
import math
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/app/src/common")
from config import MODELS_PATH, COLLISION_DISTANCE_NM
from schema_utils import haversine_nm, bearing_degrees, predict_position_dr


class LiveScorer:
    """
    Scores live vessel positions with all trained ML models.
    Load once at startup, score every incoming batch.
    """

    def __init__(self):
        self.anomaly_model    = None
        self.anomaly_scaler   = None
        self.anomaly_features = None
        self.xgb_lat          = {}   # {minutes: model}
        self.xgb_lon          = {}
        self.predictor_features = None
        self.congestion_model = None
        self.congestion_enc   = None
        self.congestion_feats = None
        self._load_all()

    def _load_all(self):
        """Load all saved models. Gracefully handles missing models."""
        mp = Path(MODELS_PATH)

        # Anomaly model
        try:
            self.anomaly_model    = joblib.load(mp / "isolation_forest_v2.pkl")
            self.anomaly_scaler   = joblib.load(mp / "scaler_anomaly_v2.pkl")
            self.anomaly_features = joblib.load(mp / "anomaly_features_v2.pkl")
            print("✅  Anomaly model loaded")
        except Exception as e:
            print(f"⚠️   Anomaly model not found: {e}")

        # Position predictor
        try:
            self.predictor_features = joblib.load(
                mp / "predictor_features.pkl")
            for m in [5, 10, 15]:
                lat_f = mp / f"xgb_lat_{m}min.pkl"
                lon_f = mp / f"xgb_lon_{m}min.pkl"
                if lat_f.exists() and lon_f.exists():
                    self.xgb_lat[m] = joblib.load(lat_f)
                    self.xgb_lon[m] = joblib.load(lon_f)
            print(f"✅  Predictor models loaded: "
                  f"{list(self.xgb_lat.keys())} min")
        except Exception as e:
            print(f"⚠️   Predictor models not found: {e}")

        # Congestion model
        try:
            self.congestion_model = joblib.load(mp / "congestion_rf.pkl")
            self.congestion_enc   = joblib.load(mp / "congestion_encoder.pkl")
            self.congestion_feats = joblib.load(mp / "congestion_features.pkl")
            print("✅  Congestion model loaded")
        except Exception as e:
            print(f"⚠️   Congestion model not found: {e}")

    def _compute_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sog_change, heading_change, distance_nm per vessel if not already present."""
        need_sog  = "sog_change"    not in df.columns
        need_hdg  = "heading_change" not in df.columns
        need_dist = "distance_nm"   not in df.columns
        if not need_sog and not need_hdg and not need_dist:
            return df

        df = df.copy()
        ts_col = next(
            (c for c in ("timestamp", "ts", "base_date_time") if c in df.columns),
            None,
        )
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.sort_values(["mmsi", ts_col])

        grp = df.groupby("mmsi", sort=False)
        if need_sog:
            df["sog_change"] = grp["sog"].diff().fillna(0)
        if need_hdg:
            raw_diff = grp["heading"].diff().fillna(0)
            df["heading_change"] = ((raw_diff + 180) % 360 - 180).abs()
        if need_dist and {"lat", "lon"}.issubset(df.columns):
            lat1 = grp["lat"].shift(1)
            lon1 = grp["lon"].shift(1)
            valid = ~lat1.isna()
            R_nm  = 3440.065
            rlat1 = np.radians(lat1.values)
            rlat2 = np.radians(df["lat"].values)
            rlon1 = np.radians(lon1.values)
            rlon2 = np.radians(df["lon"].values)
            dlat  = rlat2 - rlat1
            dlon  = rlon2 - rlon1
            a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
            dist  = R_nm * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            df["distance_nm"] = np.where(valid, dist, 0.0)
        elif need_dist:
            df["distance_nm"] = 0.0
        return df

    def score_anomaly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score each vessel position for anomaly.
        Adds: is_anomaly, anomaly_score, anomaly_type columns.
        """
        df = self._compute_delta_features(df)
        df["is_anomaly"]   = False
        df["anomaly_score"] = 0.0
        df["anomaly_type"]  = ""

        # Rule-based (always runs, no model needed)
        # Sudden stop
        if "sog_change" in df.columns:
            mask = df["sog_change"].fillna(0) < -5.0
            df.loc[mask, "is_anomaly"]   = True
            df.loc[mask, "anomaly_score"] = 0.8
            df.loc[mask, "anomaly_type"]  = "SUDDEN_STOP"

        # Unusual speed
        mask = df["sog"] > 30.0
        df.loc[mask, "is_anomaly"]   = True
        df.loc[mask, "anomaly_score"] = df.loc[mask, "sog"] / 60.0
        df.loc[mask, "anomaly_type"]  = "UNUSUAL_SPEED"

        # Sharp turn
        if "heading_change" in df.columns:
            mask = (df["heading_change"].fillna(0) > 45) & (df["sog"] > 2)
            df.loc[mask, "is_anomaly"]   = True
            df.loc[mask, "anomaly_score"] = \
                (df.loc[mask, "heading_change"] / 180).clip(0, 1)
            df.loc[mask, "anomaly_type"]  = "SHARP_TURN"

        # Stationary inside US port geofences or restricted shipping channels
        if "in_us_port_zone" in df.columns:
            mask = (df["sog"] < 0.5) & (df["in_us_port_zone"] == True)
            df.loc[mask, "is_anomaly"]    = True
            df.loc[mask, "anomaly_score"]  = 0.9
            df.loc[mask, "anomaly_type"]   = "STATIONARY_RISK"

        # ML-based (if model available)
        if self.anomaly_model is not None:
            feat_cols = [c for c in self.anomaly_features
                         if c in df.columns]
            X = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
            X_scaled = self.anomaly_scaler.transform(X)
            preds  = self.anomaly_model.predict(X_scaled)   # -1 or 1
            scores = self.anomaly_model.score_samples(X_scaled)
            norm   = 1 - (scores - scores.min()) / (
                scores.max() - scores.min() + 1e-9)

            ml_mask = preds == -1
            # Only override if ML score is higher than rule score
            upgrade = ml_mask & (norm > df["anomaly_score"])
            df.loc[upgrade, "is_anomaly"]   = True
            df.loc[upgrade, "anomaly_score"] = norm[upgrade]
            df.loc[upgrade & (df["anomaly_type"] == ""),
                   "anomaly_type"] = "ML_ANOMALY"

        return df

    def predict_position(self, lat: float, lon: float,
                         sog: float, cog: float,
                         heading: float,
                         minutes: int = 10) -> dict:
        """
        Predict vessel position N minutes ahead.
        Uses XGBoost if available, else dead reckoning.
        """
        # Try XGBoost first
        if minutes in self.xgb_lat and self.predictor_features:
            try:
                feat_vals = {
                    "lat": lat, "lon": lon,
                    "sog": sog, "cog": cog, "heading": heading,
                    "sog_change": 0, "heading_change": 0,
                    "time_delta_sec": minutes * 60,
                    "hour": datetime.utcnow().hour,
                    "month": datetime.utcnow().month,
                }
                X = np.array([[feat_vals.get(f, 0)
                               for f in self.predictor_features]])
                d_lat = float(self.xgb_lat[minutes].predict(X)[0])
                d_lon = float(self.xgb_lon[minutes].predict(X)[0])
                return {
                    "predicted_lat": round(lat + d_lat, 6),
                    "predicted_lon": round(lon + d_lon, 6),
                    "method":        "xgboost",
                    "minutes_ahead": minutes,
                    "confidence":    0.82,
                }
            except Exception:
                pass

        # Fall back to dead reckoning
        pred_lat, pred_lon = predict_position_dr(
            lat, lon, sog, heading, minutes
        )
        return {
            "predicted_lat": pred_lat,
            "predicted_lon": pred_lon,
            "method":        "dead_reckoning",
            "minutes_ahead": minutes,
            "confidence":    0.65,
        }

    def detect_collisions(self, df: pd.DataFrame) -> list:
        """
        Detect collision risks between nearby vessels.
        Returns list of alert dicts.
        """
        _THRESHOLD_NM = 0.1   # main detection radius

        risks  = []
        latest = df.drop_duplicates("mmsi", keep="last")

        for i, v1 in latest.iterrows():
            for j, v2 in latest.iterrows():
                if i >= j:
                    continue                           # avoid duplicates / self
                if str(v1["mmsi"]) == str(v2["mmsi"]):
                    continue                           # same vessel guard
                if v1.get("sog", 0) < 1.0 or v2.get("sog", 0) < 1.0:
                    continue                           # skip stationary vessels

                dist = haversine_nm(
                    v1["lat"], v1["lon"],
                    v2["lat"], v2["lon"]
                )

                if dist == 0.0:
                    continue                           # exact same position — bad data
                if dist > _THRESHOLD_NM * 10:
                    continue                           # skip distant pairs

                # Check convergence
                bearing = bearing_degrees(
                    v1["lat"], v1["lon"],
                    v2["lat"], v2["lon"]
                )
                h_diff = abs(v1.get("heading", 0) - bearing)
                if h_diff > 180:
                    h_diff = 360 - h_diff
                converging = h_diff < 45

                # Classify risk within the 0.1 nm detection zone
                if dist < _THRESHOLD_NM * 0.5:   # < 0.05 nm
                    severity = "CRITICAL"
                elif dist < _THRESHOLD_NM and converging:
                    severity = "HIGH"
                elif dist < _THRESHOLD_NM:
                    severity = "MEDIUM"
                else:
                    continue

                risks.append({
                    "alert_type":   "COLLISION_RISK",
                    "severity":     severity,
                    "mmsi_1":       v1["mmsi"],
                    "mmsi_2":       v2["mmsi"],
                    "vessel_name_1":v1.get("vessel_name", ""),
                    "vessel_name_2":v2.get("vessel_name", ""),
                    "distance_nm":  round(dist, 3),
                    "converging":   converging,
                    "lat":          (v1["lat"] + v2["lat"]) / 2,
                    "lon":          (v1["lon"] + v2["lon"]) / 2,
                    "description":  (
                        f"Vessels {v1['mmsi']} and {v2['mmsi']} "
                        f"are {dist:.2f} nm apart"
                        + (" and converging" if converging else "")
                    ),
                    "detected_at":  datetime.utcnow().isoformat(),
                })

        return risks

    def predict_congestion(self, lat_bin: float,
                           lon_bin: float,
                           vessel_count: int,
                           avg_sog: float,
                           stopped_count: int) -> str:
        """Predict congestion level for a grid cell."""
        if self.congestion_model is None:
            # Rule-based fallback
            if vessel_count >= 15:
                return "HIGH"
            if vessel_count >= 5:
                return "MEDIUM"
            return "LOW"

        now = datetime.utcnow()
        feat_vals = {
            "vessel_count":  vessel_count,
            "avg_sog":       avg_sog,
            "stopped_count": stopped_count,
            "hour":          now.hour,
            "day_of_week":   now.weekday(),
            "is_weekend":    int(now.weekday() >= 5),
            "lat_bin":       lat_bin,
            "lon_bin":       lon_bin,
        }
        X = np.array([[feat_vals.get(f, 0)
                       for f in self.congestion_feats]])
        pred = self.congestion_model.predict(X)[0]
        return self.congestion_enc.inverse_transform([pred])[0]


# Singleton — load once
_scorer = None

def get_scorer() -> LiveScorer:
    global _scorer
    if _scorer is None:
        _scorer = LiveScorer()
    return _scorer
