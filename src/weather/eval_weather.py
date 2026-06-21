"""
eval_weather.py — Weather Feature Evaluation
Compares a baseline congestion classifier against an augmented one that
adds weather_severity as an additional feature.

BOTH models use the same chronological (time-based) 80/20 train/test split —
earlier dates for training, latest dates for testing.  The split cutoff is
printed so you can verify it.

Saves ONLY the augmented model to models/congestion_rf_weather.pkl.
NEVER overwrites models/congestion_rf.pkl (the production baseline).

Usage:
    docker compose exec producer python src/weather/eval_weather.py
"""
from __future__ import annotations
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Resolve app paths for both Docker and host contexts ───────────────────────
sys.path.insert(0, "/app/src/common")
sys.path.insert(0, "/app/src")

from config import MODELS_PATH, PARQUET_DATA_PATH
from ml.train_congestion import (
    FEATURES,
    load_and_aggregate,
    add_future_target,
    time_based_split,
    congestion_label,
)
from weather.weather_features import build_congestion_with_weather

_MODELS_DIR    = Path(MODELS_PATH)
_AUGMENTED_PKL = _MODELS_DIR / "congestion_rf_weather.pkl"   # NEW path — safe
_BASELINE_PKL  = _MODELS_DIR / "congestion_rf.pkl"           # NEVER overwritten

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [eval_weather] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Augmented feature set: baseline + weather severity only.
WEATHER_FEATURE = "weather_severity"
AUG_FEATURES    = FEATURES + [WEATHER_FEATURE]


def _train_rf(X: np.ndarray, y: np.ndarray):
    """Thin wrapper — same hyper-params as the production baseline."""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def _evaluate(model, X: np.ndarray, y: np.ndarray, le, label: str) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    preds  = model.predict(X)
    acc    = float(accuracy_score(y, preds))
    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    report = classification_report(
        y, preds, target_names=le.classes_, zero_division=0,
    )
    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"{'─' * 55}")
    print(report)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {macro_f1:.4f}")
    return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4)}


def main() -> None:
    from sklearn.preprocessing import LabelEncoder

    start_ts = datetime.utcnow()
    print("=" * 60)
    print("  Maritime AI — Weather Feature Evaluation")
    print("=" * 60)

    # ── 1. Load base density grid (same as train_congestion.py) ───────────────
    density, data_layer = load_and_aggregate()
    density = add_future_target(density)

    if len(density) < 200:
        raise RuntimeError(
            f"Only {len(density)} consecutive-hour rows — too few to evaluate. "
            "Ensure PARQUET_DATA_PATH has multi-day data."
        )

    # ── 2. Time-based split (80 % train / 20 % test) ──────────────────────────
    # Both models use IDENTICAL split so the comparison is fair.
    train_base, test_base = time_based_split(density, test_frac=0.20)

    print(f"\n  Train date range : {train_base['hour_bucket'].min()} → {train_base['hour_bucket'].max()}")
    print(f"  Test  date range : {test_base['hour_bucket'].min()}  → {test_base['hour_bucket'].max()}")
    print(f"  Train rows       : {len(train_base):,}")
    print(f"  Test  rows       : {len(test_base):,}")

    # ── 3. Encode labels ──────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(density["future_congestion"])

    # ── 4. Baseline model (no weather) ────────────────────────────────────────
    base_feats = [c for c in FEATURES if c in train_base.columns]
    X_tr_b = train_base[base_feats].fillna(0.0).values
    X_te_b = test_base[base_feats].fillna(0.0).values
    y_tr   = le.transform(train_base["future_congestion"])
    y_te   = le.transform(test_base["future_congestion"])

    print(f"\n  Classes : {list(le.classes_)}")
    print(f"  Baseline features ({len(base_feats)}): {base_feats}")

    print("\n[Baseline] Training …")
    baseline_model = _train_rf(X_tr_b, y_tr)
    base_metrics   = _evaluate(baseline_model, X_te_b, y_te, le, "BASELINE (no weather)")

    # ── 5. Augmented model (+ weather_severity) ───────────────────────────────
    print("\n[Augmented] Joining weather data from dim_weather …")
    train_aug = build_congestion_with_weather(train_base)
    test_aug  = build_congestion_with_weather(test_base)

    # Fill NaN weather_severity with 0.0: rows with no weather data are treated
    # as "no weather impact", which is a conservative and semantically correct default.
    train_aug["weather_severity"] = train_aug["weather_severity"].fillna(0.0)
    test_aug["weather_severity"]  = test_aug["weather_severity"].fillna(0.0)

    aug_feats = [c for c in AUG_FEATURES if c in train_aug.columns]
    missing   = [c for c in AUG_FEATURES if c not in train_aug.columns]
    if missing:
        print(f"  Warning: augmented features not available: {missing}")

    X_tr_a = train_aug[aug_feats].fillna(0.0).values
    X_te_a = test_aug[aug_feats].fillna(0.0).values

    print(f"  Augmented features ({len(aug_feats)}): {aug_feats}")

    # Report weather coverage on the test set.
    w_col = test_aug.get("weather_severity", pd.Series(dtype=float))
    n_nonzero = int((w_col > 0).sum())
    print(
        f"  Weather coverage on test: {n_nonzero}/{len(test_aug)} rows "
        f"with severity > 0  ({100.0*n_nonzero/max(len(test_aug),1):.1f}%)"
    )

    print("\n[Augmented] Training …")
    aug_model    = _train_rf(X_tr_a, y_tr)
    aug_metrics  = _evaluate(aug_model, X_te_a, y_te, le, "AUGMENTED (+ weather_severity)")

    # ── 6. Delta report ───────────────────────────────────────────────────────
    acc_delta = aug_metrics["accuracy"] - base_metrics["accuracy"]
    f1_delta  = aug_metrics["macro_f1"] - base_metrics["macro_f1"]

    print(f"\n{'═' * 55}")
    print("  COMPARISON SUMMARY")
    print(f"{'═' * 55}")
    print(f"  {'Metric':<20} {'Baseline':>10} {'Augmented':>10} {'Delta':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8}")
    print(
        f"  {'Accuracy':<20} {base_metrics['accuracy']:>10.4f} "
        f"{aug_metrics['accuracy']:>10.4f} {acc_delta:>+8.4f}"
    )
    print(
        f"  {'Macro F1':<20} {base_metrics['macro_f1']:>10.4f} "
        f"{aug_metrics['macro_f1']:>10.4f} {f1_delta:>+8.4f}"
    )
    print()

    if abs(f1_delta) < 0.005:
        print(
            "  ⚠  Weather adds little to macro-F1 (|Δ| < 0.005). "
            "This is expected when:\n"
            "     • The date range is too short for weather to vary meaningfully.\n"
            "     • Most density is driven by port schedules, not weather.\n"
            "     • Weather fetch coverage was low (check dim_weather row count).\n"
            "  The augmented model is saved anyway as a challenger for future evaluation."
        )
    elif f1_delta > 0:
        print(
            f"  ✓  Weather improves macro-F1 by {f1_delta:+.4f} "
            "— augmented model is worth promoting."
        )
    else:
        print(
            f"  ✗  Weather hurts macro-F1 by {f1_delta:+.4f} "
            "— do NOT promote the augmented model."
        )

    # Feature importances for augmented model.
    importance = sorted(
        zip(aug_feats, aug_model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\n  Feature importances (augmented model, descending):")
    for feat, imp in importance:
        bar = "█" * max(1, int(imp * 40))
        print(f"    {feat:<22} {bar} {imp:.4f}")

    # ── 7. Save augmented model — NEVER touch congestion_rf.pkl ──────────────
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(aug_model, _AUGMENTED_PKL)
    joblib.dump(le,        _MODELS_DIR / "congestion_encoder_weather.pkl")
    joblib.dump(aug_feats, _MODELS_DIR / "congestion_features_weather.pkl")

    print(f"\n  Augmented model  → {_AUGMENTED_PKL}")
    print(f"  Baseline model     {_BASELINE_PKL}  (untouched)")
    assert not _BASELINE_PKL.samefile(_AUGMENTED_PKL), "Path collision guard"

    # ── 8. Persist evaluation report ─────────────────────────────────────────
    artifacts_dir = Path("/app/mlops/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path   = artifacts_dir / "evaluation_report_weather.json"

    report = {
        "generated_at":     datetime.utcnow().isoformat(),
        "data_layer":       data_layer,
        "split_type":       "time_based_chronological_80_20",
        "train_date_range": [
            str(train_base["hour_bucket"].min()),
            str(train_base["hour_bucket"].max()),
        ],
        "test_date_range":  [
            str(test_base["hour_bucket"].min()),
            str(test_base["hour_bucket"].max()),
        ],
        "train_samples":    int(len(X_tr_b)),
        "test_samples":     int(len(X_te_b)),
        "weather_coverage_test_pct": round(
            100.0 * n_nonzero / max(len(test_aug), 1), 2
        ),
        "baseline": {
            "features":  base_feats,
            "accuracy":  base_metrics["accuracy"],
            "macro_f1":  base_metrics["macro_f1"],
            "model_path": str(_BASELINE_PKL),
        },
        "augmented": {
            "features":       aug_feats,
            "accuracy":       aug_metrics["accuracy"],
            "macro_f1":       aug_metrics["macro_f1"],
            "model_path":     str(_AUGMENTED_PKL),
            "weather_feature": WEATHER_FEATURE,
        },
        "delta": {
            "accuracy": round(acc_delta, 4),
            "macro_f1": round(f1_delta, 4),
        },
        "feature_importances_augmented": {
            feat: round(float(imp), 6) for feat, imp in importance
        },
    }

    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Evaluation report → {report_path}")

    elapsed = (datetime.utcnow() - start_ts).seconds
    print(f"\n  Total time: {elapsed}s")
    print("Done.")


if __name__ == "__main__":
    main()
