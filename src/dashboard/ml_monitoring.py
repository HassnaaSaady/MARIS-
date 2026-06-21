"""
ml_monitoring.py — Maritime Navigation AI System
Standalone Streamlit page for MLflow experiment monitoring.

Shows: experiments, run history, metric trends, model registry versions,
       feature importance charts, and latest evaluation report.

Run:
    streamlit run src/dashboard/ml_monitoring.py

Behaviour:
  - MLflow available + reachable → live data from tracking server
  - MLflow unavailable or not configured → mock data with info banner
  - evaluation_report.json exists → shows latest evaluation metrics

Does NOT modify streamlit_app.py.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
for _p in [
    "/app/src/common", "/app/src",
    str(_PROJECT_ROOT / "src" / "common"),
    str(_PROJECT_ROOT / "src"),
    str(_PROJECT_ROOT / "mlops"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from configs.mlflow_config import (
        is_mlflow_available, get_tracking_uri, ARTIFACTS_DIR,
        ANOMALY_EXPERIMENT, CONGESTION_EXPERIMENT, PREDICTOR_EXPERIMENT,
        ANOMALY_MODEL_NAME, CONGESTION_MODEL_NAME, PREDICTOR_MODEL_NAME,
    )
    from model_registry.model_loader import summarise as model_summary
    _MLOPS_IMPORT_OK = True
except ImportError:
    _MLOPS_IMPORT_OK = False
    def is_mlflow_available(): return False
    ARTIFACTS_DIR       = _PROJECT_ROOT / "mlops" / "artifacts"
    ANOMALY_EXPERIMENT  = "maritime-anomaly-detector"
    CONGESTION_EXPERIMENT = "maritime-congestion-classifier"
    PREDICTOR_EXPERIMENT  = "maritime-position-predictor"
    ANOMALY_MODEL_NAME    = "maritime-anomaly-detector"
    CONGESTION_MODEL_NAME = "maritime-congestion-classifier"
    PREDICTOR_MODEL_NAME  = "maritime-position-predictor"
    def model_summary(): return {}

logger = logging.getLogger(__name__)

USE_MLFLOW = _MLOPS_IMPORT_OK and is_mlflow_available()

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
EXP_COLORS = {
    ANOMALY_EXPERIMENT:    "#EF4444",
    CONGESTION_EXPERIMENT: "#F59E0B",
    PREDICTOR_EXPERIMENT:  "#3B82F6",
    "maritime-model-evaluation": "#10B981",
}

# ---------------------------------------------------------------------------
# MLflow data fetchers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_experiments() -> list:
    if not USE_MLFLOW:
        return []
    try:
        import mlflow
        mlflow.set_tracking_uri(get_tracking_uri())
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        exps = client.search_experiments()
        return [{"id": e.experiment_id, "name": e.name,
                 "lifecycle": e.lifecycle_stage} for e in exps]
    except Exception as exc:
        logger.warning("fetch_experiments: %s", exc)
        return []


@st.cache_data(ttl=60, show_spinner=False)
def fetch_runs(experiment_name: str, max_results: int = 100) -> pd.DataFrame:
    if not USE_MLFLOW:
        return pd.DataFrame()
    try:
        import mlflow
        mlflow.set_tracking_uri(get_tracking_uri())
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        exp    = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return pd.DataFrame()
        runs = client.search_runs(
            exp.experiment_id,
            order_by=["start_time DESC"],
            max_results=max_results,
        )
        rows = []
        for r in runs:
            row = {
                "run_id":    r.info.run_id[:8],
                "run_name":  r.info.run_name or r.info.run_id[:8],
                "status":    r.info.status,
                "start_time":datetime.fromtimestamp(r.info.start_time / 1000)
                             if r.info.start_time else None,
                **{f"p_{k}": v for k, v in r.data.params.items()},
                **{f"m_{k}": v for k, v in r.data.metrics.items()},
            }
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning("fetch_runs(%s): %s", experiment_name, exc)
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_registered_models() -> list:
    if not USE_MLFLOW:
        return []
    try:
        from mlflow.tracking import MlflowClient
        import mlflow
        mlflow.set_tracking_uri(get_tracking_uri())
        client = MlflowClient()
        models = client.search_registered_models()
        result = []
        for m in models:
            versions = client.get_latest_versions(m.name)
            for v in versions:
                result.append({
                    "name":          m.name,
                    "version":       v.version,
                    "stage":         v.current_stage,
                    "run_id":        v.run_id[:8] if v.run_id else "—",
                    "created_at":    datetime.fromtimestamp(
                                         v.creation_timestamp / 1000
                                     ).strftime("%Y-%m-%d %H:%M")
                                     if v.creation_timestamp else "—",
                    "description":   v.description or "",
                })
        return result
    except Exception as exc:
        logger.warning("fetch_registered_models: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Mock data (when MLflow is not configured)
# ---------------------------------------------------------------------------

def _mock_runs(experiment: str, n: int = 20) -> pd.DataFrame:
    rows = []
    base = datetime.utcnow()
    for i in range(n):
        t = base - timedelta(hours=i * 6)
        rows.append({
            "run_id":       f"run{i:04d}",
            "run_name":     f"run-{i:03d}",
            "status":       "FINISHED",
            "start_time":   t,
            "m_f1_score":       round(0.72 + (i % 5) * 0.02, 3) if "anomaly" in experiment else None,
            "m_accuracy":       round(0.81 + (i % 4) * 0.02, 3) if "congestion" in experiment else None,
            "m_total_mae_nm":   round(1.2 - i * 0.03, 3) if "predictor" in experiment else None,
            "m_training_time_s":round(45 + i * 2, 1),
            "p_n_estimators":   "200",
            "p_contamination":  "0.01" if "anomaly" in experiment else None,
        })
    return pd.DataFrame(rows)


def _mock_registry() -> list:
    return [
        {"name": ANOMALY_MODEL_NAME,    "version": "2", "stage": "Production",
         "run_id": "abc12345", "created_at": "2025-05-07 10:30", "description": ""},
        {"name": ANOMALY_MODEL_NAME,    "version": "1", "stage": "Staging",
         "run_id": "def67890", "created_at": "2025-05-06 08:00", "description": ""},
        {"name": CONGESTION_MODEL_NAME, "version": "1", "stage": "Production",
         "run_id": "ghi11223", "created_at": "2025-05-07 11:00", "description": ""},
        {"name": f"{PREDICTOR_MODEL_NAME}-5min-lat",  "version": "3",
         "stage": "Production", "run_id": "jkl44556",
         "created_at": "2025-05-07 12:00", "description": ""},
        {"name": f"{PREDICTOR_MODEL_NAME}-10min-lat", "version": "2",
         "stage": "Production", "run_id": "mno77889",
         "created_at": "2025-05-07 12:05", "description": ""},
    ]


# ---------------------------------------------------------------------------
# Evaluation report loader
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120, show_spinner=False)
def load_eval_report() -> dict:
    path = ARTIFACTS_DIR / "evaluation_report.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Feature importance loader
# ---------------------------------------------------------------------------

def load_importance_csv(name: str) -> pd.DataFrame:
    path = ARTIFACTS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# =============================================================================
# Page layout
# =============================================================================

st.set_page_config(
    page_title="ML Monitoring — Maritime AIS",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stMetric"] {
    background:#1C2333; border-radius:10px;
    padding:12px 18px; border:1px solid #2D3748;
}
[data-testid="stMetricLabel"] { color:#A0AEC0 !important; font-size:13px; }
[data-testid="stMetricValue"] { color:#FFFFFF !important; font-size:22px; font-weight:700; }
.badge { display:inline-block; padding:2px 10px; border-radius:12px;
         font-size:12px; font-weight:600; margin-bottom:12px; }
.production { background:#22C55E; color:#000; }
.staging    { background:#F59E0B; color:#000; }
.archived   { background:#6B7280; color:#fff; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🤖 ML Monitoring — Maritime AIS Models")

if USE_MLFLOW:
    tracking_uri = get_tracking_uri()
    st.success(f"MLflow connected · {tracking_uri}", icon="✅")
else:
    st.info(
        "MLflow is not configured. Showing mock data. "
        "Install mlflow and set `MLFLOW_TRACKING_URI` to enable live tracking. "
        "See `mlops/README.md` for setup instructions.",
        icon="ℹ️",
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    max_runs = st.slider("Max runs to display", 10, 200, 50)
    show_mock = not USE_MLFLOW
    st.markdown("---")
    st.subheader("Model paths")
    try:
        from config import MODELS_PATH as _mp
    except ImportError:
        _mp = os.getenv("MODELS_PATH", "/app/models")
    st.code(_mp)
    st.markdown("---")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

# =============================================================================
# Section 1 — Model Registry
# =============================================================================

st.subheader("Model Registry")

registry_data = fetch_registered_models() if USE_MLFLOW else _mock_registry()
pkl_info      = model_summary() if _MLOPS_IMPORT_OK else {}

if registry_data:
    reg_df = pd.DataFrame(registry_data)

    # Badges
    col1, col2, col3 = st.columns(3)
    production_count = sum(1 for r in registry_data if r["stage"] == "Production")
    staging_count    = sum(1 for r in registry_data if r["stage"] == "Staging")
    model_names      = reg_df["name"].nunique()
    col1.metric("Registered Models",     model_names)
    col2.metric("Production Versions",   production_count)
    col3.metric("Staging Versions",      staging_count)

    # Styled table
    def _stage_badge(s):
        cls = {"Production": "production", "Staging": "staging"}.get(s, "archived")
        return f'<span class="badge {cls}">{s}</span>'

    reg_df["Stage"] = reg_df["stage"].apply(_stage_badge)
    st.write(
        reg_df[["name", "version", "Stage", "run_id", "created_at"]]
        .rename(columns={"name": "Model", "version": "Ver",
                         "run_id": "Run ID", "created_at": "Created"})
        .to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )
else:
    st.info("No models in registry yet. Run an mlops/experiments/ training script.")

# .pkl availability
if pkl_info.get("models"):
    with st.expander("Local .pkl availability"):
        st.json(pkl_info)

# =============================================================================
# Section 2 — Experiment run history
# =============================================================================

st.markdown("---")
st.subheader("Experiment Run History")

exp_tab1, exp_tab2, exp_tab3 = st.tabs([
    "Anomaly Detector", "Congestion Classifier", "Position Predictor"
])

def _render_run_tab(experiment: str, metric_col: str, metric_label: str):
    df = fetch_runs(experiment, max_runs) if USE_MLFLOW else _mock_runs(experiment, max_runs)
    if df.empty:
        st.info(f"No runs yet for experiment `{experiment}`.")
        return
    df = df.sort_values("start_time", ascending=False) if "start_time" in df.columns else df

    m_col = f"m_{metric_col}"
    if m_col in df.columns:
        valid = df[df[m_col].notna()].copy()
        valid[m_col] = pd.to_numeric(valid[m_col], errors="coerce")
        if not valid.empty:
            fig = px.line(valid.sort_values("start_time"),
                          x="start_time", y=m_col,
                          markers=True,
                          labels={"start_time": "Run time", m_col: metric_label},
                          title=f"{metric_label} over training runs",
                          color_discrete_sequence=[EXP_COLORS.get(experiment, "#888")])
            fig.update_layout(height=300,
                              plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    display_cols = ["run_name", "start_time", "status"] + [
        c for c in df.columns if c.startswith("m_") and df[c].notna().any()
    ]
    st.dataframe(df[display_cols].head(max_runs), use_container_width=True)

with exp_tab1:
    _render_run_tab(ANOMALY_EXPERIMENT, "f1_score", "F1 Score")
with exp_tab2:
    _render_run_tab(CONGESTION_EXPERIMENT, "accuracy", "Accuracy")
with exp_tab3:
    _render_run_tab(PREDICTOR_EXPERIMENT, "total_mae_nm", "Total MAE (nm)")

# =============================================================================
# Section 3 — Latest evaluation report
# =============================================================================

st.markdown("---")
st.subheader("Latest Evaluation Report (test split)")

report = load_eval_report()

if report:
    st.caption(f"Generated: {report.get('generated_at','?')}  |  "
               f"Test rows: {report.get('test_rows', 0):,}")

    r1, r2, r3 = st.columns(3)
    a = report.get("anomaly", {})
    if a.get("status") == "ok":
        r1.metric("Anomaly F1",       f"{a.get('f1_score', 0):.3f}")
        r1.metric("Anomaly Rate",     f"{a.get('anomaly_rate', 0):.4f}")
        r1.metric("Est. FP Rate",     f"{a.get('estimated_false_positive_rate', 0):.4f}")

    c = report.get("congestion", {})
    if c.get("status") == "ok":
        r2.metric("Congestion Acc",   f"{c.get('accuracy', 0):.3f}")
        r2.metric("Macro F1",         f"{c.get('macro_f1', 0):.3f}")

    pred = report.get("predictor", {})
    for key in ["5min", "10min", "15min"]:
        v = pred.get(key, {})
        if isinstance(v, dict) and v.get("status") == "ok":
            r3.metric(f"MAE {key}",   f"{v.get('total_mae_nm', 0):.3f} nm")

    with st.expander("Full JSON report"):
        st.json(report)
else:
    st.info(
        "No evaluation report found. "
        "Run `python mlops/model_registry/evaluate_models.py` to generate one."
    )

# =============================================================================
# Section 4 — Feature importance charts
# =============================================================================

st.markdown("---")
st.subheader("Feature Importance")

fi_tab1, fi_tab2, fi_tab3, fi_tab4 = st.tabs(
    ["Anomaly", "Congestion", "Predictor 5min", "Predictor 10min"]
)

def _render_importance_tab(csv_name: str, title: str):
    df = load_importance_csv(csv_name)
    if df.empty:
        st.info(
            f"No importance data yet. "
            f"Run `python mlops/experiments/feature_importance.py`"
        )
        return
    df = df.sort_values("importance", ascending=True).tail(12)
    fig = px.bar(df, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale="Blues",
                 title=title)
    fig.update_layout(height=350, showlegend=False,
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with fi_tab1:
    _render_importance_tab("anomaly_feature_importance.csv",
                           "IsolationForest — Feature Importance")
with fi_tab2:
    _render_importance_tab("congestion_feature_importance.csv",
                           "RandomForest — Congestion Feature Importance")
with fi_tab3:
    _render_importance_tab("predictor_5min_feature_importance.csv",
                           "XGBoost 5-min — Feature Importance (gain)")
with fi_tab4:
    _render_importance_tab("predictor_10min_feature_importance.csv",
                           "XGBoost 10-min — Feature Importance (gain)")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Maritime Navigation AI System · ML Monitoring · "
    f"{'MLflow live' if USE_MLFLOW else 'Mock data'} · "
    f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
)
