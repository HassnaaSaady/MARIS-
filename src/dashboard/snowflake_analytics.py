"""
snowflake_analytics.py — Maritime Navigation AI System
Standalone Streamlit analytics page powered by Snowflake.

Run:
    streamlit run src/dashboard/snowflake_analytics.py

Behaviour:
  - If SNOWFLAKE_ACCOUNT is set: queries run against Snowflake warehouse.
  - If SNOWFLAKE_ACCOUNT is not set: falls back to PostgreSQL for every metric
    so the page renders useful data in the Docker-only environment.
  - If neither is configured: page renders with empty charts and a clear banner.

This file does NOT modify streamlit_app.py.  It is a separate page that can
be run standalone or added to a Streamlit multi-page app by placing it in a
pages/ directory.

Geographic focus: US coastal waters (East Coast, West Coast, Gulf of Mexico,
Great Lakes) — distinct from the Suez Canal monitoring in the main dashboard.
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — works inside Docker (/app/src) and in local dev
# ---------------------------------------------------------------------------
for _p in ["/app/src", "/app/api",
           os.path.join(os.path.dirname(__file__), "..", ".."),
           os.path.join(os.path.dirname(__file__), "..")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from common.config import POSTGRES_URL, US_WATERS
except ImportError:
    POSTGRES_URL = os.getenv(
        "POSTGRES_URL",
        "postgresql://maritime:maritime123@postgres:5432/maritime"
    )
    US_WATERS = {"lat_min": 24.0, "lat_max": 49.0,
                 "lon_min": -125.0, "lon_max": -66.0}

try:
    from snowflake.snowflake_queries import (
        get_fleet_summary,
        get_busiest_lanes,
        get_anomaly_trends,
        get_congestion_by_hour,
        get_top_risky_vessels,
        get_collision_stats,
        _is_configured as sf_configured,
    )
    _SF_IMPORT_OK = True
except ImportError:
    _SF_IMPORT_OK = False
    def sf_configured(): return False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# US region colours
# ---------------------------------------------------------------------------
REGION_COLORS = {
    "East Coast":     "#3B82F6",
    "West Coast":     "#10B981",
    "Gulf of Mexico": "#F59E0B",
    "Great Lakes":    "#8B5CF6",
    "Other US Waters":"#6B7280",
}

RISK_COLORS = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#22C55E"}

# =============================================================================
# PostgreSQL fallback helpers
# =============================================================================

@st.cache_resource(ttl=300)
def _pg_engine():
    from sqlalchemy import create_engine
    return create_engine(POSTGRES_URL, pool_pre_ping=True, pool_size=5)


def _pg_available() -> bool:
    try:
        eng = _pg_engine()
        with eng.connect() as c:
            c.execute(__import__("sqlalchemy").text("SELECT 1"))
        return True
    except Exception:
        return False


def _pg_fleet_summary() -> dict:
    """PostgreSQL equivalent of get_fleet_summary(), filtered to US waters."""
    try:
        import sqlalchemy as sa
        eng = _pg_engine()
        sql = sa.text("""
            SELECT
                COUNT(DISTINCT mmsi)                                         AS total_vessels,
                COUNT(DISTINCT CASE WHEN risk_level='HIGH'  THEN mmsi END)   AS high_risk_vessels,
                COUNT(DISTINCT CASE WHEN is_anomaly         THEN mmsi END)   AS anomalous_vessels,
                ROUND(AVG(sog)::NUMERIC, 2)                                  AS fleet_avg_speed_kn,
                COUNT(DISTINCT CASE WHEN lon BETWEEN -82  AND -66  THEN mmsi END) AS east_coast_vessels,
                COUNT(DISTINCT CASE WHEN lon BETWEEN -125 AND -117 THEN mmsi END) AS west_coast_vessels,
                COUNT(DISTINCT CASE WHEN lon BETWEEN -97  AND -80
                                     AND lat BETWEEN 24   AND 31  THEN mmsi END) AS gulf_vessels,
                MAX(updated_at)                                              AS data_freshness
            FROM fact_vessel_latest
            WHERE lat BETWEEN :lat_min AND :lat_max
              AND lon BETWEEN :lon_min AND :lon_max
        """)
        with eng.connect() as c:
            row = c.execute(sql, {
                "lat_min": US_WATERS["lat_min"], "lat_max": US_WATERS["lat_max"],
                "lon_min": US_WATERS["lon_min"], "lon_max": US_WATERS["lon_max"],
            }).fetchone()
        if row:
            d = dict(row._mapping)
            d["snowflake_available"] = False
            return d
    except Exception as exc:
        logger.warning("PG fleet summary failed: %s", exc)
    return {"total_vessels": 0, "high_risk_vessels": 0, "anomalous_vessels": 0,
            "fleet_avg_speed_kn": 0.0, "east_coast_vessels": 0,
            "west_coast_vessels": 0, "gulf_vessels": 0,
            "data_freshness": None, "snowflake_available": False}


def _pg_busiest_lanes(top_n: int = 50) -> pd.DataFrame:
    try:
        import sqlalchemy as sa
        sql = sa.text("""
            SELECT
                ROUND(lat_bin::NUMERIC, 1)         AS lat_grid,
                ROUND(lon_bin::NUMERIC, 1)         AS lon_grid,
                SUM(vessel_count)                  AS total_vessels,
                SUM(unique_vessels)                AS unique_vessels,
                ROUND(AVG(avg_sog)::NUMERIC, 2)    AS avg_speed_kn,
                MAX(congestion_level)              AS peak_congestion,
                CASE
                    WHEN lon_bin BETWEEN -82  AND -66  THEN 'East Coast'
                    WHEN lon_bin BETWEEN -125 AND -117 THEN 'West Coast'
                    WHEN lon_bin BETWEEN -97  AND -80
                     AND lat_bin BETWEEN 24   AND 31   THEN 'Gulf of Mexico'
                    WHEN lon_bin BETWEEN -93  AND -76
                     AND lat_bin BETWEEN 41   AND 49   THEN 'Great Lakes'
                    ELSE 'Other US Waters'
                END                                AS us_region
            FROM fact_traffic_density
            WHERE lat_bin BETWEEN :lat_min AND :lat_max
              AND lon_bin BETWEEN :lon_min AND :lon_max
            GROUP BY 1, 2, 7
            ORDER BY total_vessels DESC
            LIMIT :n
        """)
        with _pg_engine().connect() as c:
            return pd.DataFrame(c.execute(sql, {
                "lat_min": US_WATERS["lat_min"], "lat_max": US_WATERS["lat_max"],
                "lon_min": US_WATERS["lon_min"], "lon_max": US_WATERS["lon_max"],
                "n": top_n,
            }).fetchall(), columns=["lat_grid","lon_grid","total_vessels",
                                    "unique_vessels","avg_speed_kn",
                                    "peak_congestion","us_region"])
    except Exception as exc:
        logger.warning("PG busiest_lanes failed: %s", exc)
        return pd.DataFrame()


def _pg_daily_stats(days: int = 30) -> pd.DataFrame:
    try:
        import sqlalchemy as sa
        sql = sa.text("""
            SELECT stat_date, total_vessels, avg_sog AS avg_speed_kn,
                   high_risk_count, anomaly_count
            FROM fact_daily_stats
            WHERE stat_date >= CURRENT_DATE - :days
            ORDER BY stat_date
        """)
        with _pg_engine().connect() as c:
            return pd.DataFrame(c.execute(sql, {"days": days}).fetchall(),
                                columns=["event_date","total_vessels",
                                         "avg_speed_kn","high_risk_count",
                                         "anomaly_count"])
    except Exception as exc:
        logger.warning("PG daily_stats failed: %s", exc)
        return pd.DataFrame()


def _pg_congestion_by_hour() -> pd.DataFrame:
    try:
        import sqlalchemy as sa
        sql = sa.text("""
            SELECT
                EXTRACT(HOUR FROM hour_bucket)::INT  AS hour_of_day,
                congestion_level,
                COUNT(*)                             AS grid_cells,
                ROUND(AVG(vessel_count)::NUMERIC, 1) AS avg_vessels_per_cell,
                ROUND(AVG(avg_sog)::NUMERIC, 2)      AS avg_speed_kn,
                CASE
                    WHEN lon_bin BETWEEN -82  AND -66  THEN 'East Coast'
                    WHEN lon_bin BETWEEN -125 AND -117 THEN 'West Coast'
                    WHEN lon_bin BETWEEN -97  AND -80
                     AND lat_bin BETWEEN 24   AND 31   THEN 'Gulf of Mexico'
                    ELSE 'Other US Waters'
                END AS us_region
            FROM fact_traffic_density
            WHERE lat_bin BETWEEN :lat_min AND :lat_max
              AND lon_bin BETWEEN :lon_min AND :lon_max
            GROUP BY 1, 2, 6
            ORDER BY 6, 1
        """)
        with _pg_engine().connect() as c:
            return pd.DataFrame(c.execute(sql, {
                "lat_min": US_WATERS["lat_min"], "lat_max": US_WATERS["lat_max"],
                "lon_min": US_WATERS["lon_min"], "lon_max": US_WATERS["lon_max"],
            }).fetchall(), columns=["hour_of_day","congestion_level","grid_cells",
                                    "avg_vessels_per_cell","avg_speed_kn","us_region"])
    except Exception as exc:
        logger.warning("PG congestion_by_hour failed: %s", exc)
        return pd.DataFrame()


# =============================================================================
# Data fetchers — try Snowflake first, fall back to PostgreSQL
# =============================================================================

USE_SNOWFLAKE = _SF_IMPORT_OK and sf_configured()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_fleet_summary() -> dict:
    if USE_SNOWFLAKE:
        return get_fleet_summary()
    return _pg_fleet_summary()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_busiest_lanes(top_n: int) -> pd.DataFrame:
    if USE_SNOWFLAKE:
        return get_busiest_lanes(top_n)
    return _pg_busiest_lanes(top_n)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_anomaly_trends(days: int) -> pd.DataFrame:
    if USE_SNOWFLAKE:
        return get_anomaly_trends(days)
    return _pg_daily_stats(days)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_congestion_by_hour() -> pd.DataFrame:
    if USE_SNOWFLAKE:
        return get_congestion_by_hour()
    return _pg_congestion_by_hour()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_top_risky(top_n: int, days: int) -> pd.DataFrame:
    if USE_SNOWFLAKE:
        return get_top_risky_vessels(top_n, days)
    return pd.DataFrame()   # complex join — Snowflake only


@st.cache_data(ttl=300, show_spinner=False)
def fetch_collision_stats(weeks: int) -> pd.DataFrame:
    if USE_SNOWFLAKE:
        return get_collision_stats(weeks)
    return pd.DataFrame()


# =============================================================================
# Page layout
# =============================================================================

st.set_page_config(
    page_title="Maritime Analytics — US Coastal Waters",
    page_icon="🌊",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stMetric"] {
    background:#1C2333; border-radius:10px;
    padding:12px 18px; border:1px solid #2D3748;
}
[data-testid="stMetricLabel"] { color:#A0AEC0 !important; font-size:13px; }
[data-testid="stMetricValue"] { color:#FFFFFF !important; font-size:24px; font-weight:700; }
.source-badge {
    display:inline-block; padding:2px 10px; border-radius:12px;
    font-size:12px; font-weight:600; margin-bottom:12px;
}
.sf-badge  { background:#29B5E8; color:#000; }
.pg-badge  { background:#336791; color:#fff; }
.na-badge  { background:#4B5563; color:#fff; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🌊 US Coastal Waters — Analytics Dashboard")

if USE_SNOWFLAKE:
    st.markdown('<span class="source-badge sf-badge">Data source: Snowflake</span>',
                unsafe_allow_html=True)
elif _pg_available():
    st.markdown('<span class="source-badge pg-badge">Data source: PostgreSQL (Snowflake not configured)</span>',
                unsafe_allow_html=True)
    st.info(
        "Snowflake is not configured. Set `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, "
        "`SNOWFLAKE_PASSWORD` to enable warehouse analytics. "
        "See `.env.snowflake.example` and `src/snowflake/README.md`.",
        icon="ℹ️",
    )
else:
    st.markdown('<span class="source-badge na-badge">No data source configured</span>',
                unsafe_allow_html=True)
    st.warning(
        "Neither Snowflake nor PostgreSQL is reachable. "
        "Start the Docker stack (`docker compose up`) or configure Snowflake credentials.",
        icon="⚠️",
    )

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    days_back  = st.slider("Days of history", 7, 90, 30)
    top_n      = st.slider("Top N vessels / lanes", 10, 100, 25)
    weeks_back = st.slider("Collision stats (weeks)", 2, 16, 8)
    region_filter = st.multiselect(
        "US Regions",
        ["East Coast", "West Coast", "Gulf of Mexico", "Great Lakes", "Other US Waters"],
        default=["East Coast", "West Coast", "Gulf of Mexico", "Great Lakes"],
    )
    st.markdown("---")
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.rerun()

# =============================================================================
# Section 1 — Fleet KPI strip
# =============================================================================

st.subheader("Fleet Overview — US Coastal Waters")
summary = fetch_fleet_summary()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Vessels",    f"{summary.get('total_vessels', 0):,}")
c2.metric("High Risk",        f"{summary.get('high_risk_vessels', 0):,}")
c3.metric("Anomalous",        f"{summary.get('anomalous_vessels', 0):,}")
c4.metric("Avg Speed (kn)",   f"{summary.get('fleet_avg_speed_kn', 0.0):.1f}")
c5.metric("East Coast",       f"{summary.get('east_coast_vessels', 0):,}")
c6.metric("Gulf of Mexico",   f"{summary.get('gulf_vessels', 0):,}")

freshness = summary.get("data_freshness")
if freshness:
    st.caption(f"Data freshness: {freshness}")

# =============================================================================
# Section 2 — Busiest shipping lanes map
# =============================================================================

st.markdown("---")
st.subheader("Busiest Shipping Lanes")

lanes_df = fetch_busiest_lanes(top_n)

if not lanes_df.empty:
    if region_filter:
        lanes_df = lanes_df[lanes_df["us_region"].isin(region_filter)]

    col_map, col_bar = st.columns([3, 1])
    with col_map:
        fig_map = px.scatter_mapbox(
            lanes_df,
            lat="lat_grid", lon="lon_grid",
            size="total_vessels",
            color="us_region",
            color_discrete_map=REGION_COLORS,
            hover_data={"total_vessels": True, "avg_speed_kn": True,
                        "peak_congestion": True},
            size_max=30,
            zoom=3,
            center={"lat": 37.0, "lon": -95.0},
            mapbox_style="carto-darkmatter",
            title="Traffic Density by Grid Cell",
            height=500,
        )
        fig_map.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0},
                              legend_title_text="Region")
        st.plotly_chart(fig_map, use_container_width=True)

    with col_bar:
        region_totals = (lanes_df.groupby("us_region")["total_vessels"]
                         .sum().reset_index()
                         .sort_values("total_vessels", ascending=True))
        fig_bar = px.bar(
            region_totals, x="total_vessels", y="us_region",
            orientation="h",
            color="us_region", color_discrete_map=REGION_COLORS,
            title="Vessels by Region",
        )
        fig_bar.update_layout(showlegend=False, height=500,
                              plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No lane data available for the selected filters.")

# =============================================================================
# Section 3 — Anomaly trend + daily stats
# =============================================================================

st.markdown("---")
st.subheader(f"Anomaly & Risk Trends — Last {days_back} Days")

trends_df = fetch_anomaly_trends(days_back)

if not trends_df.empty:
    date_col = "event_date" if "event_date" in trends_df.columns else trends_df.columns[0]

    if "anomaly_count" in trends_df.columns:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trends_df[date_col], y=trends_df["total_vessels"],
            name="Total Vessels", mode="lines", line=dict(color="#3B82F6")))
        fig_trend.add_trace(go.Bar(
            x=trends_df[date_col], y=trends_df["anomaly_count"],
            name="Anomalies", marker_color="#EF4444", opacity=0.7,
            yaxis="y2"))
        fig_trend.update_layout(
            yaxis=dict(title="Vessels", color="#3B82F6"),
            yaxis2=dict(title="Anomalies", overlaying="y", side="right",
                        color="#EF4444"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1),
            height=350,
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    elif "anomaly_count" in trends_df.columns or "anomaly_type" in trends_df.columns:
        # Snowflake version has per-type breakdown
        fig_trend = px.bar(trends_df, x=date_col, y="anomaly_count",
                           color="anomaly_type", barmode="stack",
                           title="Daily Anomalies by Type")
        st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No trend data available.")

# =============================================================================
# Section 4 — Congestion by hour of day
# =============================================================================

st.markdown("---")
st.subheader("Congestion Patterns by Hour of Day")

hour_df = fetch_congestion_by_hour()

if not hour_df.empty:
    if region_filter:
        hour_df = hour_df[hour_df["us_region"].isin(region_filter)]

    pivot = (hour_df.groupby(["hour_of_day", "us_region"])
             ["avg_vessels_per_cell"].mean().reset_index())

    fig_heat = px.density_heatmap(
        pivot, x="hour_of_day", y="us_region",
        z="avg_vessels_per_cell",
        color_continuous_scale="Blues",
        title="Average Vessels per Grid Cell by Hour (UTC)",
        labels={"hour_of_day": "Hour (UTC)", "us_region": "Region",
                "avg_vessels_per_cell": "Avg Vessels"},
    )
    fig_heat.update_layout(height=300,
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(hour_df, use_container_width=True)
else:
    st.info("No congestion data available.")

# =============================================================================
# Section 5 — Top risky vessels (Snowflake only)
# =============================================================================

st.markdown("---")
st.subheader(f"Top {top_n} High-Risk Vessels — Last {days_back} Days")

if USE_SNOWFLAKE:
    risky_df = fetch_top_risky(top_n, days_back)
    if not risky_df.empty:
        if region_filter and "predominant_risk_zone" in risky_df.columns:
            risky_df = risky_df[risky_df["predominant_risk_zone"].isin(region_filter)]

        col_tbl, col_scatter = st.columns([2, 1])
        with col_tbl:
            display_cols = [c for c in [
                "mmsi","vessel_name","vessel_type_label","risk_count",
                "avg_risk_sog","predominant_risk_zone","last_sog",
            ] if c in risky_df.columns]
            st.dataframe(
                risky_df[display_cols].rename(columns={
                    "mmsi":"MMSI","vessel_name":"Vessel","vessel_type_label":"Type",
                    "risk_count":"Risk Events","avg_risk_sog":"Avg SOG (kn)",
                    "predominant_risk_zone":"Primary Zone","last_sog":"Last SOG",
                }),
                use_container_width=True, height=400,
            )
        with col_scatter:
            if "last_lat" in risky_df.columns:
                fig_scatter = px.scatter_mapbox(
                    risky_df, lat="last_lat", lon="last_lon",
                    size="risk_count", color="predominant_risk_zone",
                    color_discrete_map=REGION_COLORS,
                    hover_data={"vessel_name": True, "risk_count": True},
                    zoom=3, center={"lat": 37.0, "lon": -95.0},
                    mapbox_style="carto-darkmatter",
                    height=400,
                )
                fig_scatter.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                                          showlegend=False)
                st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No high-risk vessel data in the selected period.")
else:
    st.info(
        "Top risky vessels requires Snowflake (multi-table join). "
        "Configure `SNOWFLAKE_ACCOUNT` to enable this section."
    )

# =============================================================================
# Section 6 — Collision risk statistics
# =============================================================================

st.markdown("---")
st.subheader(f"Collision Risk Statistics — Last {weeks_back} Weeks")

if USE_SNOWFLAKE:
    collision_df = fetch_collision_stats(weeks_back)
    if not collision_df.empty:
        if region_filter and "us_region" in collision_df.columns:
            collision_df = collision_df[collision_df["us_region"].isin(region_filter)]

        col_a, col_b = st.columns(2)
        with col_a:
            fig_col = px.bar(
                collision_df, x="week_start", y="collision_alerts",
                color="severity",
                color_discrete_map={"HIGH":"#EF4444","MEDIUM":"#F59E0B","LOW":"#22C55E"},
                barmode="stack", title="Weekly Collision Alerts by Severity",
            )
            fig_col.update_layout(height=300,
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_col, use_container_width=True)
        with col_b:
            region_col = (collision_df.groupby("us_region")["collision_alerts"]
                          .sum().reset_index())
            fig_pie = px.pie(region_col, names="us_region",
                             values="collision_alerts",
                             color="us_region",
                             color_discrete_map=REGION_COLORS,
                             title="Collision Alerts by Region")
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        total_cpa = collision_df["avg_cpa_nm"].mean() if "avg_cpa_nm" in collision_df.columns else 0
        outstanding = collision_df["outstanding"].sum() if "outstanding" in collision_df.columns else 0
        st.metric("Avg Closest Point of Approach", f"{total_cpa:.3f} nm")
        st.metric("Outstanding (Unresolved) Alerts", int(outstanding))
    else:
        st.info("No collision data in the selected period.")
else:
    st.info(
        "Collision statistics require Snowflake. "
        "Configure `SNOWFLAKE_ACCOUNT` to enable this section."
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Maritime Navigation AI System | US Coastal Waters Analytics | "
    f"Source: {'Snowflake' if USE_SNOWFLAKE else 'PostgreSQL'} | "
    f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
)
