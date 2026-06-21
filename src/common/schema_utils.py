"""
schema_utils.py — Maritime Navigation AI System
Handles full MarineCadastre AIS column normalisation.
Includes geographic utility functions used across all services.
"""
from __future__ import annotations
import math
import pandas as pd

# ── Canonical field names (used in ALL layers) ─────────────────────────────────
CANONICAL_FIELDS = [
    "mmsi", "base_datetime", "lat", "lon",
    "sog", "cog", "heading",
    "vessel_name", "imo", "call_sign",
    "vessel_type", "status",
    "length", "width", "draft",
    "cargo", "transceiver_class",
]


# ── Column name aliases (MarineCadastre → canonical) ──────────────────────────
COLUMN_ALIASES = {
    "mmsi":             ["mmsi", "MMSI"],
    "base_datetime":    ["BaseDateTime", "base_date_time",
                         "base_datetime", "timestamp", "time"],
    "lat":              ["LAT", "lat", "latitude", "Latitude"],
    "lon":              ["LON", "lon", "longitude", "Longitude"],
    "sog":              ["SOG", "sog", "speed", "Speed"],
    "cog":              ["COG", "cog", "course"],
    "heading":          ["Heading", "heading", "true_heading"],
    "vessel_name":      ["VesselName", "vessel_name",
                         "vessel_nm", "name"],
    "imo":              ["IMO", "imo"],
    "call_sign":        ["CallSign", "call_sign", "callsign"],
    "vessel_type":      ["VesselType", "vessel_type",
                         "vessel_ny_status", "type"],
    "status":           ["Status", "status", "nav_status"],
    "length":           ["Length", "length"],
    "width":            ["Width", "width"],
    "draft":            ["Draft", "draft"],
    "cargo":            ["Cargo", "cargo"],
    "transceiver_class":["TransceiverClass", "transceiver_class", "transceiver"],
}

# ── Vessel type lookup (numeric code → human readable) ────────────────────────
VESSEL_TYPE_LABELS = {
    "0": "Not Available",  "20": "Wing in Ground",
    "30": "Fishing",       "31": "Towing",
    "32": "Towing Large",  "33": "Dredging",
    "35": "Military",      "36": "Sailing",
    "37": "Pleasure Craft","40": "High Speed Craft",
    "50": "Pilot Vessel",  "51": "SAR Vessel",
    "52": "Tug",           "53": "Port Tender",
    "55": "Law Enforcement","60": "Passenger",
    "70": "Cargo",         "71": "Cargo Hazardous A",
    "80": "Tanker",        "81": "Tanker Hazardous A",
    "89": "Tanker",        "90": "Other",
}

# ── Navigation status lookup ───────────────────────────────────────────────────
NAV_STATUS_LABELS = {
    "0": "Underway using engine",
    "1": "At anchor",
    "2": "Not under command",
    "3": "Restricted manoeuvrability",
    "5": "Moored",
    "6": "Aground",
    "7": "Engaged in fishing",
    "8": "Underway sailing",
    "15": "Not defined",
}


# ── MMSI fix (scientific notation → string) ───────────────────────────────────
def fix_mmsi(val) -> str:
    """
    Convert MMSI from any format to clean string.
    Handles: 3.68E+08 → '368000000'
             368000000.0 → '368000000'
             '368000000' → '368000000'
    """
    try:
        return str(int(float(str(val).strip())))
    except (ValueError, TypeError):
        return str(val).strip()


# ── Safe numeric conversion ────────────────────────────────────────────────────
def safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


# ── Column resolver ────────────────────────────────────────────────────────────
def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw CSV columns to canonical names.
    Works with any MarineCadastre column naming variant.
    """
    # Build lowercase lookup
    col_lower = {c.lower().strip(): c for c in df.columns}
    rename = {}

    for canonical, variants in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue  # already correct
        for variant in variants:
            raw = col_lower.get(variant.lower())
            if raw and raw not in rename.values():
                rename[raw] = canonical
                break

    return df.rename(columns=rename)


# ── Full record normaliser ─────────────────────────────────────────────────────
def normalize_ais_record(record: dict) -> dict:
    """
    Map any raw AIS dict → canonical schema with safe defaults.
    Used by kafka_producer.py for every row sent to Kafka.
    """
    lower = {str(k).lower().strip(): v for k, v in record.items()}

    def pick(*names, default=None):
        for name in names:
            val = lower.get(name.lower())
            if val is not None and str(val).strip() not in (
                "", "nan", "NaT", "None", "none", "null"
            ):
                return val
        return default

    return {
        "mmsi":              fix_mmsi(pick(*COLUMN_ALIASES["mmsi"], default="")),
        "base_datetime":     str(pick(*COLUMN_ALIASES["base_datetime"], default="")).strip(),
        "lat":               safe_float(pick(*COLUMN_ALIASES["lat"])),
        "lon":               safe_float(pick(*COLUMN_ALIASES["lon"])),
        "sog":               safe_float(pick(*COLUMN_ALIASES["sog"])),
        "cog":               safe_float(pick(*COLUMN_ALIASES["cog"])),
        "heading":           safe_float(pick(*COLUMN_ALIASES["heading"])),
        "vessel_name":       str(pick(*COLUMN_ALIASES["vessel_name"],   default="")).strip(),
        "imo":               str(pick(*COLUMN_ALIASES["imo"],           default="")).strip(),
        "call_sign":         str(pick(*COLUMN_ALIASES["call_sign"],     default="")).strip(),
        "vessel_type":       str(pick(*COLUMN_ALIASES["vessel_type"],   default="")).strip(),
        "status":            str(pick(*COLUMN_ALIASES["status"],        default="")).strip(),
        "length":            safe_float(pick(*COLUMN_ALIASES["length"])),
        "width":             safe_float(pick(*COLUMN_ALIASES["width"])),
        "draft":             safe_float(pick(*COLUMN_ALIASES["draft"])),
        "cargo":             str(pick(*COLUMN_ALIASES["cargo"],         default="")).strip(),
        "transceiver_class": str(pick(*COLUMN_ALIASES["transceiver_class"], default="")).strip(),
    }


# ── Position validation ────────────────────────────────────────────────────────
def is_valid_position(record: dict) -> bool:
    """True only when record has a valid, non-zero lat/lon."""
    try:
        lat = float(record.get("lat", 0))
        lon = float(record.get("lon", 0))
        return (
            lat != 0.0 and lon != 0.0
            and not math.isnan(lat) and not math.isnan(lon)
            and -90 <= lat <= 90
            and -180 <= lon <= 180
        )
    except (TypeError, ValueError):
        return False


# ── Geographic utilities ───────────────────────────────────────────────────────
def haversine_nm(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    """
    Calculate distance in nautical miles between two coordinates.
    Uses Haversine formula.
    1 nautical mile = 1852 metres = 1/60 degree of latitude
    """
    R = 3440.065  # Earth radius in nautical miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def bearing_degrees(lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
    """Calculate bearing in degrees from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def predict_position_dr(lat: float, lon: float,
                        sog: float, heading: float,
                        minutes: float) -> tuple[float, float]:
    """
    Dead Reckoning position prediction.
    Returns (predicted_lat, predicted_lon).
    """
    heading_rad = math.radians(heading)
    dist_nm     = sog * (minutes / 60.0)
    delta_lat   = (dist_nm * math.cos(heading_rad)) / 60.0
    delta_lon   = (dist_nm * math.sin(heading_rad)) / (
        60.0 * math.cos(math.radians(lat)) + 1e-9
    )
    return round(lat + delta_lat, 6), round(lon + delta_lon, 6)


def classify_risk(sog: float, lat: float, lon: float,
                  us_port_zones: list = None) -> str:
    """Classify vessel risk level based on speed and location."""
    from config import US_PORT_ZONES
    zones = us_port_zones or US_PORT_ZONES

    in_zone = any(
        z["lat_min"] <= lat <= z["lat_max"] and
        z["lon_min"] <= lon <= z["lon_max"]
        for z in zones
    )

    if sog < 0.5 and in_zone:
        return "HIGH"
    if sog < 2.0:
        return "MEDIUM"
    return "LOW"


def get_vessel_type_label(code: str) -> str:
    """Convert numeric vessel type code to human-readable label."""
    return VESSEL_TYPE_LABELS.get(str(code), f"Type {code}")


def get_status_label(code: str) -> str:
    """Convert numeric nav status to human-readable label."""
    return NAV_STATUS_LABELS.get(str(code), f"Status {code}")
