"""
silver_job.py — Maritime Navigation AI System
Bronze → Silver transformation:
  A) Physical-validity cleaning BEFORE feature engineering:
       - AIS sentinel nulling: sog=102.3 → null, heading=511 → null
       - Kinematic bounds: sog in [0,60], cog in [0,360), heading in [0,360)
       - Position bounds: lat in [-90,90], lon in [-180,180]
       - Teleport/GPS-glitch filter: implied speed > 100 kn dropped
  B) distance_nm uses true haversine (R=3440.065 nm), not Euclidean*60
  C) Delta-feature window is mmsi-only (no day partition), continuous across
     midnight — matches scorer.py exactly, eliminates day-boundary skew
  D) First pings dropped before writing Silver so ML never trains on
     fabricated zeros that were previously inserted via .fillna(0)
  - Deduplication, vessel info fill, human-readable labels, fine grid bins

Run after bronze_job.py:
    docker compose exec spark-master \\
      /opt/spark/bin/spark-submit \\
        --packages io.delta:delta-core_2.12:2.4.0 \\
        src/processing/silver_job.py
"""
import sys
sys.path.insert(0, "/opt/spark/app/src/common")

from config import (
    SPARK_MASTER,
    DELTA_BRONZE_PATH, DELTA_SILVER_PATH,
)
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, lag, abs as spark_abs, sqrt,
    radians, sin, cos, asin,
    unix_timestamp, lit, when,
    round as spark_round, coalesce,
    first, last, row_number,
    year as spark_year, month as spark_month, dayofmonth,
    to_timestamp,
)

# ── AIS sentinel values ────────────────────────────────────────────────────────
# These are not "unknown" — they are explicit "not available" codes in the AIS spec.
SOG_SENTINEL     = 102.3   # AIS spec: sog = 102.3 → "not available"
HEADING_SENTINEL = 511     # AIS spec: heading = 511 → "not available"

R_NM = 3440.065            # Earth radius in nautical miles

# Human-readable vessel type labels
VESSEL_TYPE_MAP = {
    "0": "Not Available",  "30": "Fishing",
    "31": "Towing",        "36": "Sailing",
    "37": "Pleasure Craft","50": "Pilot Vessel",
    "52": "Tug",           "60": "Passenger",
    "70": "Cargo",         "80": "Tanker",
    "35": "Military",      "51": "SAR",
    "90": "Other",
}

STATUS_MAP = {
    "0": "Underway/Engine", "1": "At Anchor",
    "2": "Not Under Command", "5": "Moored",
    "6": "Aground",         "7": "Fishing",
    "8": "Underway/Sailing",
}


def build_spark():
    return (
        SparkSession.builder
        .appName("Silver_AIS_Clean")
        .master(SPARK_MASTER)
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.executor.memory", "3g")
        .config("spark.driver.memory",   "3g")
        .config("spark.memory.fraction",          "0.8")
        .config("spark.memory.storageFraction",   "0.3")
        .config("spark.shuffle.io.maxRetries",    "10")
        .config("spark.shuffle.io.retryWait",     "30s")
        .config("spark.sql.timestampType", "TIMESTAMP_NTZ")
        .getOrCreate()
    )


# ── A: Physical validity cleaning ─────────────────────────────────────────────

def physical_validity_clean(df):
    """
    Drop rows with impossible AIS kinematics or out-of-range positions.

    AIS sentinels (sog=102.3, heading=511) are nulled BEFORE range checks so
    those rows pass rather than being wrongly rejected by the bounds filter.
    Null values for sog/cog/heading are allowed (null = no measurement);
    lat/lon nulls are always rejected.
    """
    # Null out AIS sentinels
    df = (
        df
        .withColumn("sog",
                    when(col("sog") == SOG_SENTINEL, None).otherwise(col("sog")))
        .withColumn("heading",
                    when(col("heading") == HEADING_SENTINEL, None).otherwise(col("heading")))
    )

    # sog: null passes; otherwise must be in [0, 60]
    df = df.filter(col("sog").isNull() | ((col("sog") >= 0) & (col("sog") <= 60)))

    # cog: null passes; otherwise [0, 360)
    df = df.filter(col("cog").isNull() | ((col("cog") >= 0) & (col("cog") < 360)))

    # heading: null passes (sentinel already nulled above); otherwise [0, 360)
    df = df.filter(
        col("heading").isNull() | ((col("heading") >= 0) & (col("heading") < 360))
    )

    # Position bounds: mandatory — no null lat/lon accepted
    df = df.filter(
        (col("lat") >= -90)  & (col("lat") <= 90) &
        (col("lon") >= -180) & (col("lon") <= 180)
    )

    return df


# ── B/C helpers: position lags + teleport filter ───────────────────────────────

def _compute_position_lags(df):
    """
    First-pass lag computation for teleport detection only.
    Computes prev_lat, prev_lon, prev_time, time_delta_sec, and haversine
    distance_nm (B) using a per-mmsi window (C).
    These columns are removed by filter_teleports() after filtering.
    """
    w = Window.partitionBy("mmsi").orderBy("base_datetime")

    return (
        df
        .withColumn("prev_time", lag("base_datetime", 1).over(w))
        .withColumn("prev_lat",  lag("lat",           1).over(w))
        .withColumn("prev_lon",  lag("lon",           1).over(w))
        .withColumn("time_delta_sec",
                    unix_timestamp("base_datetime") - unix_timestamp("prev_time"))
        # B: haversine distance in nautical miles
        .withColumn("_dlat", radians(col("lat") - col("prev_lat")))
        .withColumn("_dlon", radians(col("lon") - col("prev_lon")))
        .withColumn("_a",
                    sin(col("_dlat") / 2) ** 2 +
                    cos(radians(col("prev_lat"))) * cos(radians(col("lat"))) *
                    sin(col("_dlon") / 2) ** 2)
        .withColumn("distance_nm",
                    spark_round(lit(R_NM) * 2 * asin(sqrt(col("_a"))), 4))
        .drop("_dlat", "_dlon", "_a")
    )


def filter_teleports(df, max_speed_kn: float = 100.0):
    """
    Drop GPS-jump rows whose implied speed exceeds max_speed_kn knots.
    First pings (prev_lat is null) are always kept.
    Drops the temporary lag columns after filtering so add_ml_features()
    recomputes them cleanly on the teleport-free dataset.
    """
    implied_speed = when(
        col("prev_lat").isNull(),
        lit(0.0)                           # first ping — no prior position, keep
    ).when(
        col("time_delta_sec") > 0,
        col("distance_nm") / (col("time_delta_sec") / lit(3600.0))
    ).when(
        col("distance_nm") > 0,
        lit(999_999.0)                     # zero time + non-zero distance = teleport
    ).otherwise(
        lit(0.0)
    )

    df = df.filter(implied_speed <= max_speed_kn)

    return df.drop("prev_lat", "prev_lon", "prev_time", "time_delta_sec", "distance_nm")


# ── B + C: Final ML feature engineering on clean data ─────────────────────────

def add_ml_features(df):
    """
    Engineer features needed for ML training on the teleport-filtered dataset.

    B — distance_nm uses true haversine (R=3440.065 nm), not Euclidean*60.
    C — window partitions by mmsi ONLY (no day), so deltas are continuous
        across midnight and match scorer.py exactly.
    """
    # C: per-vessel ordered by time — no day-boundary cut
    w = Window.partitionBy("mmsi").orderBy("base_datetime")

    return (
        df
        # ── Lag values ────────────────────────────────────────
        .withColumn("prev_time",    lag("base_datetime", 1).over(w))
        .withColumn("prev_lat",     lag("lat",           1).over(w))
        .withColumn("prev_lon",     lag("lon",           1).over(w))
        .withColumn("prev_sog",     lag("sog",           1).over(w))
        .withColumn("prev_heading", lag("heading",       1).over(w))

        # ── Time delta ─────────────────────────────────────────
        .withColumn("time_delta_sec",
                    unix_timestamp("base_datetime") - unix_timestamp("prev_time"))

        # ── B: Haversine distance in nautical miles ─────────────
        .withColumn("_dlat", radians(col("lat") - col("prev_lat")))
        .withColumn("_dlon", radians(col("lon") - col("prev_lon")))
        .withColumn("_a",
                    sin(col("_dlat") / 2) ** 2 +
                    cos(radians(col("prev_lat"))) * cos(radians(col("lat"))) *
                    sin(col("_dlon") / 2) ** 2)
        .withColumn("distance_nm",
                    spark_round(lit(R_NM) * 2 * asin(sqrt(col("_a"))), 4))
        .drop("_dlat", "_dlon", "_a")

        # ── Speed change ──────────────────────────────────────
        .withColumn("sog_change", col("sog") - col("prev_sog"))

        # ── Heading change (wrap-around: 350→10 = 20°, not 340°) ──
        .withColumn("raw_heading_change",
                    spark_abs(col("heading") - col("prev_heading")))
        .withColumn("heading_change",
                    when(col("raw_heading_change") > 180,
                         360 - col("raw_heading_change"))
                    .otherwise(col("raw_heading_change")))

        # ── Fine grid bins ─────────────────────────────────────
        .withColumn("lat_bin_fine", spark_round(col("lat"), 2))
        .withColumn("lon_bin_fine", spark_round(col("lon"), 2))

        # ── Coarse 0.1° grid bins (matches gold_job + live_scorer) ─────────────
        .withColumn("lat_bin", spark_round(spark_round(col("lat") / 0.1) * 0.1, 1))
        .withColumn("lon_bin", spark_round(spark_round(col("lon") / 0.1) * 0.1, 1))

        # ── Drop intermediates ─────────────────────────────────
        .drop("prev_sog", "prev_heading", "raw_heading_change",
              "prev_time", "prev_lat", "prev_lon")
    )


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    print(f"Reading Bronze Delta: {DELTA_BRONZE_PATH}")
    bronze = spark.read.format("delta").load(DELTA_BRONZE_PATH)

    # Convert base_datetime string (Bronze) to timestamp for Silver processing
    bronze = bronze.withColumn(
        "base_datetime",
        to_timestamp(col("base_datetime"))
    )

    # ── Step 1: Deduplicate ────────────────────────────────────────────────────
    deduped = bronze.dropDuplicates(["mmsi", "base_datetime"])

    # ── Step 2: Physical validity cleaning (A) ────────────────────────────────
    valid = physical_validity_clean(deduped)
    valid = valid.repartition(64, "mmsi")
    

    # ── Step 3: Fill missing vessel info ──────────────────────────────────────
    w_vessel = Window.partitionBy("mmsi")
    filled = (
        valid
        .withColumn("vessel_name",
                    coalesce(col("vessel_name"),
                             first("vessel_name", True).over(w_vessel)))
        .withColumn("vessel_type",
                    coalesce(col("vessel_type"),
                             first("vessel_type", True).over(w_vessel)))
        .withColumn("imo",
                    coalesce(col("imo"),
                             first("imo", True).over(w_vessel)))
    )

    # ── Step 4: Add human-readable labels ─────────────────────────────────────
    type_expr = col("vessel_type")
    for code, label in VESSEL_TYPE_MAP.items():
        type_expr = when(col("vessel_type") == code, label).otherwise(type_expr)
    filled = filled.withColumn("vessel_type_label", type_expr)

    status_expr = col("status")
    for code, label in STATUS_MAP.items():
        status_expr = when(col("status") == code, label).otherwise(status_expr)
    filled = filled.withColumn("status_label", status_expr)

    # ── Step 5: First-pass lags for teleport detection (A) ────────────────────
    print("\nComputing initial lags for teleport detection …")
    with_lags = _compute_position_lags(filled)

    # ── Step 6: Teleport / GPS-glitch filter (A) ──────────────────────────────
    post_teleport = filter_teleports(with_lags, max_speed_kn=100.0)

    # ── Step 7: Final ML features on clean data (B + C) ───────────────────────
    print("\nComputing final ML features on cleaned data …")
    silver = add_ml_features(post_teleport)

    # ── Step 8: Drop first pings (D) ──────────────────────────────────────────
    # time_delta_sec and distance_nm are null ONLY for first pings (no prior
    # position record). sog_change/heading_change can be null on non-first pings
    # whenever sog/heading was an AIS sentinel and got nulled — we keep those
    # rows; ML training handles remaining nulls with its own NaN guards.
    silver = silver.dropna(subset=["time_delta_sec", "distance_nm"])

    # ── Step 8b: Assign data_split from date ──────────────────────────────────
    # May 1-5 → train (71%), May 6 → test (14%), May 7 → live (15%).
    # Gold job and API both group/filter by this column.
    silver = silver.withColumn(
        "data_split",
        when(dayofmonth(col("base_datetime")) <= 5, "train")
        .when(dayofmonth(col("base_datetime")) == 6, "test")
        .otherwise("live")
    )

    # ── Step 9: Derive partition columns from base_datetime ───────────────────
    silver = (
        silver
        .withColumn("year",  spark_year(col("base_datetime")))
        .withColumn("month", spark_month(col("base_datetime")))
        .withColumn("day",   dayofmonth(col("base_datetime")))
    )

    # ── Step 10: Write Silver Delta ────────────────────────────────────────────
    print(f"\nWriting to Silver Delta: {DELTA_SILVER_PATH}")
    (
        silver.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("year", "month", "day")
        .save(DELTA_SILVER_PATH)
    )

    count = spark.read.format("delta").load(DELTA_SILVER_PATH).count()
    print(f"Silver Delta has {count:,} records")
    print(f"\nNext: run gold_job.py")
    spark.stop()


if __name__ == "__main__":
    main()
