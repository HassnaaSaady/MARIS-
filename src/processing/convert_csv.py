"""
convert_csv.py — Maritime Navigation AI System
Converts all raw CSV files to partitioned Parquet format.
Uses chunked reading to handle large files (800MB+) without memory issues.

Split plan for 7 days:
    Files 1-5  (index 0-4) → train      (71%) ← ML training data
    File  6    (index 5)   → test       (14%) ← ML evaluation
    File  7    (index 6)   → live       (14%) ← dashboard shows this

When you add more files later (14 days):
    Files 1-10 → train
    Files 11-12 → test
    Files 13-14 → live
"""
import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from schema_utils import resolve_columns, fix_mmsi

RAW_DIR    = Path(os.getenv("RAW_DATA_PATH",    "/app/data/raw"))
OUTPUT_DIR = Path(os.getenv("PARQUET_DATA_PATH", "/app/data/parquet"))

CHUNK_SIZE = 100_000


def get_split_label(file_index: int, total_files: int) -> str:
    train_end = int(total_files * 0.71)  # 5 files
    test_end  = int(total_files * 0.86)  # 1 file
    # live = 1 file (last)
    if file_index < train_end:
        return "train"
    elif file_index < test_end:
        return "test"
    else:
        return "live"


def process_chunk(chunk: pd.DataFrame, split: str) -> pd.DataFrame:
    """Clean and enrich one chunk of rows."""

    # Standardise column names
    chunk.columns = [c.strip() for c in chunk.columns]
    chunk = resolve_columns(chunk)

    # Fix MMSI
    if "mmsi" in chunk.columns:
        chunk["mmsi"] = chunk["mmsi"].apply(fix_mmsi)

    # Parse timestamp
    # if "base_datetime" in chunk.columns:
    #     chunk["base_datetime"] = pd.to_datetime(
    #         chunk["base_datetime"], errors="coerce"
    #     )
    # else:
    #     for col in chunk.columns:
    #         if "date" in col.lower() or "time" in col.lower():
    #             chunk["base_datetime"] = pd.to_datetime(
    #                 chunk[col], errors="coerce"
    #             )
    #             break
    if "base_datetime" in chunk.columns:
        chunk["base_datetime"] = pd.to_datetime(
            chunk["base_datetime"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )
    else:
        for col in chunk.columns:
            if "date" in col.lower() or "time" in col.lower():
                chunk["base_datetime"] = pd.to_datetime(
                    chunk[col],
                    format="%Y-%m-%d %H:%M:%S",
                    errors="coerce"
                )
                break

    # CRITICAL FIX: Cast to microseconds so Spark 3.4 can read Parquet
    # Pandas default = nanoseconds → Spark 3.4 crashes with TIMESTAMP(NANOS)
    if "base_datetime" in chunk.columns:
        chunk["base_datetime"] = chunk["base_datetime"].astype("datetime64[us]")

    # Date parts for partitioning
    if "base_datetime" in chunk.columns:
        chunk["year"]  = chunk["base_datetime"].dt.year
        chunk["month"] = chunk["base_datetime"].dt.month
        chunk["day"]   = chunk["base_datetime"].dt.day
        chunk["hour"]  = chunk["base_datetime"].dt.hour

    # Numeric columns
    for col in ["lat", "lon", "sog", "cog", "heading",
                "length", "width", "draft"]:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(
                chunk[col], errors="coerce"
            ).fillna(0.0)

    # String columns
    for col in ["vessel_name", "imo", "call_sign",
                "vessel_type", "status", "cargo", "transceiver_class"]:
        if col in chunk.columns:
            chunk[col] = chunk[col].fillna("").astype(str).str.strip()
        else:
            chunk[col] = ""

    # Quality filters
    chunk = chunk.dropna(subset=["lat", "lon", "base_datetime"])
    chunk = chunk[
        (chunk["lat"] != 0.0) & (chunk["lon"] != 0.0) &
        (chunk["lat"].between(-90, 90)) &
        (chunk["lon"].between(-180, 180))
    ]
    if "mmsi" in chunk.columns:
        chunk = chunk[chunk["mmsi"].str.strip() != ""]
        chunk = chunk[chunk["mmsi"] != "nan"]
    if "sog" in chunk.columns:
        chunk = chunk[chunk["sog"].between(0, 60)]

    if len(chunk) == 0:
        return chunk

    # Split label
    chunk["data_split"] = split

    # Basic risk enrichment
    chunk["risk_level"] = "LOW"
    chunk.loc[chunk["sog"] < 2.0, "risk_level"] = "MEDIUM"
    chunk.loc[
        (chunk["sog"] < 1.0) &
        (chunk["lat"].between(29.5, 31.5)) &
        (chunk["lon"].between(31.0, 33.5)),
        "risk_level"
    ] = "HIGH"

    # Grid bins
    chunk["lat_bin"] = (chunk["lat"] / 0.1).astype(int) * 0.1
    chunk["lon_bin"] = (chunk["lon"] / 0.1).astype(int) * 0.1

    return chunk


def convert_file(csv_path: Path, file_index: int, total_files: int) -> dict:
    """Convert one large CSV file to Parquet using chunked reading."""
    split      = get_split_label(file_index, total_files)
    start_time = datetime.now()
    total_rows = 0
    total_kept = 0
    chunk_num  = 0

    print(f"  Split      : {split}")

    try:
        reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False)

        for chunk in reader:
            chunk_num  += 1
            total_rows += len(chunk)
            processed   = process_chunk(chunk, split)

            if len(processed) == 0:
                continue

            total_kept += len(processed)

            processed.to_parquet(
                OUTPUT_DIR,
                partition_cols=["year", "month", "day"],
                engine="pyarrow",
                index=False,
                existing_data_behavior="overwrite_or_ignore",
            )

            if chunk_num % 10 == 0:
                print(f"  Chunk {chunk_num:>4} | "
                      f"processed: {total_rows:>10,} | "
                      f"kept: {total_kept:>10,}")

        elapsed = (datetime.now() - start_time).seconds
        print(f"  Done — {total_rows:,} raw -> {total_kept:,} kept | {elapsed}s")

        return {"file": csv_path.name, "status": "ok",
                "rows": total_kept, "split": split}

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"file": csv_path.name, "status": "failed", "rows": 0}


def main():
    print("=" * 60)
    print("  Maritime AI - CSV to Parquet Converter")
    print("=" * 60)

    csv_files = sorted([
        f for f in RAW_DIR.glob("*.csv")
        if not str(f).endswith(".zst")
    ])

    if not csv_files:
        print(f"\nNo CSV files found in {RAW_DIR}")
        return

    total     = len(csv_files)
    train_end = int(total * 0.70)
    test_end  = int(total * 0.85)

    print(f"\nFound {total} CSV files")
    print(f"    Split plan ({total} files):")
    print(f"    Train : files 1-{train_end}  ({train_end} files)")
    print(f"    Test  : files {train_end+1}-{test_end}  ({test_end-train_end} files)")
    print(f"    Live  : files {test_end+1}-{total}  ({total-test_end} files) <- dashboard")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, f in enumerate(csv_files):
        print(f"\n[{i+1}/{total}] Processing: {f.name}")
        results.append(convert_file(f, i, total))

    print("\n" + "=" * 60)
    ok         = [r for r in results if r["status"] == "ok"]
    total_kept = sum(r.get("rows", 0) for r in ok)

    print(f"Converted: {len(ok)}/{total} files")
    print(f"Total rows: {total_kept:,}")
    for split in ["train", "test", "live"]:
        sf = [r for r in ok if r.get("split") == split]
        sr = sum(r.get("rows", 0) for r in sf)
        marker = " <- dashboard" if split == "live" else ""
        print(f"  {split:<8}: {len(sf)} files, {sr:,} rows{marker}")


if __name__ == "__main__":
    main()