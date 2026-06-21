"""
kafka_producer.py — Maritime Navigation AI System
Streams LIVE split AIS data to Kafka file by file.
Never loads more than one parquet file at a time — no memory crash.

Start with:
    docker compose exec producer python src/producer/kafka_producer.py
"""
import json, os, sys, time
import pandas as pd
from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

sys.path.insert(0, "/app/src/common")
from config import (
    KAFKA_BOOTSTRAP_SERVERS, AIS_TOPIC,
    PARQUET_DATA_PATH, STREAM_DELAY_SECONDS, LOOP,
)
from schema_utils import normalize_ais_record, is_valid_position

FLUSH_EVERY = 200


def wait_for_kafka(retries=30, delay=3) -> KafkaProducer:
    for attempt in range(1, retries + 1):
        try:
            p = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(
                    v, default=str).encode("utf-8"),
                key_serializer=lambda v: str(v).encode("utf-8"),
                linger_ms=5,
                batch_size=32768,
                acks="all",
                retries=5,
            )
            print(f"Kafka connected (attempt {attempt})")
            return p
        except (NoBrokersAvailable, Exception) as e:
            print(f"Waiting for Kafka ({attempt}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError("Kafka unreachable after retries")


def get_live_files(parquet_path: str) -> list:
    """Find parquet files belonging to live split by scanning data_split column only."""
    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    all_files  = sorted(p.rglob("*.parquet"))
    live_files = []

    print(f"Scanning {len(all_files)} parquet files for live split...")

    for f in all_files:
        try:
            # Read only data_split column — very fast, no memory issue
            sample = pd.read_parquet(f, columns=["data_split"])
            if "live" in sample["data_split"].unique():
                live_files.append(f)
        except Exception:
            # No data_split column — include file
            live_files.append(f)

    if not live_files:
        print("No live split found — using first 3 files as fallback")
        live_files = all_files[:3]

    print(f"Found {len(live_files)} live split files")
    return live_files


def stream_file(parquet_file: Path, producer: KafkaProducer) -> tuple:
    """Load ONE parquet file and stream its live rows to Kafka."""
    try:
        df = pd.read_parquet(parquet_file)

        # Filter to live split
        if "data_split" in df.columns:
            df = df[df["data_split"] == "live"]

        if len(df) == 0:
            return 0, 0

        # Sort by time
        if "base_datetime" in df.columns:
            df["base_datetime"] = pd.to_datetime(
                df["base_datetime"], errors="coerce"
            )
            df = df.sort_values("base_datetime")

        total   = len(df)
        sent    = 0
        skipped = 0

        print(f"\n  File: {parquet_file.name} — {total:,} rows")

        for i, row in enumerate(df.itertuples(index=False), 1):
            event = normalize_ais_record(row._asdict())
            event["data_split"] = "live"

            if not event["mmsi"] or not is_valid_position(event):
                skipped += 1
                continue

            producer.send(AIS_TOPIC, key=event["mmsi"], value=event)
            sent += 1

            if sent % FLUSH_EVERY == 0:
                producer.flush()
                print(f"  [{sent:>8,}/{total:>8,}] "
                      f"{str(event.get('vessel_name',''))[:15]:<15} "
                      f"SOG={event['sog']:5.1f}kn")

            time.sleep(STREAM_DELAY_SECONDS)

        producer.flush()
        return sent, skipped

    except Exception as e:
        print(f"  Error: {e}")
        return 0, 0


def main():
    live_files = get_live_files(PARQUET_DATA_PATH)
    producer   = wait_for_kafka()
    run        = 0

    print(f"\nStarting stream -> '{AIS_TOPIC}'")
    print(f"Rate : ~{1/STREAM_DELAY_SECONDS:.0f} msg/sec")
    print(f"Files: {len(live_files)}")
    print(f"Loop : {LOOP}")

    while True:
        run += 1
        print(f"\n--- Run #{run} ---")
        total_sent = 0

        for f in live_files:
            sent, skipped = stream_file(f, producer)
            total_sent   += sent
            print(f"  Done: {f.name} -> sent {sent:,} / skipped {skipped:,}")

        print(f"\nRun #{run} complete — {total_sent:,} total sent")

        if not LOOP:
            break

        print(f"Loop #{run} complete — restarting from beginning")

    producer.close()
    print("Producer shut down.")


if __name__ == "__main__":
    main()