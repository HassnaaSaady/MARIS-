# Maritime Navigation AI System — Technical Knowledge Report
## PART 3: 120 Q&A DEFENSE GUIDE, PRESENTATION FLOW & FINAL ASSESSMENT

---

## SECTION 12 — DISCUSSION & DEFENSE PREPARATION: 120 QUESTIONS

### 12.1 Data Engineering Questions (40 Questions)

---

**Q1: Why did you choose the Medallion Architecture (Bronze/Silver/Gold)?**

**Answer:** The Medallion Architecture provides a structured data quality progression. Bronze stores raw, immutable data — our "source of truth" for reprocessing. Silver applies all transformations (cleaning, deduplication, feature engineering) to produce production-quality data. Gold pre-aggregates for serving. This separation means: if we find a bug in Silver cleaning, we rerun Silver from Bronze without re-ingesting source data. Multiple consumers (ML training, dashboard, API) read the tier appropriate to their needs. The pattern is industry-standard, used by Databricks, Uber, Netflix, and scales naturally as data volume grows.

**Follow-up:** "Could you have done this in one step?" Yes, but a monolithic pipeline is fragile — a failure in aggregation would require restarting the entire ingestion. Separation of concerns improves reliability and maintainability.

**Senior engineer perspective:** They'll ask how you handle schema evolution in Bronze when the AIS data format changes. Answer: Delta Lake supports `mergeSchema=true` on write, allowing new columns to be added without breaking existing readers.

---

**Q2: What is Delta Lake and why did you use it instead of plain Parquet?**

**Answer:** Delta Lake is an open-source storage layer on top of Parquet that adds ACID transactions, schema enforcement, and time travel. We chose it over plain Parquet because: (1) ACID — concurrent reads and writes are safe without corruption risks; (2) Schema enforcement — rejects records with wrong column types at write time, preventing silent data quality degradation; (3) Time travel — we can query historical states via `versionAsOf` for debugging and reprocessing; (4) Upsert support — `merge into` enables efficient record-level updates, important for the Gold vessel_latest table. Plain Parquet has none of these guarantees — concurrent writers can corrupt files, there's no schema check, and no time travel.

**Follow-up:** "What's the difference between Delta Lake and Apache Iceberg?" Both provide ACID transactions and time travel on top of Parquet. Delta Lake has stronger Spark integration and is the Databricks ecosystem choice. Iceberg offers better multi-engine support (Flink, Spark, Trino all read the same table). For our purely Spark-based pipeline, Delta Lake is the natural choice.

---

**Q3: Why do you partition the Silver layer by year/month/day?**

**Answer:** Date partitioning allows Spark to skip entire directory trees when reading time-range queries. If a query asks for "data from May 5," Spark reads only `/silver/year=2025/month=5/day=5/` instead of scanning all 5M rows. This is called partition pruning. Without partitioning, every query would scan the full dataset. The performance difference is 10–100x for selective date queries.

**Follow-up:** "Why not partition by MMSI?" Partitioning by MMSI would create 1,000+ partition directories. With small files in each, this creates the "small files problem" — excessive metadata overhead. Date partitioning is better for our primary query patterns (time-range analytics).

---

**Q4: Why does the Silver window partition by MMSI only (not MMSI + day)?**

**Answer:** If we partitioned by (MMSI, day), the first record of each day would have no previous record (NULL prev values), even though the vessel's last position from yesterday is available. This means every vessel would generate a "first ping" at midnight, wasting training data — roughly 50% of records near midnight would be dropped. By partitioning only by MMSI, the lag window continues across day boundaries. A vessel's 00:00:01 record correctly sees its 23:59:45 record as its predecessor. This was a deliberate design decision to maximize training data quality, at the cost of higher executor memory pressure (all records for one MMSI must be in the same partition).

**Senior engineer follow-up:** "What if one vessel has 500,000 records? Wouldn't that cause a memory spill?" Yes. For this dataset (~5M rows, ~1,000 vessels), average records per vessel is ~5,000 — manageable. For global AIS data (100M+ rows/day), we'd need to partition by (MMSI hash bucket + date) with a custom overlap-handling strategy.

---

**Q5: Explain the haversine formula and why you didn't use Euclidean distance.**

**Answer:** The haversine formula computes great-circle distance on a sphere:
```
a = sin²(Δlat/2) + cos(lat₁)×cos(lat₂)×sin²(Δlon/2)
d = 2R × atan2(√a, √(1-a))
```
where R = 3440.065 nautical miles (Earth radius in nm). We use haversine instead of Euclidean because lat/lon are angular coordinates, not a flat grid. A 1° longitude difference equals ~60nm at the equator but ~30nm at 60°N. Euclidean distance ignores this, creating errors up to 50% at high latitudes. For our US waters data (25°N–50°N), Euclidean error ranges from 13% to 36%. Haversine is accurate to < 0.5% for maritime distances. Vincenty's formula is more accurate (to millimeters) but 100× slower — unnecessary for our use case.

---

**Q6: What is the teleport filter and how does it work?**

**Answer:** The teleport filter removes GPS glitch records where a vessel appears to "jump" impossibly large distances between consecutive pings. We compute: `implied_speed = (distance_nm / time_delta_sec) × 3600`. If implied_speed > 100 knots, the record is dropped. 100 knots was chosen because it exceeds the maximum speed of any commercial vessel (~60 knots for military hydrofoils) while providing buffer for measurement error. Without this filter, GPS glitches — where a vessel appears to jump 500nm in 5 seconds — would train the position predictor on impossible trajectories, causing it to predict absurd future positions.

**Follow-up:** "What if a legitimate fast military vessel is in your dataset?" It would be filtered out at implied speeds above 100 knots, which even naval vessels don't exceed. The risk is negligible for commercial maritime monitoring.

---

**Q7: How do you handle AIS sentinel values (102.3 for SOG, 511 for Heading)?**

**Answer:** The ITU-R M.1371 AIS standard reserves specific values to indicate "data not available": SOG=102.3 knots and Heading=511 degrees. These are not real measurements — they're protocol codes. We detect them in the Silver job and replace with NULL: `sog = NULLIF(sog, 102.3)`. If we treated 102.3 as a real speed, it would appear as an extreme outlier and be flagged as an anomaly by the Isolation Forest — creating massive false positive rates. After nullification, records with NULL SOG are either dropped or handled with the model's NULL-safe logic.

---

**Q8: Why do you deduplicate on (MMSI, base_datetime) rather than on all fields?**

**Answer:** AIS signals are broadcast over VHF radio and received by multiple shore receivers simultaneously. The same physical broadcast may arrive from Station A and Station B, creating two identical rows in the dataset. Deduplication on (MMSI, base_datetime) — the natural primary key for AIS data — removes these exact duplicates. We don't include lat/lon in the dedup key because two receivers might decode slightly different positions from the same broadcast (signal noise), but these should be treated as the same event. Using all fields would leave these near-duplicates in, causing Δt=0 pairs and infinite implied speeds in the teleport filter.

---

**Q9: What does your Bronze layer NOT do, and why?**

**Answer:** Bronze does NOT: (1) remove duplicates — deduplication requires sorting by MMSI+datetime, which is a cross-record operation; Bronze keeps raw data. (2) Compute lag features — these require window functions across vessel tracks, complex and expensive; reserved for Silver. (3) Apply ML scoring — ML models operate on clean, feature-engineered data; Bronze is raw. (4) Aggregate — aggregation belongs in Gold. Bronze's philosophy: ingest, validate minimally, enrich lightly, preserve. The rule is: "Don't lose data in Bronze; clean it in Silver." This ensures we can always reprocess if cleaning logic changes.

---

**Q10: How would you make the Silver job handle late-arriving data (data arriving after its timestamp)?**

**Answer:** Our current Silver job is a batch job — it processes Bronze data up to now. If a vessel's AIS message was delayed (received by satellite AIS 2 hours after broadcast), it would arrive in the next Bronze batch and be processed in the next Silver run. For strict correctness, we would: (1) Run Silver on a micro-batch schedule (every 15 minutes), (2) Use a watermark in Structured Streaming: `withWatermark("base_datetime", "2 hours")` to tolerate 2-hour late arrivals, (3) Use Delta Lake's MERGE INTO to upsert records into Silver rather than just appending. Our current implementation accepts up to 1 batch-cycle of latency (~daily), which is acceptable for analytics use cases.

---

**Q11: Explain the data split strategy and why temporal splitting matters.**

**Answer:** We split 14 days of data: Days 1-8 = TRAIN (57%), Days 9-10 = VALIDATION (14%), Days 11-12 = TEST (14%), Days 13-14 = LIVE (15%). This is a temporal split — training data comes before validation, which comes before test. Random splitting would commit temporal leakage: if Day 10 rows appear in training while Day 7 rows appear in test, the model sees "future" information during training. For time-series models (position predictor, congestion predictor), this inflates metrics by 20-40% in our estimation. Temporal splitting ensures test performance reflects true generalization to future, unseen data — the real production scenario.

---

**Q12: What is the fact_vessel_latest table and why is it necessary?**

**Answer:** fact_vessel_latest contains exactly one row per unique MMSI, representing the vessel's most recent position and ML scores. Without it, a live map query would require: `SELECT mmsi, lat, lon, ... FROM fact_ais_track WHERE base_datetime = (SELECT MAX(base_datetime) FROM fact_ais_track WHERE mmsi = ...)` for each of 1,000+ vessels — this would take 30+ seconds. fact_vessel_latest reduces this to a simple `SELECT * FROM fact_vessel_latest WHERE data_split='live'` — milliseconds. The trade-off is staleness: if we fail to update fact_vessel_latest for a vessel, the map shows its last known position until the next live_scorer run (< 5 seconds). This is acceptable for our 30-second dashboard refresh interval.

---

**Q13: How does the Kafka producer simulate a live AIS stream?**

**Answer:** The producer reads Parquet files from the LIVE split (Days 13-14), sorts each file by base_datetime, and publishes records to the Kafka `ais_raw` topic with a configurable delay between messages (`STREAM_DELAY_SECONDS=0.002`, producing ~500 messages/second). The Kafka key is set to MMSI, ensuring all messages from the same vessel go to the same Kafka partition (preserving per-vessel ordering). The producer loops through files (if `LOOP=true`) to simulate continuous streaming. The simulation doesn't replicate the exact timing of the original broadcasts — it plays them back at a uniform rate, which is sufficient for demonstrating the real-time pipeline.

---

**Q14: How does Spark Structured Streaming write to Bronze Delta?**

**Answer:** The `spark_streaming_consumer.py` reads from Kafka using Spark Structured Streaming's built-in Kafka connector. Schema is defined explicitly (not inferred) for performance. Data is filtered (null MMSI, invalid lat/lon rejected), enriched (speed flags, port zones, risk level added), then written to Bronze Delta using `writeStream.format("delta").outputMode("append").option("checkpointLocation", "/delta/checkpoints/bronze").trigger(processingTime="10 seconds").start()`. The 10-second trigger means Spark processes all Kafka messages received in the last 10 seconds as one micro-batch. The checkpoint directory stores Kafka offsets and stream state, enabling exactly-once semantics and recovery from failures.

---

**Q15: Why do you drop first pings in Silver (records with time_delta_sec = NULL)?**

**Answer:** First pings have no previous record — they are the first AIS broadcast captured for a vessel (or after a coverage gap). Without a previous record, we cannot compute: time_delta_sec, prev_lat/lon, sog_change, heading_change, or distance_nm. These are 5 of our 6 anomaly detection features. Imputing them (filling with zeros or means) would be misleading: a zero distance_nm for a first ping doesn't mean the vessel is stationary; it means we have no information. Including first pings with NULL or imputed features would corrupt ML training. We drop them to ensure every Silver record has a complete, meaningful feature set. The quantity dropped: approximately 1 per vessel per session start — ~1,000-5,000 rows out of 5M (< 0.1%).

---

**Q16: What is the difference between SOG and STW (Speed Through Water)?**

**Answer:** SOG (Speed over Ground) is the vessel's speed relative to the Earth's surface — measured by GPS. STW (Speed Through Water) is the vessel's speed relative to the water mass — measured by a mechanical log or Doppler. SOG = STW + current_speed_component. In a 2-knot favorable current, a vessel at 10 knots STW achieves 12 knots SOG. AIS broadcasts SOG (not STW). This matters for our anomaly detection: a vessel might have an unusual SOG not because of its own behavior but because of unusual ocean currents. Our system cannot distinguish these without current data — a known limitation.

---

**Q17: How do you handle the case where a vessel stops transmitting for hours, then reappears?**

**Answer:** This creates a coverage gap — the vessel may have been outside AIS receiver range (common in open ocean), turned off its transponder (AIS dark event — suspicious for certain vessel types), or experienced transponder malfunction. In our pipeline: the first record after a gap has time_delta_sec = hours (large value), distance_nm = large (wherever the vessel traveled in silence), and implied_speed may exceed 100 knots → teleport filter drops the record. The record is effectively treated as a new first ping. This is conservative — we don't flag AIS dark events explicitly. A production system would monitor for "expected vessel not transmitting" and generate an alert after a configurable silence period.

---

**Q18: Explain your fact_traffic_density schema and how it enables the heatmap.**

**Answer:** fact_traffic_density aggregates vessel counts per 0.1° grid cell per 1-hour time bucket. Schema: (lat_bin FLOAT, lon_bin FLOAT, hour_bucket DATETIME, vessel_count BIGINT, unique_vessels BIGINT, avg_sog FLOAT, stopped_count BIGINT, congestion_level TEXT). The Streamlit dashboard queries: `SELECT lat_bin, lon_bin, vessel_count, congestion_level FROM fact_traffic_density WHERE hour_bucket = '2025-05-06 14:00:00'`. Each row maps directly to a geographic tile in the heatmap renderer — the higher the vessel_count, the more intense the heat color. Without pre-aggregation, the dashboard would need to aggregate 5M Silver rows live on every refresh, taking 30+ seconds.

---

**Q19: Why is PostgreSQL write using JDBC and not a Python ORM?**

**Answer:** Gold job is a PySpark job running in the Spark execution environment. SQLAlchemy (Python ORM) doesn't integrate with Spark's distributed execution model — it runs on the driver only and would serialize all data to the driver before writing (losing the distributed advantage). JDBC writing allows Spark executors to write directly to PostgreSQL in parallel: `df.write.format("jdbc").option("url", POSTGRES_URL).option("dbtable", "fact_traffic_density").mode("overwrite").save()`. This parallelizes the write across executor nodes, significantly faster for large DataFrames. The FastAPI separately uses SQLAlchemy for read queries, which is appropriate since the API is a single-threaded Python process.

---

**Q20: How does dim_vessel get populated and updated?**

**Answer:** The Gold job aggregates vessel metadata from the Silver layer: `SELECT mmsi, FIRST(vessel_name), FIRST(vessel_type), FIRST(imo), MIN(base_datetime) as first_seen, MAX(base_datetime) as last_seen, COUNT(*) as total_records FROM silver GROUP BY mmsi`. This is written to dim_vessel with `mode="overwrite"` on each Gold job run. This implements SCD Type 1 (overwrite) — if a vessel changes its name, the new name overwrites the old one in dim_vessel. A production system would use SCD Type 2 (insert new row with valid_from/valid_to timestamps) to track name changes over time. We chose Type 1 for simplicity, acknowledging that vessel name history is lost between Gold job runs.

---

**Q21: Explain MMSI scientific notation problem and how you solve it.**

**Answer:** In the raw MarineCadastre parquet files, MMSI values are sometimes encoded as float in scientific notation (e.g., 368000001 → 3.68E+08). When pandas reads parquet and casts float to string, you get "368000000.0" or "3.68e+08". Our `fix_mmsi()` function in `schema_utils.py` handles this:
```python
def fix_mmsi(val):
    try:
        return str(int(float(val)))  # "3.68E+08" → 368000000 → "368000000"
    except:
        return None
```
Without this, MMSI "3.68E+08" and MMSI "368000000" would appear as different vessels — breaking vessel tracking, splitting one vessel's track into two, and corrupting all lag/delta features.

---

**Q22: What happens to your pipeline if Kafka goes down?**

**Answer:** Three components are affected:
1. **Spark Streaming Consumer:** Uses checkpoints — when Kafka recovers, it resumes from the last committed offset. No data loss (up to Kafka's 7-day retention). 
2. **Live Scorer:** Reconnects automatically with retry logic. Kafka messages accumulated during downtime are processed in order when reconnected.
3. **Streamlit Dashboard:** Switches to demo mode (synthetic vessel generation) when Kafka is unavailable. Users see fake vessels on the map — clearly labeled "DEMO MODE."
The batch pipeline (Bronze/Silver/Gold Spark jobs) is unaffected — they read from Delta Lake, not Kafka.

---

**Q23: What is a Kafka consumer group and why does it matter?**

**Answer:** A Kafka consumer group is a set of consumers sharing a topic subscription, with each partition assigned to exactly one consumer in the group. Messages are delivered to only one consumer per group. We have two consumer groups: `spark-bronze-consumer` (Spark Streaming reading for Bronze), and `live-scorer-group` (live_scorer.py reading for ML scoring). Because they are different groups, each gets a full copy of every message. If they shared a group, Spark would receive some messages and the live scorer would receive others — both would have incomplete data. Consumer groups enable fan-out: multiple independent processors reading the same stream.

---

**Q24: How does your system handle the "small files problem" in Delta Lake?**

**Answer:** Delta Lake's `OPTIMIZE` command compacts small files into larger ones (target: 1GB per file). `Z-ORDER BY (mmsi, base_datetime)` physically co-locates related data for faster queries. Our Silver job is a batch job that writes to date partitions — each partition receives one write, producing one Parquet file per partition, avoiding the small files problem at write time. The Structured Streaming Bronze job writes micro-batches every 10 seconds, creating many small files. In production, a scheduled `OPTIMIZE` command every few hours would compact these. For the demo, the files are small enough that this is not a bottleneck.

---

**Q25: What is the difference between `append` and `overwrite` mode in Delta Lake?**

**Answer:**
- `append`: Adds new data without modifying existing data. Used for Bronze (immutable log of all received records) and fact_ais_track (historical record of all positions — never delete).
- `overwrite`: Replaces all existing data. Used for Gold tables (vessel_latest, traffic_density, daily_stats) which are regenerated from scratch on each Gold job run with the latest data.
- `merge` (not currently used but should be): Upserts — update existing rows if key matches, insert if new. Ideal for fact_vessel_latest but currently implemented as overwrite for simplicity.

**Interview trap:** "Isn't overwriting fact_vessel_latest slow?" Yes — it truncates and rewrites all ~1,000 rows on every Gold job. With 1,000 rows this is fast. At 1M vessels, MERGE INTO would be required.

---

**Q26: How do you ensure exactly-once semantics in your pipeline?**

**Answer:** Our current implementation provides at-least-once semantics (not exactly-once):
- Kafka → Spark Structured Streaming: checkpoints prevent reprocessing on recovery, but transient errors can cause duplicate processing within a micro-batch
- Spark → Delta Lake: Delta Lake write transactions are atomic (either full micro-batch committed or nothing), preventing partial writes
- Gold job → PostgreSQL: `overwrite` mode is idempotent — running twice produces the same result
- Live Scorer → PostgreSQL: UPSERT on MMSI primary key means re-processing the same message simply overwrites with identical data — effectively idempotent
For production exactly-once: configure Kafka idempotent producer + enable Spark's `spark.streaming.kafka.allowNonConsecutiveOffsets=false` and use Delta Lake `foreachBatch` with merge semantics.

---

**Q27: Why is the congestion prediction target the NEXT hour, not the current hour?**

**Answer:** This is a leakage prevention decision. The current congestion level can be computed trivially from the current vessel_count — it's already known (it's in our features). A model that predicts current congestion from current vessel_count would achieve ~100% accuracy but be completely useless (it tells you what you already know). By shifting the target to the NEXT hour's vessel_count, the model learns to predict FUTURE congestion from CURRENT signals. This is operationally valuable: "Given that this grid cell has 12 vessels at 14:00, will it reach HIGH congestion at 15:00?" The consecutive-hour-only filter ensures we only train on valid hour-to-next-hour pairs.

---

**Q28: What is the JDBC batch size and why does it matter for PostgreSQL write performance?**

**Answer:** When writing via JDBC, data is batched into SQL INSERT statements. The default batch size in Spark JDBC writer is 1,000 rows per batch. For our fact_traffic_density table (~100,000 rows), this means ~100 sequential INSERT batches. Setting `batchsize=10000` reduces this to 10 batches — significantly faster. We can set: `.option("batchsize", "10000")`. PostgreSQL performance with batched INSERTs: ~100,000 rows/second with optimal batching. Without batching (one INSERT per row): ~1,000 rows/second — 100× slower. This optimization matters for Gold job runtime.

---

**Q29: How does your Silver job handle NULL values in SOG when computing sog_change?**

**Answer:** `sog_change = current_sog - prev_sog`. If either is NULL (sentinel-nullified or original NULL), the SQL subtraction returns NULL. NULL features cannot be passed to ML models — sklearn raises ValueError. Our training code includes NaN guards: `df.dropna(subset=ANOMALY_FEATURES)` removes rows with ANY NULL in the 6 anomaly features before training. At inference, the scorer checks for NaN/Inf before scoring: if the feature vector contains NaN, the record is scored with rules only (no ML). This fail-loud approach (raising RuntimeError if NaN/Inf appear after cleaning) ensures data quality issues are caught during development, not silently producing incorrect scores.

---

**Q30: What would break if you removed the heading_change wrap-around correction?**

**Answer:** Without the wrap-around correction, vessels crossing the 0°/360° boundary would appear to make extreme turns:
- Vessel turning from 355° to 005°: raw = 005-355 = -350°; abs(-350) = 350° — appears to be a 350° turn (near-full circle)
- Correct value: 10° turn (5° left of true north, then 5° right)
Impact on anomaly detection: every vessel approaching a north-heading waterway (heading near 360°→0°) would trigger SHARP_TURN anomaly (>45°) falsely. At Houston Ship Channel (which has north-running sections), this would create thousands of false alerts. The wrap-around correction is a critical correctness fix — not an optimization.

---

**Q31 - Q40: Additional Data Engineering Q&A (Summary Format)**

**Q31:** Why use `FIRST_VALUE IGNORE NULLS` for vessel name filling?
**A:** Forward-fills the first non-NULL vessel name to all subsequent records for the same MMSI, so even records with blank name fields show the vessel's known name. `IGNORE NULLS` skips null values in the window aggregation.

**Q32:** What is partition pruning and when does it apply?
**A:** Spark reads only the partitions that satisfy a WHERE clause filter. `WHERE base_datetime BETWEEN '2025-05-05' AND '2025-05-07'` reads only year=2025/month=5/day=5 and day=7. Requires that the filter column matches the partition column exactly.

**Q33:** Why do you store both Delta Lake and PostgreSQL (isn't that redundant)?
**A:** Delta Lake is the durable data lake — the source of truth, optimized for large-scale batch processing. PostgreSQL is the operational serving layer, optimized for low-latency queries from the dashboard and API. Two stores with different access patterns are the norm in modern data architectures (OLAP vs. OLTP).

**Q34:** How many unique vessels are in your dataset?
**A:** ~1,000+ unique MMSIs in the LIVE split visible on the map. The full Silver dataset contains substantially more — all vessels observed in US waters during May 1-7, 2025.

**Q35:** How long does the Silver job take to run?
**A:** Approximately 15-30 minutes for 36M Bronze rows → 5M Silver rows on our 2-worker cluster (2×5GB, 2×4 cores). Most time is spent on the MMSI-partitioned window functions (shuffle-intensive).

**Q36:** What is the checkpoint in Spark Streaming and why is it critical?
**A:** The checkpoint stores Kafka consumer offsets and streaming state to disk. On failure/restart, Spark resumes exactly where it left off. Without checkpoints, a restart would reprocess Kafka from the beginning (or latest offset, depending on config), causing data duplication or loss.

**Q37:** Why is your Bronze layer "append-only"?
**A:** The Bronze layer is the immutable raw log — it should never be modified. If we need to correct an error, we add a corrected record (not replace the original). This enables full auditability: "show me every record we ever received" is always answerable from Bronze.

**Q38:** How do you know when Bronze → Silver → Gold pipeline has completed successfully?
**A:** Currently, each job logs completion metrics: rows written, rows dropped, timing. In production, we'd use Apache Airflow to orchestrate these as a DAG: Bronze job finishes → trigger Silver → Silver finishes → trigger Gold → Gold finishes → send completion notification. With MLflow, metrics would be logged automatically.

**Q39:** What does `data_split` column mean in your tables?
**A:** It indicates which time-based split the record belongs to: 'train' (Days 1-8), 'validation' (Days 9-10), 'test' (Days 11-12), 'live' (Days 13-14). This allows the dashboard to show only LIVE data on the map (`WHERE data_split='live'`) while ML training queries only TRAIN data (`WHERE data_split='train'`), all from the same table.

**Q40:** What is the estimated storage size of each layer?
**A:** Bronze: ~36M rows × ~200 bytes per row (columnar Parquet) ≈ ~7GB. Silver: ~5M rows × ~200 bytes ≈ ~1GB (partitioned). Gold: ~100K rows (density table) + small tables ≈ ~50MB. PostgreSQL: ~5M rows in fact_ais_track + small Gold tables ≈ ~2GB. Total system storage: ~10-12GB for 7 days of US waters data.

---

### 12.2 Machine Learning Questions (40 Questions)

---

**Q41: Why did you choose Isolation Forest for anomaly detection?**

**Answer:** We chose Isolation Forest because: (1) We have no labeled anomaly data — Isolation Forest is unsupervised; (2) It scales efficiently to millions of records — O(n log n) training, O(log n) inference; (3) It stores only the trees, not the training data, making inference memory-efficient; (4) It handles high-dimensional tabular data without feature engineering specific to the algorithm; (5) It's well-validated in the industry for maritime and network anomaly detection. Alternatives (One-Class SVM, LOF, Autoencoder) either require labels, scale poorly, or require significantly more engineering for marginal improvement.

**Follow-up:** "Is Isolation Forest state-of-the-art?" Not for all domains. For tabular data anomaly detection, extended isolation forest (EIF) and COPOD have shown improvements. For deep learning on sequence data, LSTM autoencoders outperform. Our choice is appropriate for the data type and engineering constraint.

---

**Q42: What does the contamination parameter of 0.002 mean, and how did you choose it?**

**Answer:** Contamination is the expected proportion of outliers in the training data — it sets the decision threshold. With contamination=0.002, the top 0.2% most-isolated records are labeled as anomalies. We chose 0.2% through domain reasoning: maritime AIS data is overwhelmingly normal (vessels following routine routes); true anomalies are rare. We tested contamination=0.01 (1%) and observed too many false positives (normal port maneuvers flagged). At 0.002, only extreme behaviors are flagged. In production, this would be calibrated using operator feedback on alert usefulness.

---

**Q43: How does XGBoost's gradient boosting differ from Random Forest?**

**Answer:** Both are tree ensembles, but the learning mechanism differs fundamentally:
- **Random Forest (bagging):** Trains N trees independently on random data subsamples. Final answer = majority vote (classification) or mean (regression). High variance reduction, some bias.
- **XGBoost (boosting):** Trains trees sequentially. Tree t focuses on the errors of trees 1 to t-1 (by fitting the gradient/residuals of the loss function). Final answer = sum of all trees. Low bias achievable, but requires careful regularization to avoid overfitting.

We use XGBoost for position prediction (regression) because sequential error correction achieves lower bias — important for precise spatial predictions. We use Random Forest for congestion (classification) because bagging is more robust to noisy class boundaries created by hard thresholds (5 vessels = MEDIUM, 6 vessels = MEDIUM, 14 vessels = MEDIUM, 15 vessels = HIGH — these hard boundaries create label noise at the boundary).

---

**Q44: Why do you predict delta_lat/delta_lon instead of absolute lat/lon?**

**Answer:** Predicting delta_lat focuses the model on the kinematic change — "how much does this vessel type, at this speed, on this course, move in 5 minutes?" rather than "where on Earth is a vessel that's at lat=29.7, lon=-95.1, speed=15?" Predicting absolute lat requires the model to learn the geographic distribution of vessel positions (25°N to 50°N for US waters), a much harder problem with a much larger output range. Delta_lat typically ranges from -0.05° to +0.05° (±3nm in 5 min at 35 kn), a much narrower, more learnable range. At inference: `predicted_lat = current_lat + delta_lat_prediction`.

---

**Q45: What would a 4.24 nm MAE mean for a port operator?**

**Answer:** 4.24 nautical miles = 7.85 kilometers. For a container ship operator planning berth allocation (berths are ~500m long), a 7.85km position uncertainty 5 minutes ahead is insufficient for precise docking guidance. However, for: (1) traffic flow monitoring — is this vessel heading toward congested Port A or bypassing to Port B? (4.24 nm error is acceptable for this decision); (2) collision risk monitoring at the traffic management level — will this vessel reach the channel approach in the next 5 minutes? (acceptable); (3) early warning of unusual routing — will this vessel deviate from its expected trajectory significantly? (acceptable). We explicitly state in the dashboard that predictions are for traffic management awareness, not for navigation.

---

**Q46: Why doesn't the anomaly detector have evaluation metrics (accuracy, F1)?**

**Answer:** Anomaly detection evaluation requires ground truth labels — records that a domain expert has verified as "true anomaly" or "true normal." We don't have this labeled dataset. This is not an oversight — it's the fundamental challenge of unsupervised learning evaluation in operational settings. What we can evaluate: (1) Precision by sampling flagged records and having a maritime expert judge if they're truly anomalous; (2) Distribution of anomaly types (are rules catching sensible events?); (3) False positive rate by checking if high-scoring records have business explanations. In production, operator alert acknowledgment data ("this alert was useful/not useful") would enable feedback-based evaluation.

---

**Q47: Explain why vessel_count is the most important feature in the congestion model (~90% importance).**

**Answer:** Traffic density is highly autocorrelated over 1-hour intervals — this is a known property of maritime traffic. A cell with 20 vessels at 14:00 will likely still have substantial traffic at 15:00, because: vessels move slowly (~15 kn), grid cells are ~11km wide, and a vessel takes ~44 minutes to cross a cell. High current count strongly predicts high next-hour count. This is NOT data leakage because: features represent CURRENT hour, target is NEXT hour's count — these are genuinely different values. The model has learned a valid leading indicator. Feature importance confirms that our primary signal (current density) is the right one; other features provide contextual refinement.

---

**Q48: What is class_weight='balanced' and why is it important for your congestion model?**

**Answer:** `class_weight='balanced'` makes the algorithm weight training samples inversely proportional to class frequency: `weight_c = total_samples / (n_classes × n_samples_c)`. With our distribution (~75% LOW, ~15% MEDIUM, ~10% HIGH), without balancing the model would heavily favor LOW (predicting LOW always gives 75% accuracy). With balancing: each HIGH training example is weighted ~7.5× more than a LOW example. Effect: the model gives more "attention" to rare HIGH events, improving HIGH recall from near-zero to 40%. The trade-off: slightly lower LOW accuracy, but significantly better HIGH recall — which is operationally more valuable.

---

**Q49: Why did you use 200 estimators for both Isolation Forest and Random Forest?**

**Answer:** For both algorithms, more estimators reduce score/prediction variance by averaging more trees. 100 estimators (sklearn default) is often insufficient for stable scores on large datasets — the same record might score differently across runs. 200 provides stable estimates. Beyond 300 estimators, returns diminish — training time doubles but performance improvement is <0.1%. We benchmarked at 50, 100, 200, 500 estimators and found the performance plateau at ~150-200, choosing 200 as the sweet spot. This is a standard hyperparameter choice in the industry.

---

**Q50: What is early stopping in XGBoost and why is it useful?**

**Answer:** Early stopping monitors validation set performance during training. If validation MAE doesn't improve for `early_stopping_rounds=30` consecutive rounds, training stops. This prevents: (1) Overfitting — the model memorizes training data rather than learning generalizable patterns; (2) Unnecessary training time — if the optimal point is at round 180, training stops there rather than completing all 300 rounds. We use validation split (Silver VALIDATION split, Days 9-10) as the early stopping monitor. XGBoost restores the model to its best iteration (not the last), ensuring we deploy the optimal model.

---

**Q51: How do you prevent data leakage in the position predictor target generation?**

**Answer:** Target for 5-minute prediction: `future_lat = current_lat + delta_lat_N_steps_ahead`. The shift is `df.groupby('mmsi')['lat'].shift(-N_steps)`. Critically: `N_steps` records ahead means we're looking at a future record in the same vessel's track. The feature is the current record; the target is a future record. This is correct — the model learns "given current kinematics, predict future displacement." Leakage would occur if: (1) we included future features in current features (we don't), (2) we shuffled before splitting (we use temporal split), (3) we computed the scaler using all data including test (we fit scaler on TRAIN only, transform on all).

---

**Q52: What is the dead reckoning fallback and when does it activate?**

**Answer:** Dead reckoning uses the vessel's current position, speed, and heading to extrapolate forward:
```
dist_nm = sog × (minutes/60)
delta_lat = dist_nm × cos(heading_rad) / 60
delta_lon = dist_nm × sin(heading_rad) / (60 × cos(lat_rad))
```
It activates when: ML model files are missing (pkl files not found), or the feature vector has NaN values that cannot be cleaned. Dead reckoning confidence = 0.65 (vs. ML confidence = 0.82). Dead reckoning works well in straight-line cruise, fails for turns and port approaches. It's the universal fallback ensuring the system always produces a prediction.

---

**Q53: Why did you train separate models for lat and lon prediction?**

**Answer:** Latitude and longitude changes have different statistical properties: (1) Latitude changes are primarily determined by the north-south component of COG/heading — relatively simpler relationship; (2) Longitude changes depend on both east-west COG component AND the cosine of latitude (a vessel at 45°N needs to move further east per degree of longitude than at 25°N) — more complex relationship; (3) Error patterns differ: longitude has higher MAE at all horizons. Training separate models allows each to learn the specific relationship between features and its target, without one model trying to balance two different output distributions simultaneously.

---

**Q54: How would you improve the position predictor?**

**Answer:** Priority improvements:
1. **LSTM/Transformer:** Process the full track history (last 10-50 records) instead of just the last pair. Multi-step memory captures turn intentions, acceleration patterns, route-following behavior.
2. **True time-based targeting:** Shift by actual time (find first record ≥ T+5 minutes) rather than record count. Current N_steps ≈ 5 min has ±1-2 minute error.
3. **Current/weather features:** NOAA sea current adds 30-50% accuracy improvement.
4. **Route embeddings:** Encode known shipping lane paths as features — vessels near known lanes follow the lane geometry.
5. **Vessel type-specific models:** A tanker's dynamics differ from a fishing vessel's; separate models per type class.

---

**Q55: Explain the Isolation Forest tree construction in simple terms.**

**Answer:** Imagine you have 1,000 vessels plotted in a 6-dimensional space (sog, cog, heading, sog_change, heading_change, distance_nm). Most cluster together in a dense "normal" region. Anomalies are isolated outliers. 

For each tree: randomly pick one of the 6 features. Randomly pick a split value within that feature's range. Put records left or right based on the split. Repeat until each record is alone in a leaf.

The key insight: to isolate an anomaly (which is already far from the cluster), you need very few random splits — maybe 2-3. To isolate a normal record (buried in the dense cluster), you need many splits — maybe 20-30. The average path length = isolation difficulty. Short path = anomaly. This is elegant and fast.

---

**Q56: Why might the congestion model perform poorly on HIGH class despite class balancing?**

**Answer:** Several reasons compound:
1. **Threshold noise:** The boundary between MEDIUM (14 vessels) and HIGH (15 vessels) is a hard, arbitrary threshold. A cell with 14 vessels in training vs. 15 in test creates identical features but different labels — confusing the model.
2. **Class imbalance persistence:** Even with balancing, HIGH represents only ~10% of data. The model sees far fewer examples to learn from.
3. **Temporal dynamics:** HIGH congestion events may be brief (an hour) and their build-up patterns are not fully captured by our 8 features.
4. **Spatial heterogeneity:** Houston Ship Channel's HIGH congestion has different patterns than New York Harbor's HIGH congestion — a single model struggles to capture both with location features alone.
5. **Feature limitation:** Current vessel_count dominates — the model doesn't know why vessels are accumulating (weather delay, berth unavailable, scheduled convoy).

---

**Q57: What is macro F1 vs. weighted F1, and which should you report?**

**Answer:**
- **Weighted F1:** `Σ(support_c / total) × F1_c` — weights each class by its frequency. With 75% LOW, weighted F1 ≈ 0.95 — falsely suggests excellent performance.
- **Macro F1:** `Σ F1_c / n_classes` — equal weight to all classes. Our result: 0.7052 — reveals that HIGH performance (0.5041) pulls the average down.
- **Which to report:** Macro F1 when all classes have equal operational importance. Weighted F1 when class frequency reflects operational importance. For our use case, HIGH congestion events are MORE important than LOW (consequences are asymmetric — missing HIGH is worse than missing LOW). We should report BOTH, but emphasize Macro F1 as the honest performance summary.

---

**Q58: How does your scoring pipeline handle a record with no prior position (first ping)?**

**Answer:** The live_scorer maintains a per-MMSI state dictionary of the previous record. When a new record arrives for MMSI X: if no previous record exists → this is a first ping → delta features (sog_change, heading_change, distance_nm, time_delta_sec) are all NULL → Isolation Forest cannot score it (NULL features). The scorer applies ONLY rule-based checks that don't require delta features: `if sog > 30 → UNUSUAL_SPEED; if sog < 0.5 AND in_us_port_zone → STATIONARY_RISK`. The record is scored with whatever rule fires; if none fires, it's marked non-anomalous with score=0. The state dictionary is updated with this record as the "previous" for the next message from MMSI X.

---

**Q59: What would you do if the Isolation Forest model is producing too many false positives?**

**Answer:** Several remediation steps, in order:
1. **Reduce contamination:** Lower from 0.002 to 0.001 — fewer records flagged overall
2. **Increase n_estimators:** More trees = more stable scores, fewer borderline cases
3. **Re-examine features:** If a specific feature is causing false positives (e.g., heading_change too sensitive), reconsider its inclusion or normalize it differently
4. **Add rule suppression:** If ML flags a record but no rule fires, suppress the ML alert if the score is below a configurable threshold (e.g., 0.6)
5. **Collect feedback:** Log which alerts operators dismiss; retrain excluding those patterns
6. **Consider context:** A "sharp turn" at a known waypoint (Houston Ship Channel turn basin) is expected — add geofenced suppression for known high-maneuver zones

---

**Q60: What is the difference between anomaly_score and is_anomaly in your output?**

**Answer:**
- **anomaly_score:** Continuous value [0, 1] indicating the degree of anomalousness. Higher = more anomalous. Rule-based: set to specific values (0.8 for sudden stop, sog/60 for speed). ML: derived from Isolation Forest path length, normalized to [0,1].
- **is_anomaly:** Boolean flag. True when: any rule fires OR Isolation Forest scores > threshold (top 0.2%). This binary flag drives the alert system and map marker color.

The separation allows filtering: "show me all vessels with anomaly_score > 0.7" for high-confidence only, vs. "show all is_anomaly=True" for everything flagged. Dashboard shows both: the map uses is_anomaly for marker icons; the detail panel shows anomaly_score for context.

---

**Q61-Q80: Additional ML Q&A (Summary Format)**

**Q61:** Why use StandardScaler before Isolation Forest (tree-based)?
**A:** Isolation Forest doesn't technically require scaling, but StandardScaler normalizes feature ranges so random split selection considers each feature equally. Without scaling, heading_change (0-180°) would be split more often than distance_nm (0-2nm) simply due to range size, biasing anomaly detection toward heading anomalies.

**Q62:** What is the confidence score in position prediction (0.82 for ML, 0.65 for DR)?
**A:** These are fixed values assigned by design, not learned probabilities. ML model confidence=0.82 means "this is a learned prediction from historical data." Dead reckoning confidence=0.65 means "this is a physics estimate without learned correction." In production, confidence should be computed from the prediction interval (standard deviation of XGBoost leaf values).

**Q63:** Could you use K-Means for anomaly detection?
**A:** K-Means could cluster vessels into behavioral groups; records far from all cluster centroids could be flagged as anomalies. However: K-Means requires choosing K (number of clusters); doesn't naturally produce anomaly scores; more sensitive to outliers during training. Isolation Forest is more principled for anomaly detection specifically.

**Q64:** What is the train/test sample count for each model?
**A:** Anomaly: 2M training samples from Silver (stratified per file). Position predictor: 2M training, ~322K test (5-min horizon). Congestion: 65,548 grid-hour cells total, 80/20 time-split.

**Q65:** Why not use DBSCAN for anomaly detection?
**A:** DBSCAN defines anomalies as "points that don't belong to any cluster." It requires choosing ε (radius) and min_samples parameters that are hard to calibrate in 6D feature space. It also struggles with varying density clusters (busy ports vs. open ocean have very different vessel densities). Isolation Forest requires only contamination and n_estimators — more practical for our use case.

**Q66:** What is feature importance from Random Forest and how is it computed?
**A:** Feature importance = mean decrease in impurity (MDI). For each tree, each feature's contribution to reducing Gini impurity at its split nodes is summed. Averaged across all 200 trees, this gives relative importance. vessel_count's dominance (~90%) means nearly every tree splits on vessel_count at the root — it provides the most information gain.

**Q67:** Why use max_depth=10 for Random Forest (not unlimited)?
**A:** Unlimited depth = trees memorize training data exactly (overfitting). max_depth=10 limits tree complexity: at depth 10, each tree can represent at most 2^10 = 1,024 distinct decision regions. With our 8 features and 65K training samples, depth 10 is sufficient for capturing the relevant patterns without memorizing individual data points. min_samples_leaf=5 adds another regularization layer.

**Q68:** If you had labeled anomaly data, what model would you use instead of Isolation Forest?
**A:** LightGBM or XGBoost classifier (for tabular data). Or, if sequence context matters: LSTM-based anomaly classifier trained on vessel tracks. With labels, we could optimize directly for precision-recall tradeoff on confirmed maritime anomaly types.

**Q69:** What does RMSE/MAE > 1 tell you about the position predictor error distribution?
**A:** RMSE/MAE = 7.25/2.77 ≈ 2.6 (for 5-min lat prediction). This ratio > 1 indicates heavy-tailed error distribution — some predictions are very wrong (large errors squaring to large RMSE). This is expected: vessels making unexpected turns cause large prediction errors, pulling RMSE up while the majority of straight-line predictions have small MAE.

**Q70:** How would MLflow improve your ML pipeline?
**A:** MLflow would: (1) Track every training run with parameters, metrics, and model artifacts; (2) Enable model versioning — "this is v3 of the anomaly model, trained on 3M records, contamination=0.001"; (3) Support A/B testing between model versions; (4) Provide a model registry for promoting models from development to production; (5) Enable parameter search logging for hyperparameter tuning. Currently our models are saved as pkl files with manual version numbering (_v2). MLflow would automate this governance.

**Q71:** What would you change about the anomaly detection rule thresholds for a fishing vessel vs. a tanker?
**A:** Fishing vessels routinely: stop suddenly (setting nets), make sharp turns (chasing schools), travel at irregular speeds (0-15 kn variability). Our current thresholds (sudden stop at Δsog<-5, sharp turn at heading_change>45°) would flag normal fishing operations repeatedly. Tankers: extremely predictable courses, rarely exceed 15 kn, turns > 20° unusual. Vessel-type-specific thresholds and Isolation Forest models would significantly improve precision for each type.

**Q72:** What is subsample=0.8 in XGBoost?
**A:** Each tree sees only a random 80% of training rows (sampled without replacement). This is stochastic gradient boosting (Friedman 1999). Benefits: reduces overfitting (model can't memorize specific rows), reduces computation time (smaller trees), adds randomness that helps escape local optima, similar to dropout in neural networks.

**Q73:** What are the three prediction horizons used for in practice?
**A:** 5-minute: collision avoidance — is vessel X about to enter vessel Y's path? 10-minute: traffic management — will the fairway be clear in time for scheduled departure? 15-minute: berth scheduling — should port tug be dispatched now to meet incoming vessel?

**Q74:** What happens if a vessel makes a 90° turn between the current record and the predicted 5-minute position?
**A:** The model predicts based on current heading/COG — it has no knowledge of the planned turn. If the turn happens within the 5-minute window, the prediction will be severely wrong (error could be 10-20 nm). This is the fundamental limitation of short-horizon prediction without route knowledge. The high RMSE (7.25 nm at 5 min) reflects these tail-end errors from unexpected maneuvers.

**Q75:** How does the live scorer handle vessel_count for congestion prediction at inference time?
**A:** The live scorer receives individual vessel records, not grid aggregations. To compute vessel_count for a grid cell, the scorer queries `fact_vessel_latest`: `SELECT COUNT(*) FROM fact_vessel_latest WHERE lat_bin = round(lat, 1) AND lon_bin = round(lon, 1) AND updated_at > NOW() - INTERVAL '1 hour'`. This gives the current count of active vessels in the 0.1° cell. Combined with avg_sog from the same query, these form the congestion prediction features. This query runs on every record — performance dependency on PostgreSQL index on (lat_bin, lon_bin).

**Q76:** Why doesn't the position predictor use vessel type as a feature?
**A:** Vessel type (Cargo, Tanker, Fishing) would be a useful contextual feature — a fishing vessel's kinematics differ from a container ship's. It's currently excluded because: (1) vessel_type is highly sparse in the AIS data (~30% missing/unknown); (2) including it would require one-hot encoding, adding 15+ columns; (3) our evaluation showed marginal MAE improvement (< 0.5 nm) with vessel type included, not worth the complexity. It's a candidate for the next improvement iteration.

**Q77:** What is the formula for XGBoost's regularized objective function?
**A:** `L(θ) = Σᵢ l(yᵢ, ŷᵢ) + Σₜ Ω(fₜ)` where:
- `l(yᵢ, ŷᵢ)` = loss (squared error for regression: (yᵢ - ŷᵢ)²/2)
- `Ω(f) = γT + (1/2)λ||w||²` (regularization: T = leaf count, w = leaf weights, γ and λ control penalty)
The regularization term penalizes complex trees (many leaves) and large weights, preventing overfitting.

**Q78:** Could you use the anomaly score as a feature for the congestion model?
**A:** Theoretically yes — cells with many anomalous vessels might have different future congestion patterns. However: (1) the anomaly score is computed at the vessel level, not the grid level; aggregating it (mean anomaly score per cell) adds engineering complexity; (2) it would require the anomaly model to run before the congestion model — creating a dependency chain; (3) the marginal value is uncertain. This is a potential future experiment, not a current implementation.

**Q79:** What is "feature drift" and how would you detect it in production?
**A:** Feature drift occurs when the statistical distribution of input features changes over time (e.g., more high-speed vessels in summer, seasonal route changes). This degrades model performance because the model was trained on a different distribution. Detection: monitor the distribution of each feature (mean, std, percentiles) over time windows and alert when KL-divergence from the training distribution exceeds a threshold. Tools: Evidently AI, Great Expectations, or custom monitoring. We don't implement drift detection — acknowledged limitation.

**Q80:** How would you add a "collision risk alert" ML component?
**A:** Current collision detection is rule-based (distance < 0.1nm). ML improvement: train a binary classifier on vessel-pair features (relative speed, relative heading, CPA = Closest Point of Approach, TCPA = Time to CPA) to predict collision probability over the next 30 minutes. CPA and TCPA are computed geometrically from both vessels' current state vectors. Training data: historical near-miss incidents (maritime incident databases). This would provide probabilistic collision risk rather than binary threshold alerts.

---

### 12.3 Architecture Questions (20 Questions)

---

**Q81: What is a star schema and why did you choose it over other models?**

**A:** A star schema organizes data into one or more fact tables (measurements, events) surrounded by dimension tables (descriptive context). Our center star: fact_vessel_latest (positions) joined to dim_vessel (vessel metadata). Benefits: query simplicity (most queries join 1-2 tables), query performance (indexes on dimension keys), clear separation of facts (events) from dimensions (context), compatibility with BI tools. Alternative: snowflake schema (further normalization of dimensions) — more normalized but more complex joins. Alternative: flat/wide table — simpler but duplicates vessel metadata on every position row.

**Q82: Why is Kafka the right choice for this architecture (not a direct file feed)?**

**A:** Kafka decouples producers from consumers, enabling: (1) Multiple independent consumers (Spark Streaming + Live Scorer) to read the same stream at their own pace; (2) Replay — if the live scorer is down, messages are retained and processed when it restarts; (3) Back-pressure — if the live scorer is slow, Kafka buffers messages without slowing the producer; (4) Scalability — adding more consumers doesn't affect the producer. A direct file feed would require all consumers to coordinate, and any consumer failure would cause data loss.

**Q83: Why do you have both a Streamlit dashboard AND a React frontend?**

**A:** Streamlit was built first (faster development for data engineers using Python). React was added as a secondary frontend for users who need more interactive/customized UI (TypeScript, component-based design). In production, one would typically be chosen. For the demo, both demonstrate the system's flexibility: the API serves both, showing separation of concerns between the serving layer (FastAPI + PostgreSQL) and the presentation layer (Streamlit/React).

**Q84: What is the architecture tradeoff between live_scorer.py (Python) and Spark Streaming?**

**A:** live_scorer.py is a Python Kafka consumer that scores individual records and writes to PostgreSQL. It's simple, fast to start (no Spark overhead), and supports the UPSERT logic (Spark Structured Streaming doesn't natively support per-row PostgreSQL upserts without complex `foreachBatch`). Spark Streaming handles Bronze writes where high throughput and Delta Lake integration matter. The division: Spark for large-scale batch-style streaming → Delta Lake; Python for low-latency, stateful ML scoring → PostgreSQL.

**Q85: How does your system scale if vessel count grows from 1,000 to 100,000?**

**A:** Current bottlenecks at 100K vessels: (1) PostgreSQL fact_vessel_latest query time grows linearly — add index on (lat, lon, risk_level); (2) Live scorer collision detection is O(n²) pairwise — switch to R-tree spatial index; (3) Gold job re-aggregation time grows — partition fact_vessel_latest by geographic region; (4) Dashboard live map rendering — use tile-based clustering (only show vessel count per zoom level, not individual markers). Kafka and Spark Structured Streaming would handle 100K vessels at high throughput without changes.

**Q86: Explain the resource allocation decisions in Docker Compose.**

**A:** Spark Workers get the most resources (5GB each, 4 CPU) because they do the heavy computation (haversine on 36M rows, window functions). Kafka gets 1.5GB for message buffer storage. PostgreSQL gets 2GB for query caching (shared_buffers). API and Streamlit get 1-2GB — they're I/O bound (database queries), not CPU bound. The live scorer (1GB) runs lightweight Python, not Spark. Total cluster: ~26GB RAM, 26 CPU. This matches a mid-range developer workstation or a small AWS EC2 instance (m5.8xlarge: 32 vCPU, 128GB RAM).

**Q87: Why not use a real-time stream processor (Flink) instead of Spark Streaming?**

**A:** Flink has superior streaming semantics (true event-time processing, lower latency, better watermark handling). We chose Spark Structured Streaming because: (1) the same Spark codebase handles both batch (Bronze/Silver/Gold jobs) and streaming (Kafka consumer) — one framework to learn and maintain; (2) Delta Lake has first-class Spark integration; (3) Flink would require a separate cluster and a different API. For our use case (10-second micro-batches are sufficient), Spark Streaming's slightly higher latency is acceptable. If sub-second latency were required, Flink would be the right choice.

**Q88: How would you add Airflow to orchestrate this pipeline?**

**A:** Define a DAG (Directed Acyclic Graph):
```
ais_ingestion_dag:
  Task 1: kafka_producer (run continuously, not scheduled)
  Task 2: spark_streaming (run continuously)
  Task 3: bronze_job → runs every 6 hours
  Task 4: silver_job → triggered on bronze_job success
  Task 5: gold_job → triggered on silver_job success
  Task 6: ml_retrain → triggered weekly on gold_job success
```
Airflow provides: visual DAG monitoring, automatic retry on failure, backfill (rerun past dates), alerting on failures. Currently our pipeline lacks orchestration — each job runs manually or via simple cron. This is the highest-priority DevOps improvement.

**Q89: Why is there no API authentication?**

**A:** This is a deliberate decision for the demo: adding OAuth2 + JWT requires a user management system, which would add weeks of development. We explicitly acknowledge it as a limitation. In production: (1) API key authentication for server-to-server (dashboard → API); (2) OAuth2 + JWT for user-facing access; (3) Rate limiting to prevent abuse; (4) TLS for all connections. The security risk in the demo: anyone can query vessel positions. In real maritime applications, this would be unacceptable (security and commercial sensitivity of shipping routes).

**Q90: What would a Kubernetes production deployment add over Docker Compose?**

**A:** Kubernetes adds: (1) Auto-scaling: if vessel count spikes, spawn more live_scorer replicas automatically; (2) Self-healing: if the API pod crashes, Kubernetes restarts it automatically; (3) Rolling updates: deploy new ML models without downtime; (4) Resource limits and requests: prevent one service from starving others; (5) Persistent volumes: PostgreSQL data survives pod restarts; (6) Ingress controller: HTTPS termination, domain routing. Our k8s/ directory contains draft manifests for all services. For the demo, Docker Compose is simpler and sufficient.

---

### 12.4 Architecture Questions (Continued) and Business Questions

**Q91:** What monitoring would you add in production?
**A:** (1) Prometheus metrics: Kafka consumer lag, live scorer throughput, API response time, PostgreSQL query time; (2) Grafana dashboards for all metrics; (3) Alerts: consumer lag > 10,000 messages → scorer is slow; API p99 latency > 1s → performance issue; model anomaly_rate spikes > 5% → possible model drift; (4) Logging: structured JSON logs to Elasticsearch; (5) Distributed tracing: OpenTelemetry across Kafka → scorer → PostgreSQL to trace slow paths.

**Q92:** How does your system handle vessel MMSI conflicts (two vessels same MMSI)?
**A:** MMSI conflicts (intentional spoofing or transponder misconfiguration) appear as one vessel's track with two different lat/lon trajectories simultaneously. Our system: (1) would apply teleport filter when the track jumps between the two vessels; (2) fact_vessel_latest would show one of the two positions (whichever arrived last); (3) dim_vessel would show a single entry for the MMSI. Detection: flag MMSI records where simultaneous positions differ by > threshold (e.g., > 50nm). We don't implement this; it's a future enhancement.

**Q93-Q100: Architecture Q&A (Summary Format)**

**Q93:** What is connection pooling in SQLAlchemy?
**A:** Pool of pre-opened database connections reused across requests. Without pooling, each API request opens and closes a TCP connection to PostgreSQL (100-500ms overhead). With pool_size=20: up to 20 concurrent requests share pre-existing connections (< 1ms overhead per request). pool_pre_ping=True verifies connections before use (prevents stale connection errors after PostgreSQL restart).

**Q94:** How would you add data lineage tracking?
**A:** Use Apache Atlas or custom metadata tagging. For each record: write source_file, ingestion_time, bronze_version, silver_version to the record. Delta Lake's transaction log already provides table-level lineage. For column-level lineage, use tools like Marquez (OpenLineage) that integrate with Spark to track which input columns produce which output columns.

**Q95:** What would Snowflake add to this architecture?
**A:** Snowflake as cold storage: Delta Lake (hot: last 30 days) + Snowflake (cold: all historical data). Benefits: Snowflake can query multi-year AIS history efficiently; its separable compute and storage model means you pay for storage cheaply, compute only when querying; built-in BI tool connectors (Tableau, Looker). The architecture has Snowflake integration drafted but disabled for the demo.

**Q96:** Why does fact_ais_track have a lat_bin, lon_bin column alongside exact lat, lon?
**A:** lat_bin (rounded to 0.1°) and lon_bin enable efficient grid-based queries: `WHERE lat_bin = 29.7 AND lon_bin = -95.1` uses index on (lat_bin, lon_bin) to find all records in a grid cell without a full scan. Exact lat/lon enables precise vessel-level queries. Both are needed for different query patterns.

**Q97:** What is the difference between the 0.1° density grid and the 0.5° congestion model grid?
**A:** 0.1° (≈6nm): visualization resolution. Shows fine-grained vessel concentration for the heatmap — enough resolution to see individual port approach lanes. 0.5° (≈30nm): model training resolution. Coarser cells ensure enough vessel observations per cell per hour for statistical stability (≥5 vessels). At 0.1°, most cells have 0-2 vessels, making the target label (HIGH/MEDIUM/LOW) unreliable for training.

**Q98:** How is the UPSERT to fact_vessel_latest implemented in live_scorer?
**A:** PostgreSQL UPSERT syntax: `INSERT INTO fact_vessel_latest (mmsi, lat, lon, ...) VALUES (...) ON CONFLICT (mmsi) DO UPDATE SET lat=EXCLUDED.lat, lon=EXCLUDED.lon, ...`. This is atomic: if MMSI exists, update; if not, insert. The `ON CONFLICT` clause specifies the unique constraint to check (PRIMARY KEY on mmsi). This ensures fact_vessel_latest always contains exactly one row per vessel.

**Q99:** What is the role of dim_status table?
**A:** dim_status is a lookup table for AIS navigation status codes: {status_id: 0, status_code: "0", status_label: "Under way using engine", is_underway: True, is_stopped: False}. It normalizes status labels — instead of joining text strings on every query, the fact table stores an integer status_id and JOINs to dim_status for the label. This follows Kimball data warehousing best practices.

**Q100:** What is the system's end-to-end latency from AIS broadcast to dashboard update?
**A:** Kafka producer → Kafka: ~50ms. Kafka → live_scorer (batch window): up to 5 seconds (SCORER_BATCH_WINDOW). live_scorer → PostgreSQL UPSERT: ~100ms. Dashboard query: 30-second refresh interval. Total: up to ~35 seconds from vessel position broadcast to dashboard update. The dominant latency is the 30-second dashboard refresh, not the pipeline. In production, WebSockets would push updates to the dashboard in real-time, eliminating the polling delay.

---

### 12.5 Business Questions (20 Questions)

**Q101:** What is the business value of anomaly detection in maritime?
**A:** (1) Safety: early detection of vessels in distress or behaving erratically prevents accidents and reduces Coast Guard response time; (2) Security: suspicious vessel behavior (AIS spoofing, unexpected stops in restricted areas) flagged for law enforcement; (3) Compliance: environmental agencies identify vessels stopping in protected marine areas; (4) Insurance: insurers identify high-risk vessels for premium adjustment; (5) Port efficiency: anomalous speed near berths (too fast) triggers immediate alert to port authority. Industry data: IMO estimates 90% of world trade moves by sea — even 1% efficiency improvement = billions in value.

**Q102:** Who would pay for this system and how much?
**A:** Potential buyers: (1) Port authorities: US has 360+ commercial ports, typical IT budget $1-10M/year — our system could be licensed at $50-500K/year; (2) Coast Guard agencies: government contracts $500K-5M/year for maritime intelligence systems; (3) Shipping companies: fleet management software $10-50K/year per operator; (4) Marine insurance companies: risk data subscription $100-500K/year. Comparable commercial systems (MarineTraffic Enterprise, Exact Earth): $10K-100K/year. Our system's ML layer differentiation (anomaly + prediction) could command a premium.

**Q103:** What regulatory requirements does maritime monitoring need to satisfy?
**A:** (1) IMO SOLAS Chapter V: AIS carriage requirements; (2) GDPR: AIS data includes vessel owner identity — EU data protection if European vessels; (3) US Maritime Security Act: restrictions on sharing vessel position data in certain contexts; (4) ISPS Code (International Ship and Port Facility Security Code): security monitoring requirements for port facilities; (5) US Coast Guard NVIC 01-16: cyber security guidance for maritime operational systems. Our demo system doesn't address these — production deployment requires legal review.

**Q104:** What is the ROI of congestion prediction for a port authority?
**A:** A vessel waiting outside a congested port burns ~$50,000/day in fuel and port fees. If our congestion predictor (90% accuracy) allows the port to schedule 10 additional vessels per month to arrive at optimal times (saving 1 day waiting each): 10 × $50,000 = $500,000/month = $6M/year saved. System cost: ~$500K/year. ROI: 12:1. Real-world: Port of Los Angeles handles 5M containers/year — even a 0.5% efficiency improvement = $25M value.

**Q105:** How would you explain anomaly detection to a non-technical maritime manager?
**A:** "Our system is like an experienced maritime traffic controller who has watched thousands of vessels for months and knows what normal behavior looks like. When a vessel does something unusual — stops suddenly in open water, makes a sharp turn for no apparent reason, or moves at an unusual speed — our system flags it immediately for human review. We don't claim to know why the vessel is behaving unusually; we surface it so your team can investigate. Think of it as a junior analyst that never sleeps and monitors every vessel simultaneously."

**Q106:** What are the limitations we must disclose to a client?
**A:** (1) US waters only; (2) Historical data simulated as live — not connected to real AIS feed; (3) No weather integration; (4) 40% recall on HIGH congestion — we miss 60% of true HIGH events; (5) Position prediction has 4-7 nm error — not suitable for navigation; (6) No security/authentication; (7) 5 hardcoded US port zones only; (8) No labeled anomaly evaluation; (9) 7-day training data — seasonal patterns not captured.

**Q107:** How would you prioritize development if given 3 months and a small team (3 engineers)?
**A:** Month 1: Real AIS data feed (MarineTraffic API), API authentication, weather integration. Month 2: LSTM position predictor, global port zone database, MLflow tracking. Month 3: Airflow orchestration, Kubernetes production deployment, drift monitoring. This delivers a production-ready MVP focused on the most impactful improvements (real data, security, better prediction).

**Q108:** What happened to vessel tracks at the US/Mexico border in your dataset?
**A:** The AIS receiver network has gaps in coverage. US coastal receivers cover territorial waters (12nm from shore) + exclusive economic zone (200nm). Vessels in Mexican waters that enter US coverage start new tracks (first pings). Our teleport filter prevents jumps from Mexican waters to US waters from appearing as single tracks. This is a data coverage limitation, not a pipeline error.

**Q109:** Why is AIS data sometimes inaccurate or spoofed?
**A:** AIS spoofing is the deliberate manipulation of AIS broadcasts to hide true position, fake identity, or disguise cargo type. Common reasons: evading sanctions (vessels claiming to be in safe waters while in sanctioned zones), illegal fishing (hiding location from coast guards), insurance fraud (claiming vessel was in safe waters during an incident). Technical: AIS signals are unauthenticated — any radio transmitter can broadcast any MMSI and coordinates. Our system cannot detect sophisticated spoofing (we can detect position impossibilities via teleport filter, but not fabricated plausible positions).

**Q110:** What is CPA (Closest Point of Approach) and why is it more important than current distance for collision detection?
**A:** CPA is the minimum distance that will exist between two vessels based on their current trajectories, if they maintain constant speed and course. Current distance = 0.5nm doesn't indicate collision risk if both vessels are heading away from each other. CPA = 0.05nm means they WILL be within 0.05nm in 3 minutes — that's a collision risk even if currently 5nm apart. Our system uses current distance (simpler), missing CPA and TCPA (Time to CPA) calculations that real ARPA systems use. This is a known limitation in our collision detection.

**Q111-Q120: Additional Business Q&A (Summary Format)**

**Q111:** What would differentiate your system from existing commercial AIS platforms?
**A:** MarineTraffic and Vessel Finder show where vessels are but don't predict or detect anomalies. Our differentiation: ML-powered anomaly detection, position prediction, congestion forecasting — all in near-real-time. The gap is bridging "where ships are" to "where they're going and whether they're behaving normally."

**Q112:** What data sources could enrich the system?
**A:** NOAA weather API (free), OSCAR ocean currents (free), World Port Index port database (free), Lloyd's List voyage intelligence (commercial), Port authority berth schedules (partner APIs), Satellite imagery for port congestion (commercial), Sentinel-1 SAR imagery for dark vessel detection (free from ESA).

**Q113:** How would you explain MAE=4.24nm to a shipping company executive?
**A:** "If your vessel is 4 nautical miles from port, we can predict its position in 5 minutes with an error of about 4 nautical miles — roughly within a shipping lane width. This is useful for understanding traffic flow and scheduling but should not replace your navigation officers' judgment. Think of it as a planning tool, not a navigation tool."

**Q114:** What is the difference between AIS Class A and Class B?
**A:** Class A: mandatory for larger vessels (>300 GT), broadcasts every 2-10 seconds when underway, includes voyage data (destination, ETA). Class B: voluntary for smaller vessels, broadcasts every 30 seconds, less data. Our MarineCadastre data includes both; we store `transceiver_class` in dim_vessel. Class A vessels are more reliably tracked; Class B may have gaps.

**Q115:** What would prevent this system from being deployed in production today?
**A:** (1) No real AIS data subscription; (2) No API security; (3) Hardcoded database credentials; (4) No production monitoring/alerting; (5) No disaster recovery plan; (6) No data retention policy; (7) Legal review for GDPR/maritime regulations; (8) No user acceptance testing with maritime domain experts; (9) No SLA definition; (10) Single-host deployment (no redundancy).

**Q116:** What is the business cost of a false negative in anomaly detection (missing a real anomaly)?
**A:** Depends on the anomaly type. Missed collision risk: potential vessel loss ($50-500M for a large vessel, environmental cleanup $1B+). Missed smuggling event: contraband passage, regulatory fines, reputational damage to port authority. Missed vessel in distress: loss of life, multi-million dollar search and rescue. This asymmetric cost (false negative >> false positive) argues for lower contamination threshold (flag more, not fewer) and additional rule-based catches.

**Q117:** How does this project demonstrate Big Data skills specifically?
**A:** (1) Scale: 36M raw records, distributed processing with Spark; (2) Architecture: Medallion Architecture (industry standard), Delta Lake, star schema; (3) Streaming: Kafka + Spark Structured Streaming (Lambda/Kappa architecture elements); (4) ML at scale: training on millions of records, real-time inference; (5) End-to-end: from raw data ingestion to production dashboard — not just a notebook experiment; (6) Production patterns: Docker orchestration, API serving, monitoring considerations.

**Q118:** What would you tell a senior engineer who says "you could have done this in a Jupyter notebook"?
**A:** "A Jupyter notebook would handle the analysis but not the production system. A notebook cannot: stream real-time data from Kafka, serve predictions to thousands of API requests, maintain a live map with 1,000 vessels, recover from failures automatically, process 36M records efficiently, or support multiple simultaneous dashboard users. Our architecture is designed for operational use at scale, not one-time analysis."

**Q119:** What is the environmental impact of maritime AI optimization?
**A:** A large container ship burns 100-300 tons of fuel per day ($50,000-150,000). A 5% speed optimization (informed by better traffic prediction) saves 5-15 tons/day = $2,500-7,500/day per vessel. Globally, shipping accounts for 2.5% of world CO2 emissions (almost 1 billion tons/year). Better route and congestion optimization could reduce this by 5-15% = 50-150 million tons CO2 reduction. This is a significant environmental argument for maritime AI systems.

**Q120:** If a company wants to buy this system, what's the first thing you'd tell them to invest in?
**A:** A real-time AIS data subscription (MarineTraffic Terrestrial AIS API: ~$500-2,000/month for US coverage). Without real data, the system demonstrates its architecture but doesn't provide real operational value. The second investment: NOAA weather integration (free). These two changes transform the demo into a production tool. Then: API authentication, global port zones, and Airflow orchestration complete the MVP.

---

## SECTION 13 — PRESENTATION PREPARATION

### 13.1 Slide-by-Slide Flow

**Slide 1: Title**
- Project name: Maritime Navigation AI System
- Team names, internship program
- Date

**Slide 2: Problem Statement (Business)**
- Opening hook: "Every day, 50,000+ vessels navigate US waters. AIS generates 10M+ records daily. Nobody is analyzing this intelligently in real time."
- Who is affected (port authorities, coast guard, shipping companies)
- Business cost of the problem (congestion, collision risk, inefficiency)
- Speaking note: Start with a striking number. "A vessel waiting outside Los Angeles port costs $50,000 per day."

**Slide 3: Solution Overview**
- One-diagram system architecture (simplified)
- "Three questions our AI answers: Where will this vessel be in 5 minutes? Is this vessel behaving normally? Will this port approach be congested in an hour?"
- Speaking note: Position as answers to specific business questions, not technology features.

**Slide 4: Dataset**
- MarineCadastre AIS data: 7 days, May 2025, US waters
- Key AIS fields: MMSI, lat/lon, SOG, COG, heading, timestamp
- Volume: 36M raw records → 5M clean records
- Speaking note: Brief and business-focused. "We track vessels from Houston to New York Harbor."

**Slide 5: Architecture (Technical)**
- Full architecture diagram (from Section 2.1 of this report)
- Label each component with technology
- Speaking note: Walk left to right: data flows from AIS → Kafka → Spark → Delta Lake → PostgreSQL → Dashboard/API.

**Slide 6: Bronze Layer**
- Input: raw Parquet, Output: Delta Lake
- Key validations: lat/lon range, SOG range, MMSI not null
- Enrichments: speed flags, port zones, risk level
- "What Bronze guarantees: the data exists and makes physical sense"

**Slide 7: Silver Layer**
- Input: Bronze Delta, Output: clean Silver Delta
- Key transformations (use before/after table)
- Feature engineering: haversine, sog_change, heading_change
- "What Silver guarantees: the data is clean, deduplicated, and ML-ready"

**Slide 8: Gold Layer + PostgreSQL**
- Three Gold tables + dim_vessel
- Star schema diagram
- "What Gold enables: real-time dashboard in milliseconds"

**Slide 9: Anomaly Detection**
- Model: Isolation Forest (unsupervised)
- 6 features, contamination=0.2%
- Rule-based + ML hybrid
- Show example: vessel with SOG=35 kn, score=0.58, labeled UNUSUAL_SPEED

**Slide 10: Position Prediction**
- Model: XGBoost regressor
- Three horizons: 5/10/15 min
- Results: MAE = 4.24 / 6.18 / 6.86 nm
- Show example map with predicted track vs. actual track

**Slide 11: Congestion Prediction**
- Model: Random Forest classifier
- 3 classes: HIGH/MEDIUM/LOW
- Results: Accuracy 90.4%, Macro F1 0.705
- Show heatmap screenshot

**Slide 12: Live Demo**
- Streamlit dashboard live
- Show: live vessel map, risk-colored markers, anomaly feed, heatmap
- Demo flow: click a vessel → show prediction → show anomaly score

**Slide 13: Model Evaluation**
- Table of all metrics: congestion precision/recall/F1 per class
- Position predictor MAE at each horizon
- Honest interpretation: "What works well and what we know to improve"

**Slide 14: System Limitations**
- Proactively list weaknesses (shows maturity)
- Top 5 limitations with business impact
- "We know these; here's our roadmap to address them"

**Slide 15: Future Improvements**
- Priority table (HIGH/MEDIUM/LOW)
- Top 3: Real AIS feed, LSTM predictor, Weather integration
- Timeline estimate

**Slide 16: Team Contributions**
- Who did what (be specific and honest)
- Technologies each team member worked on

**Slide 17: Q&A**
- Thank you + Q&A invitation

---

### 13.2 Team Member Role Allocation

| Role | Slides to Own | Q&A Topics |
|---|---|---|
| Team Member 1 (Data Engineer) | Slides 5-8 (Architecture, Bronze/Silver/Gold, PostgreSQL) | All Q1-Q40, Q81-Q100 |
| Team Member 2 (ML Engineer) | Slides 9-11, 13 (Anomaly, Prediction, Congestion, Evaluation) | All Q41-Q80 |
| Team Member 3 (Product/Business) | Slides 2-4, 14-16 (Problem, Dataset, Limitations, Future) | All Q101-Q120 |

**Note:** All team members should be able to answer any question. The allocation is for primary responsibility in the presentation, not exclusive knowledge.

---

### 13.3 Common Mistakes to Avoid

1. **"We achieved 90% accuracy"** → Always qualify: "90% overall accuracy, but macro F1 is 0.705 and HIGH class recall is only 40%"

2. **"Our system is real-time"** → Clarify: "Near-real-time, with ~30-second end-to-end latency (5-second pipeline + 30-second dashboard refresh)"

3. **"This detects anomalies"** → Add: "with 0.2% contamination rate — we flag the top 0.2% most anomalous records; we have no labeled ground truth to measure precision/recall directly"

4. **"XGBoost is the best algorithm"** → Say: "XGBoost is appropriate for our tabular data at this scale. An LSTM would capture longer-term trajectory patterns better, which is our next improvement"

5. **"The system is production-ready"** → Say: "The architecture is production-grade; several components need work for true production deployment (real data feed, authentication, monitoring)"

6. **Don't memorize answers** → Understand them. A senior engineer will ask follow-up questions; a memorized answer breaks under follow-up.

7. **Don't say "I don't know"** → Say "That's beyond what we implemented, but the approach would be..." and describe a plausible direction.

---

### 13.4 Likely Questions After Each Slide

| Slide | Most Likely Question |
|---|---|
| Architecture (5) | "Why Spark? Couldn't you use pandas?" |
| Bronze (6) | "What happens to records that fail validation?" |
| Silver (7) | "Why do you drop first pings instead of imputing?" |
| Gold (8) | "Why not query Delta Lake directly from the dashboard?" |
| Anomaly (9) | "How do you measure the accuracy of your anomaly detection?" |
| Position Prediction (10) | "4.24 nm error is pretty large. Is this useful?" |
| Congestion (11) | "Why is HIGH recall only 40%?" |
| Live Demo (12) | "What would happen if I streamed 10,000 vessels?" |
| Evaluation (13) | "What's your baseline? How much better is ML than a simple rule?" |
| Limitations (14) | "If you had 3 more months, what would you fix first?" |

---

## SECTION 14 — FINAL TECHNICAL ASSESSMENT

### 14.1 Architecture Score: 8/10

**Strengths:**
- Medallion Architecture is correctly implemented with proper layer separation
- Delta Lake for all three layers — right tool for the job
- Star schema is well-designed for the query patterns
- Kafka decoupling enables clean fan-out to multiple consumers
- Docker Compose provides reproducible deployment
- FastAPI + SQLAlchemy is a clean, modern serving layer

**Weaknesses:**
- No Airflow orchestration — pipeline dependencies managed manually; fragile
- Gold → PostgreSQL write uses overwrite mode (not MERGE) — inefficient at scale
- No partition pruning for fact_ais_track (full table scans for MMSI queries)
- Hardcoded port zones — should be a database table with polygon geometry
- No API authentication — security gap that would block production deployment
- Shuffle partitions set correctly but not documented in code comments

**What a Senior Architect would commend:** "You understood why each technology was chosen and how they fit together. The Medallion Architecture is not just a buzzword — you implemented its principles correctly. The JDBC write optimization awareness shows depth."

**What they would push back on:** "Where's your SLA? What's the acceptable data latency? How do you handle Bronze/Silver version mismatch after a schema change? What's your disaster recovery plan?"

---

### 14.2 Data Engineering Score: 7.5/10

**Strengths:**
- MMSI scientific notation fix — catches a real-world data quality issue
- AIS sentinel value handling (102.3, 511) — shows domain knowledge
- Haversine formula correctly implemented (not Euclidean)
- Wrap-around-aware heading_change — catches a subtle but impactful bug
- MMSI-only window partition (no day split) — correct design for cross-midnight continuity
- Chronological train/test split — temporal leakage correctly avoided
- NaN/Inf guards in ML training — fail-loud quality gates

**Weaknesses:**
- Target generation for position predictor uses record-count shift (not time-based) — introduces ±1-2 minute horizon inaccuracy
- No data quality metrics logged (what % dropped at each stage, why)
- No SCD Type 2 for dim_vessel — vessel history is lost
- No OPTIMIZE / VACUUM commands for Delta Lake maintenance
- First-ping dropping logic is correct but means ~0.1% of records are untraceable at inference (first record of a Kafka session)
- Congestion model uses fixed bin size (0.5°) — this should be adaptive to vessel density

**What a Senior Data Engineer would say:** "Your cleaning logic is solid and shows you understand the data domain. The MMSI window partition decision is particularly mature — most beginners get this wrong. The position target generation approximation is the main technical debt I'd prioritize fixing."

---

### 14.3 Machine Learning Score: 7/10

**Strengths:**
- Correct unsupervised approach for unlabeled anomaly data
- Rule-based + ML hybrid is the right architecture (interpretability + coverage)
- Leakage prevention in congestion model (next-hour target) — correctly implemented
- class_weight='balanced' for imbalanced congestion classes — appropriate
- Separate lat/lon models — valid design decision
- Dead reckoning fallback — system always produces a prediction
- StandardScaler before Isolation Forest — shows understanding of why scaling matters

**Weaknesses:**
- No cross-validation (single train/test split — results may vary with different split points)
- No hyperparameter search (GridSearchCV, Optuna) — current params are reasonable defaults, not optimized
- Contamination=0.002 is domain-reasonable but not empirically calibrated with feedback
- 40% recall on HIGH congestion is genuinely weak — not glossed over, but the root cause is not fully addressed
- No SHAP values for model explainability — operators can't know why a vessel is flagged
- Position predictor confidence (0.82) is hardcoded, not computed from prediction intervals
- No detection of model drift or performance degradation over time

**What a Senior ML Engineer would say:** "You've applied appropriate algorithms for each task and correctly handled the key pitfalls (temporal leakage, class imbalance, unsupervised evaluation). The Isolation Forest choice is defensible and the XGBoost configuration is solid. The main gap is evaluation rigor — you need cross-validation and empirical hyperparameter tuning to be confident in your reported metrics."

---

### 14.4 Production Readiness Score: 5/10

**Score rationale:** The architecture has production-grade components, but several critical gaps prevent actual production deployment.

**Production-ready aspects:**
- Docker Compose → Kubernetes manifests provided
- Checkpoint-based streaming recovery
- Graceful degradation (demo mode, DR fallback, rule-based fallback)
- PostgreSQL connection pooling
- Structured logging
- CORS configuration (though too permissive)

**Production gaps:**
- No API authentication (critical security gap)
- Hardcoded database credentials
- No monitoring/alerting stack (Prometheus/Grafana not included)
- No data retention policy
- No CI/CD pipeline (no automated testing on code changes)
- No integration tests (pipeline tested manually)
- Simulated data stream (not real AIS)
- No SLA defined
- No performance benchmarks documented

**What a Technical Manager would say:** "The system demonstrates real architectural thinking, but I wouldn't approve production deployment without authentication, monitoring, and real data. The foundation is solid — these gaps are 4-6 weeks of engineering work, not architectural redesign."

---

### 14.5 Internship Project Score: 9/10

**Context:** This is evaluated as a Big Data internship graduation project, not a production system.

**What makes this exceptional:**
1. **True end-to-end:** Data ingestion → ETL → ML → Dashboard → API. Many student projects cover only one or two layers.
2. **Three ML components:** Anomaly detection (unsupervised), regression (position), classification (congestion) — demonstrates breadth across ML problem types.
3. **Streaming architecture:** Kafka + Spark Structured Streaming is beyond typical student projects; most use batch-only pipelines.
4. **Domain knowledge:** AIS sentinel values, haversine formula, heading wrap-around correction — these show genuine engagement with the maritime domain.
5. **Design decisions are justified:** The project team can explain WHY each choice was made (temporal split, MMSI window partition, Delta Lake) not just WHAT was done.
6. **Honest limitations:** Acknowledging HIGH recall=40% and no authenticated API shows maturity; hiding weaknesses would be less impressive to experienced engineers.
7. **Modern stack:** Kafka, Spark, Delta Lake, XGBoost, FastAPI, Docker — exactly the tools used in industry today.

**What would push it to 10/10:**
- Cross-validation for ML evaluation
- At least one labeled evaluation (sample-based anomaly precision)
- Airflow DAG
- API authentication
- One improvement actually implemented (even weather data as a simple REST call)

**Final word from a Senior Data Architect perspective:** "For three recent graduates in a Big Data internship, this is genuinely impressive. The architecture demonstrates systems thinking beyond academic exercise. The ML models are appropriately chosen and correctly evaluated. The pipeline has real data quality depth. The team clearly owns this work — they can explain every decision. If asked whether I'd hire these three to an entry-level Big Data Engineering role: yes, without hesitation, pending the technical interview confirmation of the knowledge in this report."

---

*End of Part 3 — Complete Technical Knowledge Report.*

*Files in this series:*
- *TECHNICAL_KNOWLEDGE_REPORT_PART1.md — Executive Summary, Architecture, Dataset, Data Engineering Pipeline*
- *TECHNICAL_KNOWLEDGE_REPORT_PART2.md — Feature Engineering, ML Models, Evaluation, Design Decisions, Limitations, Future Improvements*
- *TECHNICAL_KNOWLEDGE_REPORT_PART3.md — 120 Q&A, Presentation Guide, Final Technical Assessment (this file)*
