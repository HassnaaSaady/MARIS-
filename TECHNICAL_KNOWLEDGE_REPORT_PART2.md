# Maritime Navigation AI System — Technical Knowledge Report
## PART 2: FEATURE ENGINEERING, MACHINE LEARNING, EVALUATION & DESIGN DECISIONS

---

## SECTION 6 — FEATURE ENGINEERING DEEP DIVE

### 6.1 Why Feature Engineering Matters

Raw AIS fields tell you where a vessel is. Engineered features tell you **how** it is moving — and it is the "how" that reveals anomalies, enables prediction, and measures congestion. Raw lat/lon alone is useless for an anomaly detector: all positions between 25°N and 50°N are "normal" for US waters. The anomaly is in the pattern of movement.

### 6.2 All Engineered Features — Full Explanation

#### distance_nm (Haversine Distance)

**Formula:**
```
a = sin²((lat₂-lat₁)/2 × π/180) + cos(lat₁×π/180) × cos(lat₂×π/180) × sin²((lon₂-lon₁)/2 × π/180)
distance_nm = 2 × 3440.065 × atan2(√a, √(1-a))
```

**What it measures:** Physical distance traveled since last AIS broadcast, in nautical miles.

**Why ML models need it:** 
- Anomaly detection: a vessel that moves 50 nm between two 10-second pings has an implied speed of 18,000 kn — a clear GPS glitch or AIS replay error.
- Position predictor: distance_nm combined with time_delta_sec gives actual measured speed, which is more reliable than the broadcast SOG (which can be stale or inaccurate).
- Indirectly captures: trajectory smoothness. Normal vessels have small, consistent distance_nm values. Erratic vessels have high variance.

**Expected range (after teleport filter):** 0 to ~2.5 nm (for a 10-second interval at 30 kn: 30/3600 × 10 = 0.083 nm per second × 10 = 0.83 nm. At 60 kn ceiling: 0.17 nm × 10 = 1.67 nm).

**What high distance_nm means:** Either fast vessel, or long time since last ping (vessel at anchor for hours then updates), or residual GPS error not caught by teleport filter.

---

#### sog_change (Speed Change)

**Formula:** `sog_change = current_sog - prev_sog`

**Units:** Knots (change per ping interval, not per second — so not acceleration in the physics sense)

**What it measures:** The rate of speed change between consecutive AIS records.

**Why important:**
- Large negative values (sudden deceleration): emergency stop, collision avoidance maneuver, engine failure
- Large positive values (sudden acceleration): unusual; most vessels accelerate slowly
- Near-zero: steady-state cruising (normal)

**Rule-based anomaly trigger:** `sog_change < -5.0 knots` → SUDDEN_STOP anomaly with score 0.8

**Isolation Forest role:** sog_change is one of 6 features. Combined with sog, a vessel at high SOG that suddenly drops 10 knots stands out clearly in the 6D feature space.

**Expected distribution:** Centered near 0, with heavy tails. Most pings are steady cruising → Δsog ≈ 0. Turns and port approaches → |Δsog| up to 5 kn.

**Example:**
```
Record 1: MMSI=368001, SOG=18.5 kn
Record 2: MMSI=368001, SOG=7.2 kn
sog_change = 7.2 - 18.5 = -11.3 kn → SUDDEN_STOP anomaly triggered
```

---

#### heading_change (Turn Magnitude)

**Formula (wrap-around aware):**
```python
raw_diff = current_heading - prev_heading
if raw_diff > 180:  raw_diff -= 360
if raw_diff < -180: raw_diff += 360
heading_change = abs(raw_diff)   # degrees, 0-180
```

**What it measures:** How sharply the vessel turned between two consecutive pings.

**Why wrap-around matters:** Without it, a vessel turning from 355° to 005° (a gentle 10° starboard turn) computes as |355-5| = 350° — appearing to have turned almost all the way around.

**Rule-based anomaly trigger:** `heading_change > 45° AND sog > 2 kn` → SHARP_TURN anomaly

**Why SOG > 2 condition?** A vessel at anchor rotates with wind/current without actually maneuvering — we do not want to flag a swinging anchor chain as a sharp turn.

**ML role:** Sharp turns in open ocean are anomalous (why is a tanker turning 90° with no port nearby?). In channel approaches, 45° turns are expected. The Isolation Forest learns this context from training data — cells near ports have higher expected heading_change variance.

**Expected range:** 0° (straight line) to 180° (U-turn). Most values: < 15° (slight course adjustments).

---

#### time_delta_sec (Time Since Last Record)

**Formula:** `time_delta_sec = UNIX_TIMESTAMP(base_datetime) - UNIX_TIMESTAMP(prev_base_datetime)`

**Units:** Seconds

**Expected values:**
- Underway vessel: 2–10 seconds (AIS Class A broadcasts every 2–10 sec based on speed)
- Slow vessel: 10–30 seconds
- At anchor: up to 180 seconds (3-minute broadcast interval)
- Long gap (coverage gap or outage): 600+ seconds

**Why important for position predictor:** The predictor learns `delta_lat = f(sog, cog, heading, time_delta_sec, ...)`. If a vessel moves at 15 kn for 10 seconds, `delta_lat` is much smaller than for 60 seconds at the same speed. Without time_delta_sec, the model cannot distinguish these cases.

**Why important for anomaly detection:** A vessel with normal SOG but very long time_delta (e.g., 3600 seconds = 1 hour gap) is unusual — it may have turned off its transponder (AIS dark), which is suspicious for certain vessel types.

**Why NULL for first pings?** The first record per vessel track has no prior record. UNIX_TIMESTAMP(prev_time) is NULL → time_delta_sec = NULL. These records are dropped from Silver (and ML training/scoring).

---

#### hour (Hour of Day)

**Formula:** `hour = HOUR(base_datetime)` → integer 0–23

**Why important for congestion model:**
- Port traffic follows strong daily cycles: rush of vessel arrivals at 06:00–10:00, lulls at 02:00–04:00
- Cargo handling shifts operate on fixed schedules
- The Random Forest learns these patterns: "if hour=08 AND vessel_count=12, HIGH congestion is likely"

**Why important for position predictor:**
- Vessel speeds and headings correlate with time of day (morning port departure, evening anchorage)
- Fishing vessels follow tide patterns tied to hour
- Reduces unexplained variance in position delta

---

#### month (Month of Year)

**Formula:** `month = MONTH(base_datetime)` → integer 1–12

**Why important:** Maritime traffic is seasonal:
- Hurricane season (June–November): different routing patterns in Gulf of Mexico
- Fishing seasons: different vessel type distributions
- Cruise season: different passenger traffic patterns

**Our dataset limitation:** We only have May data (month=5). The model cannot learn seasonal variation from one month. This is acknowledged as a limitation.

---

#### day_of_week

**Formula:** `day_of_week = DAYOFWEEK(base_datetime)` → 0=Monday, 6=Sunday

**Why important for congestion model:**
- Weekday vs. weekend traffic differs significantly
- Commercial cargo operates 7 days but with different intensity
- Recreational/pleasure craft peaks on weekends

---

#### is_weekend

**Formula:** `is_weekend = 1 if day_of_week >= 5 else 0` (Saturday=5, Sunday=6)

**Why a separate binary feature?** Random forests can learn `day_of_week >= 5 → high recreational traffic`, but providing `is_weekend` explicitly reduces the required tree depth for this split, improving computational efficiency and interpretability.

---

#### lat_bin, lon_bin (Grid Cell)

**Formula:**
```python
lat_bin = ROUND(lat, 1)   # for density (0.1° ≈ 6 nm)
# OR
lat_bin = ROUND(lat / 0.5) * 0.5  # for congestion model (0.5° ≈ 30 nm)
```

**Why important for congestion model:** Port locations matter. Grid cell (29.5N, 95.0W) is Houston Ship Channel — inherently high-traffic. Grid cell (35.0N, 70.0W) is open Atlantic — inherently low-traffic. Including lat_bin and lon_bin allows the model to learn location-specific traffic patterns.

**Feature importance:** `vessel_count` dominates (~90%), but lat/lon provide the geographic context. The model learns: "low vessel_count at Houston grid cell at 08:00 → predict MEDIUM (it will fill up)"; "low vessel_count at open ocean → predict LOW (it stays empty)."

---

#### prev_lat, prev_lon (Lag Position)

**Formula:** `prev_lat = LAG(lat, 1) OVER (PARTITION BY mmsi ORDER BY base_datetime)`

**Why in Silver (for model training):** The position predictor's target is `delta_lat = future_lat - current_lat`. The current and previous positions together define the current trajectory direction and speed vector, which is the most predictive input.

**Note:** prev_lat and prev_lon are used internally to compute distance_nm but are not directly passed as features to ML models (distance_nm and sog_change capture the same information more compactly).

---

### 6.3 Feature Importance Discussion

**Anomaly Detection (Isolation Forest — 6 features):**
```
sog            — absolute speed; ultra-fast = anomaly
sog_change     — sudden speed change; most discriminating for sudden events
heading_change — sharp turns; combined with sog detects dangerous maneuvers
distance_nm    — corroborates SOG; catches GPS glitches
cog            — course; constant COG with erratic position is suspicious
heading        — combined with COG; large COG-heading divergence = drift/current issue
```

**Position Predictor (XGBoost — 10 features):**
The model predicts delta_lat and delta_lon. Most important features (by XGBoost feature importance):
1. `sog` — speed determines how far the vessel moves
2. `cog` — course determines direction of movement
3. `heading` — true heading modifies the direction
4. `time_delta_sec` — time interval scales the distance
5. `lat, lon` — current position (ocean curvature affects exact computation)
6. `sog_change, heading_change` — trend: is vessel accelerating/turning?
7. `hour, month` — temporal context

**Congestion Predictor (Random Forest — 8 features):**
Feature importance ranking:
1. `vessel_count` (~90%) — dominant: current density is the best predictor of next hour density
2. `lat_bin, lon_bin` (~5%) — location baseline traffic level
3. `hour` (~3%) — time-of-day patterns
4. `avg_sog` (~1%) — slower vessels linger longer, increasing future count
5. `day_of_week, is_weekend, stopped_count` (<1%) — minor contextual factors

**Interview question:** "Why does vessel_count dominate congestion prediction?"
**Answer:** "Traffic density is highly autocorrelated over 1-hour intervals. A cell with 20 vessels at 08:00 will likely still have many vessels at 09:00 — ships don't disappear instantaneously. This makes current vessel_count the single best predictor of next-hour count. This is valid because our target is the NEXT hour's count (not the current one), so it is not data leakage — it is a valid leading indicator."

---

## SECTION 7 — MACHINE LEARNING LAYER

### 7.1 Anomaly Detection — Isolation Forest

#### 7.1.1 What Is Isolation Forest?

Isolation Forest is an **unsupervised anomaly detection algorithm** based on the principle that anomalies are few and different — they are easier to "isolate" from the rest of the data using random splits.

**Core Idea:**
1. Build N random decision trees (default: 100, we use 200)
2. For each tree: randomly select a feature, randomly select a split value within the feature's range
3. Continue splitting until each record is isolated in its own leaf
4. Anomalies are isolated in fewer splits (shorter path length) than normal records
5. Anomaly score = normalized average path length across all trees
   - Score near 1.0 → very anomalous (isolated quickly)
   - Score near 0.5 → normal (requires many splits)
   - Score < 0.5 → very normal (dense cluster)

**Mathematical formula for anomaly score:**
```
s(x, n) = 2^(-E[h(x)] / c(n))

where:
  h(x) = average path length of x across all trees
  c(n) = expected average path length for a sample of size n (normalization)
       = 2×H(n-1) - (2(n-1)/n)   [H is harmonic number]
  E[h(x)] = expected path length
```

#### 7.1.2 Why Unsupervised (Not Supervised Anomaly Detection)?

We don't have labeled anomaly data. Nobody has gone through 5M AIS records and tagged "anomaly / not anomaly" for every row. Unsupervised learning trains on the distribution of normal behavior and identifies deviations.

**Alternatives considered:**

| Method | Why Not Used |
|---|---|
| One-Class SVM | Scales poorly to millions of records; kernel computation is O(n²) |
| Local Outlier Factor (LOF) | Memory-intensive: requires storing all training points; slow inference |
| Autoencoder (deep learning) | Requires GPU for training; higher engineering complexity; harder to explain |
| k-Nearest Neighbors anomaly | O(n) inference per record; too slow for real-time |
| Supervised classification | Requires labeled data; no labeled maritime anomaly dataset available |

**Isolation Forest advantages:**
- O(n log n) training, O(log n) inference per record — fast
- Linear memory: only stores the trees, not the data
- Works well with mixed numeric features
- Highly parallelizable (each tree is independent)
- Well-established in industry for tabular data anomaly detection

**Isolation Forest disadvantages:**
- Cannot explain WHY a record is anomalous (black box — partially mitigated by our rule-based overlay)
- Contamination parameter requires domain knowledge to set correctly
- Struggles with anomalies in dense regions (normal records near anomalies may be scored higher)
- Treats all features equally by default (no feature weighting)

#### 7.1.3 Contamination = 0.002 (0.2%)

**What it means:** The expected proportion of anomalies in the training data. This parameter sets the threshold: the top 0.2% most-isolated records are labeled as anomalies.

**Why 0.2% (not 1%, not 5%)?**

Maritime AIS data is overwhelmingly normal. Most vessels follow predictable routes at consistent speeds. True anomalies (suspicious behavior, GPS errors that passed the teleport filter, unusual maneuvers) are rare. Industry experience suggests 0.5–2% for AIS data.

We set 0.2% after observing that 1% produced too many false positives on normal port approach maneuvers (speed reductions, heading changes near berths are normal but appeared anomalous). At 0.2%, only the most extreme behaviors are flagged.

**If set too high (e.g., 5%):** Normal port maneuvers flagged as anomalies → alert fatigue → operators ignore alerts → safety system becomes useless.

**If set too low (e.g., 0.01%):** Real anomalies missed → safety risk.

**Our compromise at 0.2%:** ~10,000 anomalous records out of 5M Silver rows. Combined with rule-based detection, this provides reasonable coverage.

#### 7.1.4 Training Configuration

```python
IsolationForest(
    n_estimators=200,      # 200 trees (vs default 100) — better score stability
    contamination=0.002,   # 0.2% expected anomaly rate
    max_samples='auto',    # auto: min(256, n_samples) — sub-sampling for speed
    random_state=42        # reproducibility
)
```

**Why 200 estimators?** With 100 trees, anomaly scores have higher variance (the same record might score 0.7 one run and 0.65 another). 200 trees stabilizes scores. Beyond 300 trees, diminishing returns.

**Why max_samples='auto'?** Each tree is built on a random subsample (default 256 records). This is intentional: Isolation Forest specifically benefits from subsampling — it makes anomaly isolation more discriminative and speeds up training.

**StandardScaler:** Features are scaled before training. Isolation Forest itself does not require scaling (it's a tree-based method), but scaling ensures features with large ranges (distance_nm: 0-2 nm, heading_change: 0-180°) don't dominate the random feature selection. With scaling, each feature contributes equally to the random split selection.

#### 7.1.5 Inference Pipeline

```
New AIS record arrives via Kafka
    ↓
normalize_ais_record() → canonical field names
    ↓
compute_delta_features() → sog_change, heading_change, distance_nm, time_delta_sec
    ↓ (skip if first ping — no delta features available)
score_anomaly(record) called
    ↓
[Rule-based checks run first]
  1. sog > 30 → UNUSUAL_SPEED, score = sog/60
  2. sog_change < -5 → SUDDEN_STOP, score = 0.8
  3. heading_change > 45 AND sog > 2 → SHARP_TURN, score = 0.7
  4. sog < 0.5 AND in_us_port_zone → STATIONARY_RISK, score = 0.6
    ↓
[ML check]
  feature_vec = [sog, cog, heading, sog_change, heading_change, distance_nm]
  scaled_vec = scaler.transform(feature_vec)
  if isolation_forest.predict(scaled_vec) == -1:  # anomaly
      ml_score = compute_score()
      if ml_score > rule_score:
          override with ML anomaly
    ↓
Return: is_anomaly (bool), anomaly_score (float), anomaly_type (string)
```

**Why rule-based first, ML second?** Rules are interpretable — an operator understands "this vessel is going 35 knots near a port." ML catches subtler patterns rules miss — the combination provides both interpretability and coverage.

---

### 7.2 Position Prediction — XGBoost

#### 7.2.1 What Is XGBoost?

**XGBoost (eXtreme Gradient Boosting)** is a supervised ensemble method that builds decision trees sequentially, where each new tree corrects the errors of the previous ensemble.

**Core mechanism (gradient boosting):**
```
F₀(x) = initial prediction (e.g., mean of target)
For t = 1 to T:
  1. Compute residuals: rᵢ = -∂L/∂Fₜ₋₁(xᵢ)  [gradient of loss]
  2. Fit a decision tree hₜ(x) to the residuals
  3. Update: Fₜ(x) = Fₜ₋₁(x) + η × hₜ(x)
     where η = learning_rate (0.05 in our config)
FT(x) = final prediction
```

**XGBoost adds to standard gradient boosting:**
- L1 and L2 regularization (prevents overfitting)
- Second-order gradient statistics (better convergence)
- Column and row subsampling (reduces variance)
- Out-of-core computation (handles data larger than RAM)
- GPU support (not used here)
- Early stopping (stops when validation error stops improving)

#### 7.2.2 Why XGBoost for Position Prediction?

| Requirement | Why XGBoost Fits |
|---|---|
| Non-linear relationships | lat/lon prediction is non-linear (ships curve around obstacles, follow shipping lanes) |
| Feature interactions | SOG × time_delta interacts to produce distance; XGBoost captures this automatically |
| Tabular data | XGBoost is state-of-the-art for structured/tabular regression |
| Interpretability | Feature importance scores available (via gain, weight, SHAP) |
| Training speed | ~10 minutes on 2M records — acceptable for daily retraining |
| Inference speed | Single record: microseconds — essential for real-time scoring |

**Why not a physics-based (dead reckoning) model only?**

Dead reckoning: `new_lat = lat + (sog × cos(cog_rad)) / 60 × (time_min/60)`

This works perfectly in open water with no currents and constant heading. It fails when:
- Vessel is approaching a port (decelerating non-linearly)
- Vessel is in a traffic separation scheme (follows lane geometry, not straight line)
- Current is significant (COG ≠ intended course)
- Vessel performs a turn (heading changes during the interval)

XGBoost learns these corrections from historical data. That's why it achieves 4.24 nm MAE vs. dead reckoning's ~8 nm MAE (estimated).

**Why not LSTM/Transformer (deep learning)?**
- LSTM requires sequential input with fixed padding — complex preprocessing
- Training time: hours on GPU vs. minutes for XGBoost on CPU
- Our dataset (2M records) is not large enough to fully leverage deep learning capacity
- XGBoost is already near-optimal for this feature set at this scale
- Interpretability: XGBoost has feature importance; LSTM is a black box

**Alternative: Simple Linear Regression?**
Linear regression assumes the relationship between features and delta_lat is linear. In reality, a vessel's position change depends on sin(COG), cos(COG) — which are non-linear functions of COG. Linear regression would require manually computing sin/cos features. XGBoost learns these non-linearities automatically from the data.

#### 7.2.3 Target Engineering

**Why predict delta_lat instead of absolute future_lat?**

If we predicted absolute future_lat, the model would need to learn the entire geographic distribution of vessel positions (25°N–50°N for US waters). This is a much harder problem. By predicting delta_lat (the change), the model focuses on the kinematic pattern — how much does this vessel typically move in 5 minutes given its current speed, course, and heading.

**Target generation:**
```python
# For 5-minute horizon:
N_steps = round(5 × 60 / 28)  # ≈ 10-11 records (assuming 28-sec average interval)

df['future_lat'] = df.groupby('mmsi')['lat'].shift(-N_steps)
df['delta_lat']  = df['future_lat'] - df['lat']
df['delta_lon']  = df.groupby('mmsi')['lon'].shift(-N_steps) - df['lon']
```

**Why shift by records (not by time)?** The dataset has irregular timestamps. Shifting by N records provides an approximate time horizon, but the actual time varies. A better approach would be to shift by true time: find the first record ≥ T+5 minutes. The current implementation is an approximation — this is an acknowledged limitation.

**Impact:** Some training pairs represent 4-minute horizons, others 6-minute horizons. The model learns an average 5-minute behavior. For a more precise system, time-based shifting should be implemented.

#### 7.2.4 Hyperparameter Configuration

```python
XGBRegressor(
    n_estimators=300,        # 300 boosting rounds
    max_depth=6,             # tree depth: balance complexity vs. overfitting
    learning_rate=0.05,      # small step size: more trees, better generalization
    subsample=0.8,           # 80% row sampling per tree: reduces overfitting
    colsample_bytree=0.8,    # 80% feature sampling per tree: reduces correlation
    random_state=42,         # reproducibility
    early_stopping_rounds=30,# stop if val error flat for 30 rounds
    eval_set=[(X_val, y_val)],# validation set for early stopping
)
```

**Why learning_rate=0.05?** Lower learning rate = more trees needed but better generalization. With 300 trees and lr=0.05, the model converges smoothly. Higher lr (e.g., 0.3) would converge in ~50 trees but risk overfitting.

**Why max_depth=6?** Deeper trees can learn more complex patterns but overfit more easily. Depth 6 creates trees with up to 64 leaf nodes — sufficient for our 10-feature input space.

**Why subsample=0.8?** Stochastic gradient boosting (Friedman 1999). Randomly sampling 80% of rows per tree introduces variance that reduces overfitting — similar to dropout in neural networks.

**Early stopping:** If validation MAE doesn't improve for 30 consecutive rounds, training stops. This prevents overfitting without requiring a fixed n_estimators.

#### 7.2.5 Three Prediction Horizons

| Horizon | Model Files | Test MAE (nm) | Use Case |
|---|---|---|---|
| 5 min | xgb_lat_5min.pkl, xgb_lon_5min.pkl | 4.24 | Collision avoidance, immediate alert |
| 10 min | xgb_lat_10min.pkl, xgb_lon_10min.pkl | 6.18 | Traffic management, berth scheduling |
| 15 min | xgb_lat_15min.pkl, xgb_lon_15min.pkl | 6.86 | Port approach planning |

**Error growth:** MAE grows sub-linearly (4.24 → 6.18 → 6.86) rather than linearly. This is because:
- At 5 min: current kinematics strongly predict position
- At 10+ min: vessel may have altered course; current information loses predictive power
- The model learns to be more conservative at longer horizons

**MAE interpretation:**
- 5-minute MAE = 4.24 nm = 7.85 km
- A standard container ship is ~300m long, ~45m wide
- 4.24 nm error means the prediction circle has radius of 7.85 km — clearly not precise enough for safe navigation guidance but useful for traffic management
- Collision avoidance systems (AIS SART, ARPA radar) use 6-minute CPA (Closest Point of Approach) calculations with much more precise data
- Our system is useful for planning, not life-safety decisions — this must be clearly stated

**Dead reckoning fallback:**
```python
def predict_position_dr(lat, lon, sog, heading, minutes):
    dist_nm = sog * (minutes / 60)
    delta_lat = (dist_nm * cos(heading_rad)) / 60
    delta_lon = (dist_nm * sin(heading_rad)) / (60 * cos(lat_rad))
    return lat + delta_lat, lon + delta_lon
```
Confidence: 0.65 (lower than ML's 0.82)

---

### 7.3 Congestion Prediction — Random Forest

#### 7.3.1 What Is Random Forest?

Random Forest is an ensemble of decision trees trained using **bagging** (Bootstrap Aggregating):

1. For each of the N trees:
   a. Draw a bootstrap sample (random 63% of training data, with replacement)
   b. At each node split, only consider a random subset of features (√features for classification)
   c. Grow tree to max_depth (or until min_samples_leaf is met)
2. Final prediction: majority vote across all N trees (classification)

**Key difference from XGBoost:**
- XGBoost builds trees sequentially (each corrects previous errors) → boosting
- Random Forest builds trees in parallel (each independently) → bagging
- Random Forest is more robust to noise; XGBoost achieves lower bias

#### 7.3.2 Why Random Forest for Congestion (Not XGBoost)?

Both could work. Random Forest was chosen because:
- **Robustness:** Congestion labels are derived from thresholds (≥15 = HIGH), which introduces label noise at the boundaries. Random Forest handles label noise better due to averaging.
- **Class imbalance:** `class_weight='balanced'` is natively supported and proven effective in sklearn's Random Forest.
- **Speed:** Parallel training makes Random Forest faster for this 3-class problem with 8 features.
- **Interpretability:** Feature importances from Random Forest are more stable than XGBoost's gain-based importances for this feature set.

**Alternative: Logistic Regression?** Cannot model non-linear interactions between vessel_count and location (a count of 12 vessels is HIGH risk at Houston but LOW risk at open ocean).

**Alternative: Neural Network?** Overkill for 8 features and 65,000 training samples. Worse interpretability, longer training, no performance gain expected.

#### 7.3.3 Target Engineering — Preventing Data Leakage

**Critical design decision:**

```python
# Wrong (data leakage):
target = current_vessel_count  # predicting what we already know

# Correct (shift by 1 hour):
df_sorted = df.sort_values(['lat_bin', 'lon_bin', 'hour_bucket'])
df['next_vessel_count'] = df.groupby(['lat_bin', 'lon_bin'])['vessel_count'].shift(-1)
df['target'] = df['next_vessel_count'].apply(classify_congestion)
```

**What leakage would look like:** If we trained on current vessel_count to predict current congestion_level (which is derived from the same vessel_count), the model would achieve ~100% accuracy trivially. In production, it would fail because it would predict current state (already known) rather than future state (what we need).

**Consecutive hour check:**
```python
# Remove rows where next record is not exactly 1 hour later
df['hour_diff'] = (df['next_hour_bucket'] - df['hour_bucket']).dt.seconds / 3600
df = df[df['hour_diff'] == 1.0]
```
**Why?** If a grid cell has no vessels for several hours then one vessel appears, the "shift -1" would pair the empty cell with a distant future hour, creating a misleading training example. Only consecutive hour pairs are valid training instances.

#### 7.3.4 Class Imbalance Handling

**Class distribution in Gold density data:**
```
LOW:    ~75% of all grid-hour cells (most ocean is empty)
MEDIUM: ~15%
HIGH:   ~10%
```

**`class_weight='balanced'`:** Automatically adjusts sample weights so each class contributes equally to the loss function:
```
weight_LOW    = total_samples / (3 × n_LOW)
weight_MEDIUM = total_samples / (3 × n_MEDIUM)
weight_HIGH   = total_samples / (3 × n_HIGH)
```

**Effect:** The model sees HIGH congestion events as "heavier" training examples, preventing the model from just predicting LOW for everything (which would give 75% accuracy but zero recall on HIGH).

**Why this matters operationally:** False negatives on HIGH congestion are more costly than false positives. A port authority that misses an incoming HIGH congestion event cannot prepare berths in time.

#### 7.3.5 Chronological Train/Test Split

```python
# WRONG for time-series:
X_train, X_test = train_test_split(df, test_size=0.2, shuffle=True)

# CORRECT:
split_idx = int(len(df) × 0.8)
df_sorted = df.sort_values('hour_bucket')
X_train = df_sorted.iloc[:split_idx]
X_test  = df_sorted.iloc[split_idx:]
```

**Why no shuffling?** Same reason as the overall data split — future knowledge must not contaminate training. A model trained on congestion at hour 50 should not know about congestion at hour 20 if hour 20 is in the test set.

**Why min_samples_leaf=5?** Prevents the model from memorizing individual grid-hour cells with very few observations. Any leaf must represent at least 5 training samples, ensuring predictions are based on statistical patterns not individual examples.

---

## SECTION 8 — MODEL EVALUATION

### 8.1 Evaluation Metrics — Complete Definitions

#### Accuracy

**Formula:** `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

**What it measures:** Fraction of all predictions that are correct.

**When to use:** Only meaningful when classes are balanced.

**Why misleading for congestion:** If 75% of cells are LOW, a model that always predicts LOW achieves 75% accuracy while being useless for HIGH detection. This is why we report Macro F1 alongside accuracy.

**Our result:** 90.44% — high accuracy, but partially driven by excellent LOW performance.

---

#### Precision

**Formula:** `Precision = TP / (TP + FP)`

**What it measures:** Of all records predicted as class X, what fraction actually are class X?

**Interpretation:** Precision of HIGH = 0.6708 means: "When we predict HIGH congestion, we're right 67% of the time." 33% of HIGH predictions are false alarms.

**Business impact of low precision:** Too many false alarms → alert fatigue → operators stop trusting the system.

---

#### Recall

**Formula:** `Recall = TP / (TP + FN)`

**What it measures:** Of all actual class X records, what fraction did we correctly identify?

**Interpretation:** Recall of HIGH = 0.4037 means: "We catch only 40% of actual HIGH congestion events. We miss 60%."

**Business impact of low recall:** Missed HIGH events → port not prepared → vessel delays, fuel waste, potential incidents.

**Our HIGH recall problem (40%):** This is the most significant weakness. The model misses 60% of true HIGH congestion events. The root cause is class imbalance — even with balanced weights, HIGH events in training have high variance due to scarcity.

---

#### F1 Score

**Formula:** `F1 = 2 × (Precision × Recall) / (Precision + Recall)` — harmonic mean

**Why harmonic mean (not arithmetic)?** Harmonic mean penalizes extreme imbalances. If Precision=1.0 and Recall=0.0 (predict everything as positive), arithmetic mean = 0.5; harmonic mean = 0 (correctly reflects uselessness).

**F1 for HIGH = 0.5041:** Moderate. The system is somewhat useful for HIGH detection but cannot be relied upon for critical decisions.

---

#### Macro F1

**Formula:** `Macro F1 = (F1_LOW + F1_MEDIUM + F1_HIGH) / 3`

**What it measures:** Unweighted average F1 across all classes. Each class contributes equally regardless of its frequency.

**Our result:** `(0.9674 + 0.6442 + 0.5041) / 3 = 0.7052`

**Why Macro F1 (not Weighted F1)?** Weighted F1 weights by class frequency. With 75% LOW, weighted F1 would be dominated by LOW performance (~0.95), hiding the poor HIGH performance. Macro F1 gives equal weight to each class — the right choice when all classes matter operationally.

**Interpretation of 0.7052:** Moderate overall performance. Excellent on LOW (0.9674), acceptable on MEDIUM (0.6442), weak on HIGH (0.5041). This is the most honest summary of the model's capabilities.

---

#### MAE (Mean Absolute Error) — Position Predictor

**Formula:** `MAE = (1/n) × Σ|predicted_i - actual_i|`

**Units:** Nautical miles (nm) for our position predictor

**Why MAE (not RMSE)?** RMSE squares errors, making it sensitive to large outliers. A single GPS-glitched vessel appearing 200 nm away would dominate RMSE. MAE treats all errors equally — more representative of typical performance across the fleet.

**Our results:**
```
5-minute MAE:  4.24 nm (= 7.85 km)
10-minute MAE: 6.18 nm (= 11.45 km)
15-minute MAE: 6.86 nm (= 12.70 km)
```

**Contextual interpretation:**
- AIS accuracy itself is ~10-100 meters
- Our 5-minute prediction has ~7.8 km uncertainty
- For context: the Gulf of Mexico is ~1,800 km wide
- For container ship berth planning (berth width ~100m): insufficient for docking
- For traffic density monitoring (cell = 11km × 11km): the vessel will likely remain in the same cell — adequate for congestion assessment
- For collision risk assessment at 0.1 nm (185m): insufficient — AIS ARPA systems do this with radar

**RMSE comparison:**
```
5-minute RMSE lat: 7.247 nm, RMSE lon: 8.074 nm
```
RMSE > MAE confirms there are outliers with large prediction errors (some vessels make unexpected turns that the model misses). The ratio RMSE/MAE ≈ 1.9 indicates the error distribution has heavy tails.

---

### 8.2 Our Real Model Performance — Interpretation

#### Congestion Model Scorecard

| Class | Precision | Recall | F1 | Interpretation |
|---|---|---|---|---|
| HIGH (≥15 vessels) | 0.6708 | 0.4037 | 0.5041 | Catches 40% of actual congestion. When it alerts, 67% correct |
| MEDIUM (5-14 vessels) | 0.5339 | 0.8117 | 0.6442 | Catches 81% of medium events. Many false positives |
| LOW (<5 vessels) | 0.9880 | 0.9476 | 0.9674 | Excellent. Nearly perfect on empty cells |
| **Macro F1** | **0.7309** | **0.7210** | **0.7052** | Moderate overall — driven by LOW dominance |

**Overall Accuracy: 90.44%** — inflated by LOW performance. True system capability = Macro F1 = 0.7052.

**The High-MEDIUM confusion:** The model sometimes predicts MEDIUM when actual is HIGH, and HIGH when actual is MEDIUM. These boundary cases (vessel_count = 12-17) are inherently ambiguous given our hard thresholds at 15.

#### Position Predictor Scorecard

| Horizon | MAE Total (nm) | MAE lat (nm) | MAE lon (nm) | RMSE lat (nm) |
|---|---|---|---|---|
| 5 min | 4.24 | 2.77 | 3.22 | 7.25 |
| 10 min | 6.18 | 3.93 | 4.77 | 9.35 |
| 15 min | 6.86 | 4.15 | 5.46 | 9.39 |

**Note:** MAE lon > MAE lat at every horizon. This is because longitude movement is harder to predict — east-west course components interact more with ocean currents, port approach vectors, and traffic separation scheme boundaries.

#### Anomaly Detection — Why No Metrics?

Anomaly detection metrics (AUC-ROC, precision-recall) require ground truth labels. We have no labeled dataset of "confirmed maritime anomalies." This is a fundamental limitation of unsupervised learning evaluation.

**What we can say instead:**
- The model flags 0.2% of records (by contamination parameter)
- Manual inspection of flagged records shows: unusual speed vessels, erratic course changes, vessels in unexpected areas
- Rule-based anomalies provide an auditable, interpretable complement

**How we would evaluate it in production:** Collect operator feedback ("this alert was useful / not useful"), compute precision from feedback, and retrain with the contamination parameter adjusted.

---

## SECTION 9 — DESIGN DECISIONS

### 9.1 Why Medallion Architecture (Bronze/Silver/Gold)?

**What it is:** A data organization pattern where data progresses through three quality tiers:
- **Bronze:** Raw, minimally validated, append-only — the source of truth
- **Silver:** Cleaned, deduplicated, enriched — production-quality
- **Gold:** Pre-aggregated, analytics-optimized — serving layer

**Why this specific pattern?**

1. **Reprocessability:** If a bug is found in the Silver cleaning logic, we can rerun the Silver job from Bronze without re-ingesting from source. Without Bronze, we'd need to re-fetch all AIS data.

2. **Data quality escalation:** Each layer adds quality guarantees. Bronze: "the data exists." Silver: "the data is clean and consistent." Gold: "the data is aggregated and query-ready."

3. **Multiple consumers:** The Streamlit dashboard queries Gold (fast, pre-aggregated). ML training reads Silver (clean, detailed). Debugging reads Bronze (raw, complete). Different consumers get the tier they need.

4. **Auditability:** Any Gold anomaly can be traced back through Silver to Bronze to the original AIS broadcast.

5. **Industry standard:** Databricks coined this pattern; it's now used at Uber, Netflix, Lyft for their data lakes.

**Why Delta Lake for all three layers?**
- ACID transactions: concurrent reads and writes are safe
- Schema enforcement: wrong data types rejected at write time
- Time travel: query Bronze from 5 days ago for debugging: `spark.read.format("delta").option("versionAsOf", "5").load(path)`
- Optimistic concurrency: multiple Spark jobs can write to different partitions simultaneously

---

### 9.2 Why PostgreSQL as Serving Layer (Not Parquet/Delta directly)?

**Problem with querying Delta Lake from the dashboard:** Spark must be started, a session created, a DataFrame query executed. This takes 30–60 seconds — unacceptable for a real-time dashboard.

**PostgreSQL solution:** Pre-computed Gold data pushed to PostgreSQL. Dashboard queries: `SELECT * FROM fact_vessel_latest WHERE risk_level='HIGH'` — returns in milliseconds.

**Trade-off:** Data duplication (same data in Delta Lake and PostgreSQL). Accepted trade-off for query performance.

**Why not ClickHouse/DuckDB for serving?** PostgreSQL is universally supported by SQLAlchemy, has mature tooling, and is sufficient at our scale (1,000 vessels, 5M tracks). ClickHouse is better at 100M+ rows but requires more setup.

---

### 9.3 Why Star Schema (Not Normalized / Flat Table)?

**Star schema advantages for this project:**

1. **Separates concerns:** `dim_vessel` stores vessel metadata; `fact_vessel_latest` stores current position. Updating a vessel's name only requires one update in `dim_vessel`, not updating every fact row.

2. **Query performance:** Dashboard asks "show all HIGH-risk Tankers." With a star schema: `JOIN dim_vessel ON mmsi WHERE vessel_type='Tanker' AND risk_level='HIGH'` — using indexes on both tables.

3. **Extensibility:** Adding a new dimension (e.g., `dim_route` for shipping lanes) doesn't require changing the fact tables.

4. **Analytics tool compatibility:** BI tools (Tableau, Metabase) auto-detect star schemas and build joins automatically.

**Alternative — single flat table:** All columns in one table. Simpler to query but: duplicates vessel metadata on every fact row (vessel_name repeated 10,000 times), harder to maintain consistency, larger storage footprint.

---

### 9.4 Why Docker Compose for Orchestration?

**Docker Compose** defines all services (Kafka, Spark, PostgreSQL, API, Dashboard) as containers in a single YAML file, allowing `docker-compose up -d` to start the entire system.

**Why not Kubernetes?** For a demo/development environment, Docker Compose is simpler. Kubernetes is provided as an optional production deployment (`k8s/` directory) for when the system needs scaling, rolling deployments, and self-healing. Using Kubernetes for a demo would add significant setup complexity without benefit.

**Why not bare-metal / virtual machines?** Containers ensure reproducibility — the system runs identically on any machine with Docker, eliminating "works on my machine" problems during the presentation demo.

---

### 9.5 Why Spark Shuffle Partitions = 8 (Not Default 200)?

**Problem with default 200:** With ~5M Silver rows (~2GB), distributing across 200 partitions creates 200 tasks of ~10MB each. The overhead of scheduling, shuffling, and tracking 200 tasks exceeds the computation time. This is known as the "small file problem" in Spark.

**Our setting of 8:** Creates 8 tasks of ~250MB each — better balance between parallelism and overhead for our 2-worker cluster.

**How to choose the right number:** Rule of thumb = 2×(number of executor cores). Our cluster: 2 workers × 4 cores = 8 executor slots → 8 shuffle partitions.

**Risk of too few partitions (e.g., 2):** Not enough parallelism; one worker sits idle. Risk of too many (e.g., 1000): task overhead > computation time.

---

### 9.6 Why These Specific Anomaly Rule Thresholds?

| Rule | Threshold | Reasoning |
|---|---|---|
| UNUSUAL_SPEED | SOG > 30 kn | Commercial vessel max ~27 kn; 30 gives buffer for rounding/measurement |
| SUDDEN_STOP | Δsog < -5 kn | 5 kn deceleration is aggressive but not impossible; <2 kn is normal slowing |
| SHARP_TURN | heading_change > 45° | 45° is substantial; normal ocean course corrections: <10° |
| STATIONARY_RISK | SOG < 0.5 kn in port | 0.5 kn is operationally "stopped"; stopped in monitored port = high risk |

**All thresholds are tunable** via `config.py`. In production, they would be calibrated based on operator feedback and historical incident data.

---

## SECTION 10 — SYSTEM LIMITATIONS

### 10.1 No Weather/Environmental Data

**What's missing:** Wind speed, wave height, ocean current, visibility, storm tracks.

**Business impact:**
- A vessel at 8 knots in 10-foot seas is behaving normally; the same vessel at 8 knots in flat calm may be slowing for a reason
- Our anomaly detector cannot distinguish weather-induced speed changes from mechanical failures
- Position predictor ignores ocean currents — a vessel in the Gulf Stream drifts significantly east even with engines off

**Technical impact:** MAE of position predictor would decrease significantly (estimated 30-50%) with current data.

**Solution:** Integrate NOAA weather API (free) and OSCAR ocean current data (free) into Silver feature engineering.

---

### 10.2 No Vessel Route/Schedule Information

**What's missing:** Published shipping schedules, expected routes, port call appointments.

**Business impact:**
- A vessel 50 nm off its expected route is anomalous; our system doesn't know the expected route
- Many "anomalies" flagged by our system (unexpected heading) are vessels detouring to avoid weather — not suspicious

**Technical impact:** Precision of anomaly detection would improve dramatically if expected routes were known.

**Solution:** Integrate with Lloyd's List shipping intelligence API or port authority berth-scheduling systems.

---

### 10.3 No Deep Learning Trajectory Model

**What's missing:** LSTM or Transformer-based sequence models that process the full vessel track history (not just the last two records).

**Business impact:** Our XGBoost predictor only looks at the current + previous record. A vessel that has been traveling northeast for 4 hours is more likely to continue northeast than a vessel that just changed course. LSTMs capture this multi-step memory.

**Why not implemented:** Added engineering complexity (PyTorch/TensorFlow stack, GPU requirement, sequence padding, variable-length tracks). For an internship project, XGBoost with lag features is a pragmatic choice. This is the highest-priority technical improvement.

---

### 10.4 Hardcoded Port Zones (Only 5 US Ports)

**What's missing:** The full universe of global ports (there are 4,000+ commercial ports worldwide).

**Business impact:** `in_us_port_zone` flag only applies to 5 specific US ports. A vessel behaving suspiciously near Rotterdam or Singapore is not flagged.

**Technical impact:** Risk classification is incorrect for non-US waters; STATIONARY_RISK anomaly only fires in these 5 zones.

**Solution:** Use World Port Index (free dataset), load port zones dynamically from a database table. Implement PostGIS for proper polygon-based geofencing.

---

### 10.5 No Real-Time AIS Feed (Simulated Stream)

**What's missing:** A real AIS data subscription (commercial providers: MarineTraffic, ExactEarth, Orbcomm — $500-5000/month).

**Our simulation:** Kafka producer reads LIVE split parquet files (historical data from Days 13-14) and streams them at 500 msg/sec. This simulates real-time but does not reflect actual current vessel positions.

**Business impact:** The demo map shows historical positions labeled as "live." In a real deployment, subscribers to AIS-b (terrestrial) or AIS satellite feeds would provide true real-time data.

---

### 10.6 Limited Data (7 Days, US Waters Only)

**What's missing:**
- Full year of data (to capture seasonal patterns)
- Global coverage (not just US coastal waters)
- Multiple vessel types in proportion (fishing vs. cargo vs. tanker behaviors differ significantly)

**ML impact:**
- Congestion predictor cannot learn seasonal traffic variations
- Position predictor may generalize poorly to non-US vessel routing conventions
- Isolation Forest trained on May US patterns may flag as anomalous vessels that are normal in June or in European waters

---

### 10.7 No Authentication or Security

**What's missing:** API authentication, role-based access control, data encryption.

**Risk:** The FastAPI has `CORS: allow all origins` — any website can query vessel positions. This is acceptable for a demo but unacceptable in production (maritime security / AIS spoofing concerns).

**Solution:** Add OAuth2 + JWT tokens to API; restrict CORS to known dashboards; encrypt PostgreSQL connections with TLS.

---

### 10.8 PostgreSQL Password Hardcoded

**Config.py:** `POSTGRES_PASSWORD = "maritime123"`

This is a critical security vulnerability in production. Acceptable for demo. Solution: use Docker secrets, Kubernetes secrets, or AWS Secrets Manager.

---

## SECTION 11 — FUTURE IMPROVEMENTS (PRIORITIZED)

| Priority | Improvement | Effort | Impact | Architecture Change Required |
|---|---|---|---|---|
| HIGH | Real AIS data feed (MarineTraffic API) | 2 weeks | Enables real-time production deployment | Kafka producer replacement |
| HIGH | LSTM/Transformer trajectory model | 4 weeks | MAE reduction ~30%, multi-step memory | GPU server + PyTorch stack |
| HIGH | Weather data integration (NOAA API) | 1 week | Anomaly false positive reduction ~40% | Silver job feature addition |
| HIGH | Global port zone database (PostGIS) | 1 week | Correct risk classification globally | Add PostGIS, dynamic zone loading |
| HIGH | Labeled anomaly dataset + supervised model | 8 weeks | Precision/recall measurement possible | Training pipeline overhaul |
| MEDIUM | SHAP values for anomaly explanation | 1 week | "This vessel is anomalous because SOG=35 (score +0.4)" | Model wrapper addition |
| MEDIUM | Route deviation detection | 3 weeks | Detect vessels off published routes | New route database integration |
| MEDIUM | API authentication (OAuth2/JWT) | 1 week | Production security compliance | FastAPI middleware |
| MEDIUM | Airflow/Prefect pipeline orchestration | 2 weeks | Scheduled jobs, retry logic, DAG visualization | Replace shell scripts with DAG |
| MEDIUM | Streaming Silver (Structured Streaming) | 3 weeks | Near-real-time clean features | Silver job rewrite |
| MEDIUM | Vessel behavior clustering (K-Means) | 2 weeks | Profile vessel archetypes for better anomaly context | New ML component |
| LOW | Snowflake integration (long-term archival) | 1 week | Cost-efficient cold storage | Snowflake connector |
| LOW | Kubernetes production deployment | 2 weeks | Auto-scaling, rolling updates | k8s manifests (already drafted) |
| LOW | MLflow experiment tracking | 1 week | Model versioning, parameter comparison | MLflow server deployment |
| LOW | Predictive maintenance indicators | 6 weeks | Abnormal stop/speed patterns → engine fault prediction | Domain expert input needed |

---

*End of Part 2. Continue reading TECHNICAL_KNOWLEDGE_REPORT_PART3.md for the complete 120-question Q&A, Presentation Guide, and Final Technical Assessment.*
