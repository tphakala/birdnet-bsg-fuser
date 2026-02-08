# BSG Finland Model — Technical Reference

Technical documentation for the BSG (Bird Sounds Global) Finnish Birds classification
system. Covers the model architecture, inference pipeline, calibration, and species
distribution model (SDM) algorithms as implemented in the
[BSG](https://github.com/luomus/BSG) project by the University of Jyväskylä,
developed as part of the **Muuttolintujen kevät** (Spring of Migrating Birds)
citizen science initiative.

## Model Architecture

BSG uses a two-stage pipeline built on top of BirdNET v2.4:

```
audio [batch, 144000] @ 48 kHz (3 seconds)
  ┌──────────────────────────────────────┐
  │  BirdNET v2.4 Backbone               │
  │  ─ Mel spectrogram                   │
  │  ─ EfficientNet CNN                  │
  │  ─ Global average pooling            │
  │  → embeddings [batch, 1024]          │
  └──────────────────────────────────────┘
  ┌──────────────────────────────────────┐
  │  BSG Classification Head             │
  │  ─ Dense(1024 → hidden) + ReLU       │
  │  ─ Dense(hidden → 265) + Sigmoid     │
  │  → predictions [batch, 265]          │
  └──────────────────────────────────────┘
```

### Audio Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 48,000 Hz |
| Segment duration | 3.0 seconds |
| Samples per segment | 144,000 |
| Default overlap | 1.0 seconds |
| Input dtype | float32 |

### Output

The BSG classifier outputs 265 values per segment, each in `[0, 1]` (sigmoid
activation). These are **not** softmax probabilities — they are independent
per-species confidence scores.

**Important:** The fused ONNX model already includes the sigmoid. Applying sigmoid
again produces a "double sigmoid" that compresses all scores toward 0.50.

## Class Structure (265 Classes)

| Index | Type | Description |
|-------|------|-------------|
| 0 | Meta-class | "No bird" — background noise |
| 1 | Meta-class | "Human" — speech and human noise |
| 2–264 | Bird species | 263 Finnish bird species |

### Label File Format

The BSG label file (`classes.csv`) has 266 rows (header + 265 data rows):

```csv
species_code,common_name,scientific_name,luomus_name,suomenkielinen_nimi,class
"nobird","No bird","No bird",,"ei lintu",0
"human","Human","Human",,"ihminen",1
"lotduc","Long-tailed Duck","Clangula hyemalis","Clangula hyemalis","alli",2
"merlin","Merlin","Falco columbarius","Falco columbarius","ampuhaukka",3
...
```

For inference with `rust-birdnet-onnx`, a simplified text label file is used
(one label per line, 265 lines).

## Calibration (Platt Scaling)

Raw model outputs are calibrated using per-species logistic (Platt) scaling.

### Formula

```
P_calibrated = 1 / (1 + exp(-(β₀ + β₁ × P_raw)))
```

Where for each species `i`:
- `β₀` = `calibration_params[i, 0]` (intercept)
- `β₁` = `calibration_params[i, 1]` (slope)
- `P_raw` = raw sigmoid output from the model

### Parameters

Loaded from `calibration_params.npy`, shape `(265, 2)`:

| Column | Name | Description |
|--------|------|-------------|
| 0 | intercept (β₀) | Logistic intercept per species |
| 1 | slope (β₁) | Logistic slope per species |

Meta-classes (indices 0–1) have `(0, 0)`, meaning calibration is a no-op
(`1 / (1 + exp(0)) = 0.5` when `P_raw = 0`).

### Example Values

```
Species 0 (nobird):          β₀ =  0.000,  β₁ =  0.000
Species 1 (human):           β₀ =  0.000,  β₁ =  0.000
Species 2 (Long-tailed Duck): β₀ = -8.055,  β₁ = 10.845
Species 3 (Merlin):          β₀ = -4.895,  β₁ =  7.505
```

## Species Distribution Model (SDM)

The SDM adjusts calibrated predictions based on when and where the observation
was made. It combines three signals:

1. **Migration probability** — is the species expected at this latitude on this day?
2. **Geographic presence** — is the species known to occur at this location?
3. **Seasonal map** — time-of-year-specific geographic presence

SDM is applied only to bird species (indices 2–264). Meta-classes (0–1) are skipped.

### Migration Parameters

Loaded from `migration_params.npy`, shape `(265, 8)`:

| Index | Name | Description |
|-------|------|-------------|
| 0 | `arrival_intercept` | Spring arrival day at latitude 0 |
| 1 | `arrival_slope` | Spring arrival latitude coefficient |
| 2 | `departure_intercept` | Fall departure day at latitude 0 |
| 3 | `departure_slope` | Fall departure latitude coefficient |
| 4 | `arrival_std` | Spring arrival standard deviation (before halving) |
| 5 | `departure_std` | Fall departure standard deviation (before halving) |
| 6 | `seasonal_start` | Day of year when seasonal map (map_b) becomes active |
| 7 | `seasonal_end` | Day of year when seasonal map (map_b) becomes inactive |

### Migration Probability

Uses the Normal CDF to model arrival and departure as sigmoid-like transitions:

```python
arrival_mean = arrival_intercept + arrival_slope × latitude
arrival_prob = Φ(day_of_year; μ=arrival_mean, σ=arrival_std / 2)

departure_mean = departure_intercept + departure_slope × latitude
departure_prob = 1 - Φ(day_of_year; μ=departure_mean, σ=departure_std / 2)

migration_probability = min(arrival_prob, departure_prob)
```

Where `Φ(x; μ, σ)` is the Normal CDF. Note the standard deviation is **halved**
before use.

### Distribution Maps

Two GeoTIFF maps per species for indices 2–264 (263 species × 2 = 526 files):

| Map | File Pattern | Description |
|-----|-------------|-------------|
| map_a | `{index}_a.tif` | Year-round geographic presence probability |
| map_b | `{index}_b.tif` | Seasonal geographic presence probability |

All maps share the same grid:

| Property | Value |
|----------|-------|
| CRS | EPSG:4326 (WGS 84) |
| Width | 93 pixels |
| Height | 132 pixels |
| Origin (NW corner) | lon ≈ 17.60°E, lat ≈ 70.59°N |
| Pixel scale | ≈ 0.215° lon, ≈ 0.089° lat |
| Coverage | Finland and surrounding area |
| Dtype | float32 |
| NoData handling | NaN → 1.0 (species assumed possible) |

### Geographic Presence (map_a)

Sample `map_a` at the observation coordinates:

```
geo_presence = sample(map_a, longitude, latitude)
if isnan(geo_presence):
    geo_presence = 1.0  # assume species possible where no data
```

### Seasonal Map Logic (map_b)

The seasonal map is only consulted during a species-specific date range,
which can wrap across the year boundary:

```python
use_map_b = 0

if seasonal_start < seasonal_end:
    # Non-wrapping range (e.g., day 91 to 280)
    if seasonal_start <= day_of_year <= seasonal_end:
        use_map_b = 1

if seasonal_start > seasonal_end:
    # Wrapping range (e.g., day 300 to 60 — winter species)
    if day_of_year >= seasonal_start or day_of_year <= seasonal_end:
        use_map_b = 1

if use_map_b:
    time_presence = sample(map_b, longitude, latitude)
    if isnan(time_presence):
        time_presence = 1.0
```

### Combined Presence Probability

```
presence = migration_prob × (geo_presence + (1 - geo_presence) × use_map_b × time_presence)
```

This blends the year-round geographic presence with the seasonal map. When
`use_map_b = 0`, it simplifies to:

```
presence = migration_prob × geo_presence
```

### Score Adjustment

The presence probability is converted to a score adjustment on a logarithmic scale:

```python
adj = clamp(log10(presence) + 1, -10, 0)

adjusted_score = max(0, score + adj × 0.25) / max(0.0001, 1 + adj × 0.25)
```

| Constant | Value | Purpose |
|----------|-------|---------|
| `0.25` | Adjustment scaling factor | Controls SDM influence strength |
| `-10` | Minimum adjustment | Prevents `-inf` from `log10(0)` |
| `0.0001` | Denominator floor | Prevents division by zero |

**Behavior:**
- `presence = 1.0` → `adj = 0` → score unchanged
- `presence = 0.1` → `adj = 0` → score unchanged (log10(0.1) + 1 = 0)
- `presence = 0.01` → `adj = -1` → score reduced
- `presence ≈ 0` → `adj = -10` → score heavily suppressed

### Day of Year Handling

```
if day_of_year > 365:
    day_of_year = 365
```

Leap year day 366 is clamped to 365. Valid range: 1–366 (input), 1–365 (effective).

### Coordinate Validation

```
latitude  must be in [-90, 90]
longitude must be in [-180, 180]
```

## Processing Pipeline Order

The full inference pipeline for a single audio segment:

```
1. Audio → BirdNET+BSG model → raw predictions [265]     (sigmoid output)
2. Calibrate predictions (Platt scaling)                   (per-species)
3. Threshold filter (default ≥ 0.5)                        (remove low scores)
4. SDM adjustment (if location/date provided)              (per-species, indices 2-264)
5. Output: species predictions with adjusted confidence
```

## Default Parameters

| Parameter | Default | Range |
|-----------|---------|-------|
| Confidence threshold | 0.5 | 0.0–1.0 |
| Overlap | 1.0 s | — |
| Chunk size | 600 s | 10–1200 s |
| SDM enabled | false | — |

## File Inventory

### Model Files

| File | Description |
|------|-------------|
| `BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite` | BirdNET v2.4 backbone (TFLite) |
| `BSG_birds_Finland_v4_4.keras` | BSG classification head (Keras) |
| `BSG_birds_Finland_v4_4_fp32.onnx` | BSG classification head (ONNX, standalone) |
| `BSG_birds_Finland_v4_4_fused_fp32.onnx` | End-to-end fused model (ONNX) |

### Prediction Adjustment Data

Local directory: `data/`

| File | Shape/Count | Description |
|------|-------------|-------------|
| `data/BSG_calibration.csv` | (265, 2) | Platt scaling intercept + slope |
| `data/BSG_migration.csv` | (265, 8) | Migration curve parameters |
| `data/distribution_maps/*.tif` | 526 files | Geographic and seasonal presence maps |

### Converted Binary Maps (for rust-birdnet-onnx)

Produced by `convert_bsg_data.py`:

| File | Size | Description |
|------|------|-------------|
| `data/BSG_distribution_maps.bin` | 25.8 MB | Packed binary with 64-byte header |

## References

- [BSG](https://github.com/luomus/BSG) — BSG Finnish Birds Model (University of Jyväskylä)
- [BirdNET](https://github.com/kahst/BirdNET-Analyzer) — BirdNET v2.4 by Cornell Lab of Ornithology
- [rust-birdnet-onnx](https://github.com/tphakala/rust-birdnet-onnx) — Rust inference library
- [birdnet-onnx-converter](https://github.com/tphakala/birdnet-onnx-converter) — BirdNET model conversion tool
