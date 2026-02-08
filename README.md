# BirdNET + BSG Model Fuser

Tools for preparing BSG Finland models for use with
[rust-birdnet-onnx](https://github.com/tphakala/rust-birdnet-onnx):

1. **`fuse_models.py`** — Fuse BirdNET v2.4 backbone + BSG classifier into a
   single end-to-end ONNX model
2. **`convert_bsg_data.py`** — Convert BSG prediction adjustment data
   (calibration, migration, distribution maps) to formats consumed by
   rust-birdnet-onnx

## Setup

```bash
pip install -r requirements.txt
```

## Fusing models

BirdNET v2.4 classifies 6,522 bird species globally. BSG classifiers are
regional models trained on BirdNET embeddings for better local accuracy. Running
them requires a two-stage pipeline: extract embeddings from BirdNET, then feed
them to the BSG model.

`fuse_models.py` merges both stages into a single ONNX model:

```
audio [batch, 144000] @ 48kHz, 3 seconds
  -> BirdNET backbone (mel spectrogram + EfficientNet CNN)
  -> embeddings [batch, 1024]
  -> BSG classifier (Dense -> ReLU -> Dense -> Sigmoid)
  -> predictions [batch, 265]
```

### Prerequisites

- **BirdNET ONNX model** — converted by
  [birdnet-onnx-converter](https://github.com/tphakala/birdnet-onnx-converter)
  from the official TFLite model
- **BSG classifier ONNX model** — the regional classifier head from
  [BSG](https://github.com/luomus/BSG) (e.g.
  `BSG_birds_Finland_v4_4_fp32.onnx`)

### Usage

```bash
python fuse_models.py \
    --birdnet /path/to/BirdNET_fp32.onnx \
    --bsg /path/to/BSG_birds_Finland_v4_4_fp32.onnx \
    --output BSG_birds_Finland_v4_4_fused_fp32.onnx
```

The script automatically verifies the fused model against the two-stage pipeline
(add `--skip-verify` to skip).

### How it works

1. Loads the BirdNET ONNX model and removes the final classification head
   (MatMul + Add for 6,522 global species)
2. Exposes the 1,024-dim embedding from the global average pooling layer
3. Rewires the BSG classifier input to consume that embedding tensor
4. Merges both graphs into a single ONNX model
5. Verifies the fused output matches the two-stage pipeline (tolerance < 1e-5)

## Converting prediction adjustment data

The BSG project includes per-species calibration parameters, migration curves,
and geographic distribution maps used for post-processing predictions. These are
stored as NumPy arrays and GeoTIFF files. `convert_bsg_data.py` converts them
to the CSV and packed binary formats used by `BsgPostProcessor` in
rust-birdnet-onnx.

### Source data

From the [BSG](https://github.com/luomus/BSG) project, directory
`scripts/Pred_adjustment/`:

| File | Description |
|------|-------------|
| `calibration_params.npy` | Per-species Platt scaling parameters (265 x 2) |
| `migration_params.npy` | Migration curve parameters (265 x 8) |
| `distribution_maps/*.tif` | Geographic presence maps, 526 GeoTIFFs (263 species x 2 maps) |

### Usage

```bash
# Convert all three data files
python convert_bsg_data.py \
    --bsg-data /path/to/BSG/scripts/Pred_adjustment \
    --output-dir ./output

# Skip distribution maps if rasterio is not available
python convert_bsg_data.py \
    --bsg-data /path/to/BSG/scripts/Pred_adjustment \
    --output-dir ./output \
    --skip-maps
```

### Output files

| File | Size | Description |
|------|------|-------------|
| `BSG_calibration.csv` | 7 KB | Intercept + slope per species (CSV with header) |
| `BSG_migration.csv` | 17 KB | 8 migration parameters per species (CSV with header) |
| `BSG_distribution_maps.bin` | 25.8 MB | Packed binary: 64-byte header + 263 species x 2 maps x 132x93 float32 |

### Distribution maps binary format

```
Header (64 bytes):
  magic:           4 bytes   "BSDM"
  version:         u32 LE    (1)
  num_species:     u32 LE    (263)
  width:           u32 LE    (93)
  height:          u32 LE    (132)
  origin_lon:      f64 LE    west edge longitude
  origin_lat:      f64 LE    north edge latitude
  pixel_scale_lon: f64 LE    degrees east per pixel
  pixel_scale_lat: f64 LE    degrees south per pixel
  _reserved:       12 bytes  zeros

Data (per species, ordered by class index 2-264):
  map_a: height * width * f32 LE
  map_b: height * width * f32 LE
```

NaN values in the source TIFFs are replaced with 1.0 (assume species possible
where no data exists).

## Compatibility

Tested with:
- BirdNET v2.4 ONNX (from `BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite`)
- BSG Finnish Birds classifier v4.4

## References

- [BSG](https://github.com/luomus/BSG) — BSG Finnish Birds Model by the University of Jyväskylä (part of the **Muuttolintujen kevät** project)
- [rust-birdnet-onnx](https://github.com/tphakala/rust-birdnet-onnx) — Rust library for BirdNET, Perch, and BSG Finland inference
- [birdnet-onnx-converter](https://github.com/tphakala/birdnet-onnx-converter) — Convert and optimize BirdNET models for ONNX Runtime
