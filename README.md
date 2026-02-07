# BirdNET + BSG Model Fuser

Fuses the BirdNET v2.4 feature extractor with a BSG regional classifier into a
single end-to-end ONNX model.

## What it does

BirdNET v2.4 classifies 6,522 bird species globally. BSG classifiers are
regional models trained on BirdNET embeddings for better local accuracy. Running
them requires a two-stage pipeline: extract embeddings from BirdNET, then feed
them to the BSG model.

This tool merges both stages into a single ONNX model:

```
audio [batch, 144000] @ 48kHz, 3 seconds
  -> BirdNET backbone (mel spectrogram + EfficientNet CNN)
  -> embeddings [batch, 1024]
  -> BSG classifier (Dense -> ReLU -> Dense -> Sigmoid)
  -> predictions [batch, 265]
```

## Prerequisites

- **BirdNET ONNX model** — converted by
  [birdnet-onnx-converter](https://github.com/tphakala/birdnet-onnx-converter)
  from the official TFLite model
- **BSG classifier ONNX model** — the regional classifier head from
  [BSG](https://github.com/luomus/BSG) (e.g.
  `BSG_birds_Finland_v4_4_fp32.onnx`)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python fuse_models.py \
    --birdnet /path/to/BirdNET_fp32.onnx \
    --bsg /path/to/BSG_birds_Finland_v4_4_fp32.onnx \
    --output BSG_birds_Finland_v4_4_fused_fp32.onnx
```

The script automatically verifies the fused model against the two-stage pipeline
(add `--skip-verify` to skip).

## How it works

1. Loads the BirdNET ONNX model and removes the final classification head
   (MatMul + Add for 6,522 global species)
2. Exposes the 1,024-dim embedding from the global average pooling layer
3. Rewires the BSG classifier input to consume that embedding tensor
4. Merges both graphs into a single ONNX model
5. Verifies the fused output matches the two-stage pipeline (tolerance < 1e-5)

## Compatibility

Tested with:
- BirdNET v2.4 ONNX (from `BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite`)
- BSG Finnish Birds classifier v4.4

The script validates model structure on load and will fail with clear errors if
the models don't match expected shapes.

## References

- [BSG](https://github.com/luomus/BSG) — BSG Finnish Birds Model by the Finnish Museum of Natural History (Luomus)
- [birdnet-onnx-converter](https://github.com/tphakala/birdnet-onnx-converter) — Convert and optimize BirdNET models for ONNX Runtime inference on GPUs, CPUs, and embedded devices
