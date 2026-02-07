#!/usr/bin/env python3
"""
Fuse BirdNET v2.4 feature extractor with BSG Finnish Birds classifier.

Creates an end-to-end ONNX model: audio (144000 samples) -> predictions (265 classes)

The BirdNET ONNX model (converted from TFLite by birdnet-onnx-converter) provides
the feature extraction backbone. Its final classification head (MatMul + Add for
6522 global species) is removed, exposing the 1024-dim embedding from the global
average pooling layer. The BSG classifier head (trained on 263 Finnish bird species)
is then attached to produce 265-class predictions (263 species + 2 meta-classes).

Architecture:
    audio [batch, 144000] @ 48kHz
      -> BirdNET mel spectrogram + EfficientNet backbone
      -> embeddings [batch, 1024]
      -> BSG classifier (Dense -> ReLU -> Dense -> Sigmoid)
      -> predictions [batch, 265]

Usage:
    python fuse_models.py --birdnet BirdNET_fp32.onnx --bsg BSG_fp32.onnx

Requirements:
    onnx>=1.14
    onnxruntime>=1.16
    numpy
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


# BirdNET classification head nodes to remove (MatMul + BiasAdd for 6522 classes).
# These are the final two nodes in the BirdNET ONNX graph produced by
# birdnet-onnx-converter from BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite.
BIRDNET_HEAD_NODE_NAMES = {
    "Identity",   # MatMul: embedding [1024] -> logits [6522]
    "Add__100",   # Add: bias for 6522 logits
}

# Weight/bias initializers belonging to the classification head.
BIRDNET_HEAD_INITIALIZER_NAMES = {
    "const_fold_opt__1616",                                      # MatMul weights [1024, 6522]
    "model/CLASS_DENSE_LAYER/BiasAdd/ReadVariableOp/resource",   # Bias [6522]
}

# The tensor connecting BirdNET backbone to BSG classifier —
# output of the ReduceMean (global average pooling) node.
EMBEDDING_TENSOR_NAME = "model/GLOBAL_AVG_POOL/Mean_Squeeze__1647:0"


def load_and_validate_birdnet(path: str | Path) -> onnx.ModelProto:
    """Load BirdNET ONNX model and validate its structure."""
    model = onnx.load(str(path))

    assert len(model.graph.input) == 1, f"Expected 1 input, got {len(model.graph.input)}"
    input_shape = [d.dim_value or d.dim_param for d in model.graph.input[0].type.tensor_type.shape.dim]
    assert input_shape[1] == 144000, f"Expected input dim 144000, got {input_shape[1]}"

    assert len(model.graph.output) == 1, f"Expected 1 output, got {len(model.graph.output)}"
    output_shape = [d.dim_value or d.dim_param for d in model.graph.output[0].type.tensor_type.shape.dim]
    assert output_shape[1] == 6522, f"Expected output dim 6522, got {output_shape[1]}"

    # Verify the embedding tensor exists
    all_outputs = {out for node in model.graph.node for out in node.output}
    assert EMBEDDING_TENSOR_NAME in all_outputs, (
        f"Embedding tensor '{EMBEDDING_TENSOR_NAME}' not found in model"
    )

    # Verify the head nodes exist
    node_names = {node.name for node in model.graph.node}
    for name in BIRDNET_HEAD_NODE_NAMES:
        assert name in node_names, f"Expected head node '{name}' not found"

    print(f"BirdNET model validated: input {input_shape}, output {output_shape}")
    print(f"  Nodes: {len(model.graph.node)}, Initializers: {len(model.graph.initializer)}")
    return model


def load_and_validate_bsg(path: str | Path) -> onnx.ModelProto:
    """Load BSG classifier ONNX model and validate its structure."""
    model = onnx.load(str(path))

    assert len(model.graph.input) == 1, f"Expected 1 input, got {len(model.graph.input)}"
    input_shape = [d.dim_value or d.dim_param for d in model.graph.input[0].type.tensor_type.shape.dim]
    assert input_shape[1] == 1024, f"Expected input dim 1024, got {input_shape[1]}"

    assert len(model.graph.output) == 1, f"Expected 1 output, got {len(model.graph.output)}"
    output_shape = [d.dim_value or d.dim_param for d in model.graph.output[0].type.tensor_type.shape.dim]
    assert output_shape[1] == 265, f"Expected output dim 265, got {output_shape[1]}"

    print(f"BSG model validated: input {input_shape}, output {output_shape}")
    print(f"  Nodes: {len(model.graph.node)}, Initializers: {len(model.graph.initializer)}")
    return model


def extract_birdnet_backbone(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove BirdNET's classification head, keeping only the feature extractor.

    Removes the final MatMul + Add nodes that map 1024-dim embeddings to
    6522-class logits, and their associated weight/bias initializers.
    """
    model = copy.deepcopy(model)

    # Remove classification head nodes
    nodes_to_keep = [n for n in model.graph.node if n.name not in BIRDNET_HEAD_NODE_NAMES]
    removed_count = len(model.graph.node) - len(nodes_to_keep)
    assert removed_count == len(BIRDNET_HEAD_NODE_NAMES), (
        f"Expected to remove {len(BIRDNET_HEAD_NODE_NAMES)} nodes, removed {removed_count}"
    )
    del model.graph.node[:]
    model.graph.node.extend(nodes_to_keep)

    # Remove classification head initializers
    inits_to_keep = [i for i in model.graph.initializer if i.name not in BIRDNET_HEAD_INITIALIZER_NAMES]
    removed_init_count = len(model.graph.initializer) - len(inits_to_keep)
    del model.graph.initializer[:]
    model.graph.initializer.extend(inits_to_keep)

    # Replace the output with the embedding tensor
    del model.graph.output[:]
    embedding_output = helper.make_tensor_value_info(
        EMBEDDING_TENSOR_NAME, TensorProto.FLOAT, ["batch", 1024]
    )
    model.graph.output.append(embedding_output)

    print(f"BirdNET backbone extracted:")
    print(f"  Removed {removed_count} head nodes, {removed_init_count} initializers")
    print(f"  Output: {EMBEDDING_TENSOR_NAME} [batch, 1024]")
    return model


def rewire_bsg_inputs(model: onnx.ModelProto, embedding_tensor: str) -> onnx.ModelProto:
    """Rewire BSG model to accept the BirdNET embedding tensor as input.

    Replaces references to the BSG 'input' tensor with the BirdNET embedding
    tensor name throughout the graph.
    """
    model = copy.deepcopy(model)
    bsg_input_name = model.graph.input[0].name

    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == bsg_input_name:
                node.input[i] = embedding_tensor

    return model


def fuse_models(
    birdnet: onnx.ModelProto,
    bsg: onnx.ModelProto,
) -> onnx.ModelProto:
    """Fuse BirdNET feature extractor with BSG classifier into a single model.

    Pipeline: audio [batch, 144000] -> BirdNET backbone -> embeddings [batch, 1024]
              -> BSG classifier -> predictions [batch, 265]
    """
    backbone = extract_birdnet_backbone(birdnet)
    bsg_rewired = rewire_bsg_inputs(bsg, EMBEDDING_TENSOR_NAME)

    fused_nodes = list(backbone.graph.node) + list(bsg_rewired.graph.node)
    fused_initializers = list(backbone.graph.initializer) + list(bsg_rewired.graph.initializer)

    fused_input = backbone.graph.input[0]
    fused_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 265])

    fused_graph = helper.make_graph(
        fused_nodes,
        "birdnet_bsg_fused",
        [fused_input],
        [fused_output],
        initializer=fused_initializers,
    )

    opset = backbone.opset_import[0]
    fused_model = helper.make_model(fused_graph, opset_imports=[opset])
    fused_model.ir_version = backbone.ir_version

    print(f"\nFused model created:")
    print(f"  Nodes: {len(fused_nodes)} (backbone: {len(backbone.graph.node)}, BSG: {len(bsg_rewired.graph.node)})")
    print(f"  Initializers: {len(fused_initializers)}")
    print(f"  Input: {fused_input.name} [batch, 144000]")
    print(f"  Output: output [batch, 265]")

    return fused_model


def verify_fused_model(
    fused_path: str | Path,
    birdnet_path: str | Path,
    bsg_path: str | Path,
) -> None:
    """Verify the fused model produces identical results to the two-stage pipeline."""
    print("\n=== Verification ===")

    rng = np.random.default_rng(42)
    audio = (rng.standard_normal((1, 144000)) * 0.1).astype(np.float32)

    # Two-stage pipeline (ground truth)
    birdnet_model = onnx.load(str(birdnet_path))
    embedding_out = helper.make_tensor_value_info(EMBEDDING_TENSOR_NAME, TensorProto.FLOAT, None)
    birdnet_model.graph.output.append(embedding_out)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = f.name
        onnx.save(birdnet_model, tmp_path)

    birdnet_session = ort.InferenceSession(tmp_path)
    birdnet_results = birdnet_session.run(None, {"input": audio})
    embedding = birdnet_results[1]
    print(f"Two-stage embedding shape: {embedding.shape}")

    Path(tmp_path).unlink()

    bsg_session = ort.InferenceSession(str(bsg_path))
    bsg_input_name = bsg_session.get_inputs()[0].name
    bsg_result = bsg_session.run(None, {bsg_input_name: embedding})
    two_stage_output = bsg_result[0]
    print(f"Two-stage output shape: {two_stage_output.shape}")

    # Fused model
    fused_session = ort.InferenceSession(str(fused_path))
    fused_result = fused_session.run(None, {"input": audio})
    fused_output = fused_result[0]
    print(f"Fused output shape: {fused_output.shape}")

    # Compare
    assert two_stage_output.shape == fused_output.shape, (
        f"Shape mismatch: {two_stage_output.shape} vs {fused_output.shape}"
    )

    max_diff = np.abs(two_stage_output - fused_output).max()
    mean_diff = np.abs(two_stage_output - fused_output).mean()
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")

    assert max_diff < 1e-5, f"Results differ too much: max_diff={max_diff}"
    print("PASS: Fused model output matches two-stage pipeline")

    assert fused_output.shape == (1, 265), f"Expected (1, 265), got {fused_output.shape}"
    print("PASS: Output shape is (1, 265)")

    assert np.all(fused_output >= 0) and np.all(fused_output <= 1), (
        f"Output values outside [0, 1]: min={fused_output.min()}, max={fused_output.max()}"
    )
    print(f"PASS: Output values in [0, 1] (min={fused_output.min():.6f}, max={fused_output.max():.6f})")
    print(f"  Output sum: {fused_output.sum():.4f} (sigmoid outputs, not softmax)")

    # Batch test
    audio_batch = rng.standard_normal((3, 144000)).astype(np.float32) * 0.1
    batch_result = fused_session.run(None, {"input": audio_batch})
    assert batch_result[0].shape == (3, 265), f"Batch test failed: {batch_result[0].shape}"
    print("PASS: Dynamic batch size works (tested batch=3)")

    fused_size = Path(fused_path).stat().st_size / 1e6
    print(f"\nFused model size: {fused_size:.1f} MB")
    print("\nAll verification checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse BirdNET v2.4 feature extractor with BSG Finnish Birds classifier"
    )
    parser.add_argument(
        "--birdnet", required=True,
        help="Path to BirdNET ONNX model (from birdnet-onnx-converter)",
    )
    parser.add_argument(
        "--bsg", required=True,
        help="Path to BSG classifier ONNX model (fp32)",
    )
    parser.add_argument(
        "--output", default="BSG_birds_Finland_v4_4_fused_fp32.onnx",
        help="Output path for fused model (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip verification step",
    )

    args = parser.parse_args()

    for name, path in [("BirdNET", args.birdnet), ("BSG", args.bsg)]:
        if not Path(path).exists():
            print(f"Error: {name} model not found at {path}", file=sys.stderr)
            sys.exit(1)

    print("Loading models...")
    birdnet = load_and_validate_birdnet(args.birdnet)
    bsg = load_and_validate_bsg(args.bsg)

    print("\nFusing models...")
    fused = fuse_models(birdnet, bsg)

    print(f"\nSaving fused model to {args.output}...")
    onnx.save(fused, args.output)
    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"Saved ({size_mb:.1f} MB)")

    if not args.skip_verify:
        verify_fused_model(args.output, args.birdnet, args.bsg)


if __name__ == "__main__":
    main()
