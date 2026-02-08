#!/usr/bin/env python3
"""
Convert BSG prediction adjustment data to formats used by rust-birdnet-onnx.

Converts:
  1. calibration_params.npy  -> BSG_calibration.csv
  2. migration_params.npy    -> BSG_migration.csv
  3. distribution_maps/*.tif -> BSG_distribution_maps.bin

Source files are from the BSG project:
  https://github.com/luomus/BSG
  → scripts/Pred_adjustment/

Usage:
    python convert_bsg_data.py --bsg-data /path/to/Pred_adjustment --output-dir ./output

Requirements:
    numpy
    rasterio (for reading GeoTIFF distribution maps)
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
from pathlib import Path

import numpy as np


# ── Calibration ──────────────────────────────────────────────────────────────

def convert_calibration(npy_path: Path, output_path: Path) -> None:
    """Convert calibration_params.npy (265, 2) to CSV with header."""
    data = np.load(npy_path)
    assert data.shape[1] == 2, f"Expected 2 columns, got {data.shape[1]}"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["intercept", "slope"])
        for row in data:
            writer.writerow([f"{row[0]:.10g}", f"{row[1]:.10g}"])

    print(f"Calibration: {data.shape[0]} species -> {output_path}")


# ── Migration ────────────────────────────────────────────────────────────────

MIGRATION_COLUMNS = [
    "arrival_intercept",
    "arrival_slope",
    "departure_intercept",
    "departure_slope",
    "arrival_std",
    "departure_std",
    "seasonal_start",
    "seasonal_end",
]


def convert_migration(npy_path: Path, output_path: Path) -> None:
    """Convert migration_params.npy (265, 8) to CSV with header."""
    data = np.load(npy_path)
    assert data.shape[1] == 8, f"Expected 8 columns, got {data.shape[1]}"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MIGRATION_COLUMNS)
        for row in data:
            writer.writerow([f"{v:.10g}" for v in row])

    print(f"Migration: {data.shape[0]} species -> {output_path}")


# ── Distribution Maps ────────────────────────────────────────────────────────

BSDM_MAGIC = b"BSDM"
BSDM_VERSION = 1
# Species indices in the BSG model: 0-1 are meta-classes (no maps),
# 2-264 are bird species with distribution maps.
SPECIES_INDEX_START = 2
SPECIES_INDEX_END = 264  # inclusive


def convert_distribution_maps(maps_dir: Path, output_path: Path) -> None:
    """Pack distribution map GeoTIFFs into a single binary file.

    Reads {index}_a.tif and {index}_b.tif for species indices 2-264 (263 species).
    All TIFFs must share the same dimensions and geo transform.

    Binary format:
        Header (64 bytes):
            magic:           4 bytes  "BSDM"
            version:         u32 LE   (1)
            num_species:     u32 LE   (263)
            width:           u32 LE   (93)
            height:          u32 LE   (132)
            origin_lon:      f64 LE   (west edge)
            origin_lat:      f64 LE   (north edge)
            pixel_scale_lon: f64 LE   (positive, degrees per pixel east)
            pixel_scale_lat: f64 LE   (positive, degrees per pixel south)
            _reserved:       12 bytes (zeros, pad to 64)

        Data (per species, ordered by class index 2-264):
            map_a: height * width * f32 LE  (NaN replaced with 1.0)
            map_b: height * width * f32 LE  (NaN replaced with 1.0)
    """
    import rasterio

    num_species = SPECIES_INDEX_END - SPECIES_INDEX_START + 1
    assert num_species == 263

    # Verify all expected files exist
    missing = []
    for idx in range(SPECIES_INDEX_START, SPECIES_INDEX_END + 1):
        for suffix in ("a", "b"):
            path = maps_dir / f"{idx}_{suffix}.tif"
            if not path.exists():
                missing.append(path.name)
    if missing:
        print(f"Error: missing {len(missing)} TIFFs: {missing[:5]}...", file=sys.stderr)
        sys.exit(1)

    # Read geo parameters from the first file
    ref_path = maps_dir / f"{SPECIES_INDEX_START}_a.tif"
    with rasterio.open(ref_path) as ds:
        height, width = ds.shape
        t = ds.transform
        origin_lon = t.c      # west edge
        origin_lat = t.f      # north edge
        pixel_scale_lon = t.a  # positive: degrees east per pixel
        pixel_scale_lat = abs(t.e)  # make positive: degrees south per pixel

    print(f"Distribution maps: {num_species} species, {width}x{height} pixels")
    print(f"  Origin: ({origin_lon:.6f}, {origin_lat:.6f})")
    print(f"  Scale: ({pixel_scale_lon:.10f}, {pixel_scale_lat:.10f})")

    # Write binary file
    with open(output_path, "wb") as f:
        # Header (64 bytes)
        f.write(BSDM_MAGIC)
        f.write(struct.pack("<I", BSDM_VERSION))
        f.write(struct.pack("<I", num_species))
        f.write(struct.pack("<I", width))
        f.write(struct.pack("<I", height))
        f.write(struct.pack("<d", origin_lon))
        f.write(struct.pack("<d", origin_lat))
        f.write(struct.pack("<d", pixel_scale_lon))
        f.write(struct.pack("<d", pixel_scale_lat))
        f.write(b"\x00" * 12)  # reserved (pad to 64 bytes)

        header_size = f.tell()
        assert header_size == 64, f"Header is {header_size} bytes, expected 64"

        nan_total = 0
        for idx in range(SPECIES_INDEX_START, SPECIES_INDEX_END + 1):
            for suffix in ("a", "b"):
                tif_path = maps_dir / f"{idx}_{suffix}.tif"
                with rasterio.open(tif_path) as ds:
                    assert ds.shape == (height, width), (
                        f"{tif_path.name}: shape {ds.shape} != expected ({height}, {width})"
                    )
                    data = ds.read(1).astype(np.float32)

                    # Replace NaN with 1.0 (assume species possible where no data)
                    nan_count = np.isnan(data).sum()
                    nan_total += nan_count
                    np.nan_to_num(data, nan=1.0, copy=False)

                    f.write(data.tobytes())

    file_size = output_path.stat().st_size
    expected_size = 64 + num_species * 2 * height * width * 4
    assert file_size == expected_size, (
        f"File size {file_size} != expected {expected_size}"
    )

    print(f"  NaN values replaced with 1.0: {nan_total}")
    print(f"  Output: {output_path} ({file_size / 1e6:.1f} MB)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BSG prediction adjustment data for rust-birdnet-onnx"
    )
    parser.add_argument(
        "--bsg-data", required=True, type=Path,
        help="Path to BSG Pred_adjustment directory containing "
             "calibration_params.npy, migration_params.npy, and distribution_maps/",
    )
    parser.add_argument(
        "--output-dir", default=Path("."), type=Path,
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--skip-maps", action="store_true",
        help="Skip distribution maps conversion (requires rasterio)",
    )
    args = parser.parse_args()

    bsg_data: Path = args.bsg_data
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calibration
    cal_npy = bsg_data / "calibration_params.npy"
    if not cal_npy.exists():
        print(f"Error: {cal_npy} not found", file=sys.stderr)
        sys.exit(1)
    convert_calibration(cal_npy, output_dir / "BSG_calibration.csv")

    # Migration
    mig_npy = bsg_data / "migration_params.npy"
    if not mig_npy.exists():
        print(f"Error: {mig_npy} not found", file=sys.stderr)
        sys.exit(1)
    convert_migration(mig_npy, output_dir / "BSG_migration.csv")

    # Distribution maps
    if args.skip_maps:
        print("Skipping distribution maps (--skip-maps)")
    else:
        maps_dir = bsg_data / "distribution_maps"
        if not maps_dir.is_dir():
            print(f"Error: {maps_dir} not found", file=sys.stderr)
            sys.exit(1)
        convert_distribution_maps(maps_dir, output_dir / "BSG_distribution_maps.bin")

    print("\nDone.")


if __name__ == "__main__":
    main()
