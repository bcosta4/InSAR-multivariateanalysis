
"""
05_qc_aligned_layers.py (v2 - NoData fix)
─────────────────────────
Automated QC check for ALL aligned predictor GeoTIFFs after the alignment
step in the pipeline (run_nX_YYYYMMDD/aligned/).

FIX vs v1: the previous version only masked NaN for validity, which does
nothing on integer-typed rasters (int16/int32 cannot hold NaN). Categorical
layers stored as int16 with a -9999 NoData sentinel were therefore counted
as "100% valid" and their -9999 minimum triggered false-positive
NEGATIVE_WHERE_UNEXPECTED flags (cavm, dsw_trends, ground_ice, landcover,
pan_arctic_wetlands, permafrost_landcover_cls, wetland_class,
woodyplant_dominance - all 8 in the first run were false positives).

This version reads the actual NoData value from each file's own rasterio
profile (src.nodata) and excludes it from the valid-pixel mask, in addition
to excluding NaN for float rasters. This makes pct_valid, min/max, and
n_unique all computed only over genuinely valid data, regardless of dtype.

Outputs (written into the same run folder):
- aligned_layers_qc_report.csv
- aligned_layers_qc_summary.txt
"""

import re
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from collections import Counter

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"E:\Processing\InSAR_multivariate_analysis")
PIPE_OUT = PROJECT_DIR / "outputs"
RUN_DIR_OVERRIDE = None   # set to a Path(...) to force a specific run folder

LOW_VALID_THRESHOLD = 50.0
NEAR_CONSTANT_MAX_UNIQUE = 3

TEMP_NAME_HINTS = ["t2m", "stl", "cci_t", "cci_gst", "temp", "_t1m", "_t5m", "_t10m"]
CATEGORICAL_NAME_HINTS = ["landcover", "wetland_class", "cavm", "surficial_geology",
                           "ground_ice", "permafrost_landcover_cls", "dsw_trends",
                           "pan_arctic_wetlands", "woodyplant_dominance",
                           "canopy_height_lt5m"]

# Fallback sentinel values to also treat as NoData even if src.nodata is
# unset in the file's own metadata (common with some rasterize/resample
# pipelines that write -9999 as a convention without tagging it formally).
FALLBACK_SENTINELS = [-9999, -9999.0]


def parse_run_key(path):
    m = re.match(r"run_n(\d+)_(\d{8})", path.name)
    return (-1, "00000000") if not m else (int(m.group(1)), m.group(2))


def get_run_dir():
    if RUN_DIR_OVERRIDE is not None:
        return Path(RUN_DIR_OVERRIDE)
    runs = sorted([p for p in PIPE_OUT.glob("run_n*_*") if p.is_dir()], key=parse_run_key)
    if not runs:
        raise RuntimeError(f"No run folders found in {PIPE_OUT}.")
    return runs[-1]


def is_temp_like(name):
    n = name.lower()
    return any(h in n for h in TEMP_NAME_HINTS)


def is_categorical_like(name):
    n = name.lower()
    return any(h in n for h in CATEGORICAL_NAME_HINTS)


def build_valid_mask(arr, nodata_val, dtype):
    """Combines float-NaN masking with explicit NoData-value masking, so
    integer rasters with a -9999 sentinel are correctly excluded."""
    if np.issubdtype(dtype, np.floating):
        mask = ~np.isnan(arr)
    else:
        mask = np.ones_like(arr, dtype=bool)

    if nodata_val is not None:
        mask &= (arr != nodata_val)

    for sentinel in FALLBACK_SENTINELS:
        try:
            mask &= (arr != sentinel)
        except Exception:
            pass

    return mask


def check_raster(path, mode_shape=None):
    row = {"file": path.name, "status": None, "shape": None, "dtype": None,
           "nodata": None, "n_valid": None, "pct_valid": None, "min": None,
           "max": None, "n_unique": None, "flags": ""}

    if not path.exists():
        row["status"] = "MISSING"
        row["flags"] = "FILE_NOT_FOUND"
        return row

    try:
        with rasterio.open(path) as src:
            arr = src.read(1)
            nodata_val = src.nodata
    except Exception as e:
        row["status"] = "ERROR"
        row["flags"] = f"READ_ERROR: {type(e).__name__}: {e}"
        return row

    row["status"] = "OK"
    row["shape"] = arr.shape
    row["dtype"] = str(arr.dtype)
    row["nodata"] = nodata_val

    valid_mask = build_valid_mask(arr, nodata_val, arr.dtype)
    n_total = arr.size
    n_valid = int(np.sum(valid_mask))
    pct_valid = round(100 * n_valid / n_total, 1) if n_total else 0.0
    row["n_valid"] = n_valid
    row["pct_valid"] = pct_valid

    flags = []

    if n_valid == 0:
        row["flags"] = "ALL_NODATA"
        return row

    vals = arr[valid_mask]
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    n_unique = int(len(np.unique(vals)))
    row["min"] = round(vmin, 6)
    row["max"] = round(vmax, 6)
    row["n_unique"] = n_unique

    if n_unique == 1:
        flags.append("CONSTANT")
    elif n_unique <= NEAR_CONSTANT_MAX_UNIQUE:
        flags.append("NEAR_CONSTANT")

    if vmin == vmax:
        flags.append("MIN_EQ_MAX")

    if mode_shape is not None and arr.shape != mode_shape:
        flags.append(f"SHAPE_MISMATCH(got {arr.shape}, expected {mode_shape})")

    if pct_valid < LOW_VALID_THRESHOLD:
        flags.append(f"LOW_VALID_PCT({pct_valid}%)")

    stem = path.stem.replace("_aligned", "")
    if is_temp_like(stem):
        if vmin < -100 or vmax < -100 or vmax > 60:
            flags.append("SUSPECT_TEMP_KELVIN_OR_BAD_RANGE")

    if is_categorical_like(stem):
        if vmin < 0:
            flags.append("NEGATIVE_WHERE_UNEXPECTED")

    row["flags"] = "; ".join(flags) if flags else "OK"
    return row


def main():
    run_dir = get_run_dir()
    aligned_dir = run_dir / "aligned"
    print(f"Run dir (auto-detected latest): {run_dir}")
    print(f"Aligned dir: {aligned_dir}")

    if not aligned_dir.exists():
        print(f"ERROR: aligned dir not found: {aligned_dir}")
        return

    tif_files = sorted(aligned_dir.glob("*.tif")) + sorted(aligned_dir.glob("*.tiff"))
    if not tif_files:
        print(f"No .tif files found in {aligned_dir}")
        return

    print(f"Found {len(tif_files)} aligned raster(s)\n")

    shapes = []
    for p in tif_files:
        try:
            with rasterio.open(p) as src:
                shapes.append(src.read(1).shape)
        except Exception:
            shapes.append(None)
    valid_shapes = [s for s in shapes if s is not None]
    mode_shape = Counter(valid_shapes).most_common(1)[0][0] if valid_shapes else None
    print(f"Modal aligned grid shape across all layers: {mode_shape}\n")

    results = [check_raster(p, mode_shape=mode_shape) for p in tif_files]
    df = pd.DataFrame(results)

    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))

    out_csv = run_dir / "aligned_layers_qc_report.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    flagged = df[df["flags"] != "OK"]
    summary_lines = [f"Aligned Layers QC Summary — {run_dir.name}\n",
                      f"Modal shape: {mode_shape}\n",
                      f"Total files checked: {len(df)}\n",
                      f"Files with flags: {len(flagged)}\n\n"]
    if len(flagged) == 0:
        summary_lines.append("No issues detected. All aligned layers passed QC.\n")
    else:
        summary_lines.append("FLAGGED FILES:\n")
        for _, r in flagged.iterrows():
            summary_lines.append(f"  {r['file']}: {r['flags']}\n")

    out_txt = run_dir / "aligned_layers_qc_summary.txt"
    out_txt.write_text("".join(summary_lines), encoding="utf-8")
    print(f"Saved: {out_txt}")

    if len(flagged) > 0:
        print(f"\n*** {len(flagged)} of {len(df)} aligned layers flagged for review ***")
        for _, r in flagged.iterrows():
            print(f"  - {r['file']}: {r['flags']}")
    else:
        print("\nAll aligned layers passed QC with no flags.")


if __name__ == "__main__":
    main()
