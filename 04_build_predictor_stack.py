r"""
04_build_predictor_stack.py
──────────────────────────
Purpose
-------
Assembles all aligned predictor rasters from a given pipeline run into a single multi-band stack and a flattened tabular dataset restricted to valid InSAR pixels, alongside automated data-quality diagnostics.

Method
------
Each aligned raster is read and its own declared nodata sentinel (NaN, -9999, 0, etc., regardless of source dtype) is converted to NaN prior to stacking, so downstream statistics are computed only over genuinely valid data. 
Per-variable diagnostics compute the percentage of nodata pixels within the InSAR valid-pixel mask, min/max, and unique-value counts, and flag variables as constant, near-constant (a single value covering more than a fixed dominant-fraction threshold), or entirely nodata across the valid mask - a diagnostic step to catch predictors that  would otherwise silently produce zero variable importance and blank VIF scores in later modeling stages.

Inputs
------
- outputs/run_nX_YYYYMMDD/aligned/*_aligned.tif: co-registered predictor rasters from script 03.
- valid_mask.tif, subsidence_clean.tif (from 01): valid-pixel mask and target variable.

Outputs
-------
- predictor_stack.tif: multi-band GeoTIFF of all aligned predictors.
- predictor_flat.parquet: flattened table (one row per valid InSAR pixel) with pixel row/col, target variable, and all predictor values.
- band_names.json: ordered list of predictor variable names corresponding to stack band order.
- qc_variable_summary.csv: per-variable nodata percentage, min/max, unique-value count, and dead-variable diagnosis.
- dead_variables_report.json: subset of variables flagged as constant, near-constant, or entirely missing.
- predictor_histograms.png: distribution plot for every predictor over valid pixels.

Dependencies
------------
rasterio, numpy, pandas, matplotlib.

Reproducibility notes
----------------------
- Random seed: not applicable (deterministic aggregation and diagnostics).
- Deterministic given identical inputs: Yes.
"""

import re
import json
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"E:\Processing\InSAR_multivariate_analysis")
PIPE_OUT = PROJECT_DIR / "outputs"
REF_DIR = PROJECT_DIR / "scripts" / "outputs"
VALID_MASK = REF_DIR / "valid_mask.tif"
SUBSIDENCE = REF_DIR / "subsidence_clean.tif"
RUN_DIR = None   # None = auto-pick latest run_nX_YYYYMMDD
NEAR_CONSTANT_DOMINANT_FRAC = 0.999  # flag if one value covers >99.9% of valid pixels
# ──────────────────────────────────────────────────────────────────────────────

def parse_run_key(path):
    m = re.match(r"run_n(\d+)_(\d{8})", path.name)
    if not m:
        return (-1, "00000000")
    return (int(m.group(1)), m.group(2))

def get_run_dir():
    if RUN_DIR is not None:
        return Path(RUN_DIR)
    runs = [p for p in PIPE_OUT.glob("run_n*_*") if p.is_dir()]
    if not runs:
        raise RuntimeError(f"No run folders found in {PIPE_OUT}. Run 03_align_predictors.py first.")
    runs = sorted(runs, key=parse_run_key)
    return runs[-1]

def read_as_float_with_nan(path):
    """Read band 1, cast to float32, and convert the file's OWN nodata sentinel
    (whatever it is: NaN, -9999, 0, 255, etc.) into NaN. This is required
    because categorical/int-dtype aligned rasters (landcover, wetland_class,
    surficial_geology, etc.) use integer nodata sentinels like -9999 or 0,
    which are NOT automatically NaN after casting to float32."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata_val = src.nodata

    if nodata_val is None:
        return arr

    if np.isnan(nodata_val):
        return arr

    nodata_mask = np.isclose(arr, float(nodata_val), equal_nan=False)
    arr[nodata_mask] = np.nan
    return arr

run_dir = get_run_dir()
ALIGNED_DIR = run_dir / "aligned"
if not ALIGNED_DIR.exists():
    raise RuntimeError(f"Aligned folder not found: {ALIGNED_DIR}")

print("=== 04 Build Predictor Stack (v2, dead-variable check) ===")
print(f"Run dir      : {run_dir}")
print(f"Aligned dir  : {ALIGNED_DIR}")

aligned_files = sorted(ALIGNED_DIR.glob("*_aligned.tif"))
if not aligned_files:
    raise RuntimeError(f"No aligned GeoTIFFs found in {ALIGNED_DIR}. Run script 03 first.")

band_names = [f.stem.replace("_aligned", "") for f in aligned_files]
print(f"Found {len(aligned_files)} aligned layers.")

with rasterio.open(VALID_MASK) as src:
    mask = src.read(1).astype(bool)

with rasterio.open(SUBSIDENCE) as src:
    sub = src.read(1).astype(np.float32)
    meta = src.meta.copy()

mask &= ~np.isnan(sub)
n_valid = int(mask.sum())
print(f"Valid InSAR pixels: {n_valid:,}")

stack_meta = meta.copy()
stack_meta.update(count=len(aligned_files), dtype="float32", compress="lzw", nodata=np.nan)
stack_path = run_dir / "predictor_stack.tif"

arrays = []
qc_rows = []
dead_vars = []

with rasterio.open(stack_path, "w", **stack_meta) as dst:
    for i, (f, name) in enumerate(zip(aligned_files, band_names), 1):
        arr = read_as_float_with_nan(f)

        vals = arr[mask]
        nan_frac = float(np.isnan(vals).mean() * 100) if vals.size else np.nan
        valid_vals = vals[~np.isnan(vals)]
        vmin = float(np.nanmin(valid_vals)) if valid_vals.size else np.nan
        vmax = float(np.nanmax(valid_vals)) if valid_vals.size else np.nan

        # Dead-variable diagnostic (merged from 04b)
        n_unique = int(np.unique(valid_vals).size) if valid_vals.size else 0
        is_all_nan = valid_vals.size == 0
        is_constant = (not is_all_nan) and (n_unique == 1)
        is_near_constant = False
        dominant_frac = np.nan
        if not is_all_nan and not is_constant and n_unique <= 5:
            uniq, counts = np.unique(valid_vals, return_counts=True)
            dominant_frac = float(counts.max() / valid_vals.size)
            is_near_constant = dominant_frac >= NEAR_CONSTANT_DOMINANT_FRAC

        diagnosis = None
        if is_all_nan:
            diagnosis = "ALL_NAN"
        elif is_constant:
            diagnosis = f"CONSTANT (value={valid_vals[0]:.4g})"
        elif is_near_constant:
            diagnosis = f"NEAR_CONSTANT ({dominant_frac*100:.2f}% one value)"

        qc_rows.append({
            "variable": name, "nan_pct": nan_frac, "min": vmin, "max": vmax,
            "n_unique": n_unique, "is_dead": diagnosis is not None,
            "diagnosis": diagnosis or "",
        })

        if diagnosis is not None:
            dead_vars.append({"variable": name, "diagnosis": diagnosis, "nan_pct": nan_frac})
            print(f"Band {i:02d}: {name:<28} [WARNING: {diagnosis} - will produce blank VIF/importance]")
        else:
            print(f"Band {i:02d}: {name:<28} NaN={nan_frac:.1f}% [{vmin:.3f}, {vmax:.3f}] n_unique={n_unique}")

        dst.write(arr, i)
        arrays.append(arr)

print(f"\nStack saved: {stack_path}")

if dead_vars:
    print(f"\n[!] {len(dead_vars)} variable(s) flagged as dead/near-constant:")
    for d in dead_vars:
        print(f"    - {d['variable']}: {d['diagnosis']}")
    print("    These will produce blank VIF scores and zero importance downstream.")
    print("    Review dead_variables_report.json and consider removing them from")
    print("    pruned_feature_list.json before running 05_multicollinearity_vif.py.")
else:
    print("\n[OK] No dead/near-constant variables detected.")

with open(run_dir / "dead_variables_report.json", "w") as f:
    json.dump(dead_vars, f, indent=2)

with open(run_dir / "band_names.json", "w") as f:
    json.dump(band_names, f, indent=2)

rows, cols = np.where(mask)
df = pd.DataFrame({
    "row": rows,
    "col": cols,
    "subsidence_mm_yr": sub[mask]
})
for name, arr in zip(band_names, arrays):
    df[name] = arr[mask]

parquet_path = run_dir / "predictor_flat.parquet"
df.to_parquet(parquet_path, index=False)
print(f"Flat parquet: {parquet_path} ({len(df):,} rows x {len(df.columns)} cols)")

qc_df = pd.DataFrame(qc_rows)
qc_csv = run_dir / "qc_variable_summary.csv"
qc_df.to_csv(qc_csv, index=False)
print(f"QC CSV saved: {qc_csv}")

n = len(band_names)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
axes = np.atleast_1d(axes).flatten()

for i, (name, arr) in enumerate(zip(band_names, arrays)):
    valid_vals = arr[mask & ~np.isnan(arr)]
    if len(valid_vals) > 0:
        axes[i].hist(valid_vals, bins=60, color="#2196F3", alpha=0.8, edgecolor="none")
        axes[i].set_title(name, fontsize=8)
        axes[i].set_xlabel("Value", fontsize=7)
        axes[i].set_ylabel("Count", fontsize=7)
        axes[i].tick_params(labelsize=7)
    else:
        axes[i].set_title(f"{name}\n(no valid data)", fontsize=8)
        axes[i].set_axis_off()

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Predictor Histograms (valid InSAR pixels)", fontsize=10, y=1.01)
plt.tight_layout()
hist_path = run_dir / "predictor_histograms.png"
plt.savefig(hist_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Histogram PNG saved: {hist_path}")
print("Done.")