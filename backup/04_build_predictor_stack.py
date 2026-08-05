"""
04_build_predictor_stack.py
─────────────────────────────
Stacks all aligned predictor GeoTIFFs + subsidence into:
  1. predictor_stack.tif        — multi-band GeoTIFF (all variables)
  2. predictor_stack_valid.parquet  — flat table of valid pixels only
     (1 row per pixel, columns: x_coord, y_coord, pixel_row, pixel_col,
      subsidence, <all predictors>)

Also runs quality checks:
  - NaN counts per variable
  - Value ranges
  - High-correlation pairs (|r| > 0.8)
  - Subsidence distribution histogram
  - Aspect sin/cos sanity check
"""

import os
import sys
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── CONFIG ───────────────────────────────────────────────────────────────────
REFERENCE_PATH  = r"E:\Processing\InSAR_validation\data\aligned\subsidence_masked.tif"
VALID_MASK_PATH = r"E:\Processing\InSAR_validation\data\aligned\valid_mask.tif"
ALIGNED_DIR     = r"E:\Processing\InSAR_validation\data\aligned"
OUTPUT_DIR      = r"E:\Processing\InSAR_validation\data\stack"
CORR_THRESHOLD  = 0.80   # flag pairs above this absolute correlation

# Variables to include in the stack. Edit if some are still missing.
# Key = column name in parquet; Value = filename in ALIGNED_DIR
VARIABLE_FILES = {
    "subsidence"    : "subsidence_masked.tif",
    "alt"           : "alt_mean.tif",
    "magt"          : "magt.tif",
    "ddt"           : "ddt.tif",
    "ground_ice"    : "ground_ice.tif",
    "slope"         : "slope.tif",
    "aspect_sin"    : "aspect_sin.tif",
    "aspect_cos"    : "aspect_cos.tif",
    "curvature"     : "curvature.tif",
    "twi"           : "twi.tif",
    "tpi"           : "tpi.tif",
    "tri"           : "tri.tif",
    "ndvi"          : "ndvi_mean.tif",
    "flood_freq"    : "flood_freq.tif",
    "landcover"     : "landcover.tif",
    "geology"       : "geology.tif",
    "wetlands"      : "wetlands.tif",
}
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("04  BUILD PREDICTOR STACK")
print("=" * 60)

# ── 1. Load valid mask ────────────────────────────────────────────────────────
with rasterio.open(VALID_MASK_PATH) as src:
    valid_mask = src.read(1).astype(bool)
print(f"\nValid pixels: {valid_mask.sum():,}")

# ── 2. Load reference metadata ───────────────────────────────────────────────
with rasterio.open(REFERENCE_PATH) as ref:
    ref_meta      = ref.meta.copy()
    ref_transform = ref.transform
    ref_shape     = (ref.height, ref.width)

# ── 3. Load all variables ─────────────────────────────────────────────────────
arrays = {}
missing_files = []

for name, fname in VARIABLE_FILES.items():
    fpath = os.path.join(ALIGNED_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  [MISSING] {name:20s} — {fpath}")
        missing_files.append(name)
        continue
    with rasterio.open(fpath) as src:
        arr = src.read(1).astype(np.float32)
    arr[~valid_mask] = np.nan
    arrays[name] = arr
    pct_nan = np.isnan(arr[valid_mask]).mean() * 100
    print(f"  [OK] {name:20s}  NaN within mask: {pct_nan:.1f}%")

available = list(arrays.keys())
print(f"\nLoaded {len(available)} variables  ({len(missing_files)} missing: {missing_files})")

# ── 4. Write multi-band GeoTIFF ───────────────────────────────────────────────
stack_path = os.path.join(OUTPUT_DIR, "predictor_stack.tif")
stack_meta = ref_meta.copy()
stack_meta.update(count=len(available), dtype="float32",
                  nodata=float("nan"), compress="lzw")

with rasterio.open(stack_path, "w", **stack_meta) as dst:
    for i, name in enumerate(available, start=1):
        dst.write(arrays[name], i)
    dst.update_tags(BAND_NAMES=",".join(available))

print(f"\nSaved multi-band stack: {stack_path}")
print(f"  Bands: {available}")

# ── 5. Build flat pixel table ─────────────────────────────────────────────────
print("\nBuilding flat pixel table ...")
rows_idx, cols_idx = np.where(valid_mask)
xs, ys = rasterio.transform.xy(ref_transform, rows_idx, cols_idx)

data = {
    "pixel_row" : rows_idx.astype(np.int32),
    "pixel_col" : cols_idx.astype(np.int32),
    "x_coord"   : np.array(xs, dtype=np.float64),
    "y_coord"   : np.array(ys, dtype=np.float64),
}
for name in available:
    data[name] = arrays[name][valid_mask]

df = pd.DataFrame(data)
df = df.dropna(subset=["subsidence"])
print(f"  Rows (valid pixels): {len(df):,}")

parquet_path = os.path.join(OUTPUT_DIR, "predictor_stack_valid.parquet")
df.to_parquet(parquet_path, index=False)
print(f"Saved flat table: {parquet_path}")

# ── 6. Quality checks ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("QUALITY CHECKS")
print("=" * 60)

# 6a. Value ranges
print("\nVariable statistics (valid pixels only):")
numeric_cols = [c for c in available if c not in ["landcover", "geology", "wetlands"]]
stats = df[numeric_cols].describe().T[["mean", "std", "min", "max"]]
print(stats.to_string())

# 6b. NaN counts
print("\nNaN counts within valid mask:")
nan_counts = df[available].isna().sum().sort_values(ascending=False)
print(nan_counts[nan_counts > 0].to_string() if nan_counts.any() else "  None.")

# 6c. High correlation pairs
print(f"\nCorrelation pairs |r| > {CORR_THRESHOLD}:")
cont_cols = [c for c in numeric_cols if c != "subsidence"]
if len(cont_cols) >= 2:
    corr_matrix = df[cont_cols].corr(method="spearman")
    pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    pairs.columns = ["var1", "var2", "r"]
    high = pairs[pairs["r"].abs() > CORR_THRESHOLD].sort_values("r", ascending=False)
    if len(high):
        print(high.to_string(index=False))
        print("\n  ⚠ Review these pairs in 05_multicollinearity.py before modelling.")
    else:
        print("  No highly correlated pairs above threshold.")

# 6d. Subsidence distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["subsidence"].dropna(), bins=200, color="#01696f", edgecolor="none", alpha=0.85)
axes[0].set_xlabel("Subsidence rate (cm/yr)")
axes[0].set_ylabel("Pixel count")
axes[0].set_title("Subsidence Rate Distribution")
axes[0].axvline(df["subsidence"].mean(), color="tomato", lw=1.5, label=f"Mean: {df['subsidence'].mean():.3f}")
axes[0].legend()

axes[1].hist(df["subsidence"].dropna(), bins=200, color="#01696f", edgecolor="none", alpha=0.85)
axes[1].set_yscale("log")
axes[1].set_xlabel("Subsidence rate (cm/yr)")
axes[1].set_ylabel("Pixel count (log scale)")
axes[1].set_title("Subsidence Rate Distribution (log Y)")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "subsidence_distribution.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"\nSaved distribution plot: {plot_path}")

print("\n✓ 04 complete. Proceed to 05_multicollinearity_vif.py")
