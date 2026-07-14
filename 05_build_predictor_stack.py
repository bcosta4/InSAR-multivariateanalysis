"""
05_build_predictor_stack.py
────────────────────────────
Stacks all aligned GeoTIFFs (from ./outputs/aligned/) into:
  - predictor_stack.tif    : multi-band GeoTIFF (1 band = 1 variable)
  - predictor_flat.parquet : flattened table of valid pixels only
  - band_names.json        : ordered list of variable names

Also exports QC: per-variable NaN count, value range, histogram PNG.
"""

import json, numpy as np, pandas as pd, rasterio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ALIGNED_DIR = Path(__file__).parent / "outputs" / "aligned"
VALID_MASK  = Path(__file__).parent / "outputs" / "valid_mask.tif"
SUBSIDENCE  = Path(__file__).parent / "outputs" / "subsidence_clean.tif"
OUT_DIR     = Path(__file__).parent / "outputs"
# ──────────────────────────────────────────────────────────────────────────────

print("=== 05 Build Predictor Stack ===")

# Discover all aligned layers
aligned_files = sorted(ALIGNED_DIR.glob("*_aligned.tif"))
if not aligned_files:
    raise RuntimeError(f"No aligned GeoTIFFs found in {ALIGNED_DIR}. Run script 03 first.")

band_names = [f.stem.replace("_aligned", "") for f in aligned_files]
print(f"  Found {len(aligned_files)} aligned layers: {band_names}")

# Load valid mask and subsidence
with rasterio.open(VALID_MASK) as src:
    mask = src.read(1).astype(bool)

with rasterio.open(SUBSIDENCE) as src:
    sub = src.read(1).astype(np.float32)
    meta = src.meta.copy()

mask &= ~np.isnan(sub)
n_valid = mask.sum()
print(f"  Valid pixels (InSAR mask): {n_valid:,}")

# ── Stack bands ───────────────────────────────────────────────────────────────
stack_meta = meta.copy()
stack_meta.update(count=len(aligned_files), dtype="float32", compress="lzw")
stack_path = OUT_DIR / "predictor_stack.tif"

arrays = []
qc_rows = []

with rasterio.open(stack_path, "w", **stack_meta) as dst:
    for i, (f, name) in enumerate(zip(aligned_files, band_names), 1):
        with rasterio.open(f) as src:
            arr = src.read(1).astype(np.float32)

        nan_frac = np.isnan(arr[mask]).mean() * 100
        vmin = np.nanmin(arr[mask]) if (~np.isnan(arr[mask])).any() else np.nan
        vmax = np.nanmax(arr[mask]) if (~np.isnan(arr[mask])).any() else np.nan
        qc_rows.append({"variable": name, "nan_pct": nan_frac,
                         "min": vmin, "max": vmax})
        dst.write(arr, i)
        arrays.append(arr)
        print(f"  Band {i:02d}: {name:<28} NaN={nan_frac:.1f}%  [{vmin:.3f}, {vmax:.3f}]")

print(f"\n  Stack saved: {stack_path}")

# Save band names
with open(OUT_DIR / "band_names.json", "w") as f:
    json.dump(band_names, f, indent=2)

# ── Flat parquet ──────────────────────────────────────────────────────────────
rows, cols = np.where(mask)
df = pd.DataFrame({"row": rows, "col": cols, "subsidence_mm_yr": sub[mask]})
for name, arr in zip(band_names, arrays):
    df[name] = arr[mask]

parquet_path = OUT_DIR / "predictor_flat.parquet"
df.to_parquet(parquet_path, index=False)
print(f"  Flat parquet : {parquet_path}  ({len(df):,} rows × {len(df.columns)} cols)")

# ── QC histograms ─────────────────────────────────────────────────────────────
qc_df = pd.DataFrame(qc_rows)
qc_df.to_csv(OUT_DIR / "qc_variable_summary.csv", index=False)
print(f"  QC CSV saved.")

n = len(band_names)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
axes = axes.flatten()

for i, (name, arr) in enumerate(zip(band_names, arrays)):
    valid_vals = arr[mask & ~np.isnan(arr)]
    if len(valid_vals) > 0:
        axes[i].hist(valid_vals, bins=60, color="#2196F3", alpha=0.8, edgecolor="none")
    axes[i].set_title(name, fontsize=8)
    axes[i].set_xlabel("Value", fontsize=7)
    axes[i].set_ylabel("Count", fontsize=7)
    axes[i].tick_params(labelsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Predictor Histograms (valid InSAR pixels)", fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "predictor_histograms.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Histogram PNG saved.")
print("  Done. → Next: run 06_multicollinearity_vif.py")
