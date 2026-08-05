"""
01_prepare_reference_grid.py
────────────────────────────
Purpose
-------
Establishes the canonical reference grid that every subsequent script in the pipeline aligns to, derived from the DS-InSAR subsidence velocity map.

Method
------
Reads the InSAR velocity raster, identifies nodata pixels by matching the source sentinel value (with tolerance) and an additional out-of-range check, and replaces them with NaN. Produces a binary valid-pixel mask (1 = valid InSAR measurement) and exports the grid's CRS, affine transform, shape, and resolution as metadata for reuse by all downstream alignment steps.

Inputs
------
- Network2_ifg_unw_Stack_TC1_velocity_map_masked_slope5deg.tif: DS-InSAR velocity map (source CRS, resolution, and extent define the reference grid).

Outputs
-------
- reference_grid_meta.json: CRS (WKT + EPSG), affine transform, shape, resolution, and valid-pixel count.
- valid_mask.tif: binary raster, 1 = valid InSAR pixel, 0/255 = nodata.
- subsidence_clean.tif: velocity map with nodata sentinel replaced by NaN (float32).

Dependencies
------------
rasterio, numpy.

Reproducibility notes
----------------------
- Random seed: not applicable (deterministic I/O and masking only).
- Deterministic given identical inputs: Yes.
"""

import os
import json, numpy as np, rasterio
from rasterio.enums import Resampling
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INSAR_PATH  = r"E:/Processing/InSAR/results/Network2_ifg_unw_Stack_TC1_velocity_map_masked_slope5deg.tif"
NODATA_IN   = -3.40282e+38       # original nodata sentinel
NODATA_OUT  = np.nan
OUT_DIR     = Path(__file__).parent / "outputs"

# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(exist_ok=True)

print("=== 01 Prepare Reference Grid ===")
print(f"  Input : {INSAR_PATH}")

with rasterio.open(INSAR_PATH) as src:
    meta = src.meta.copy()
    data = src.read(1).astype(np.float32)
    transform = src.transform
    crs       = src.crs
    height, width = src.height, src.width

    print(f"  CRS    : {crs}")
    print(f"  Shape  : {height} rows x {width} cols")
    print(f"  Res    : {transform.a:.2f} m")

# ── valid mask ────────────────────────────────────────────────────────────────
nodata_mask = np.isclose(data, NODATA_IN, rtol=1e-5) | (data < -1e+30)
data[nodata_mask] = np.nan

valid_mask = (~np.isnan(data)).astype(np.uint8)
n_valid = valid_mask.sum()
print(f"  Valid pixels : {n_valid:,} / {height*width:,}  ({100*n_valid/(height*width):.1f}%)")

# ── write valid_mask.tif ──────────────────────────────────────────────────────
mask_meta = meta.copy()
mask_meta.update(dtype="uint8", nodata=255, count=1)
mask_path = OUT_DIR / "valid_mask.tif"
with rasterio.open(mask_path, "w", **mask_meta) as dst:
    dst.write(valid_mask, 1)
print(f"  Saved : {mask_path}")

# ── write subsidence_clean.tif ────────────────────────────────────────────────
sub_meta = meta.copy()
sub_meta.update(dtype="float32", nodata=np.nan, count=1)
sub_path = OUT_DIR / "subsidence_clean.tif"
with rasterio.open(sub_path, "w", **sub_meta) as dst:
    dst.write(data, 1)
print(f"  Saved : {sub_path}")

# ── write reference grid JSON ─────────────────────────────────────────────────
grid_info = {
    "crs_wkt" : crs.to_wkt(),
    "epsg"    : 3155,
    "transform": list(transform)[:6],
    "height"  : height,
    "width"   : width,
    "res_m"   : abs(transform.a),
    "n_valid" : int(n_valid),
    "nodata"  : "NaN",
}
json_path = OUT_DIR / "reference_grid_meta.json"
with open(json_path, "w") as f:
    json.dump(grid_info, f, indent=2)
print(f"  Saved : {json_path}")

print("  Done.")