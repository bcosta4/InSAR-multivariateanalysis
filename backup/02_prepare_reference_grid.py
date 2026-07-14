"""
02_prepare_reference_grid.py
─────────────────────────────
Loads the DS-InSAR subsidence raster (your reference grid), extracts
the valid pixel mask (quality filter + lake exclusion), and saves both
the mask and the subsidence array as GeoTIFFs ready for alignment.

INPUTS  (edit CONFIG block below):
    DS_INSAR_PATH   : path to your DS-InSAR velocity raster (cm/yr)
    LANDCOVER_PATH  : path to a land cover raster (for water masking)
                      Set to None if not yet available → will use NDWI
    NDWI_PATH       : path to NDWI raster (only used if LANDCOVER_PATH=None)
    WATER_LC_VALUE  : land cover class integer for open water (ESA CCI: 80)

OUTPUTS (written to OUTPUT_DIR):
    valid_mask.tif          : binary uint8 (1=valid, 0=excluded)
    subsidence_masked.tif   : float32 subsidence (NaN outside valid mask)
    reference_grid_info.txt : CRS, transform, shape, pixel count
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.crs import CRS

# ─── CONFIG — edit these paths ────────────────────────────────────────────────
DS_INSAR_PATH   = r"E:\Processing\InSAR_validation\data\dsInSAR_subsidence_rate.tif"
LANDCOVER_PATH  = None          # Set to path string or leave None
NDWI_PATH       = None          # Set to path string if available (no landcover)
WATER_LC_VALUE  = 80            # ESA CCI open water class; adjust if needed
NDWI_WATER_THR  = 0.2           # NDWI > this value → water pixel
OUTPUT_DIR      = r"E:\Processing\InSAR_validation\data\aligned"
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("02  PREPARE REFERENCE GRID")
print("=" * 60)

# ── 1. Load DS-InSAR raster ───────────────────────────────────────────────────
print(f"\nLoading DS-InSAR raster: {DS_INSAR_PATH}")
with rasterio.open(DS_INSAR_PATH) as src:
    ref_profile   = src.profile.copy()
    ref_crs       = src.crs
    ref_transform = src.transform
    ref_shape     = (src.height, src.width)
    ref_nodata    = src.nodata
    subsidence    = src.read(1).astype(np.float32)
    ref_res       = src.res

print(f"  CRS        : {ref_crs}")
print(f"  Shape      : {ref_shape[0]} rows × {ref_shape[1]} cols")
print(f"  Resolution : {ref_res[0]:.1f} m × {ref_res[1]:.1f} m")
print(f"  NoData     : {ref_nodata}")
print(f"  Value range: {np.nanmin(subsidence):.4f} – {np.nanmax(subsidence):.4f} cm/yr")

# ── 2. DS-InSAR valid mask (non-NaN, non-nodata) ──────────────────────────────
if ref_nodata is not None:
    insar_valid = (subsidence != ref_nodata) & np.isfinite(subsidence)
else:
    insar_valid = np.isfinite(subsidence)

print(f"\n  InSAR valid pixels  : {insar_valid.sum():>12,}")
print(f"  InSAR masked pixels : {(~insar_valid).sum():>12,}")

# ── 3. Water / lake mask ──────────────────────────────────────────────────────
water_mask = np.zeros(ref_shape, dtype=bool)   # True = water (to exclude)

if LANDCOVER_PATH and os.path.exists(LANDCOVER_PATH):
    print(f"\nBuilding water mask from land cover: {LANDCOVER_PATH}")
    with rasterio.open(LANDCOVER_PATH) as lc_src:
        lc = lc_src.read(1)
        # Warn if shape doesn't match (should have been pre-aligned)
        if lc.shape != ref_shape:
            print(f"  [WARNING] Land cover shape {lc.shape} ≠ reference {ref_shape}")
            print("  Run 03_align_predictors.py first, then rerun this script.")
        else:
            water_mask = (lc == WATER_LC_VALUE)
            print(f"  Water pixels (LC={WATER_LC_VALUE}): {water_mask.sum():,}")

elif NDWI_PATH and os.path.exists(NDWI_PATH):
    print(f"\nBuilding water mask from NDWI: {NDWI_PATH}")
    with rasterio.open(NDWI_PATH) as nd_src:
        ndwi = nd_src.read(1).astype(np.float32)
        water_mask = (ndwi > NDWI_WATER_THR)
        print(f"  Water pixels (NDWI > {NDWI_WATER_THR}): {water_mask.sum():,}")
else:
    print("\n  [INFO] No land cover or NDWI provided.")
    print("  Water masking skipped. Thermokarst lakes will remain in dataset.")
    print("  Re-run this script after 03_align_predictors.py with LANDCOVER_PATH set.")

# ── 4. Combined valid mask ────────────────────────────────────────────────────
valid_mask = insar_valid & ~water_mask

print(f"\n  Final valid pixel count : {valid_mask.sum():,}")
print(f"  Lake / water excluded   : {(insar_valid & water_mask).sum():,}")

# ── 5. Write valid_mask.tif ───────────────────────────────────────────────────
mask_profile = ref_profile.copy()
mask_profile.update(dtype="uint8", count=1, nodata=255, compress="lzw")

mask_path = os.path.join(OUTPUT_DIR, "valid_mask.tif")
with rasterio.open(mask_path, "w", **mask_profile) as dst:
    dst.write(valid_mask.astype(np.uint8), 1)
print(f"\nSaved: {mask_path}")

# ── 6. Write subsidence_masked.tif ───────────────────────────────────────────
sub_masked = subsidence.copy()
sub_masked[~valid_mask] = np.nan

sub_profile = ref_profile.copy()
sub_profile.update(dtype="float32", count=1, nodata=float("nan"), compress="lzw")

sub_path = os.path.join(OUTPUT_DIR, "subsidence_masked.tif")
with rasterio.open(sub_path, "w", **sub_profile) as dst:
    dst.write(sub_masked, 1)
print(f"Saved: {sub_path}")

# ── 7. Write reference grid info ─────────────────────────────────────────────
info_path = os.path.join(OUTPUT_DIR, "reference_grid_info.txt")
with open(info_path, "w") as f:
    f.write(f"DS-InSAR Reference Grid Info\n")
    f.write(f"{'=' * 40}\n")
    f.write(f"Source       : {DS_INSAR_PATH}\n")
    f.write(f"CRS          : {ref_crs}\n")
    f.write(f"Transform    : {ref_transform}\n")
    f.write(f"Shape        : {ref_shape[0]} × {ref_shape[1]}\n")
    f.write(f"Resolution   : {ref_res[0]:.2f} m\n")
    f.write(f"Total pixels : {ref_shape[0] * ref_shape[1]:,}\n")
    f.write(f"Valid pixels : {valid_mask.sum():,}\n")
    f.write(f"NoData value : {ref_nodata}\n")
    f.write(f"Subsidence   : {np.nanmin(sub_masked):.4f} – {np.nanmax(sub_masked):.4f} cm/yr\n")
    f.write(f"Subsidence μ : {np.nanmean(sub_masked):.4f} cm/yr\n")
    f.write(f"Subsidence σ : {np.nanstd(sub_masked):.4f} cm/yr\n")

print(f"Saved: {info_path}")
print("\n✓ 02 complete. Proceed to 03_align_predictors.py")

