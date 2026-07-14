"""
03_align_predictors.py
─────────────────────────────
Reprojects and resamples every predictor raster to the DS-InSAR
reference grid (27 m, same CRS, same extent). Handles:
  - Continuous variables  → bilinear resampling
  - Categorical variables → nearest-neighbour resampling
  - Aspect                → decomposed into sin/cos components
  - Temporal variables    → per-year stack or pre-computed mean

INPUTS  (edit PREDICTORS dict below):
    All raw raster paths, keyed by output variable name.

OUTPUTS:
    aligned/<name>.tif  — one GeoTIFF per variable, same grid as reference
    aligned/aspect_sin.tif, aligned/aspect_cos.tif  (from raw aspect)
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
REFERENCE_PATH = r"E:\Processing\InSAR_validation\data\aligned\subsidence_masked.tif"
OUTPUT_DIR     = r"E:\Processing\InSAR_validation\data\aligned"

# ─── PREDICTOR MANIFEST ───────────────────────────────────────────────────────
# Key     = output variable name (used in parquet column headers)
# "path"  = path to raw source raster
# "type"  = "continuous" | "categorical" | "aspect"
#            aspect → automatically split into sin/cos
# "note"  = brief description for logging
#
# ⚠ EDIT PATHS to match your actual file locations.
# Leave "path" as None for datasets not yet downloaded — script will skip and warn.

PREDICTORS = {
    # ── Permafrost CCI ────────────────────────────────────────────────────────
    "alt_mean": {
        "path": None,   # e.g. r"E:\Data\PermafrostCCI\ALT_mean_2019_2024.tif"
        "type": "continuous",
        "note": "Active Layer Thickness mean 2019-2024 (m)"
    },
    "magt": {
        "path": None,   # e.g. r"E:\Data\PermafrostCCI\MAGT_2019.tif"
        "type": "continuous",
        "note": "Mean Annual Ground Temperature 2019 (°C)"
    },
    "ddt": {
        "path": None,   # e.g. r"E:\Data\PermafrostCCI\DDT_mean_2019_2024.tif"
        "type": "continuous",
        "note": "Thaw Degree Days mean 2019-2024"
    },
    "ground_ice": {
        "path": None,   # e.g. r"E:\Data\PermafrostCCI\GroundIce.tif"
        "type": "continuous",
        "note": "Volumetric ground ice content (%)"
    },

    # ── Arctic DEM derivatives ────────────────────────────────────────────────
    "slope": {
        "path": None,   # e.g. r"E:\Data\ArcticDEM\slope_27m.tif"
        "type": "continuous",
        "note": "Slope (degrees) from ArcticDEM"
    },
    "aspect": {
        "path": None,   # e.g. r"E:\Data\ArcticDEM\aspect_27m.tif"
        "type": "aspect",
        "note": "Aspect (degrees) → split into sin/cos"
    },
    "curvature": {
        "path": None,   # e.g. r"E:\Data\ArcticDEM\curvature_27m.tif"
        "type": "continuous",
        "note": "Profile curvature from ArcticDEM"
    },
    "twi": {
        "path": None,   # e.g. r"E:\Data\ArcticDEM\TWI_27m.tif"
        "type": "continuous",
        "note": "Topographic Wetness Index"
    },
    "tpi": {
        "path": None,   # e.g. r"E:\Data\ArcticDEM\TPI_27m.tif"
        "type": "continuous",
        "note": "Topographic Position Index"
    },
    "tri": {
        "path": None,   # e.g. r"E:\Data\ArcticDEM\TRI_27m.tif"
        "type": "continuous",
        "note": "Terrain Roughness Index"
    },

    # ── Spectral / land cover ─────────────────────────────────────────────────
    "ndvi_mean": {
        "path": None,   # e.g. r"E:\Data\Sentinel2\NDVI_summer_mean_2019_2024.tif"
        "type": "continuous",
        "note": "NDVI summer mean 2019-2024 (Jun-Aug)"
    },
    "landcover": {
        "path": None,   # e.g. r"E:\Data\LandCover\ESA_CCI_LC_2020.tif"
        "type": "categorical",
        "note": "Land cover (categorical, ESA CCI 2020)"
    },
    "wetlands": {
        "path": None,   # e.g. r"E:\Data\Wetlands\wetlands_binary.tif"
        "type": "categorical",
        "note": "Wetland presence (binary)"
    },
    "geology": {
        "path": None,   # e.g. r"E:\Data\Geology\geology_classes.tif"
        "type": "categorical",
        "note": "Geological units (categorical)"
    },

    # ── Flood inventory ───────────────────────────────────────────────────────
    "flood_freq": {
        "path": None,   # e.g. r"E:\Data\FloodInventory\flood_frequency.tif"
        "type": "continuous",
        "note": "Flood event frequency count per pixel 2019-2024"
    },

    # ── Optional additions ────────────────────────────────────────────────────
    "soil_organic_carbon": {
        "path": None,   # SoilGrids250 SOC — download from ISRIC
        "type": "continuous",
        "note": "Soil organic carbon 0-5 cm (g/kg) — SoilGrids250"
    },
    "era5_soil_temp": {
        "path": None,   # ERA5-Land mean summer soil temp 0-7cm 2019-2024
        "type": "continuous",
        "note": "ERA5-Land mean summer soil temp (K)"
    },
}
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load reference grid metadata ─────────────────────────────────────────────
print("=" * 60)
print("03  ALIGN PREDICTORS TO REFERENCE GRID")
print("=" * 60)

with rasterio.open(REFERENCE_PATH) as ref:
    ref_meta      = ref.meta.copy()
    ref_transform = ref.transform
    ref_crs       = ref.crs
    ref_shape     = (ref.height, ref.width)

print(f"\nReference grid: {ref_shape[0]} × {ref_shape[1]} @ {ref_transform.a:.1f} m")
print(f"CRS: {ref_crs}\n")


def align_raster(src_path, out_path, resampling=Resampling.bilinear):
    """Reproject + resample src_path to match reference grid."""
    with rasterio.open(src_path) as src:
        src_dtype = src.dtypes[0]
        out_array = np.full(ref_shape, np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=out_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling,
            dst_nodata=np.nan,
        )

    out_meta = ref_meta.copy()
    out_meta.update(dtype="float32", count=1, nodata=float("nan"), compress="lzw")

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(out_array, 1)

    nan_pct = np.isnan(out_array).mean() * 100
    return out_array, nan_pct


skipped = []
processed = []

for name, cfg in PREDICTORS.items():
    src_path = cfg["path"]
    vtype    = cfg["type"]
    note     = cfg["note"]

    if src_path is None or not os.path.exists(str(src_path)):
        print(f"  [SKIP]  {name:25s} — path not set or file missing")
        skipped.append(name)
        continue

    print(f"  Processing: {name:25s} ({vtype}) — {note}")

    if vtype == "aspect":
        # Special case: decompose aspect into sin and cos components
        tmp_path = os.path.join(OUTPUT_DIR, f"_aspect_raw.tif")
        aspect_arr, nan_pct = align_raster(src_path, tmp_path, Resampling.bilinear)

        aspect_rad  = np.deg2rad(aspect_arr)
        aspect_sin  = np.sin(aspect_rad)
        aspect_cos  = np.cos(aspect_rad)
        aspect_sin[np.isnan(aspect_arr)] = np.nan
        aspect_cos[np.isnan(aspect_arr)] = np.nan

        for comp_name, comp_arr in [("aspect_sin", aspect_sin), ("aspect_cos", aspect_cos)]:
            out_path = os.path.join(OUTPUT_DIR, f"{comp_name}.tif")
            out_meta = ref_meta.copy()
            out_meta.update(dtype="float32", count=1, nodata=float("nan"), compress="lzw")
            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(comp_arr, 1)
            print(f"    → Saved {comp_name}.tif  (NaN: {nan_pct:.1f}%)")
            processed.append(comp_name)

        os.remove(tmp_path)

    elif vtype == "categorical":
        out_path = os.path.join(OUTPUT_DIR, f"{name}.tif")
        arr, nan_pct = align_raster(src_path, out_path, Resampling.nearest)
        print(f"    → Saved {name}.tif  (NaN: {nan_pct:.1f}%)")
        processed.append(name)

    else:  # continuous
        out_path = os.path.join(OUTPUT_DIR, f"{name}.tif")
        arr, nan_pct = align_raster(src_path, out_path, Resampling.bilinear)
        print(f"    → Saved {name}.tif  (NaN: {nan_pct:.1f}%)")
        processed.append(name)

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Processed : {len(processed)} variables → {OUTPUT_DIR}")
if skipped:
    print(f"Skipped   : {len(skipped)} variables (paths not set):")
    for s in skipped:
        print(f"            • {s}  — {PREDICTORS[s]['note']}")
print("\n✓ 03 complete. Proceed to 04_build_predictor_stack.py")
