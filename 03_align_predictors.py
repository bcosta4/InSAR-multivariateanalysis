"""
03_align_predictors.py
───────────────────────
Reprojects and resamples every predictor layer to the canonical InSAR
reference grid (EPSG:3155, ~27m, same extent/shape as subsidence_clean.tif).

All outputs written to ./outputs/aligned/
Skips any layer whose source path is None.
"""

import os, json, numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as R
from pathlib import Path

# ─── CONFIG: add / remove variables here ─────────────────────────────────────
VARIABLES = {
    # key : { "path": r"...", "resample": R.bilinear|R.nearest, "dtype": "float32"|"int16" }

    # ── Terrain (from script 02 outputs) ──────────────────────────────────────
    "dem":              {"path": r"E:\Processing\InSAR_multivariate_analysis\variables_data\HRDEM_dsm_2m_ArcticDEM_merged.tif",   "resample": R.bilinear,  "dtype": "float32"},
    "slope":            {"path": r"E:\Processing\InSAR_multivariate_analysis\variables_data\HRDEM_dsm_2m_ArcticDEM_merged_slope.tif", "resample": R.bilinear, "dtype": "float32"},
    "aspect_sin":       {"path": None,  "resample": R.bilinear,  "dtype": "float32"},   # → outputs/terrain/aspect_sin_2m.tif after script 02
    "aspect_cos":       {"path": None,  "resample": R.bilinear,  "dtype": "float32"},
    "curvature_profile":{"path": None,  "resample": R.bilinear,  "dtype": "float32"},
    "curvature_plan":   {"path": None,  "resample": R.bilinear,  "dtype": "float32"},
    "twi":              {"path": None,  "resample": R.bilinear,  "dtype": "float32"},
    "tpi":              {"path": None,  "resample": R.bilinear,  "dtype": "float32"},
    "tri":              {"path": None,  "resample": R.bilinear,  "dtype": "float32"},

    # ── Permafrost / Climate ──────────────────────────────────────────────────
    "alt_cci":          {"path": None,  "resample": R.bilinear,  "dtype": "float32"},   # ESA CCI ALT (see download guide)
    "magt_cci":         {"path": None,  "resample": R.bilinear,  "dtype": "float32"},   # ESA CCI MAGT
    "ground_ice":       {"path": r"E:\Processing\InSAR_multivariate_analysis\variables_data\Ground_ice_abundance_v1_1.tif", "resample": R.bilinear, "dtype": "float32"},

    # ── ERA5-Land (processed NetCDF → GeoTIFF via script 04) ──────────────────
    "era5_stl1_mean":   {"path": None,  "resample": R.bilinear,  "dtype": "float32"},
    "era5_t2m_mean":    {"path": None,  "resample": R.bilinear,  "dtype": "float32"},
    "era5_ddt":         {"path": None,  "resample": R.bilinear,  "dtype": "float32"},   # derived in script 04

    # ── Vegetation / Land cover ───────────────────────────────────────────────
    "landcover":        {"path": r"E:\Processing\InSAR_multivariate_analysis\variables_data\landcover-2020-classification.tif", "resample": R.nearest, "dtype": "int16"},
    "ndvi_summer":      {"path": None,  "resample": R.bilinear,  "dtype": "float32"},   # GEE export (see download guide)

    # ── Hydrology ────────────────────────────────────────────────────────────
    "flood_index":      {"path": r"E:\Processing\InSAR_multivariate_analysis\variables_data\FS-national-2015-index.tif", "resample": R.bilinear, "dtype": "float32"},

    # ── Geology / Surficial ───────────────────────────────────────────────────
    "surficial_geology":{"path": None,  "resample": R.nearest,   "dtype": "int16"},    # GSC rasterised shapefile (see download guide)
}

# ── Paths that script 02 populates automatically ──────────────────────────────
TERRAIN_DIR = Path(__file__).parent / "outputs" / "terrain"
AUTO_TERRAIN = {
    "aspect_sin":        TERRAIN_DIR / "aspect_sin_2m.tif",
    "aspect_cos":        TERRAIN_DIR / "aspect_cos_2m.tif",
    "curvature_profile": TERRAIN_DIR / "curvature_profile_2m.tif",
    "curvature_plan":    TERRAIN_DIR / "curvature_plan_2m.tif",
    "twi":               TERRAIN_DIR / "twi_2m.tif",
    "tpi":               TERRAIN_DIR / "tpi_2m.tif",
    "tri":               TERRAIN_DIR / "tri_2m.tif",
}
for k, p in AUTO_TERRAIN.items():
    if p.exists() and VARIABLES[k]["path"] is None:
        VARIABLES[k]["path"] = str(p)
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR  = Path(__file__).parent / "outputs" / "aligned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load reference grid
REF_PATH = Path(__file__).parent / "outputs" / "subsidence_clean.tif"
with rasterio.open(REF_PATH) as ref:
    ref_crs   = ref.crs
    ref_trans = ref.transform
    ref_h, ref_w = ref.height, ref.width

print("=== 03 Align Predictors ===")
print(f"  Reference grid : {ref_h}×{ref_w} @ {abs(ref_trans.a):.1f}m  {ref_crs}")

skipped, done = [], []

for name, cfg in VARIABLES.items():
    src_path = cfg["path"]
    if src_path is None or not Path(src_path).exists():
        skipped.append(name)
        continue

    out_path = OUT_DIR / f"{name}_aligned.tif"
    resample_method = cfg["resample"]
    dtype = cfg["dtype"]

    with rasterio.open(src_path) as src:
        dst_meta = {
            "driver":    "GTiff",
            "crs":       ref_crs,
            "transform": ref_trans,
            "width":     ref_w,
            "height":    ref_h,
            "count":     1,
            "dtype":     dtype,
            "nodata":    np.nan if dtype == "float32" else -9999,
            "compress":  "lzw",
        }
        with rasterio.open(out_path, "w", **dst_meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_trans,
                dst_crs=ref_crs,
                resampling=resample_method,
            )
    print(f"  [OK]    {name:<25} → {out_path.name}")
    done.append(name)

print(f"\n  Aligned  : {len(done)}  layers")
print(f"  Skipped  : {len(skipped)}  layers (path=None or missing)")
if skipped:
    print(f"             {skipped}")
print("  Done. → Next: run 04_era5_to_geotiff.py (if ERA5 NetCDF not yet converted)")
