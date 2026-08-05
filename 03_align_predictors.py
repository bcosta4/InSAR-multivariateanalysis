r"""
03_align_predictors.py
─────────────────────
Reprojects/resamples all predictor rasters from variables_data to the canonical
InSAR reference grid (EPSG:3155, same extent/shape as subsidence_clean.tif).

Inputs:
- Source predictors in E:\Processing\InSAR_multivariate_analysis\variables_data
- Reference grid from 01_prepare_reference_grid.py

Outputs:
- Aligned rasters written to a run folder, e.g.E:\Processing\InSAR_multivariate_analysis\outputs\run_nX_YYYYMMDD\aligned\

Features:
- Uses median by default for variables with median/mean alternatives
- Optional nearest-valid fill AFTER reprojection for selected CONTINUOUS variables
- Categorical/nominal variables use nearest-neighbor resampling and are NEVER bilinear-filled (averaging category codes is meaningless)
- Uses rasterized surficial geology from 02_files_to_geotiff.py
"""

import re
import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling as R
from scipy.ndimage import distance_transform_edt
from pathlib import Path
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"E:\Processing\InSAR_multivariate_analysis")
DATA_DIR = PROJECT_DIR / "variables_data"
PIPE_OUT = PROJECT_DIR / "outputs"
REF_PATH = PROJECT_DIR / "scripts" / "outputs" / "subsidence_clean.tif"
VALID_MASK_PATH = PROJECT_DIR / "scripts" / "outputs" / "valid_mask.tif"

STAT_PREFERENCE = "median"  # or "mean"
RUN_NUMBER = None           # None = auto-detect next run_nX_YYYYMMDD
DATE_TAG = datetime.now().strftime("%Y%m%d")
# ──────────────────────────────────────────────────────────────────────────────
def assert_north_up(transform_or_path, label):
    """Confirms north-up orientation (row 0 = northernmost latitude).
    Accepts either an Affine transform directly or a path to a GeoTIFF."""
    if isinstance(transform_or_path, (str, Path)):
        with rasterio.open(transform_or_path) as check:
            e = check.transform.e
    else:
        e = transform_or_path.e
    assert e < 0, (
        f"[GEOMETRY CHECK FAILED] {label}: transform.e={e} should be "
        f"negative (north-up convention) - orientation is flipped."
    )

def get_run_dir():
    PIPE_OUT.mkdir(parents=True, exist_ok=True)
    if RUN_NUMBER is None:
        existing = []
        for p in PIPE_OUT.glob("run_n*_*"):
            m = re.match(r"run_n(\d+)_\d{8}", p.name)
            if m:
                existing.append(int(m.group(1)))
        n = max(existing) + 1 if existing else 1
    else:
        n = RUN_NUMBER
    run_dir = PIPE_OUT / f"run_n{n}_{DATE_TAG}"
    (run_dir / "aligned").mkdir(parents=True, exist_ok=True)
    return run_dir

def pick(median_path, mean_path):
    chosen = median_path if STAT_PREFERENCE == "median" else mean_path
    fallback = mean_path if STAT_PREFERENCE == "median" else median_path
    if chosen is not None and Path(chosen).exists():
        return str(chosen)
    if fallback is not None and Path(fallback).exists():
        return str(fallback)
    return None

def nearest_fill(arr, mask_valid):
    filled = arr.copy()
    nan_mask = np.isnan(filled) & mask_valid
    if not nan_mask.any():
        return filled
    known_mask = ~np.isnan(filled)
    if not known_mask.any():
        return filled
    _, (inds_r, inds_c) = distance_transform_edt(~known_mask, return_indices=True)
    nearest_vals = filled[inds_r, inds_c]
    return np.where(nan_mask, nearest_vals, filled)

def safe_src_nodata_fallback(src_dtype):
    dt = np.dtype(src_dtype)
    if np.issubdtype(dt, np.floating):
        return np.nan
    if np.issubdtype(dt, np.unsignedinteger):
        return int(np.iinfo(dt).max)
    if np.issubdtype(dt, np.integer):
        return int(np.iinfo(dt).min)
    return None

VARIABLES = {
    # Terrain
    "dem":               {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "slope":             {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_slope_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "aspect_sin":        {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_aspect_sin_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "aspect_cos":        {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_aspect_cos_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "curvature_profile": {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_profilecurvature_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "curvature_plan":    {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_plancurvature_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "twi":               {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_TWI_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "tpi":               {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_TPI_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "tri":               {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_TRI_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "wind_effect":       {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_derivatewindeffect_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},

    # CCI permafrost (continuous depth/temperature layers)
    "cci_alt":   {"path": str(DATA_DIR / f"cci_alt_{STAT_PREFERENCE}_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "cci_gst":   {"path": str(DATA_DIR / f"cci_gst_{STAT_PREFERENCE}_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "cci_t1m":   {"path": str(DATA_DIR / f"cci_t1m_{STAT_PREFERENCE}_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "cci_t2m":   {"path": str(DATA_DIR / f"cci_t2m_{STAT_PREFERENCE}_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "cci_t5m":   {"path": str(DATA_DIR / f"cci_t5m_{STAT_PREFERENCE}_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "cci_t10m":  {"path": str(DATA_DIR / f"cci_t10m_{STAT_PREFERENCE}_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},

    # ERA5-derived rasters from 02 (all continuous)
    "era5_stl1_mean":       {"path": str(DATA_DIR / "era5_stl1_mean_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_stl1_jjas_mean":  {"path": str(DATA_DIR / "era5_stl1_JJAS_mean_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_t2m_mean":        {"path": str(DATA_DIR / "era5_t2m_mean_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_t2m_jjas_mean":   {"path": str(DATA_DIR / "era5_t2m_JJAS_mean_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_ddt":             {"path": str(DATA_DIR / "era5_t2m_ddt_annual_mean_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_ddf":             {"path": str(DATA_DIR / "era5_t2m_ddf_annual_mean_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_ssr_mean":        {"path": str(DATA_DIR / "era5_ssr_mean_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_ssr_jjas_mean":   {"path": str(DATA_DIR / "era5_ssr_JJAS_mean_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_swvl1_mean":      {"path": str(DATA_DIR / "era5_swvl1_mean_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_swvl1_jjas_mean": {"path": str(DATA_DIR / "era5_swvl1_JJAS_mean_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_tp_mean":         {"path": str(DATA_DIR / "era5_tp_mean_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_tp_jjas_mean":    {"path": str(DATA_DIR / "era5_tp_JJAS_mean_2019_2024.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},

    # Optical / vegetation indices (continuous)
    "bsi":   {"path": pick(DATA_DIR / "BSI_median_JJAS_2019_2024_masked.tif",   DATA_DIR / "BSI_mean_JJAS_2019_2024_masked.tif"),   "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "mndwi": {"path": pick(DATA_DIR / "MNDWI_median_JJAS_2019_2024_masked.tif", DATA_DIR / "MNDWI_mean_JJAS_2019_2024_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "ndmi":  {"path": str(DATA_DIR / "NDMI_median_JJAS_2019_2024_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "ndvi":  {"path": str(DATA_DIR / "NDVI_median_JJAS_2019_2024_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},

    # Snow (continuous day-of-year / duration)
    "snow_end_doy":     {"path": pick(DATA_DIR / "HLS_Fmask_v1_1_snow_endL_winterYear1924_Canada_masked_median.tif", DATA_DIR / "HLS_Fmask_v1_1_snow_endL_winterYear1924_Canada_masked_avg.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "snow_length_days": {"path": pick(DATA_DIR / "HLS_Fmask_v1_1_snow_lengthT_winterYear1924_Canada_masked_median.tif", DATA_DIR / "HLS_Fmask_v1_1_snow_lengthT_winterYear1924_Canada_masked_avg.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},

    # Vegetation structure / moisture (continuous)
    "fsm_historic_mc": {"path": str(DATA_DIR / "fsm-2019-2023-historic-mc_masked_avg.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "lai":    {"path": pick(DATA_DIR / "monthly-vegetation-parameters-v1-201924-LAI_masked_median.tif", DATA_DIR / "monthly-vegetation-parameters-v1-201924-LAI_masked_avg.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "fcover": {"path": pick(DATA_DIR / "monthly-vegetation-parameters-v1-201924-fCOVER_masked_median.tif", DATA_DIR / "monthly-vegetation-parameters-v1-201924-fCOVER_masked_avg.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},

    # Categorical / nominal layers -> nearest-neighbor, no averaging
    "landcover":              {"path": str(DATA_DIR / "landcover-2020-classification_masked.tif"), "resample": R.nearest, "dtype": "int16", "nearest_fill": False, "categorical": True},
    "wetland_class":          {"path": str(DATA_DIR / "canadian-wetland-inventory-v3a-classification_masked.tif"), "resample": R.nearest, "dtype": "int16", "nearest_fill": False, "categorical": True},
    "pan_arctic_wetlands":    {"path": str(DATA_DIR / "pan_arctic_wetlands_masked.tif"), "resample": R.nearest, "dtype": "int16", "nearest_fill": False, "categorical": True},
    "canopy_height_lt5m":     {"path": str(DATA_DIR / "canopy_height_mask_lt5m_C3_masked.tif"), "resample": R.nearest, "dtype": "int16", "nearest_fill": False, "categorical": True},
    "woodyplant_dominance":   {"path": str(DATA_DIR / "woodyplant_dominance_2020_C3_masked.tif"), "resample": R.nearest, "dtype": "int16", "nearest_fill": False, "categorical": True},
    "cavm":                   {"path": str(DATA_DIR / "raster_cavm_v1_masked.tif"), "resample": R.nearest, "dtype": "int16", "nearest_fill": False, "categorical": True},
    "surficial_geology":      {"path": str(DATA_DIR / "geology_tuk_rich_DESC2_rasterized_3155.tif"), "resample": R.nearest, "dtype": "int32", "nearest_fill": False, "categorical": True},
    "ground_ice": {"path": str(DATA_DIR / "Ground_ice_abundance_v1_1_masked.tif"), "resample": R.nearest, "dtype": "int16", "nearest_fill": False, "categorical": True},
    "dsw_trends":    {"path": str(DATA_DIR / "dsw-1984-2023-trends_masked.tif"), "resample": R.nearest, "dtype": "int16", "nearest_fill": False, "categorical": True},
    "permafrost_landcover_cls": {"path": str(DATA_DIR / "ESACCI-PERMAFROST-L4-CLS-SENTINEL1_2_KMLH-AREA18_LCU_CANL1-2016-2024-fv02.0_masked.tif"), "resample": R.nearest, "dtype": "int16", "nearest_fill": False, "categorical": True},

    # Hydrology / biomass / soils
    "flood_index":   {"path": str(DATA_DIR / "FS-national-2015-index_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "dsw_frequency": {"path": str(DATA_DIR / "dsw-1984-2023-frequency_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},  # confirmed continuous (0-100 integer values)
    "plant_agb":     {"path": str(DATA_DIR / "plant_agb_2020_p500_C3_masked.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "clay_pct":      {"path": str(DATA_DIR / "clay_soilgridsorg_250m.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "sand_pct":      {"path": str(DATA_DIR / "sand_soilgridsorg_250m.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "silt_pct":      {"path": str(DATA_DIR / "silt_soilgridsorg_250m.tif"), "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
}

run_dir = get_run_dir()
ALIGNED_DIR = run_dir / "aligned"

with rasterio.open(REF_PATH) as ref:
    ref_crs = ref.crs
    ref_trans = ref.transform
    ref_h, ref_w = ref.height, ref.width

# Sanity check (v2): confirm the reference grid itself is north-up BEFORE using it as dst_transform for every single reproject() call below.
assert_north_up(ref_trans, "reference grid (subsidence_clean.tif)")
print("[CHECK] Reference grid transform confirmed north-up.")

with rasterio.open(VALID_MASK_PATH) as vm:
    valid_mask_raw = vm.read(1)

n_ones = int((valid_mask_raw == 1).sum())
n_total = valid_mask_raw.size
frac_ones = n_ones / n_total
print(f"[CHECK] valid_mask.tif: {n_ones}/{n_total} pixels == 1 ({frac_ones*100:.1f}%)")
if frac_ones > 0.95 or frac_ones < 0.01:
    print("[WARNING] valid_mask fraction looks unusual - verify polarity.")

valid_mask = (valid_mask_raw == 1)

print("=== 03 Align Predictors (v2, geometry check) ===")
print(f"Run dir         : {run_dir}")
print(f"Stat preference : {STAT_PREFERENCE}")
print(f"Reference grid  : {ref_h}x{ref_w} @ {abs(ref_trans.a):.1f}m {ref_crs}")

skipped, done, failed = [], [], []

for name, cfg in VARIABLES.items():
    src_path = cfg["path"]
    if src_path is None or not Path(src_path).exists():
        skipped.append(name)
        continue

    out_path = ALIGNED_DIR / f"{name}_aligned.tif"
    resample_method = cfg["resample"]
    dtype = cfg["dtype"]
    do_fill = cfg.get("nearest_fill", False)
    is_categorical = cfg.get("categorical", False)

    try:
        with rasterio.open(src_path) as src:
            src_nodata_val = src.nodata
            src_dtype = src.dtypes[0]


            if dtype == "float32":
                dst_arr = np.full((ref_h, ref_w), np.nan, dtype=np.float32)
                dst_nodata = np.nan
            elif dtype == "int32":
                dst_arr = np.full((ref_h, ref_w), 0, dtype=np.int32)
                dst_nodata = 0
            else:
                dst_arr = np.full((ref_h, ref_w), -9999, dtype=np.int16)
                dst_nodata = -9999


            if src_nodata_val is not None:
                reproj_src_nodata = src_nodata_val
            else:
                reproj_src_nodata = safe_src_nodata_fallback(src_dtype)


            reproject(
                source=rasterio.band(src, 1),
                destination=dst_arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_trans,
                dst_crs=ref_crs,
                src_nodata=reproj_src_nodata,
                dst_nodata=dst_nodata,
                resampling=resample_method,
            )
    except Exception as e:
        print(f"[FAIL] {name:<28} -> {type(e).__name__}: {e}")
        failed.append(name)
        continue


    if do_fill and dtype == "float32" and not is_categorical:
        n_before = int(np.isnan(dst_arr[valid_mask]).sum())
        dst_arr = nearest_fill(dst_arr, valid_mask)
        n_after = int(np.isnan(dst_arr[valid_mask]).sum())
        fill_note = f" | nearest-fill {n_before}->{n_after}"
    else:
        fill_note = " | categorical (no fill)" if is_categorical else ""


    meta = {
        "driver": "GTiff",
        "crs": ref_crs,
        "transform": ref_trans,
        "width": ref_w,
        "height": ref_h,
        "count": 1,
        "dtype": str(dst_arr.dtype),
        "nodata": dst_nodata,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(dst_arr, 1)

    # Sanity check (v2): confirm each aligned output is still north-up.
    assert_north_up(out_path, name)


    print(f"[OK] {name:<28} -> {out_path.name}{fill_note}")
    done.append(name)

print(f"\nAligned: {len(done)}")
print(f"Skipped (file not found): {len(skipped)}")
if skipped:
    print(skipped)
print(f"Failed (error during reproject): {len(failed)}")
if failed:
    print(failed)
print("Done.")
print(f"Next: 04_build_predictor_stack.py using run folder: {run_dir}")