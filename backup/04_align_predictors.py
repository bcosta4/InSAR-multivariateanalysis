
"""
03_align_predictors.py
───────────────────────
Reprojects and resamples every predictor layer to the canonical InSAR
reference grid (EPSG:3155, ~27m, same extent/shape as subsidence_clean.tif).

All outputs written to ./outputs/aligned/
Skips any layer whose source path is None or missing.

NEAREST-FILL: for variables listed in NEAREST_FILL_VARS, any remaining
NaN/nodata pixels *within the InSAR valid-pixel footprint* are filled
using nearest-valid-pixel interpolation AFTER reprojection.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling as R
from scipy.ndimage import distance_transform_edt
from pathlib import Path

# ─── STAT PREFERENCE TOGGLE ───────────────────────────────────────────────────
# "median" (default) or "mean"/"avg" — controls which version is picked for
# every variable that has both options available.
STAT_PREFERENCE = "median"
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(r"E:\Processing\InSAR_multivariate_analysis\variables_data")
ERA5_DIR = Path(__file__).parent / "outputs" / "era5"
TERRAIN_DIR = Path(__file__).parent / "outputs" / "terrain"


def pick(median_path, mean_path):
    """Return median_path or mean_path depending on STAT_PREFERENCE, whichever exists."""
    chosen = median_path if STAT_PREFERENCE == "median" else mean_path
    fallback = mean_path if STAT_PREFERENCE == "median" else median_path
    if chosen is not None and Path(chosen).exists():
        return str(chosen)
    if fallback is not None and Path(fallback).exists():
        return str(fallback)
    return None


# ─── CONFIG: VARIABLES DICT ───────────────────────────────────────────────────
VARIABLES = {
    # ── Terrain (script 02 outputs, 2m EPSG:3155) ─────────────────────────────
    "dem":               {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_masked.tif"),
                           "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "slope":             {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_slope_masked.tif"),
                           "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "aspect_sin":        {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_aspect_sin_masked.tif"),
                           "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "aspect_cos":        {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_aspect_cos_masked.tif"),
                           "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "curvature_profile": {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_profilecurvature_masked.tif"),
                           "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "curvature_plan":    {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_plancurvature_masked.tif"),
                           "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "twi":               {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_TWI_masked.tif"),
                           "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "tpi":               {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_TPI_masked.tif"),
                           "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "tri":               {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_TRI_masked.tif"),
                           "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "wind_effect":       {"path": str(DATA_DIR / "HRDEM_dsm_2m_ArcticDEM_merged_derivatewindeffect_masked.tif"),
                           "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},

    # ── Permafrost / Ground Ice (ESA CCI via script 04, + static ground ice) ──
    "cci_alt":     {"path": str(ERA5_DIR / f"cci_alt_{STAT_PREFERENCE}.tif"),
                     "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "cci_gst":     {"path": str(ERA5_DIR / f"cci_gst_{STAT_PREFERENCE}.tif"),
                     "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "cci_t1m":     {"path": str(ERA5_DIR / f"cci_t1m_{STAT_PREFERENCE}.tif"),
                     "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "cci_t2m":     {"path": str(ERA5_DIR / f"cci_t2m_{STAT_PREFERENCE}.tif"),
                     "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "cci_t5m":     {"path": str(ERA5_DIR / f"cci_t5m_{STAT_PREFERENCE}.tif"),
                     "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "cci_t10m":    {"path": str(ERA5_DIR / f"cci_t10m_{STAT_PREFERENCE}.tif"),
                     "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "ground_ice":  {"path": str(DATA_DIR / "Ground_ice_abundance_v1_1_masked.tif"),
                     "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},

    # ── ERA5-Land climate (via script 04) ─────────────────────────────────────
    "era5_stl1_mean":       {"path": str(ERA5_DIR / "era5_stl1_mean.tif"),        "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_stl1_jjas_mean":  {"path": str(ERA5_DIR / "era5_stl1_JJAS_mean.tif"),   "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_t2m_mean":        {"path": str(ERA5_DIR / "era5_t2m_mean.tif"),         "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_t2m_jjas_mean":   {"path": str(ERA5_DIR / "era5_t2m_JJAS_mean.tif"),    "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_ddt":             {"path": str(ERA5_DIR / "era5_t2m_ddt_annual.tif"),   "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_ddf":             {"path": str(ERA5_DIR / "era5_t2m_ddf_annual.tif"),   "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_ssr_mean":        {"path": str(ERA5_DIR / "era5_ssr_mean.tif"),         "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_ssr_jjas_mean":   {"path": str(ERA5_DIR / "era5_ssr_JJAS_mean.tif"),    "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_swvl1_mean":      {"path": str(ERA5_DIR / "era5_swvl1_mean.tif"),       "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_swvl1_jjas_mean": {"path": str(ERA5_DIR / "era5_swvl1_JJAS_mean.tif"),  "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_tp_mean":         {"path": str(ERA5_DIR / "era5_tp_mean.tif"),          "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "era5_tp_jjas_mean":    {"path": str(ERA5_DIR / "era5_tp_JJAS_mean.tif"),     "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},

    # ── Optical / Vegetation Indices (JJAS composites) ────────────────────────
    "bsi":   {"path": pick(DATA_DIR / "BSI_median_JJAS_2019_2024_masked.tif",   DATA_DIR / "BSI_mean_JJAS_2019_2024_masked.tif"),
              "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "mndwi": {"path": pick(DATA_DIR / "MNDWI_median_JJAS_2019_2024_masked.tif", DATA_DIR / "MNDWI_mean_JJAS_2019_2024_masked.tif"),
              "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "ndmi":  {"path": pick(DATA_DIR / "NDMI_median_JJAS_2019_2024_masked.tif",  None),
              "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},
    "ndvi":  {"path": pick(DATA_DIR / "NDVI_median_JJAS_2019_2024_masked.tif",  None),
              "resample": R.bilinear, "dtype": "float32", "nearest_fill": False},

    # ── Snow (HLS Fmask) ───────────────────────────────────────────────────────
    "snow_end_doy":    {"path": pick(DATA_DIR / "HLS_Fmask_v1_1_snow_endL_winterYear1924_Canada_masked_median.tif",
                                      DATA_DIR / "HLS_Fmask_v1_1_snow_endL_winterYear1924_Canada_masked_avg.tif"),
                         "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "snow_length_days":{"path": pick(DATA_DIR / "HLS_Fmask_v1_1_snow_lengthT_winterYear1924_Canada_masked_median.tif",
                                      DATA_DIR / "HLS_Fmask_v1_1_snow_lengthT_winterYear1924_Canada_masked_avg.tif"),
                         "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},

    # ── Soil Moisture Class (historic) ────────────────────────────────────────
    # avg is Float32 (continuous); median version was saved as Byte (classified) — use avg
    "fsm_historic_mc": {"path": str(DATA_DIR / "fsm-2019-2023-historic-mc_masked_avg.tif"),
                         "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},

    # ── Vegetation Structure (LAI, fCOVER) ────────────────────────────────────
    "lai":    {"path": pick(DATA_DIR / "monthly-vegetation-parameters-v1-201924-LAI_masked_median.tif",
                             DATA_DIR / "monthly-vegetation-parameters-v1-201924-LAI_masked_avg.tif"),
               "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "fcover": {"path": pick(DATA_DIR / "monthly-vegetation-parameters-v1-201924-fCOVER_masked_median.tif",
                             DATA_DIR / "monthly-vegetation-parameters-v1-201924-fCOVER_masked_avg.tif"),
               "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},

    # ── Land cover / vegetation classification (single-file, categorical) ────
    "landcover":      {"path": str(DATA_DIR / "landcover-2020-classification_masked.tif"),
                        "resample": R.nearest, "dtype": "int16", "nearest_fill": False},
    "wetland_class":  {"path": str(DATA_DIR / "canadian-wetland-inventory-v3a-classification_masked.tif"),
                        "resample": R.nearest, "dtype": "int16", "nearest_fill": False},
    "pan_arctic_wetlands": {"path": str(DATA_DIR / "pan_arctic_wetlands_masked.tif"),
                        "resample": R.nearest, "dtype": "int16", "nearest_fill": False},
    "canopy_height_lt5m": {"path": str(DATA_DIR / "canopy_height_mask_lt5m_C3_masked.tif"),
                        "resample": R.nearest, "dtype": "int16", "nearest_fill": False},
    "plant_agb":      {"path": str(DATA_DIR / "plant_agb_2020_p500_C3_masked.tif"),
                        "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "woodyplant_dominance": {"path": str(DATA_DIR / "woodyplant_dominance_2020_C3_masked.tif"),
                        "resample": R.nearest, "dtype": "int16", "nearest_fill": False},
    "cavm":           {"path": str(DATA_DIR / "raster_cavm_v1_masked.tif"),
                        "resample": R.nearest, "dtype": "int16", "nearest_fill": False},

    # ── Hydrology / Water Dynamics ─────────────────────────────────────────────
    "flood_index":    {"path": str(DATA_DIR / "FS-national-2015-index_masked.tif"),
                        "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "dsw_frequency":  {"path": str(DATA_DIR / "dsw-1984-2023-frequency_masked.tif"),
                        "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "dsw_trends":     {"path": str(DATA_DIR / "dsw-1984-2023-trends_masked.tif"),
                        "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},

    # ── Soil texture (SoilGrids) ────────────────────────────────────────────────
    "clay_pct": {"path": str(DATA_DIR / "clay_soilgridsorg_250m.tif"),
                 "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "sand_pct": {"path": str(DATA_DIR / "sand_soilgridsorg_250m.tif"),
                 "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},
    "silt_pct": {"path": str(DATA_DIR / "silt_soilgridsorg_250m.tif"),
                 "resample": R.bilinear, "dtype": "float32", "nearest_fill": True},

    # ── Ground Deformation (Sentinel-1 ADP) ────────────────────────────────────
    "active_deformation": {"path": str(DATA_DIR / "active_deformation_processes_sentinel1_2017-2024_asc_masked.tif"),
                            "resample": R.nearest, "dtype": "int16", "nearest_fill": False},

    # ── Surficial Geology (rasterised, if available) ───────────────────────────
    "surficial_geology": {"path": None, "resample": R.nearest, "dtype": "int16", "nearest_fill": False},
}
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).parent / "outputs" / "aligned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_PATH = Path(__file__).parent / "outputs" / "subsidence_clean.tif"
VALID_MASK_PATH = Path(__file__).parent / "outputs" / "valid_mask.tif"

with rasterio.open(REF_PATH) as ref:
    ref_crs = ref.crs
    ref_trans = ref.transform
    ref_h, ref_w = ref.height, ref.width

with rasterio.open(VALID_MASK_PATH) as vm:
    valid_mask = vm.read(1).astype(bool)

print("=== 03 Align Predictors ===")
print(f" Stat preference : {STAT_PREFERENCE}")
print(f" Reference grid  : {ref_h}x{ref_w} @ {abs(ref_trans.a):.1f}m {ref_crs}")


def nearest_fill(arr, mask_valid):
    """Fill NaNs within mask_valid using nearest valid pixel (Euclidean nearest)."""
    filled = arr.copy()
    nan_mask = np.isnan(filled) & mask_valid
    if not nan_mask.any():
        return filled
    known_mask = ~np.isnan(filled)
    if not known_mask.any():
        return filled  # nothing to fill from
    _, (inds_r, inds_c) = distance_transform_edt(~known_mask, return_indices=True)
    filled_full = filled[inds_r, inds_c]
    filled = np.where(nan_mask, filled_full, filled)
    return filled


skipped, done = [], []

for name, cfg in VARIABLES.items():
    src_path = cfg["path"]
    if src_path is None or not Path(src_path).exists():
        skipped.append(name)
        continue

    out_path = OUT_DIR / f"{name}_aligned.tif"
    resample_method = cfg["resample"]
    dtype = cfg["dtype"]
    do_fill = cfg.get("nearest_fill", False)

    with rasterio.open(src_path) as src:
        dst_nodata = np.nan if dtype == "float32" else -9999
        dst_meta = {
            "driver": "GTiff", "crs": ref_crs, "transform": ref_trans,
            "width": ref_w, "height": ref_h, "count": 1,
            "dtype": dtype, "nodata": dst_nodata, "compress": "lzw",
        }

        dst_arr = np.full((ref_h, ref_w), dst_nodata, dtype=dtype)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_trans, dst_crs=ref_crs,
            resampling=resample_method,
        )

    if do_fill and dtype == "float32":
        n_before = int(np.isnan(dst_arr[valid_mask]).sum())
        dst_arr = nearest_fill(dst_arr, valid_mask)
        n_after = int(np.isnan(dst_arr[valid_mask]).sum())
        fill_note = f" | nearest-fill: {n_before}->{n_after} NaN in valid area"
    else:
        fill_note = ""

    with rasterio.open(out_path, "w", **dst_meta) as dst:
        dst.write(dst_arr, 1)

    print(f" [OK] {name:<25} -> {out_path.name}{fill_note}")
    done.append(name)

print(f"\n Aligned : {len(done)} layers")
print(f" Skipped : {len(skipped)} layers (path=None or missing)")
if skipped:
    print(f" {skipped}")
print(" Done. -> Next: run 05_build_predictor_stack.py")
