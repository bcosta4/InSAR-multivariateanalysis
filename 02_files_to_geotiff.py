r"""
02_files_to_geotiff.py
──────────────────────
Purpose
-------
Converts source NetCDF climate/permafrost variables and one vector geology layer into analysis-ready GeoTIFF predictors, written back into the shared variables_data directory for reuse across pipeline runs.

Method
------
For ERA5-Land variables, applies one of three temporal aggregation modes depending on variable type: mean state (e.g. soil moisture) computed as a straight temporal mean; 
mean temperature (stl1, t2m) computed as a temporal mean with an automatic, idempotent Kelvin-to-Celsius conversion applied only if the input magnitude indicates Kelvin, plus derived annual degree-day totals above/below 0degC (DDT/DDF); 
and accumulation (ssr, tp), computed as a per-calendar-year sum averaged across years, to correctly represent typical annual totals for flux/accumulation variables rather than a naive full-period sum. 
All aggregations are computed for both the full year and the JJAS (June-September) subset. 
CCI permafrost depth/temperature variables are aggregated via median or mean across the study period. 
Surficial geology is rasterized from a vector polygon layer onto the InSAR reference grid using majority-category encoding, with an accompanying code-to-category lookup table. 
All raster writes explicitly convert NetCDF pixel-center coordinates to pixel-edge bounds and enforce north-up (row 0 = northernmost) orientation.

Inputs
------
- era5land_stl1_daily_grid.nc, era5land_t2m_daily_grid.nc, era5land_ssr_2019_2024_merged.nc,  era5land_swvl1_2019_2024_merged.nc, era5land_tp_2019_2024_merged.nc: ERA5-Land daily climate variables.
- ESACCI-PERMAFROST-L4-ALT/-GTD NetCDF products: active layer thickness and ground temperature at depth.
- geology_tuk_rich_masked.gpkg: surficial geology polygons (field DESC2).
- subsidence_clean.tif (from 01): reference grid for geology rasterization.

Outputs
-------
- era5_<var>_mean_2019_2024.tif / _JJAS_mean_2019_2024.tif: temporal means.
- era5_t2m_ddt/ddf_annual_mean_2019_2024.tif: annual degree-day totals.
- era5_<ssr|tp>_annual_sum_mean_2019_2024.tif / _JJAS_sum_mean_2019_2024.tif: mean annual accumulation totals.
- cci_<var>_<median|mean>_2019_2024.tif: aggregated permafrost depth/temperature layers.
- geology_tuk_rich_DESC2_rasterized_3155.tif, geology_tuk_rich_DESC2_lookup.csv: rasterized geology and its category code lookup.

Dependencies
------------
xarray, numpy, pandas, rasterio, geopandas.

Reproducibility notes
----------------------
- Random seed: not applicable (deterministic aggregation and rasterization).
- Deterministic given identical inputs: Yes. Geology category codes are
  assigned by sorted alphabetical order of unique DESC2 values, so code
  assignment is stable across runs given the same source vector file.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.features import rasterize
from pathlib import Path
import geopandas as gpd

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"E:\Processing\InSAR_multivariate_analysis")
DATA_DIR = PROJECT_DIR / "variables_data"
REF_PATH = PROJECT_DIR / "scripts" / "outputs" / "subsidence_clean.tif"

YEAR_START, YEAR_END = 2019, 2024
SUMMER_MONTHS = [6, 7, 8, 9]
STAT_PREFERENCE = "median"   # "median" or "mean"
GEOLOGY_FIELD = "DESC2"
GTD_DEPTH_VARS = ["GST", "T1m", "T2m", "T5m", "T10m"]

NC_FILES = {
    "stl1":  DATA_DIR / "era5land_stl1_daily_grid.nc",
    "t2m":   DATA_DIR / "era5land_t2m_daily_grid.nc",
    "ssr":   DATA_DIR / "era5land_ssr_2019_2024_merged.nc",
    "swvl1": DATA_DIR / "era5land_swvl1_2019_2024_merged.nc",
    "tp":    DATA_DIR / "era5land_tp_2019_2024_merged.nc",
    "alt":   DATA_DIR / "ESACCI-PERMAFROST-L4-ALT-MODISLST_CRYOGRID-AREA4_PP-2019-2024-fv05.0_merged.nc",
    "gtd":   DATA_DIR / "ESACCI-PERMAFROST-L4-GTD-MODISLST_CRYOGRID-AREA4_PP-2019-2024-fv05.0_merged.nc",
}
GEOLOGY_GPKG = DATA_DIR / "geology_tuk_rich_masked.gpkg"

# ──────────────────────────────────────────────────────────────────────────────
def kelvin_to_celsius(arr):
    """
    Convert Kelvin -> Celsius, but only if the input actually looks like
    Kelvin (mean magnitude > 150). Idempotent / self-correcting regardless
    of whether the source .nc was already pre-converted upstream.
    """
    mean_val = float(np.nanmean(arr.values if hasattr(arr, "values") else arr))
    if mean_val > 150:
        print(f"      [unit check] mean={mean_val:.1f} -> Kelvin, converting to degC")
        return arr - 273.15
    print(f"      [unit check] mean={mean_val:.1f} -> already degC, no conversion applied")
    return arr

def get_time_dim(da):
    for cand in ("time", "valid_time"):
        if cand in da.dims:
            return cand
    raise ValueError(f"No time-like dim found in dims={da.dims}")

def get_lat_lon_names(obj):
    lat_name = [d for d in obj.dims if "lat" in d.lower()][0]
    lon_name = [d for d in obj.dims if "lon" in d.lower()][0]
    return lat_name, lon_name

def annual_sum_mean(da, tdim):
    """
    For accumulation/flux variables: group by calendar year, sum within each year, then average the per-year totals across all years. 
    This gives a "typical annual total" that is robust to years with slightly different day counts / missing days, rather than naively summing everything and dividing by a fixed n_years.
    """
    grouped = da.groupby(f"{tdim}.year").sum(dim=tdim, skipna=True)
    return grouped.mean(dim="year", skipna=True)

def annual_median_mean(da, tdim):
    """
    For swvl1 (soil moisture) - per-year temporal MEDIAN, then averaged across years. 
    Median (not mean, not sum) is used because soil moisture is a bounded volumetric-fraction state variable, not a flux: summing it has no physical meaning, and the median is less sensitive than the mean to short precipitation-driven spikes in the daily time series.
    """
    grouped = da.groupby(f"{tdim}.year").median(dim=tdim, skipna=True)
    return grouped.mean(dim="year", skipna=True)

def write_geotiff_wgs84(arr2d, lats, lons, out_path):
    """Write a 2D array to GeoTIFF, correctly converting pixel-center coordinates (NetCDF convention) to pixel-edge bounds (rasterio convention).
    IMPORTANT: rasterio.transform.from_bounds() always builds a north-up transform, meaning row 0 of the array MUST correspond to the northernmost latitude. 
    If the source array's row 0 is the southernmost latitude (lats ascending), it must be flipped before writing.
    """
    arr = np.array(arr2d, dtype=np.float32)

    if lats[0] < lats[-1]:
        arr = np.flipud(arr)
        lats = lats[::-1]
    res_lat = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.1
    res_lon = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.1

    west  = float(lons.min()) - res_lon / 2
    east  = float(lons.max()) + res_lon / 2
    south = float(lats.min()) - res_lat / 2
    north = float(lats.max()) + res_lat / 2

    transform = from_bounds(west, south, east, north, len(lons), len(lats))

    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        crs=CRS.from_epsg(4326),
        transform=transform,
        width=len(lons),
        height=len(lats),
        count=1,
        dtype="float32",
        nodata=np.nan,
        compress="lzw"
    ) as dst:
        dst.write(arr, 1)
    with rasterio.open(out_path) as check:
        assert check.transform.e < 0, (
            f"{out_path.name}: transform.e={check.transform.e} should be "
            f"negative (north-up convention) - geometry may still be flipped."
        )
    print(f"    saved: {out_path}")

def process_era5_var(nc_path, var_name, out_prefix, mode="mean_state"):
    """
    mode:
      "degree_days"  -> stl1, t2m: NO mean/median/sum of raw temperature is
                        computed or written. Only DDT (thaw degree-days,
                        sum of days > 0degC) and DDF (freeze degree-days,
                        sum of days < 0degC) are produced - the only
                        physically standard way to accumulate temperature.
      "accumulate"   -> ssr, tp: per-year sum averaged across years. True
                        physical accumulation total (flux variables).
      "median_state" -> swvl1: per-year median averaged across years
                        (no unit conversion, no accumulation - soil
                        moisture is a bounded state variable, not a flux).
    """
    print(f"\n[ERA5] {nc_path.name} :: {var_name} (mode={mode})")

    if not nc_path.exists():
        print(f"    warning: file not found, skipped -> {nc_path}")
        return
    ds = xr.open_dataset(nc_path)

    if var_name not in ds.data_vars:
        print(f"    warning: variable '{var_name}' missing, skipped")
        ds.close()
        return
    da = ds[var_name]
    tdim = get_time_dim(da)
    lat_name, lon_name = get_lat_lon_names(da)

    da = da.transpose(tdim, lat_name, lon_name)
    da = da.sel({tdim: slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31")})
    da_summer = da.where(da[tdim].dt.month.isin(SUMMER_MONTHS), drop=True)

    lats = da[lat_name].values
    lons = da[lon_name].values

    if mode == "degree_days":
        da = kelvin_to_celsius(da)
        n_years = YEAR_END - YEAR_START + 1
        ddt = da.where(da > 0, 0).sum(dim=tdim) / n_years
        ddf = (-da).where(da < 0, 0).sum(dim=tdim) / n_years
        write_geotiff_wgs84(ddt.values, lats, lons, DATA_DIR / f"{out_prefix}_ddt_annual_mean_2019_2024.tif")
        write_geotiff_wgs84(ddf.values, lats, lons, DATA_DIR / f"{out_prefix}_ddf_annual_mean_2019_2024.tif")
    elif mode == "accumulate":
        annual_total = annual_sum_mean(da, tdim)
        jjas_total = annual_sum_mean(da_summer, tdim)

        write_geotiff_wgs84(annual_total.values, lats, lons,
                             DATA_DIR / f"{out_prefix}_annual_sum_mean_2019_2024.tif")
        write_geotiff_wgs84(jjas_total.values, lats, lons,
                             DATA_DIR / f"{out_prefix}_JJAS_sum_mean_2019_2024.tif")
    elif mode == "median_state":
        median_all = annual_median_mean(da, tdim)
        median_jjas = annual_median_mean(da_summer, tdim)

        write_geotiff_wgs84(median_all.values, lats, lons, DATA_DIR / f"{out_prefix}_median_2019_2024.tif")
        write_geotiff_wgs84(median_jjas.values, lats, lons, DATA_DIR / f"{out_prefix}_JJAS_median_2019_2024.tif")
    else:
        raise ValueError(f"Unknown mode '{mode}' for variable '{var_name}'")
    ds.close()

def process_depth_nc(nc_path, variables_map, agg):
    print(f"\n[CCI] {nc_path.name} :: {list(variables_map.keys())}")

    if not nc_path.exists():
        print(f"    warning: file not found, skipped -> {nc_path}")
        return
    ds = xr.open_dataset(nc_path)
    lat_name, lon_name = get_lat_lon_names(ds)

    for src_var, out_stub in variables_map.items():
        if src_var not in ds.data_vars:
            print(f"    warning: {src_var} missing, skipped")
            continue
        da = ds[src_var]
        if any(cand in da.dims for cand in ("time", "valid_time")):
            tdim = get_time_dim(da)
            da = da.transpose(tdim, lat_name, lon_name)
            da = da.sel({tdim: slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31")})
            out = da.median(dim=tdim) if agg == "median" else da.mean(dim=tdim)
        else:
            da = da.transpose(lat_name, lon_name)
            out = da
        lats = ds[lat_name].values
        lons = ds[lon_name].values
        write_geotiff_wgs84(out.values, lats, lons, DATA_DIR / f"{out_stub}_{agg}_2019_2024.tif")
    ds.close()

def rasterize_geology():
    print(f"\n[VECTOR] Rasterizing surficial geology from {GEOLOGY_GPKG.name}")
    if not REF_PATH.exists():
        print(f"    warning: reference grid not found, skipped -> {REF_PATH}")
        return
    if not GEOLOGY_GPKG.exists():
        print(f"    warning: geology file not found, skipped -> {GEOLOGY_GPKG}")
        return
    with rasterio.open(REF_PATH) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)
    gdf = gpd.read_file(GEOLOGY_GPKG)
    if GEOLOGY_FIELD not in gdf.columns:
        raise ValueError(f"Field '{GEOLOGY_FIELD}' not found. Available: {list(gdf.columns)}")
    gdf = gdf[[GEOLOGY_FIELD, "geometry"]].dropna(subset=[GEOLOGY_FIELD]).copy()
    gdf = gdf.to_crs(ref_crs)

    categories = sorted(gdf[GEOLOGY_FIELD].astype(str).unique())
    code_map = {cat: i + 1 for i, cat in enumerate(categories)}
    gdf["geo_code"] = gdf[GEOLOGY_FIELD].astype(str).map(code_map)

    shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf["geo_code"])]
    arr = rasterize(
        shapes=shapes,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,
        dtype="int32"
    )

    out_tif = DATA_DIR / "geology_tuk_rich_DESC2_rasterized_3155.tif"
    with rasterio.open(
        out_tif, "w",
        driver="GTiff",
        height=ref_shape[0], width=ref_shape[1],
        count=1, dtype="int32",
        crs=ref_crs, transform=ref_transform,
        nodata=0, compress="lzw"
    ) as dst:
        dst.write(arr, 1)
    out_csv = DATA_DIR / "geology_tuk_rich_DESC2_lookup.csv"
    pd.DataFrame({"code": list(code_map.values()), "DESC2": list(code_map.keys())}).to_csv(out_csv, index=False)

    print(f"    saved: {out_tif}")
    print(f"    saved: {out_csv}")

print("=== 02 Files -> GeoTIFF (v6, temperature = DDT/DDF only) ===")
print(f"Predictor outputs go to variables_data: {DATA_DIR}")
print(f"STAT_PREFERENCE = {STAT_PREFERENCE}")
print(f"ERA5-Land time window = {YEAR_START}-{YEAR_END}, months={SUMMER_MONTHS}")
print("NOTE: stl1/t2m now produce ONLY DDT/DDF - no mean/median/sum temperature layers are written.")

process_era5_var(NC_FILES["stl1"],  "stl1",  "era5_stl1",  mode="degree_days")
process_era5_var(NC_FILES["t2m"],   "t2m",   "era5_t2m",   mode="degree_days")
process_era5_var(NC_FILES["ssr"],   "ssr",   "era5_ssr",   mode="accumulate")
process_era5_var(NC_FILES["swvl1"], "swvl1", "era5_swvl1", mode="median_state")
process_era5_var(NC_FILES["tp"],    "tp",    "era5_tp",    mode="accumulate")

process_depth_nc(NC_FILES["alt"], {"ALT": "cci_alt"}, STAT_PREFERENCE)
process_depth_nc(NC_FILES["gtd"], {v: f"cci_{v.lower()}" for v in GTD_DEPTH_VARS}, STAT_PREFERENCE)

rasterize_geology()

print("\nDone.")
print("This script only writes reusable predictor layers to variables_data.")
print("Model outputs should later go to E:\\Processing\\InSAR_multivariate_analysis\\outputs\\run_nX_YYYYMMDD\\")