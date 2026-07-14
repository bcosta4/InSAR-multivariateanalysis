
r"""
process_era5land_variables.py
──────────────────────────────
Reads all ERA5-Land .nc files in a folder, merges yearly files per
variable, converts units, and derives:
    - DDT (Degree-Days Thawing, cumulative)
    - DDF (Degree-Days Freezing, cumulative)
    - Daily precipitation totals (mm)
    - Daily mean soil moisture
    - Daily mean net solar radiation

Folder expected structure (from your download script):
    era5land_t2m_2019.nc, era5land_t2m_2020.nc, ...
    era5land_stl1_2019.nc, ...
    era5land_swvl1_2019.nc, ...
    era5land_tp_2019.nc, ...
    era5land_ssr_2019.nc, ...

Requirements:
    pip install xarray netCDF4 numpy

Run:
    python process_era5land_variables.py
"""

import os
import glob
import xarray as xr
import numpy as np

FOLDER = r"E:\Processing\InSAR_multivariate_analysis\variables_data"
OUT_FOLDER = os.path.join(FOLDER, "derived")
os.makedirs(OUT_FOLDER, exist_ok=True)

VARS = {
    "t2m":   "2m Air Temperature",
    "stl1":  "Soil Temperature Level 1",
    "swvl1": "Volumetric Soil Water Layer 1",
    "tp":    "Total Precipitation",
    "ssr":   "Surface Net Solar Radiation",
}

def inspect_folder():
    print(f"Scanning: {FOLDER}\n")
    all_nc = sorted(glob.glob(os.path.join(FOLDER, "*.nc")))
    print(f"Found {len(all_nc)} .nc files:")
    for f in all_nc:
        print(f"  {os.path.basename(f)}")
    print()
    return all_nc

def inspect_single_file(path):
    print(f"--- Inspecting: {os.path.basename(path)} ---")
    ds = xr.open_dataset(path)
    print("Variables:", list(ds.data_vars))
    print("Dimensions:", dict(ds.dims))
    print("Coords:", list(ds.coords))
    print(ds)
    ds.close()
    print()

def merge_variable(short_name):
    pattern = os.path.join(FOLDER, f"era5land_{short_name}_*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[SKIP] No files found for variable '{short_name}' (pattern: {pattern})")
        return None
    print(f"Merging {len(files)} files for '{short_name}'...")
    ds = xr.open_mfdataset(files, combine="by_coords")
    return ds

def get_time_dim(ds):
    for candidate in ["time", "valid_time"]:
        if candidate in ds.dims:
            return candidate
    raise ValueError(f"No time dimension found. Available dims: {list(ds.dims)}")

def process_t2m():
    ds = merge_variable("t2m")
    if ds is None:
        return None
    tdim = get_time_dim(ds)
    t2m_c = ds["t2m"] - 273.15
    t2m_daily = t2m_c.resample({tdim: "1D"}).mean()

    ddt = t2m_daily.where(t2m_daily > 0, 0).cumsum(dim=tdim)
    ddf = t2m_daily.where(t2m_daily < 0, 0).cumsum(dim=tdim)

    t2m_daily.to_netcdf(os.path.join(OUT_FOLDER, "t2m_daily_celsius.nc"))
    ddt.to_netcdf(os.path.join(OUT_FOLDER, "ddt.nc"))
    ddf.to_netcdf(os.path.join(OUT_FOLDER, "ddf.nc"))
    print("Saved: t2m_daily_celsius.nc, ddt.nc, ddf.nc\n")
    return t2m_daily

def process_stl1():
    ds = merge_variable("stl1")
    if ds is None:
        return None
    tdim = get_time_dim(ds)
    stl1_c = ds["stl1"] - 273.15
    stl1_daily = stl1_c.resample({tdim: "1D"}).mean()
    stl1_daily.to_netcdf(os.path.join(OUT_FOLDER, "stl1_daily_celsius.nc"))
    print("Saved: stl1_daily_celsius.nc\n")
    return stl1_daily

def process_swvl1():
    ds = merge_variable("swvl1")
    if ds is None:
        return None
    tdim = get_time_dim(ds)
    swvl1_daily = ds["swvl1"].resample({tdim: "1D"}).mean()
    swvl1_daily.to_netcdf(os.path.join(OUT_FOLDER, "swvl1_daily_mean.nc"))
    print("Saved: swvl1_daily_mean.nc\n")
    return swvl1_daily

def process_tp():
    ds = merge_variable("tp")
    if ds is None:
        return None
    tdim = get_time_dim(ds)
    tp_mm = ds["tp"] * 1000.0
    tp_daily = tp_mm.resample({tdim: "1D"}).sum()
    tp_daily.to_netcdf(os.path.join(OUT_FOLDER, "tp_daily_mm.nc"))
    print("Saved: tp_daily_mm.nc\n")
    return tp_daily

def process_ssr():
    ds = merge_variable("ssr")
    if ds is None:
        return None
    tdim = get_time_dim(ds)
    ssr_daily = ds["ssr"].resample({tdim: "1D"}).mean()
    ssr_daily.to_netcdf(os.path.join(OUT_FOLDER, "ssr_daily_mean.nc"))
    print("Saved: ssr_daily_mean.nc\n")
    return ssr_daily

if __name__ == "__main__":
    files = inspect_folder()

    if files:
        inspect_single_file(files[0])

    print("=== Processing T2M -> DDT / DDF ===")
    process_t2m()

    print("=== Processing STL1 ===")
    process_stl1()

    print("=== Processing SWVL1 (Soil Moisture) ===")
    process_swvl1()

    print("=== Processing TP (Precipitation) ===")
    process_tp()

    print("=== Processing SSR (Net Solar Radiation) ===")
    process_ssr()

    print(f"\nAll derived outputs saved to: {OUT_FOLDER}")
