
import xarray as xr
import numpy as np
import os

data_dir = r"E:\Processing\InSAR_multivariate_analysis\variables_data"
output_dir = os.path.join(data_dir, "merged_fixed")
os.makedirs(output_dir, exist_ok=True)

# --- MANUALLY SET THESE based on gdalinfo output for dsw-1984-2023-frequency_masked.tif ---
# Replace these placeholder values with the real corner coordinates in lon/lat (EPSG:4326)
lon_min = -134.5   # <-- update
lon_max = -131.5   # <-- update
lat_min = 69.0     # <-- update
lat_max = 70.0     # <-- update

buffer = 0.05
lon_min -= buffer
lon_max += buffer
lat_min -= buffer
lat_max += buffer

print(f"Target clip extent (with buffer): lon[{lon_min:.4f}, {lon_max:.4f}] lat[{lat_min:.4f}, {lat_max:.4f}]\\n")


def fix_expver(ds):
    if "expver" in ds.dims:
        expver_vals = ds["expver"].values.tolist()
        if 1 in expver_vals and 5 in expver_vals:
            ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
        else:
            ds = ds.isel(expver=0)
        if "expver" in ds.coords:
            ds = ds.drop_vars("expver")
    return ds


def clip_to_extent(ds):
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    if lat_name not in ds.coords or lon_name not in ds.coords:
        print("  WARNING: could not find lat/lon coords to clip.")
        return ds

    lat_vals = ds[lat_name].values
    if lat_vals[0] > lat_vals[-1]:
        ds = ds.sel({lat_name: slice(lat_max, lat_min), lon_name: slice(lon_min, lon_max)})
    else:
        ds = ds.sel({lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})
    return ds


def get_compression_encoding(ds):
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {"zlib": True, "complevel": 4, "dtype": "float32"}
    return encoding


files_to_process = {
    "ssr": os.path.join(data_dir, "era5land_ssr_2019_2024_merged.nc"),
    "swvl1": os.path.join(data_dir, "era5land_swvl1_2019_2024_merged.nc"),
    "tp": os.path.join(data_dir, "era5land_tp_2019_2024_merged.nc"),
    "GTD": os.path.join(data_dir, "ESACCI-PERMAFROST-L4-GTD-MODISLST_CRYOGRID-AREA4_PP-2019-2024-fv05.0_merged.nc"),
    "ALT": os.path.join(data_dir, "ESACCI-PERMAFROST-L4-ALT-MODISLST_CRYOGRID-AREA4_PP-2019-2024-fv05.0_merged.nc"),
}

for var_key, path in files_to_process.items():
    if not os.path.exists(path):
        print(f"[{var_key}] File not found: {path}, skipping.")
        continue

    print(f"\\n=== Processing {var_key} ===")
    ds = xr.open_dataset(path)

    if "expver" in ds.dims:
        print("  Collapsing expver dimension...")
        ds = fix_expver(ds)

    print("  Clipping to reference extent...")
    ds = clip_to_extent(ds)
    print(f"  Resulting dims: {dict(ds.sizes)}")

    encoding = get_compression_encoding(ds)
    out_path = os.path.join(output_dir, f"{var_key}_fixed.nc")
    ds.to_netcdf(out_path, encoding=encoding)

    out_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved -> {out_path}  ({out_size_mb:.2f} MB)")
    ds.close()

print("\\nAll done.")
