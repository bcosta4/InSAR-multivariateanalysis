
import xarray as xr
import glob
import re
import os

data_dir = r"E:\Processing\InSAR_multivariate_analysis\variables_data"
output_dir = r"E:\Processing\InSAR_multivariate_analysis\variables_data\merged"
os.makedirs(output_dir, exist_ok=True)

def convert_gtd_vars(ds):
    for var in ["GST", "T1m", "T2m", "T5m", "T10m",
                "GST_uncertainty", "T1m_uncertainty", "T2m_uncertainty",
                "T5m_uncertainty", "T10m_uncertainty"]:
        if var in ds:
            ds[var] = ds[var] * 0.01 - 273.15
            ds[var].attrs["units"] = "degrees_C"
            ds[var].attrs["conversion_applied"] = "x*0.01-273.15"
    return ds

variables = {
    "ALT": ("ESACCI-PERMAFROST-L4-ALT-MODISLST_CRYOGRID-AREA4_PP-{year}-fv05.0.nc", None),
    "GTD": ("ESACCI-PERMAFROST-L4-GTD-MODISLST_CRYOGRID-AREA4_PP-{year}-fv05.0.nc", convert_gtd_vars),
    "ssr": ("era5land_ssr_{year}.nc", None),
    "swvl1": ("era5land_swvl1_{year}.nc", None),
    "tp": ("era5land_tp_{year}.nc", None),
}

def check_format(path):
    """Return 'netcdf' or 'grib' or 'unknown' based on file header bytes."""
    with open(path, "rb") as f:
        header = f.read(8)
    if header[:4] == b"GRIB":
        return "grib"
    if header[:3] == b"CDF" or header[:4] == b"\x89HDF":
        return "netcdf"
    return "unknown"

for var_key, (pattern, preprocess_fn) in variables.items():
    glob_pattern = os.path.join(data_dir, pattern.format(year="*"))
    all_matches = glob.glob(glob_pattern)

    # Matches years preceded by either underscore or hyphen, right before .nc or -fv05.0.nc
    files = sorted([f for f in all_matches if re.search(r"[_-]\d{4}(-fv05\.0)?\.nc$", f)])

    print(f"\n=== {var_key} ===")
    print(f"Merging {len(files)} files:")
    for f in files:
        fmt = check_format(f)
        print(f"  {os.path.basename(f)}  [{fmt}]")

    if not files:
        print(f"  WARNING: no files found for {var_key}, skipping.")
        continue

    formats = set(check_format(f) for f in files)
    if "grib" in formats:
        print(f"  WARNING: {var_key} contains GRIB file(s). Using engine='cfgrib'.")
        try:
            ds = xr.open_mfdataset(files, engine="cfgrib", combine="by_coords")
        except Exception as e:
            print(f"  ERROR opening as GRIB: {e}")
            print(f"  Install cfgrib with: conda install -c conda-forge cfgrib")
            continue
    else:
        if preprocess_fn:
            ds = xr.open_mfdataset(files, combine="by_coords", preprocess=preprocess_fn)
        else:
            ds = xr.open_mfdataset(files, combine="by_coords")

    out_path = os.path.join(output_dir, f"{var_key}_merged.nc")
    ds.to_netcdf(out_path)
    print(f"  Saved -> {out_path}")
    ds.close()

print("\nAll variables processed.")
