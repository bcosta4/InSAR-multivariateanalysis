
import os
import glob
import zipfile
import traceback

data_dir = r"E:\Processing\InSAR_multivariate_analysis\variables_data\merged_fixed"

def detect_format(path):
    with open(path, "rb") as f:
        header = f.read(16)
    if header[:4] == b"PK\x03\x04":
        return "zip"
    if header[:4] == b"GRIB":
        return "grib"
    if header[:3] == b"CDF":
        return "netcdf3"
    if header[:4] == b"\x89HDF":
        return "netcdf4/hdf5"
    return f"unknown ({header})"

nc_files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
print(f"Found {len(nc_files)} .nc files in {data_dir}\n")

try:
    import xarray as xr
    HAVE_XARRAY = True
except ImportError:
    HAVE_XARRAY = False
    print("WARNING: xarray not importable in this run.\n")

import numpy as np

report_lines = []

for path in nc_files:
    fname = os.path.basename(path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    fmt = detect_format(path)

    line = f"\n{'='*90}\nFILE: {fname}\n  Size: {size_mb:.2f} MB   Format: {fmt}"
    print(line)
    report_lines.append(line)

    if fmt == "zip":
        msg = "  STATUS: BAD - this is a ZIP archive disguised as .nc. Needs extraction."
        print(msg); report_lines.append(msg)
        continue

    if fmt == "grib":
        engine = "cfgrib"
    elif fmt in ("netcdf3", "netcdf4/hdf5"):
        engine = None
    else:
        msg = "  STATUS: BAD - unrecognized binary format. Likely corrupted/incomplete download."
        print(msg); report_lines.append(msg)
        continue

    if not HAVE_XARRAY:
        continue

    try:
        if engine:
            ds = xr.open_dataset(path, engine=engine)
        else:
            ds = xr.open_dataset(path)

        dims = dict(ds.sizes)
        variables = list(ds.data_vars)
        coords = list(ds.coords)

        msg = f"  STATUS: OK - opened successfully\n  Dimensions: {dims}\n  Coords: {coords}\n  Data variables: {variables}"
        print(msg); report_lines.append(msg)

        # time range check
        if "time" in ds.coords:
            tvals = ds["time"].values
            msg = f"  Time range: {tvals.min()} to {tvals.max()}  (n={len(tvals)})"
            print(msg); report_lines.append(msg)

        # spatial extent check
        for coord_name in ["lat", "latitude", "y"]:
            if coord_name in ds.coords:
                cvals = ds[coord_name].values
                msg = f"  {coord_name} range: {cvals.min():.4f} to {cvals.max():.4f}"
                print(msg); report_lines.append(msg)
        for coord_name in ["lon", "longitude", "x"]:
            if coord_name in ds.coords:
                cvals = ds[coord_name].values
                msg = f"  {coord_name} range: {cvals.min():.4f} to {cvals.max():.4f}"
                print(msg); report_lines.append(msg)

        # data validity check per variable (sample-based to stay fast)
        for var in variables:
            da = ds[var]
            try:
                sample = da.isel({d: slice(0, min(50, s)) for d, s in da.sizes.items()})
                sample_vals = sample.values
                n_total = sample_vals.size
                n_nan = np.isnan(sample_vals).sum() if np.issubdtype(sample_vals.dtype, np.floating) else "N/A"
                vmin = np.nanmin(sample_vals) if np.issubdtype(sample_vals.dtype, np.floating) else sample_vals.min()
                vmax = np.nanmax(sample_vals) if np.issubdtype(sample_vals.dtype, np.floating) else sample_vals.max()
                msg = f"    Var '{var}': dtype={da.dtype}, sample_min={vmin}, sample_max={vmax}, NaN_in_sample={n_nan}/{n_total}"
            except Exception as e:
                msg = f"    Var '{var}': ERROR sampling data - {e}"
            print(msg); report_lines.append(msg)

        ds.close()

    except Exception as e:
        msg = f"  STATUS: BAD - failed to open/read: {type(e).__name__}: {e}"
        print(msg); report_lines.append(msg)

with open(os.path.join(data_dir, "_nc_health_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\n\nFull report also saved to: {os.path.join(data_dir, '_nc_health_report.txt')}")
