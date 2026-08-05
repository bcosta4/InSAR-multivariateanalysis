
import xarray as xr
import numpy as np
import os

data_dir = r"E:\Processing\InSAR_multivariate_analysis\variables_data\merged_fixed"

files = {
    "ssr": "ssr_fixed.nc",
    "swvl1": "swvl1_fixed.nc",
    "tp": "tp_fixed.nc",
}

for var_key, fname in files.items():
    path = os.path.join(data_dir, fname)
    print(f"\\n=== {var_key} ===")
    ds = xr.open_dataset(path)

    print("  Before:", dict(ds.sizes), "NaN count var:", int(np.isnan(ds[var_key].values).sum()))

    if "expver" in ds.coords or "expver" in ds.dims:
        # Handle expver whether it's a real dimension or just a coordinate on valid_time
        if "expver" in ds.dims:
            expver_vals = ds["expver"].values.tolist()
            if 1 in expver_vals and 5 in expver_vals:
                ds_fixed = ds.sel(expver=1).combine_first(ds.sel(expver=5))
            else:
                ds_fixed = ds.isel(expver=0)
            ds_fixed = ds_fixed.drop_vars("expver", errors="ignore")
        else:
            # expver is a non-dim coordinate (e.g. aligned with valid_time) - just drop it
            # and instead fix NaNs by dropping/ffill along valid_time where data is truly missing
            ds_fixed = ds.drop_vars("expver", errors="ignore")
            # try ffill/bfill along valid_time to fill legitimate expver-split gaps
            ds_fixed[var_key] = ds_fixed[var_key].ffill(dim="valid_time").bfill(dim="valid_time")
    else:
        ds_fixed = ds

    nan_after = int(np.isnan(ds_fixed[var_key].values).sum())
    print("  After:", dict(ds_fixed.sizes), "NaN count var:", nan_after)

    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds_fixed.data_vars}
    out_path = os.path.join(data_dir, f"{var_key}_fixed_v2.nc")
    ds_fixed.to_netcdf(out_path, encoding=encoding)
    print(f"  Saved -> {out_path}")
    ds.close()

print("\\nDone.")
