
import xarray as xr
import os

data_dir = r"E:\Processing\InSAR_multivariate_analysis\variables_data\merged_fixed"
path = os.path.join(data_dir, "GTD_fixed.nc")
out_path = os.path.join(data_dir, "GTD_fixed_corrected.nc")

ds = xr.open_dataset(path, mask_and_scale=False)  # disable auto-masking on read too

temp_vars = ["GST", "GST_uncertainty", "T1m", "T1m_uncertainty",
             "T2m", "T2m_uncertainty", "T5m", "T5m_uncertainty",
             "T10m", "T10m_uncertainty"]

print("Before fix:")
for var in temp_vars:
    vmin, vmax = float(ds[var].min()), float(ds[var].max())
    print(f"  {var}: min={vmin:.4f}  max={vmax:.4f}")

for var in temp_vars:
    ds[var] = (ds[var] + 273.15) / 0.01
    # Strip stale valid_min/valid_max - they referred to raw scaled integers, no longer applicable
    for attr in ["valid_min", "valid_max", "_FillValue", "missing_value"]:
        if attr in ds[var].attrs:
            del ds[var].attrs[attr]
    ds[var].attrs["units"] = "degrees_C"
    ds[var].attrs["conversion_applied"] = "corrected_double_conversion"

print("\\nAfter fix:")
for var in temp_vars:
    vmin, vmax = float(ds[var].min()), float(ds[var].max())
    print(f"  {var}: min={vmin:.4f}  max={vmax:.4f}")

encoding = {}
for var in ds.data_vars:
    enc = {"zlib": True, "complevel": 4, "dtype": "float32"}
    encoding[var] = enc

ds.to_netcdf(out_path, encoding=encoding)
print(f"\\nSaved corrected file -> {out_path}")

# Verify by reopening fresh
print("\\n--- Verification: reopening saved file ---")
ds_check = xr.open_dataset(out_path)
for var in temp_vars:
    vmin, vmax = float(ds_check[var].min()), float(ds_check[var].max())
    print(f"  {var}: min={vmin:.4f}  max={vmax:.4f}")
