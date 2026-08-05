
import xarray as xr

path = r"E:\Processing\InSAR_multivariate_analysis\variables_data\merged_fixed\GTD_fixed_corrected.nc"
ds = xr.open_dataset(path)

for var in ["GST", "T1m", "T2m", "T5m", "T10m"]:
    vals = ds[var].values
    print(f"{var}: range = {float(vals.max()) - float(vals.min()):.3f} C  (min={float(vals.min()):.3f}, max={float(vals.max()):.3f})")
