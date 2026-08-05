
import xarray as xr

path = r"E:\Processing\InSAR_multivariate_analysis\variables_data\merged_fixed\ssr_fixed.nc"
ds = xr.open_dataset(path)

print("Dims:", dict(ds.sizes))
print("Coords:", list(ds.coords))
print("expver in ds.dims:", "expver" in ds.dims)
print("expver in ds.coords:", "expver" in ds.coords)
if "expver" in ds.coords:
    print("expver values:", ds["expver"].values)
    print("expver dims:", ds["expver"].dims)
