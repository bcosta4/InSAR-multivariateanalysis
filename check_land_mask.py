
import xarray as xr
import numpy as np

path = r"E:\Processing\InSAR_multivariate_analysis\variables_data\merged_fixed\ssr_fixed.nc"
ds = xr.open_dataset(path)

# Since NaN pattern is identical every timestep, take one time slice as the mask
land_mask = ~np.isnan(ds["ssr"].isel(valid_time=0).values)

print("Grid shape:", land_mask.shape)
print("Land pixels:", land_mask.sum())
print("Ocean/NaN pixels:", (~land_mask).sum())
print()
print("Land mask (True=land, False=ocean/NaN):")
print(land_mask)

lats = ds["latitude"].values
lons = ds["longitude"].values
print()
print("Latitudes:", lats)
print("Longitudes:", lons)
