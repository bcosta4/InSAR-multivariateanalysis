
import xarray as xr
import numpy as np
import pandas as pd

path = r"E:\Processing\InSAR_multivariate_analysis\variables_data\merged_fixed\ssr_fixed.nc"
ds = xr.open_dataset(path)

da = ds["ssr"]
# check NaN pattern along time - is it random, seasonal, or a contiguous block?
nan_per_time = np.isnan(da.values).sum(axis=(1, 2))  # sum over lat/lon per timestep
total_per_time = da.shape[1] * da.shape[2]
frac_nan = nan_per_time / total_per_time

times = pd.to_datetime(ds["valid_time"].values)

df = pd.DataFrame({"time": times, "frac_nan": frac_nan})
print("Overall NaN fraction:", frac_nan.mean())
print()
print("First 10 timestamps:")
print(df.head(10))
print()
print("Value counts of frac_nan (rounded):")
print(df["frac_nan"].round(2).value_counts())
print()
# check if NaN pattern correlates with specific hours (e.g. only certain hour-of-day always NaN)
df["hour"] = df["time"].dt.hour
print("Mean frac_nan by hour of day:")
print(df.groupby("hour")["frac_nan"].mean())
