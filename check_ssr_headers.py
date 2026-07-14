
import os

files = [
    r"E:\Processing\InSAR_multivariate_analysis\variables_data\era5land_ssr_2019.nc",
    r"E:\Processing\InSAR_multivariate_analysis\variables_data\era5land_ssr_2020.nc",
]

for path in files:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        header = f.read(16)
    print(path)
    print("  size (bytes):", size)
    print("  first 16 bytes:", header)
    print()
