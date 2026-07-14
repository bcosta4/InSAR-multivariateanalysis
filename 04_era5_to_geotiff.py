"""
04_era5_to_geotiff.py
──────────────────────
Processes ERA5-Land NetCDF files into analysis-ready GeoTIFFs:
  - era5land_stl1_daily_grid.nc  → Soil Temperature Level 1 (0–7 cm)
  - era5land_t2m_daily_grid.nc   → 2m air temperature

Derived variables computed here:
  - Mean annual STL1 (proxy for near-surface ground temperature)
  - Mean annual T2M
  - Degree-Day of Thawing (DDT): sum of daily T2M > 0°C (JJA months)
  - Degree-Day of Freezing (DDF): sum of |daily T2M| where T2M < 0°C

Outputs written to ./outputs/era5/
"""

import numpy as np, xarray as xr, rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
STL1_NC  = r"E:\Processing\InSAR_multivariate_analysis\variables_data\era5land_stl1_daily_grid.nc"
T2M_NC   = r"E:\Processing\InSAR_multivariate_analysis\variables_data\era5land_t2m_daily_grid.nc"

# ERA5-Land variable names (check with xr.open_dataset().data_vars if needed)
STL1_VAR = "stl1"   # soil temperature level 1, Kelvin
T2M_VAR  = "t2m"    # 2-metre temperature, Kelvin

# Time slice for averaging (adjust to match your InSAR acquisition window)
YEAR_START = 2016
YEAR_END   = 2022

OUT_DIR  = Path(__file__).parent / "outputs" / "era5"
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
print("=== 04 ERA5-Land → GeoTIFF ===")

def kelvin_to_celsius(arr):
    return arr - 273.15

def nc_to_geotiff(nc_path, var_name, label, transforms):
    """
    Load a NetCDF, apply temporal transforms, write each result as GeoTIFF.
    transforms: list of (name, func) where func(DataArray) → 2D numpy array
    """
    print(f"\n  Loading {Path(nc_path).name} [{var_name}]")
    ds = xr.open_dataset(nc_path)
    da = ds[var_name].sel(time=slice(f"{YEAR_START}", f"{YEAR_END}"))

    # Detect lat/lon dimension names
    lat_name = [d for d in da.dims if "lat" in d.lower()][0]
    lon_name = [d for d in da.dims if "lon" in d.lower()][0]

    lats = da[lat_name].values
    lons = da[lon_name].values
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    # ERA5 is WGS84
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, len(lons), len(lats))
    crs = CRS.from_epsg(4326)

    for out_name, func in transforms:
        result = func(da).astype(np.float32)
        # ERA5 lat is usually descending — flip if needed
        if lats[0] > lats[-1]:
            result = np.flipud(result)

        out_path = OUT_DIR / f"{out_name}.tif"
        with rasterio.open(out_path, "w",
                           driver="GTiff", crs=crs, transform=transform,
                           width=len(lons), height=len(lats),
                           count=1, dtype="float32", nodata=np.nan,
                           compress="lzw") as dst:
            dst.write(result, 1)
        print(f"    Saved {out_path.name}  (shape {result.shape})")

# ── STL1 transforms ───────────────────────────────────────────────────────────
stl1_transforms = [
    ("era5_stl1_mean_celsius",
     lambda da: kelvin_to_celsius(da.mean(dim="time").values)),
    ("era5_stl1_summer_mean_celsius",          # June-July-Aug mean
     lambda da: kelvin_to_celsius(
         da.sel(time=da.time.dt.month.isin([6, 7, 8])).mean(dim="time").values)),
]
nc_to_geotiff(STL1_NC, STL1_VAR, "STL1", stl1_transforms)

# ── T2M transforms ────────────────────────────────────────────────────────────
def compute_ddt(da):
    """Degree-Days of Thawing: sum of positive daily T2M"""
    tc = kelvin_to_celsius(da)
    positive = tc.where(tc > 0, 0)
    return positive.sum(dim="time").values / (YEAR_END - YEAR_START + 1)  # annual mean

def compute_ddf(da):
    """Degree-Days of Freezing: sum of |negative daily T2M|"""
    tc = kelvin_to_celsius(da)
    negative = (-tc).where(tc < 0, 0)
    return negative.sum(dim="time").values / (YEAR_END - YEAR_START + 1)

t2m_transforms = [
    ("era5_t2m_mean_celsius",
     lambda da: kelvin_to_celsius(da.mean(dim="time").values)),
    ("era5_ddt_annual_mean",    compute_ddt),
    ("era5_ddf_annual_mean",    compute_ddf),
    ("era5_t2m_summer_mean",
     lambda da: kelvin_to_celsius(
         da.sel(time=da.time.dt.month.isin([6, 7, 8])).mean(dim="time").values)),
]
nc_to_geotiff(T2M_NC, T2M_VAR, "T2M", t2m_transforms)

print("\n  All ERA5 GeoTIFFs saved to:", OUT_DIR)
print("  IMPORTANT: Now add these paths to 03_align_predictors.py VARIABLES dict,")
print("  then re-run script 03 to align them to the InSAR grid.")
print("  Done.")
