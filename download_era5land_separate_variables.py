
r"""
download_era5land_separate_variables.py
─────────────────────────────────────────
Downloads each ERA5-Land climate variable as its OWN separate
NetCDF file, one per year, to stay under CDS request size limits.

Output files (one per variable per year), e.g.:
    era5land_t2m_2019.nc, era5land_t2m_2020.nc, ...
    era5land_stl1_2019.nc, ...
    era5land_swvl1_2019.nc, ...
    era5land_tp_2019.nc, ...
    era5land_ssr_2019.nc, ...

Requirements:
    pip install cdsapi

Setup (one-time):
    Create a file at C:\Users\<you>\.cdsapirc containing:
       url: https://cds.climate.copernicus.eu/api
       key: <PASTE-YOUR-TOKEN-HERE>

Run:
    python download_era5land_separate_variables.py

Note: CDS enforces a strict item-count limit per request
      (roughly 1000-12000 fields). Splitting by year keeps
      each request well within that limit. After downloading,
      you can merge the yearly files per variable using:
      xr.open_mfdataset() or cdo mergetime.
"""

import cdsapi

# ── AOI: small polygon around the study area ───────────────────────
# Polygon bounds: lon -134.142 to -132.011, lat 69.261 to 69.816
AOI_NORTH = 69.816
AOI_WEST  = -134.143
AOI_SOUTH = 69.260
AOI_EAST  = -132.010

START_YEAR = 2019
END_YEAR   = 2024
# ────────────────────────────────────────────────────────────────────

MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{h:02d}:00" for h in range(0, 24, 6)]  # every 6 hours

AREA = [AOI_NORTH, AOI_WEST, AOI_SOUTH, AOI_EAST]  # N, W, S, E

VARIABLES = {
    "2m_temperature":                "t2m",
    "soil_temperature_level_1":      "stl1",
    "volumetric_soil_water_layer_1": "swvl1",
    "total_precipitation":           "tp",
    "surface_net_solar_radiation":   "ssr",
}

def main():
    c = cdsapi.Client()

    for var_name, short_name in VARIABLES.items():
        for year in range(START_YEAR, END_YEAR + 1):
            out_file = f"era5land_{short_name}_{year}.nc"
            print(f"\nRequesting {var_name} for {year} -> {out_file}")
            c.retrieve(
                "reanalysis-era5-land",
                {
                    "variable": [var_name],
                    "year": [str(year)],
                    "month": MONTHS,
                    "day": DAYS,
                    "time": TIMES,
                    "area": AREA,
                    "format": "netcdf",
                },
                out_file,
            )
            print(f"Done: {out_file}")

    print("\nAll variables downloaded as yearly files.")
    print("To merge each variable's yearly files into one, use xarray:")
    print('  import xarray as xr')
    print('  ds = xr.open_mfdataset("era5land_t2m_*.nc", combine="by_coords")')
    print('  ds.to_netcdf("era5land_t2m_daily_grid.nc")')

if __name__ == "__main__":
    main()
