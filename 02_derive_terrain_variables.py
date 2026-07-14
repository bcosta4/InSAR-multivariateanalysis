"""
02_derive_terrain_variables.py
───────────────────────────────
Derives all terrain variables from the 2m Arctic DEM using GDAL + WhiteboxTools:
  - Slope (already have, but re-derived here at 2m for consistency)
  - Aspect → split into sin(aspect) and cos(aspect)
  - Profile curvature
  - Plan curvature
  - TWI (Topographic Wetness Index)  ← requires WhiteboxTools
  - TPI (Topographic Position Index)
  - TRI (Terrain Ruggedness Index)

All outputs written to ./outputs/terrain/ at 2m, EPSG:3155.
Script 03 will reproject all of these to the InSAR 27m reference grid.

Dependencies: gdal (via conda), whitebox (pip install whitebox)
"""

import os, subprocess, numpy as np, rasterio
from pathlib import Path
from math import pi

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DEM_PATH   = r"E:\Processing\InSAR_multivariate_analysis\variables_data\HRDEM_dsm_2m_ArcticDEM_merged.tif"
OUT_DIR    = Path(__file__).parent / "outputs" / "terrain"

USE_WHITEBOX = True   # set False if whitebox is not installed; TWI will be skipped
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
dem = Path(DEM_PATH)

print("=== 02 Derive Terrain Variables ===")
print(f"  DEM : {dem}")


def gdal_dem(mode, out_name, extra_args=""):
    out = str(OUT_DIR / out_name)
    cmd = f'gdaldem {mode} "{dem}" "{out}" -compute_edges {extra_args} -of GTiff -co COMPRESS=LZW'
    print(f"  gdaldem {mode} → {out_name}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    ERROR: {r.stderr.strip()}")
    return out


# ── 1. Slope (degrees) ───────────────────────────────────────────────────────
slope_path = gdal_dem("slope", "slope_2m.tif")

# ── 2. Aspect (degrees) ──────────────────────────────────────────────────────
aspect_raw = gdal_dem("aspect", "aspect_raw_2m.tif", "-zero_for_flat")

# Split aspect into sin/cos components (avoids the 0°/360° discontinuity)
print("  Splitting aspect → sin/cos components")
with rasterio.open(aspect_raw) as src:
    asp  = src.read(1).astype(np.float32)
    meta = src.meta.copy()

nodata_mask = (asp < 0)
asp_rad  = np.deg2rad(asp)
sin_asp  = np.where(nodata_mask, np.nan, np.sin(asp_rad)).astype(np.float32)
cos_asp  = np.where(nodata_mask, np.nan, np.cos(asp_rad)).astype(np.float32)

meta.update(dtype="float32", nodata=np.nan)
for arr, name in [(sin_asp, "aspect_sin_2m.tif"), (cos_asp, "aspect_cos_2m.tif")]:
    with rasterio.open(OUT_DIR / name, "w", **meta) as dst:
        dst.write(arr, 1)
    print(f"    Saved {name}")

# ── 3. Profile + Plan curvature ───────────────────────────────────────────────
# gdaldem does not support curvature; use WhiteboxTools
if USE_WHITEBOX:
    try:
        import whitebox
        wbt = whitebox.WhiteboxTools()
        wbt.work_dir = str(OUT_DIR)
        wbt.verbose_mode = False

        print("  WhiteboxTools: profile curvature")
        wbt.profile_curvature(
            dem=DEM_PATH,
            output=str(OUT_DIR / "curvature_profile_2m.tif")
        )

        print("  WhiteboxTools: plan curvature")
        wbt.plan_curvature(
            dem=DEM_PATH,
            output=str(OUT_DIR / "curvature_plan_2m.tif")
        )

        print("  WhiteboxTools: TWI")
        # TWI needs filled DEM + flow accumulation first
        dem_filled = str(OUT_DIR / "dem_filled.tif")
        flow_dir   = str(OUT_DIR / "flow_dir.tif")
        flow_acc   = str(OUT_DIR / "flow_acc.tif")

        wbt.fill_depressions(dem=DEM_PATH, output=dem_filled)
        wbt.d8_pointer(dem=dem_filled, output=flow_dir)
        wbt.d8_flow_accumulation(input=dem_filled, output=flow_acc)
        wbt.wetness_index(
            sca=flow_acc,
            slope=slope_path,
            output=str(OUT_DIR / "twi_2m.tif")
        )
        print("  WhiteboxTools: complete")

    except ImportError:
        print("  WARNING: whitebox not installed — curvature and TWI skipped.")
        print("  Install with:  pip install whitebox")
    except Exception as e:
        print(f"  WARNING: WhiteboxTools error: {e}")

else:
    print("  Skipping WhiteboxTools (USE_WHITEBOX=False)")

# ── 4. TPI and TRI (native gdaldem) ──────────────────────────────────────────
gdal_dem("TPI",       "tpi_2m.tif")
gdal_dem("TRI",       "tri_2m.tif")
gdal_dem("roughness", "roughness_2m.tif")

print("\n  All terrain products written to:", OUT_DIR)
print("  Done. → Next: run 03_align_predictors.py")
