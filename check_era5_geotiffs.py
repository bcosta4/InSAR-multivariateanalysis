
import rasterio
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"E:\Processing\InSAR_multivariate_analysis\variables_data")

FILES_TO_CHECK = [
    "era5_t2m_mean_2019_2024.tif",
    "era5_t2m_JJAS_mean_2019_2024.tif",
    "era5_t2m_ddt_annual_mean_2019_2024.tif",
    "era5_t2m_ddf_annual_mean_2019_2024.tif",
    "era5_stl1_mean_2019_2024.tif",
    "era5_stl1_JJAS_mean_2019_2024.tif",
    "era5_stl1_ddt_annual_mean_2019_2024.tif",
    "era5_stl1_ddf_annual_mean_2019_2024.tif",
    "era5_ssr_annual_sum_mean_2019_2024.tif",
    "era5_ssr_JJAS_sum_mean_2019_2024.tif",
    "era5_tp_annual_sum_mean_2019_2024.tif",
    "era5_tp_JJAS_sum_mean_2019_2024.tif",
    "era5_swvl1_mean_2019_2024.tif",
    "era5_swvl1_JJAS_mean_2019_2024.tif",
]

def check_raster(path):
    if not path.exists():
        return {"file": path.name, "status": "MISSING", "shape": None, "dtype": None,
                "n_valid": None, "pct_valid": None, "min": None, "max": None,
                "n_unique": None, "flag": "FILE NOT FOUND"}
    try:
        with rasterio.open(path) as src:
            arr = src.read(1)
            n_total = arr.size
            valid_mask = ~np.isnan(arr)
            n_valid = int(np.sum(valid_mask))
            pct_valid = round(100 * n_valid / n_total, 1) if n_total else 0
            if n_valid == 0:
                return {"file": path.name, "status": "OK", "shape": arr.shape, "dtype": str(arr.dtype),
                        "n_valid": 0, "pct_valid": 0.0, "min": None, "max": None,
                        "n_unique": 0, "flag": "ALL NAN - EMPTY"}
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
            n_unique = int(len(np.unique(arr[valid_mask])))

            flags = []
            if n_unique == 1:
                flags.append("CONSTANT")
            elif n_unique <= 3:
                flags.append("NEAR_CONSTANT")
            if vmin == vmax:
                flags.append("MIN_EQ_MAX")
            if "ddt" in path.name.lower() and vmax <= 0:
                flags.append("DDT<=0 (SUSPECT - Kelvin bug not fixed)")
            if "ddf" in path.name.lower() and vmax <= 0:
                flags.append("DDF<=0 (SUSPECT)")
            if ("t2m" in path.name.lower() or "stl1" in path.name.lower()) and "ddt" not in path.name.lower() and "ddf" not in path.name.lower():
                if vmin < -100 or vmax < -100:
                    flags.append("TEMP LOOKS LIKE KELVIN NOT degC")
            if pct_valid < 50:
                flags.append(f"LOW_VALID_PCT({pct_valid}%)")

            flag_str = "; ".join(flags) if flags else "OK"

            return {"file": path.name, "status": "OK", "shape": arr.shape, "dtype": str(arr.dtype),
                    "n_valid": n_valid, "pct_valid": pct_valid, "min": round(vmin, 4), "max": round(vmax, 4),
                    "n_unique": n_unique, "flag": flag_str}
    except Exception as e:
        return {"file": path.name, "status": "ERROR", "shape": None, "dtype": None,
                "n_valid": None, "pct_valid": None, "min": None, "max": None,
                "n_unique": None, "flag": f"EXCEPTION: {type(e).__name__}: {e}"}

results = [check_raster(DATA_DIR / f) for f in FILES_TO_CHECK]

import pandas as pd
df = pd.DataFrame(results)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 200)
print(df.to_string(index=False))

df.to_csv("era5_geotiff_qc_check.csv", index=False)
