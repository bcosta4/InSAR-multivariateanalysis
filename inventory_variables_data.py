
import os
import glob
import subprocess
import json
import re
import pandas as pd

data_dir = r"E:\Processing\InSAR_multivariate_analysis\variables_data"

def detect_format(path):
    try:
        with open(path, "rb") as f:
            header = f.read(16)
    except Exception as e:
        return f"ERROR: {e}"

    if header[:4] == b"PK\x03\x04":
        return "zip"
    if header[:4] == b"GRIB":
        return "grib"
    if header[:3] == b"CDF":
        return "netcdf3"
    if header[:4] == b"\x89HDF":
        return "netcdf4/hdf5"
    if header[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "tiff/bigtiff"
    if header[:16] == b"SQLite format 3\x00":
        return "gpkg/sqlite"
    if header[:10] == b"<?xml vers" or header[:12] == b"<PAMDataset>":
        return "xml_sidecar"
    if path.lower().endswith(".csv"):
        return "csv"
    return f"unknown (header={header})"


def get_tif_info(path):
    try:
        result = subprocess.run(["gdalinfo", "-json", path], capture_output=True, text=True, timeout=30)
        info = json.loads(result.stdout)
        size = info.get("size", ["?", "?"])
        crs = info.get("coordinateSystem", {}).get("wkt", "")
        crs_short = "EPSG:" + crs.split('ID["EPSG",')[1].split(']')[0] if 'ID["EPSG",' in crs else "unknown"
        bands = info.get("bands", [])
        nodata = bands[0].get("noDataValue") if bands else None
        dtype = bands[0].get("type") if bands else None
        extent = info.get("cornerCoordinates", {})
        return {
            "width_px": size[0], "height_px": size[1],
            "crs": crs_short, "n_bands": len(bands),
            "dtype": dtype, "nodata": nodata,
            "upper_left": extent.get("upperLeft"), "lower_right": extent.get("lowerRight"),
        }
    except Exception as e:
        return {"error": str(e)}


def get_gpkg_info(path):
    try:
        result = subprocess.run(["ogrinfo", "-so", "-al", path], capture_output=True, text=True, timeout=30)
        out = result.stdout
        layers = [l.split(":")[1].strip().split(" (")[0] for l in out.splitlines() if l.startswith("Layer name")]
        feat_count = None
        for l in out.splitlines():
            if "Feature Count" in l:
                feat_count = l.split(":")[1].strip()
                break
        return {"layers": layers, "feature_count": feat_count}
    except Exception as e:
        return {"error": str(e)}


def get_nc_info(path):
    try:
        import xarray as xr
        ds = xr.open_dataset(path)
        info = {"dims": dict(ds.sizes), "data_vars": list(ds.data_vars), "coords": list(ds.coords)}
        ds.close()
        return info
    except Exception as e:
        return {"error": str(e)}


# --- Grouping logic: strip known suffix tokens to derive a base "variable group" name ---
SUFFIX_TOKENS = [
    r"_masked", r"_mean", r"_median", r"_avg", r"_JJAS", r"_2019[-_]?2024",
    r"_2019[-_]?2023", r"_2016[-_]?2024", r"_2015", r"_2020", r"_201924",
    r"_v1_1", r"_v1", r"_v3a", r"_p500", r"_C3", r"_lt5m", r"_250m",
    r"_daily_grid", r"_grid", r"_merged", r"_fixed_v2", r"_fixed_corrected",
    r"_fixed", r"_corrected", r"-201924", r"_2019-2024", r"_2019_2024",
    r"-fv\d+\.\d+", r"-fv\d+", r"_endL_winterYear\d+_Canada", r"_lengthT_winterYear\d+_Canada",
    r"_national-2015-index", r"-national-\d+-index",
]

def get_group_key(filename):
    stem = os.path.splitext(filename)[0]
    if stem.endswith(".aux"):
        stem = os.path.splitext(stem)[0]
    key = stem
    for tok in SUFFIX_TOKENS:
        key = re.sub(tok, "", key, flags=re.IGNORECASE)
    key = re.sub(r"[-_]+$", "", key)
    key = re.sub(r"[-_]{2,}", "_", key)
    return key if key else stem


all_paths = sorted(glob.glob(os.path.join(data_dir, "*")))
files_only = [f for f in all_paths if os.path.isfile(f)]

print(f"Scanning top-level only: {data_dir}")
print(f"Total files found (excluding subfolders): {len(files_only)}\n")

rows = []
for path in files_only:
    fname = os.path.basename(path)
    size_kb = os.path.getsize(path) / 1024
    fmt = detect_format(path)

    row = {"filename": fname, "size_kb": round(size_kb, 1), "format": fmt}

    if fmt == "tiff/bigtiff":
        row.update(get_tif_info(path))
    elif fmt == "gpkg/sqlite":
        row.update(get_gpkg_info(path))
    elif fmt in ("netcdf3", "netcdf4/hdf5"):
        row.update(get_nc_info(path))

    if fmt not in ("xml_sidecar",) and not fname.endswith(".ovr"):
        row["group"] = get_group_key(fname)
    else:
        row["group"] = None

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(data_dir, "variables_inventory_v2.csv"), index=False)

# --- Grouped report ---
data_rows = [r for r in rows if r["group"] is not None]
groups = {}
for r in data_rows:
    groups.setdefault(r["group"], []).append(r)

print("="*110)
print(f"GROUPED VARIABLES ({len(groups)} distinct variable groups)")
print("="*110)

for group_name in sorted(groups.keys()):
    members = groups[group_name]
    flag = "  <-- MULTIPLE VERSIONS, PICK ONE" if len(members) > 1 else ""
    print(f"\n[{group_name}]{flag}")
    for m in members:
        print(f"    {m['filename']:<75} {m['size_kb']:>10.1f} KB   [{m['format']}]")

print("\n" + "="*110)
print("SIDECAR / METADATA FILES (not variables, safe to ignore)")
print("="*110)
sidecars = [r for r in rows if r["group"] is None]
for r in sidecars:
    print(f"  {r['filename']}")

multi_groups = {k: v for k, v in groups.items() if len(v) > 1}
print(f"\n\nSUMMARY: {len(groups)} unique variable groups, {len(multi_groups)} have multiple versions needing a decision.")
print(f"Full CSV saved to: {os.path.join(data_dir, 'variables_inventory_v2.csv')}")
