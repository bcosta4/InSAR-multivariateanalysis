
import os
import glob
import zipfile

data_dir = r"E:\Processing\InSAR_multivariate_analysis\variables_data"

def detect_format(path):
    try:
        with open(path, "rb") as f:
            header = f.read(16)
    except Exception as e:
        return f"ERROR reading: {e}"

    if header[:4] == b"PK\x03\x04":
        return "zip"
    if header[:4] == b"GRIB":
        return "grib"
    if header[:3] == b"CDF":
        return "netcdf3"
    if header[:4] == b"\x89HDF":
        return "netcdf4/hdf5"
    return f"unknown (header={header})"

print(f"Scanning: {data_dir}\n")

all_files = sorted(glob.glob(os.path.join(data_dir, "*")))
files_only = [f for f in all_files if os.path.isfile(f)]

print(f"Total files found: {len(files_only)}\n")
print(f"{'Filename':<70} {'Size (KB)':>12} {'Format':<15}")
print("-" * 100)

summary = {}

for path in files_only:
    fname = os.path.basename(path)
    size_kb = os.path.getsize(path) / 1024
    fmt = detect_format(path)
    print(f"{fname:<70} {size_kb:>12.1f} {fmt:<15}")

    # group by variable prefix (strip year/extension) for summary
    key = fmt
    summary[key] = summary.get(key, 0) + 1

print("\n--- Format Summary ---")
for fmt, count in summary.items():
    print(f"  {fmt}: {count} file(s)")

# Also list any subdirectories (e.g. merged/, _extract_*, backups)
print("\n--- Subdirectories ---")
dirs_only = [f for f in all_files if os.path.isdir(f)]
for d in dirs_only:
    n_items = len(os.listdir(d))
    print(f"  {os.path.basename(d)}/  ({n_items} items)")

# Flag any .zip_backup files left over from earlier fixes
backups = [f for f in files_only if f.endswith(".zip_backup")]
if backups:
    print("\n--- Leftover .zip_backup files ---")
    for b in backups:
        print(f"  {os.path.basename(b)}")
