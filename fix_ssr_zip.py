
import zipfile
import os
import shutil

data_dir = r"E:\Processing\InSAR_multivariate_analysis\variables_data"
swvl1_files = [f"era5land_tp_{y}.nc" for y in range(2019, 2025)]

for fname in swvl1_files:
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        print(f"MISSING: {fname}")
        continue

    if zipfile.is_zipfile(path):
        print(f"{fname} IS a zip archive. Extracting...")
        extract_dir = os.path.join(data_dir, f"_extract_{fname}")
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            print("  Contents:", names)
            zf.extractall(extract_dir)

        # Find the actual .nc file inside
        nc_inside = [n for n in names if n.endswith(".nc")]
        if nc_inside:
            extracted_path = os.path.join(extract_dir, nc_inside[0])
            backup_path = path + ".zip_backup"
            shutil.move(path, backup_path)
            shutil.move(extracted_path, path)
            print(f"  Replaced {fname} with unzipped content. Original backed up as {os.path.basename(backup_path)}")
        else:
            print(f"  WARNING: no .nc file found inside zip for {fname}")
    else:
        print(f"{fname} is NOT a zip file (already fine or different issue).")
