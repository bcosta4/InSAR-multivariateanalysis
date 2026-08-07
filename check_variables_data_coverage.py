r"""
check_variables_data_coverage.py
─────────────────────────────────
Purpose
-------
Standalone audit script: lists every .tif file physically present in variables_data, then cross-references it against every file path referenced in the VARIABLES dict inside 03_align_predictors.py. 
Surfaces two classes of drift that silently break the pipeline:
  1. ORPHANED ON DISK - a .tif exists in variables_data but is NOT referenced by any VARIABLES entry (e.g. new v6 ERA5 outputs generated but the align script was never updated to point at them).
  2. BROKEN REFERENCE - a VARIABLES entry points at a path that does NOT exist on disk (e.g. old filename left behind after a rename), matching the exact bug found in era5_stl1_mean / era5_t2m_mean / etc.

This does not require running the full pipeline - it only reads directory listings and the VARIABLES dict definition, so it is safe to run at any time as a quick sanity check before or after regenerating any input rasters.

Usage
-----
Place this file anywhere and run directly:
    python check_variables_data_coverage.py

Or import ALIGN_SCRIPT_PATH override if 03_align_predictors.py has moved.

Outputs
-------
- Console report: orphaned files, broken references, and a coverage summary.
- variables_data_coverage_report.csv: one row per file/entry with status.

Dependencies
------------
None beyond the standard library (pathlib, re, importlib, csv).
"""

import re
import csv
import sys
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"E:\Processing\InSAR_multivariate_analysis")
DATA_DIR = PROJECT_DIR / "variables_data"
ALIGN_SCRIPT_PATH = PROJECT_DIR / "scripts" / "03_align_predictors.py"  # adjust if elsewhere
OUT_CSV = Path("variables_data_coverage_report.csv")
# ──────────────────────────────────────────────────────────────────────────

def extract_variables_dict(script_path):
    """Parses 03_align_predictors.py as text and extracts every quoted .tif path referenced inside the VARIABLES dict block, along with the variable key it belongs to. 
    Avoids importing the script directly since it executes reproject() logic and DATA_DIR-relative pick() calls at import time."""
    text = script_path.read_text(encoding="utf-8")

    m = re.search(r"VARIABLES\s*=\s*\{(.*?)\n\}\n", text, re.DOTALL)
    if not m:
        raise RuntimeError("Could not locate VARIABLES = {...} block in script.")
    block = m.group(1)

    entries = re.findall(r'"([a-zA-Z0-9_]+)"\s*:\s*\{[^}]*?"path"\s*:\s*([^,]+),', block, re.DOTALL)

    var_to_paths = {}
    for var_name, path_expr in entries:
        filenames = re.findall(r'"([^"]+\.tif)"', path_expr)
        var_to_paths[var_name] = filenames
    return var_to_paths

def main():
    if not DATA_DIR.exists():
        print(f"[ERROR] variables_data directory not found: {DATA_DIR}")
        sys.exit(1)
    if not ALIGN_SCRIPT_PATH.exists():
        print(f"[ERROR] 03_align_predictors.py not found at: {ALIGN_SCRIPT_PATH}")
        print("        Edit ALIGN_SCRIPT_PATH at the top of this script to point at it.")
        sys.exit(1)

    disk_files = sorted(p.name for p in DATA_DIR.glob("*.tif"))
    print(f"=== variables_data coverage check ===")
    print(f"Data dir   : {DATA_DIR}")
    print(f"Align script: {ALIGN_SCRIPT_PATH}")
    print(f"Found {len(disk_files)} .tif files on disk.\n")

    var_to_paths = extract_variables_dict(ALIGN_SCRIPT_PATH)
    print(f"VARIABLES dict has {len(var_to_paths)} entries.\n")

    referenced_filenames = set()
    broken_refs = []
    for var_name, filenames in var_to_paths.items():
        if not filenames:
            print(f"[WARN] Could not resolve filename for '{var_name}' (pick()/f-string too complex to parse statically).")
            continue
        for fn in filenames:
            referenced_filenames.add(fn)
            if not (DATA_DIR / fn).exists():
                broken_refs.append((var_name, fn))

    orphaned = [f for f in disk_files if f not in referenced_filenames]

    print(f"--- BROKEN REFERENCES ({len(broken_refs)}) ---")
    print("VARIABLES entries pointing at files that do NOT exist on disk:")
    if broken_refs:
        for var_name, fn in broken_refs:
            print(f"  [MISSING] {var_name:<28} -> {fn}")
    else:
        print("  None. All referenced paths resolve to real files.")

    print(f"\n--- ORPHANED ON DISK ({len(orphaned)}) ---")
    print("Files present in variables_data but NOT referenced by any VARIABLES entry:")
    if orphaned:
        for fn in orphaned:
            print(f"  [UNUSED] {fn}")
    else:
        print("  None. Every file on disk is referenced by some VARIABLES entry.")

    n_used = len(referenced_filenames & set(disk_files))
    print(f"\n--- SUMMARY ---")
    print(f"Files on disk               : {len(disk_files)}")
    print(f"VARIABLES entries           : {len(var_to_paths)}")
    print(f"Distinct filenames referenced: {len(referenced_filenames)}")
    print(f"Actually used (exist + referenced): {n_used}")
    print(f"Broken references (referenced but missing): {len(broken_refs)}")
    print(f"Orphaned on disk (exist but unreferenced) : {len(orphaned)}")

    rows = []
    for fn in disk_files:
        status = "USED" if fn in referenced_filenames else "ORPHANED_ON_DISK"
        rows.append({"filename": fn, "status": status, "variable_key": ""})
    used_lookup = {fn: var for var, fns in var_to_paths.items() for fn in fns}
    for row in rows:
        if row["status"] == "USED":
            row["variable_key"] = used_lookup.get(row["filename"], "")
    for var_name, fn in broken_refs:
        rows.append({"filename": fn, "status": "BROKEN_REFERENCE_MISSING_ON_DISK", "variable_key": var_name})

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "status", "variable_key"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDetailed CSV saved to: {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()