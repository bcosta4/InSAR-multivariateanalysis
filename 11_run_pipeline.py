"""
10_run_pipeline.py — MASTER PIPELINE SCRIPT
─────────────────────────────────────────────
Runs the full subsidence analysis pipeline in order.

Usage:
    python 10_run_pipeline.py              # runs all enabled stages
    python 10_run_pipeline.py --from 04    # resume from stage 04
    python 10_run_pipeline.py --only 07    # run only stage 07
    python 10_run_pipeline.py --skip 06    # skip stage 06

Stages:
    01  Prepare reference grid (InSAR -> valid mask + clean subsidence tif)
    02  Convert NetCDF and vector files to GeoTIFF (ERA5, terrain, land cover, etc.)
    03  Align all predictors to InSAR reference grid
    04  Build predictor stack + flat parquet + QC histograms
    05  Multicollinearity + VIF analysis
    06  Spatial autocorrelation (Moran's I, LISA, semivariogram)
    07  XGBoost model (spatial block CV, SHAP, permutation importance)
    08  Spatial models: OLS + SEM + GWR

Log: pipeline_run.log
"""

import subprocess, sys, os, argparse, time
from datetime import datetime
from pathlib import Path

STAGES = {
    "01": {"script": "01_prepare_reference_grid.py",   "desc": "Prepare reference grid"},
    "02": {"script": "02_files_to_geotiff.py",         "desc": "Convert NetCDF and vector files to GeoTIFF"},
    "03": {"script": "03_align_predictors.py",         "desc": "Align predictors to reference grid"},
    "04": {"script": "04_build_predictor_stack.py",    "desc": "Build predictor stack"},
    "05": {"script": "05_qc_aligned_layers.py",    "desc": "Build predictor stack"},
    "06": {"script": "06_multicollinearity_vif.py",    "desc": "Multicollinearity + VIF"},
    "07": {"script": "07_spatial_autocorrelation.py",  "desc": "Spatial autocorrelation"},
    "08": {"script": "08_xgboost_model.py",            "desc": "XGBoost model"},
    "09": {"script": "09_spatial_model_gwr.py",        "desc": "Spatial models: OLS + SEM + GWR"},
    "10": {"script": "10_consolidate_model_comparison.py", "desc": "Consolidate model comparison"},
}

# set False if done
RUN_STAGES = {
    "01": True,
    "02": True,
    "03": True,
    "04": True,
    "05": True,
    "06": True,
    "07": True,
    "08": True,
    "09": True,
    "10": True,}

SCRIPT_DIR = Path(__file__).parent
LOG_PATH   = SCRIPT_DIR / "pipeline_run.log"

parser = argparse.ArgumentParser(description="Subsidence Analysis Pipeline")
parser.add_argument("--from",  dest="from_stage", default=None)
parser.add_argument("--only",  dest="only_stage",  default=None)
parser.add_argument("--skip",  dest="skip_stages", nargs="+", default=[])
args = parser.parse_args()

if args.only_stage:
    for k in RUN_STAGES:
        RUN_STAGES[k] = (k == args.only_stage.zfill(2))
if args.from_stage:
    from_n = int(args.from_stage)
    for k in RUN_STAGES:
        RUN_STAGES[k] = int(k) >= from_n
for sk in args.skip_stages:
    key = sk.zfill(2)
    if key in RUN_STAGES:
        RUN_STAGES[key] = False

def log(msg, logfile):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    logfile.write(line + "\n")
    logfile.flush()

print("=" * 65)
print("  SUBSIDENCE ANALYSIS PIPELINE — TUKTOYAKTUK PENINSULA")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)

results_summary = []

with open(LOG_PATH, "a") as logfile:
    log("=" * 55, logfile)
    log("Pipeline started", logfile)

    for stage_id in sorted(STAGES.keys()):
        cfg     = STAGES[stage_id]
        script  = cfg["script"]
        desc    = cfg["desc"]
        enabled = RUN_STAGES.get(stage_id, False)

        if not enabled:
            msg = f"[SKIP]  Stage {stage_id}: {desc}"
            print(msg)
            log(msg, logfile)
            results_summary.append({"stage": stage_id, "desc": desc,
                                     "status": "SKIPPED", "elapsed": "-"})
            continue

        script_path = SCRIPT_DIR / script
        if not script_path.exists():
            msg = f"[ERROR] Stage {stage_id}: script not found -> {script_path}" 
            log(msg, logfile)
            results_summary.append({"stage": stage_id, "desc": desc,
                                     "status": "MISSING", "elapsed": "-"})
            continue

        log(f"[RUN]   Stage {stage_id}: {desc}", logfile)
        t0 = time.time()
        result = subprocess.run([sys.executable, str(script_path)],
                                capture_output=False)
        elapsed = time.time() - t0
        elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}min"

        if result.returncode == 0:
            status = "OK"
            msg = f"[OK]    Stage {stage_id} completed in {elapsed_str}"
        else:
            status = "FAILED"
            msg = f"[FAIL]  Stage {stage_id} failed after {elapsed_str}"

        log(msg, logfile)
        results_summary.append({"stage": stage_id, "desc": desc,
                                 "status": status, "elapsed": elapsed_str})

        if result.returncode != 0:
            print(f"\n  Pipeline halted. Resume with:")
            print(f"  python 10_run_pipeline.py --from {stage_id}")
            break

    log("Pipeline finished.", logfile)

print("\n" + "=" * 65)
print("  PIPELINE SUMMARY")
print("=" * 65)
print(f"  {'Stage':<8} {'Description':<38} {'Status':<10} {'Time'}")
print(f"  {'-'*7} {'-'*37} {'-'*9} {'-'*10}")
for r in results_summary:
    icon = {"OK": "OK", "FAILED": "X", "SKIPPED": "-", "MISSING": "?"}.get(r["status"], "?")
    print(f"  {r['stage']:<8} {r['desc']:<38} {icon} {r['status']:<8} {r['elapsed']}")
print(f"\n  Log: {LOG_PATH}")
print("=" * 65)