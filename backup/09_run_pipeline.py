"""
09_run_pipeline.py — MASTER PIPELINE SCRIPT
─────────────────────────────────────────────
Runs the full subsidence analysis pipeline in order.
Each stage can be toggled ON/OFF independently via the RUN_STAGES dict.

Usage:
    python 09_run_pipeline.py              # runs all stages
    python 09_run_pipeline.py --from 04   # resume from stage 04
    python 09_run_pipeline.py --only 07   # run only stage 07

Stages:
    01  Setup environment (install packages — run once only)
    02  Prepare reference grid + valid mask
    03  Align all predictors to 27m reference
    04  Build predictor stack + flat parquet
    05  Multicollinearity + VIF analysis
    06  Spatial autocorrelation (Moran's I, LISA, semivariogram)
    07  Random Forest model (GPU/CPU, spatial block CV, SHAP)
    08  Spatial models: OLS + SEM + GWR

Logs:  pipeline_run.log
"""

import subprocess
import sys
import os
import argparse
import time
from datetime import datetime

# ─── STAGE CONFIG ─────────────────────────────────────────────────────────────
STAGES = {
    "01": {"script": "01_setup_environment.py",       "desc": "Setup environment"},
    "02": {"script": "02_prepare_reference_grid.py",  "desc": "Prepare reference grid"},
    "03": {"script": "03_align_predictors.py",        "desc": "Align predictors"},
    "04": {"script": "04_build_predictor_stack.py",   "desc": "Build predictor stack"},
    "05": {"script": "05_multicollinearity_vif.py",   "desc": "Multicollinearity + VIF"},
    "06": {"script": "06_spatial_autocorrelation.py", "desc": "Spatial autocorrelation"},
    "07": {"script": "07_random_forest_model.py",     "desc": "Random Forest model"},
    "08": {"script": "08_spatial_model_gwr.py",       "desc": "OLS + SEM + GWR"},
}

# Toggle stages ON/OFF here — or use CLI args (see Usage above)
RUN_STAGES = {
    "01": False,   # Set True only on first run
    "02": True,
    "03": True,
    "04": True,
    "05": True,
    "06": True,
    "07": True,
    "08": True,
}
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH   = os.path.join(SCRIPT_DIR, "pipeline_run.log")

# ─── CLI argument parsing ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Subsidence Analysis Pipeline")
parser.add_argument("--from",  dest="from_stage", default=None,
                    help="Resume pipeline from this stage number (e.g. 04)")
parser.add_argument("--only",  dest="only_stage",  default=None,
                    help="Run only this stage number (e.g. 07)")
parser.add_argument("--skip",  dest="skip_stages", nargs="+", default=[],
                    help="Skip these stage numbers (e.g. --skip 01 06)")
args = parser.parse_args()

# Apply CLI overrides
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

# ─── Run stages ───────────────────────────────────────────────────────────────
def log(msg, logfile):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    log(f"Pipeline started", logfile)

    for stage_id in sorted(STAGES.keys()):
        cfg      = STAGES[stage_id]
        script   = cfg["script"]
        desc     = cfg["desc"]
        enabled  = RUN_STAGES.get(stage_id, False)

        if not enabled:
            msg = f"[SKIP]  Stage {stage_id}: {desc}"
            print(msg)
            log(msg, logfile)
            results_summary.append({"stage": stage_id, "desc": desc,
                                     "status": "SKIPPED", "elapsed": "-"})
            continue

        script_path = os.path.join(SCRIPT_DIR, script)
        if not os.path.exists(script_path):
            msg = f"[ERROR] Stage {stage_id}: script not found → {script_path}"
            log(msg, logfile)
            results_summary.append({"stage": stage_id, "desc": desc,
                                     "status": "MISSING", "elapsed": "-"})
            continue

        log(f"[RUN]   Stage {stage_id}: {desc}", logfile)
        t0 = time.time()

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False   # let stdout/stderr pass through to console
        )

        elapsed = time.time() - t0
        elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}min"

        if result.returncode == 0:
            status = "OK"
            msg = f"[OK]    Stage {stage_id} completed in {elapsed_str}"
        else:
            status = "FAILED"
            msg = f"[FAIL]  Stage {stage_id} failed (exit code {result.returncode}) after {elapsed_str}"

        log(msg, logfile)
        results_summary.append({"stage": stage_id, "desc": desc,
                                 "status": status, "elapsed": elapsed_str})

        if result.returncode != 0:
            print("\n  Pipeline halted at failed stage.")
            print("  Fix the error above, then resume with:")
            print(f"  python 09_run_pipeline.py --from {stage_id}")
            break

    log("Pipeline finished.", logfile)

# ─── Summary table ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PIPELINE SUMMARY")
print("=" * 65)
print(f"  {'Stage':<8} {'Description':<35} {'Status':<10} {'Time'}")
print(f"  {'-'*7} {'-'*34} {'-'*9} {'-'*10}")
for r in results_summary:
    icon = {"OK": "✓", "FAILED": "✗", "SKIPPED": "–", "MISSING": "?"}.get(r["status"], "?")
    print(f"  {r['stage']:<8} {r['desc']:<35} {icon} {r['status']:<8} {r['elapsed']}")
print(f"\n  Log: {LOG_PATH}")
print("=" * 65)
