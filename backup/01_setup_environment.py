"""
01_setup_environment.py
Installs all required packages for the subsidence analysis pipeline.
Run ONCE after activating the subsidence_analysis conda environment.
Handles both CPU-only fallback and RAPIDS GPU (cuML) detection.
"""

import subprocess
import sys


def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[WARNING] Command returned non-zero exit code: {result.returncode}")
    return result.returncode


# ─── Core geospatial + data stack ──────────────────────────────────────────────
conda_packages = [
    "conda install -y -c conda-forge rasterio geopandas numpy scipy pandas pyarrow",
    "conda install -y -c conda-forge matplotlib seaborn scikit-learn joblib",
    "conda install -y -c conda-forge libpysal esda spreg splot mgwr",
    "conda install -y -c conda-forge tqdm rich",
]

pip_packages = [
    "pip install shap xgboost lightgbm",
    "pip install pysal",
]

print("=" * 60)
print("STEP 1: Installing core geospatial + ML packages")
print("=" * 60)

for cmd in conda_packages:
    run(cmd)

for cmd in pip_packages:
    run(cmd)

# ─── RAPIDS cuML (GPU-accelerated Random Forest) ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Attempting RAPIDS cuML install (RTX 3090 / CUDA 12)")
print("=" * 60)

rapids_cmd = (
    "conda install -y -c rapidsai -c conda-forge -c nvidia "
    "rapids=25.02 python=3.11 cuda-version=12.0"
)

rc = run(rapids_cmd)

if rc != 0:
    print("\n[INFO] RAPIDS full install failed. Trying cuML only...")
    run("pip install cuml-cu12 --extra-index-url https://pypi.nvidia.com")
else:
    print("\n[OK] RAPIDS installed successfully — GPU acceleration enabled.")

# ─── Scalable GWR ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Scalable GWR packages")
print("=" * 60)
run("pip install mgwr")               # standard MGWR (for spatial subsamples)
run("pip install FastGWR || true")    # FastGWR if available on PyPI
run("conda install -y -c conda-forge r-scgwr || true")  # R scgwr fallback

# ─── Verify critical imports ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Verifying critical imports")
print("=" * 60)

critical = [
    "rasterio", "geopandas", "numpy", "pandas", "pyarrow",
    "sklearn", "scipy", "libpysal", "esda", "spreg", "mgwr",
    "shap", "xgboost", "matplotlib",
]

gpu_packages = ["cuml"]

failed = []
for pkg in critical:
    try:
        __import__(pkg)
        print(f"  [OK] {pkg}")
    except ImportError:
        print(f"  [MISSING] {pkg}")
        failed.append(pkg)

print("\nChecking GPU packages (optional):")
for pkg in gpu_packages:
    try:
        __import__(pkg)
        print(f"  [OK-GPU] {pkg}")
    except ImportError:
        print(f"  [NOT AVAILABLE] {pkg} — will fall back to scikit-learn CPU")

if failed:
    print(f"\n[ACTION REQUIRED] Missing critical packages: {failed}")
    print("Run: conda install -c conda-forge " + " ".join(failed))
else:
    print("\nAll critical packages available. Proceed to 02_prepare_reference_grid.py")

