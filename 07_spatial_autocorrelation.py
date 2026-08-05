r"""
07_spatial_autocorrelation.py
──────────────────────────────
Spatial autocorrelation analysis on InSAR subsidence rates:
  1. Global Moran's I (systematic grid subsample, preserves spatial adjacency)
  2. LISA cluster map (High-High, Low-Low, outliers) with FDR correction
     for multiple testing (Benjamini-Hochberg, via statsmodels)
  3. Empirical semivariogram (range, sill, nugget estimation) in real-world
     units (metres), fit with an exponential model via scikit-gstat

Notes on design choices:
- Random pixel subsampling was REPLACED with systematic (regular-grid) thinning.
  Randomly dropping pixels before building KNN weights breaks true spatial
  adjacency (neighbors are no longer physically adjacent), which distorts
  Moran's I. Systematic thinning (every Nth row/col) preserves the grid
  structure while still reducing computational cost.
- LISA local p-values are corrected for multiple testing via Benjamini-Hochberg
  FDR (this is the standard correction recommended by Anselin for LISA maps;
  without it, thousands of simultaneous local tests produce many false
  "significant" clusters by chance alone).
- The semivariogram is computed on real-world distances (metres), not pixel
  indices, using the reference raster's pixel size, so the fitted range
  parameter is scientifically interpretable (e.g. "spatial autocorrelation
  extends to ~450m") and can be directly used to justify GWR/kriging
  bandwidth choices downstream.
"""

import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import libpysal
import esda
from splot.esda import moran_scatterplot
from statsmodels.stats.multitest import fdrcorrection
import skgstat as skg
import rasterio
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"E:\Processing\InSAR_multivariate_analysis")
PIPE_OUT = PROJECT_DIR / "outputs"
REF_PATH = PROJECT_DIR / "scripts" / "outputs" / "subsidence_clean.tif"
RUN_DIR = None          # None = auto-pick latest run_nX_YYYYMMDD
K_NEIGHBORS = 8
GRID_THIN_STEP = None    # None = auto-computed to target ~SAMPLE_TARGET pixels
SAMPLE_TARGET = 30_000
FDR_ALPHA = 0.05
LISA_PERMUTATIONS = 9999
SEMIVARIOGRAM_N_LAGS = 25
SEMIVARIOGRAM_SAMPLE = 5_000   # skgstat is O(n^2)-ish on pairwise distances; subsample separately for this step only
# ──────────────────────────────────────────────────────────────────────────────

def parse_run_key(path):
    m = re.match(r"run_n(\d+)_(\d{8})", path.name)
    if not m:
        return (-1, "00000000")
    return (int(m.group(1)), m.group(2))

def get_run_dir():
    if RUN_DIR is not None:
        return Path(RUN_DIR)
    runs = [p for p in PIPE_OUT.glob("run_n*_*") if p.is_dir()]
    if not runs:
        raise RuntimeError(f"No run folders found in {PIPE_OUT}.")
    runs = sorted(runs, key=parse_run_key)
    return runs[-1]

print("=== 06 Spatial Autocorrelation ===")

run_dir = get_run_dir()
DATA_PATH = run_dir / "predictor_flat.parquet"
OUT_DIR = run_dir
print(f"Run dir: {run_dir}")

df_full = pd.read_parquet(DATA_PATH)
df_full = df_full.dropna(subset=["subsidence_mm_yr"])
n_full = len(df_full)
print(f"  Full valid dataset: {n_full:,} pixels")

with rasterio.open(REF_PATH) as ref:
    px_size_x = abs(ref.transform.a)
    px_size_y = abs(ref.transform.e)
    ref_crs = ref.crs
print(f"  Reference pixel size: {px_size_x:.2f} x {px_size_y:.2f} m ({ref_crs})")

# ── Systematic (grid-based) subsampling instead of random ───────────────────
if GRID_THIN_STEP is None:
    thin_step = max(1, int(np.sqrt(n_full / SAMPLE_TARGET)))
else:
    thin_step = GRID_THIN_STEP

df = df_full[(df_full["row"] % thin_step == 0) & (df_full["col"] % thin_step == 0)].reset_index(drop=True)
print(f"  Systematic thinning step={thin_step} -> {len(df):,} pixels retained (adjacency-preserving)")

y = df["subsidence_mm_yr"].values
coords_px = np.column_stack([df["col"].values, df["row"].values])
coords_m = coords_px * np.array([px_size_x, px_size_y])   # convert to metres for semivariogram

# ── Spatial weights (KNN, on the systematically thinned grid) ───────────────
print(f"  Building KNN weights (k={K_NEIGHBORS})...")
w = libpysal.weights.KNN.from_array(coords_px, k=K_NEIGHBORS)
w.transform = "r"

# ── Global Moran's I ──────────────────────────────────────────────────────────
moran = esda.Moran(y, w)
print(f"\n  Global Moran's I : {moran.I:.4f}")
print(f"  p-value (MC sim) : {moran.p_sim:.4f}")
print(f"  z-score          : {moran.z_sim:.4f}")

fig, ax = plt.subplots(figsize=(5, 5))
moran_scatterplot(moran, ax=ax)
ax.set_title(f"Moran's I = {moran.I:.4f}  (p={moran.p_sim:.3f})", fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "morans_scatterplot.png", dpi=300)
plt.close()
print("  Saved morans_scatterplot.png")

# ── LISA with FDR correction for multiple testing ────────────────────────────
print("  Computing LISA clusters...")
lisa = esda.Moran_Local(y, w, permutations=LISA_PERMUTATIONS)

reject, p_fdr = fdrcorrection(lisa.p_sim, alpha=FDR_ALPHA, method="indep")
n_sig_raw = int((lisa.p_sim < FDR_ALPHA).sum())
n_sig_fdr = int(reject.sum())
print(f"  Significant (raw p<0.05)      : {n_sig_raw:,} / {len(lisa.p_sim):,}")
print(f"  Significant (FDR-corrected)   : {n_sig_fdr:,} / {len(lisa.p_sim):,}")

labels = np.where(reject, lisa.q, 0)   # 0=NS, 1=HH, 2=LH, 3=LL, 4=HL
colors = {0: "#d3d3d3", 1: "#d7191c", 2: "#abd9e9", 3: "#2c7bb6", 4: "#fdae61"}
cmap_vals = np.array([colors[int(l)] for l in labels])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df["col"], df["row"], c=cmap_vals, s=1.5, marker="s")
for code, label in {1: "High-High", 2: "Low-High", 3: "Low-Low",
                    4: "High-Low", 0: "Not Significant"}.items():
    ax.scatter([], [], c=colors[code], label=label, s=20)
ax.legend(fontsize=8, loc="lower right")
ax.set_title("LISA Cluster Map — InSAR Subsidence Rate (FDR-corrected, alpha=0.05)", fontsize=10)
ax.invert_yaxis()
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUT_DIR / "lisa_cluster_map.png", dpi=300)
plt.close()
print("  Saved lisa_cluster_map.png")

# ── Empirical semivariogram (real-world distances, metres) ──────────────────
print("\n  Computing empirical semivariogram...")
if len(df) > SEMIVARIOGRAM_SAMPLE:
    vgm_idx = np.random.RandomState(42).choice(len(df), SEMIVARIOGRAM_SAMPLE, replace=False)
else:
    vgm_idx = np.arange(len(df))

coords_vgm = coords_m[vgm_idx]
values_vgm = y[vgm_idx]

V = skg.Variogram(
    coordinates=coords_vgm,
    values=values_vgm,
    model="exponential",
    n_lags=SEMIVARIOGRAM_N_LAGS,
    normalize=False,
)

params = V.parameters  # [range, sill, nugget] for exponential model in skgstat
vgm_range, vgm_sill, vgm_nugget = float(params[0]), float(params[1]), float(params[2])
print(f"  Fitted exponential variogram:")
print(f"    Range  : {vgm_range:.2f} m")
print(f"    Sill   : {vgm_sill:.4f}")
print(f"    Nugget : {vgm_nugget:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
V.plot(axes=ax, show=False)
ax.set_title(f"Semivariogram — Exponential Model (range={vgm_range:.1f}m)", fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "semivariogram.png", dpi=300)
plt.close()
print("  Saved semivariogram.png")

# ── Save all results ──────────────────────────────────────────────────────────
results = {
    "moran_I": float(moran.I),
    "p_sim": float(moran.p_sim),
    "z_sim": float(moran.z_sim),
    "n_sample": int(len(df)),
    "n_full_dataset": int(n_full),
    "grid_thin_step": int(thin_step),
    "k_neighbors": K_NEIGHBORS,
    "lisa_n_significant_raw": n_sig_raw,
    "lisa_n_significant_fdr": n_sig_fdr,
    "fdr_alpha": FDR_ALPHA,
    "semivariogram_model": "exponential",
    "semivariogram_range_m": vgm_range,
    "semivariogram_sill": vgm_sill,
    "semivariogram_nugget": vgm_nugget,
    "semivariogram_n_points": int(len(vgm_idx)),
    "pixel_size_m": [px_size_x, px_size_y],
}
with open(OUT_DIR / "spatial_autocorrelation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n  Saved spatial_autocorrelation_results.json")
print("  Done. -> Next: run 07_random_forest_model.py")