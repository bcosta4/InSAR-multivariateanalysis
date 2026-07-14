"""
06_spatial_autocorrelation.py
─────────────────────────────
Tests for spatial autocorrelation in the subsidence response variable
and informs choice between OLS vs. Spatial Error / Spatial Lag models.

Steps:
  1. Global Moran's I on subsidence (subsampled for speed)
  2. Local Moran's I (LISA) — identifies spatial clusters
  3. Semivariogram plot — shows spatial dependency range
  4. Decision output: OLS / SEM / SLM recommendation

OUTPUTS:
    qc/morans_i_result.txt
    qc/lisa_cluster_map.png
    qc/semivariogram.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PARQUET_PATH    = r"E:\Processing\InSAR_validation\data\stack\predictor_stack_valid.parquet"
OUTPUT_DIR      = r"E:\Processing\InSAR_validation\data\qc"
SAMPLE_N_MORAN  = 50_000     # subsample for Moran (faster; still statistically valid)
SAMPLE_N_VARIO  = 20_000     # subsample for semivariogram
N_NEIGHBORS_W   = 8          # k-nearest neighbours for spatial weights
N_PERMUTATIONS  = 999        # permutations for Moran significance
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("06  SPATIAL AUTOCORRELATION ANALYSIS")
print("=" * 60)

try:
    from libpysal.weights import KNN
    from esda.moran import Moran, Moran_Local
    PYSAL_AVAILABLE = True
except ImportError:
    print("[WARNING] libpysal / esda not installed. Install via:")
    print("          conda install -c conda-forge libpysal esda")
    PYSAL_AVAILABLE = False

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET_PATH, columns=["x_coord", "y_coord", "subsidence"])
df = df.dropna(subset=["subsidence"])
print(f"\nLoaded {len(df):,} valid pixels")

# Subsample for Moran
if len(df) > SAMPLE_N_MORAN:
    df_moran = df.sample(n=SAMPLE_N_MORAN, random_state=42).reset_index(drop=True)
    print(f"Subsampled to {SAMPLE_N_MORAN:,} for Moran's I")
else:
    df_moran = df.reset_index(drop=True)

coords = df_moran[["x_coord", "y_coord"]].values
y = df_moran["subsidence"].values

# ── 1. Global Moran's I ───────────────────────────────────────────────────────
if PYSAL_AVAILABLE:
    print("\nBuilding KNN spatial weights ...")
    w = KNN.from_array(coords, k=N_NEIGHBORS_W)
    w.transform = "r"

    print(f"Computing global Moran's I (n={N_PERMUTATIONS} permutations) ...")
    mi = Moran(y, w, permutations=N_PERMUTATIONS)

    print(f"\n  Moran's I = {mi.I:.6f}")
    print(f"  Expected  = {mi.EI:.6f}")
    print(f"  p-value   = {mi.p_sim:.4f}")
    print(f"  z-score   = {mi.z_sim:.4f}")

    if mi.p_sim < 0.05:
        print("\n  [RESULT] Significant spatial autocorrelation detected.")
        print("  → Use Spatial Error Model (SEM) or Spatial Lag Model (SLM)")
        print("    alongside OLS as a baseline. Pure OLS will underestimate SE.")
        recommendation = "SEM"
    else:
        print("\n  [RESULT] No significant spatial autocorrelation.")
        print("  → OLS is statistically valid, but GWR/RF still recommended")
        print("    for capturing non-linear and spatially varying relationships.")
        recommendation = "OLS"

    # Save Moran result
    result_path = os.path.join(OUTPUT_DIR, "morans_i_result.txt")
    with open(result_path, "w") as f:
        f.write(f"Global Moran's I Results\n")
        f.write(f"Sample size     : {len(df_moran):,}\n")
        f.write(f"K neighbours    : {N_NEIGHBORS_W}\n")
        f.write(f"Moran's I       : {mi.I:.6f}\n")
        f.write(f"Expected I      : {mi.EI:.6f}\n")
        f.write(f"p-value (sim)   : {mi.p_sim:.4f}\n")
        f.write(f"z-score         : {mi.z_sim:.4f}\n")
        f.write(f"Recommendation  : {recommendation}\n")
    print(f"\nSaved: {result_path}")

    # ── 2. LISA Clusters ──────────────────────────────────────────────────────
    print("\nComputing LISA (Local Moran's I) ...")
    lisa = Moran_Local(y, w, permutations=499, seed=42)

    # Quadrant labels: 1=HH, 2=LH, 3=LL, 4=HL, 0=not significant
    sig_mask = lisa.p_sim < 0.05
    quad     = np.where(sig_mask, lisa.q, 0)

    quad_labels = {0: "Not sig.", 1: "High-High", 2: "Low-High",
                   3: "Low-Low",  4: "High-Low"}
    quad_colors = {0: "#cccccc", 1: "#d7191c", 2: "#abd9e9",
                   3: "#2c7bb6",  4: "#fdae61"}

    fig, ax = plt.subplots(figsize=(10, 8))
    for q_val, label in quad_labels.items():
        mask = quad == q_val
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=quad_colors[q_val], s=0.5, alpha=0.6, label=label, rasterized=True
        )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"LISA Cluster Map — Subsidence Rate\n(n={len(df_moran):,}, p<0.05)", pad=10)
    ax.legend(markerscale=8, loc="upper right", title="LISA Quad")
    ax.set_aspect("equal")
    plt.tight_layout()
    lisa_path = os.path.join(OUTPUT_DIR, "lisa_cluster_map.png")
    plt.savefig(lisa_path, dpi=150)
    plt.close()
    print(f"Saved: {lisa_path}")

else:
    print("[SKIP] Moran's I — libpysal not available")
    recommendation = "UNKNOWN"

# ── 3. Semivariogram (empirical) ─────────────────────────────────────────────
print("\nComputing empirical semivariogram ...")

if len(df) > SAMPLE_N_VARIO:
    df_vario = df.sample(n=SAMPLE_N_VARIO, random_state=42).reset_index(drop=True)
else:
    df_vario = df.reset_index(drop=True)

coords_v = df_vario[["x_coord", "y_coord"]].values
y_v      = df_vario["subsidence"].values

tree  = cKDTree(coords_v)
pairs = tree.query_pairs(r=5000, output_type="ndarray")  # pairs within 5 km

if len(pairs) > 0:
    dists  = np.linalg.norm(coords_v[pairs[:, 0]] - coords_v[pairs[:, 1]], axis=1)
    sq_diff = 0.5 * (y_v[pairs[:, 0]] - y_v[pairs[:, 1]]) ** 2

    n_bins  = 30
    bin_edges = np.linspace(27, 5000, n_bins + 1)   # start at 1 pixel (27m)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    gamma   = np.zeros(n_bins)
    counts  = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (dists >= bin_edges[i]) & (dists < bin_edges[i + 1])
        if in_bin.sum() > 0:
            gamma[i] = sq_diff[in_bin].mean()
            counts[i] = in_bin.sum()

    valid_bins = counts > 10
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_centers[valid_bins] / 1000, gamma[valid_bins],
            "o-", color="#01696f", lw=2, ms=5)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Semivariance (cm/yr)²")
    ax.set_title("Empirical Semivariogram — Subsidence Rate")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    vario_path = os.path.join(OUTPUT_DIR, "semivariogram.png")
    plt.savefig(vario_path, dpi=150)
    plt.close()
    print(f"Saved: {vario_path}")
    print("  Note: The lag distance where the semivariogram flattens (sill) indicates")
    print("  the range of spatial dependence — use this to guide GWR bandwidth selection.")
else:
    print("  [WARN] No pairs found within 5 km — check coordinate units (should be metres).")

print("\n✓ 06 complete. Proceed to 07_random_forest_model.py")
