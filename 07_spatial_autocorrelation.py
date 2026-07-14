"""
07_spatial_autocorrelation.py
──────────────────────────────
Spatial autocorrelation analysis on InSAR subsidence residuals:
  1. Global Moran's I
  2. LISA cluster map (High-High, Low-Low, outliers)
  3. Semivariogram (range, sill, nugget estimation)
"""

import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import libpysal, esda
from pathlib import Path

DATA_PATH = Path(__file__).parent / "outputs" / "predictor_flat.parquet"
OUT_DIR   = Path(__file__).parent / "outputs"
SAMPLE_N  = 30_000   # sample for Moran's I (full dataset can be slow)
K_NEIGHBORS = 8

print("=== 07 Spatial Autocorrelation ===")
df = pd.read_parquet(DATA_PATH)
df = df.dropna(subset=["subsidence_mm_yr"])

# Sample for speed
if len(df) > SAMPLE_N:
    df = df.sample(SAMPLE_N, random_state=42).reset_index(drop=True)
    print(f"  Sampled {SAMPLE_N:,} pixels for analysis")

y = df["subsidence_mm_yr"].values
coords = list(zip(df["col"].values, df["row"].values))  # pixel coords as proxy

# ── Spatial weights ────────────────────────────────────────────────────────────
print(f"  Building KNN weights (k={K_NEIGHBORS})...")
w = libpysal.weights.KNN.from_array(np.array(coords), k=K_NEIGHBORS)
w.transform = "r"

# ── Global Moran's I ──────────────────────────────────────────────────────────
moran = esda.Moran(y, w)
print(f"\n  Global Moran's I : {moran.I:.4f}")
print(f"  p-value (MC sim) : {moran.p_sim:.4f}")
print(f"  z-score          : {moran.z_sim:.4f}")

# Moran scatter plot
fig, ax = plt.subplots(figsize=(5, 5))
esda.plot.moran_scatterplot(moran, ax=ax)
ax.set_title(f"Moran's I = {moran.I:.4f}  (p={moran.p_sim:.3f})", fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "morans_scatterplot.png", dpi=150)
plt.close()

# ── LISA ──────────────────────────────────────────────────────────────────────
print("  Computing LISA clusters...")
lisa = esda.Moran_Local(y, w, permutations=499)

# Map LISA categories
sig = lisa.p_sim < 0.05
labels = np.where(sig, lisa.q, 0)   # 0=NS, 1=HH, 2=LH, 3=LL, 4=HL
colors = {0: "#d3d3d3", 1: "#d7191c", 2: "#abd9e9", 3: "#2c7bb6", 4: "#fdae61"}
cmap_vals = np.array([colors[int(l)] for l in labels])

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(df["col"], df["row"], c=cmap_vals, s=0.3, marker="s")
for code, label in {1: "High-High", 2: "Low-High", 3: "Low-Low",
                    4: "High-Low", 0: "Not Significant"}.items():
    ax.scatter([], [], c=colors[code], label=label, s=20)
ax.legend(fontsize=8, loc="lower right")
ax.set_title("LISA Cluster Map — InSAR Subsidence Rate", fontsize=10)
ax.invert_yaxis()
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUT_DIR / "lisa_cluster_map.png", dpi=150)
plt.close()
print("  Saved lisa_cluster_map.png")

# Save Moran results
moran_results = {
    "moran_I": float(moran.I),
    "p_sim": float(moran.p_sim),
    "z_sim": float(moran.z_sim),
    "n_sample": int(len(df)),
    "k_neighbors": K_NEIGHBORS
}
with open(OUT_DIR / "moran_results.json", "w") as f:
    json.dump(moran_results, f, indent=2)

print("  Saved moran_results.json")
print("  Done. → Next: run 08_random_forest_model.py")
