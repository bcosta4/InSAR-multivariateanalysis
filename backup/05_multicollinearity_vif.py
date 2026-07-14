"""
05_multicollinearity_vif.py
─────────────────────────────
Tests for multicollinearity among continuous predictors using:
  - Spearman correlation matrix (heatmap)
  - Variance Inflation Factor (VIF) for all continuous predictors
  - Iterative VIF pruning: removes highest-VIF variable until all VIF < threshold

OUTPUTS:
    qc/correlation_heatmap.png
    qc/vif_scores.csv
    qc/vif_pruned_variables.txt   — final variable list to carry into modelling
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PARQUET_PATH = r"E:\Processing\InSAR_validation\data\stack\predictor_stack_valid.parquet"
OUTPUT_DIR   = r"E:\Processing\InSAR_validation\data\qc"
VIF_THRESHOLD = 10.0        # common threshold; use 5 for stricter pruning
SAMPLE_N      = 100_000     # subsample for VIF calc speed (≤ full dataset)
CORR_THRESHOLD = 0.80

# Continuous predictors to assess (exclude response and categoricals)
CONTINUOUS_VARS = [
    "alt", "magt", "ddt", "ground_ice",
    "slope", "aspect_sin", "aspect_cos",
    "curvature", "twi", "tpi", "tri",
    "ndvi", "flood_freq",
    # Add "soil_organic_carbon", "era5_soil_temp" when available
]
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("05  MULTICOLLINEARITY & VIF ANALYSIS")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET_PATH)
print(f"\nLoaded {len(df):,} pixels")

# Restrict to available continuous vars
avail = [v for v in CONTINUOUS_VARS if v in df.columns]
missing = [v for v in CONTINUOUS_VARS if v not in df.columns]
if missing:
    print(f"  [INFO] Missing variables (not in parquet, skipped): {missing}")
print(f"  Assessing {len(avail)} continuous predictors")

df_cont = df[avail].dropna()
print(f"  Complete cases (no NaN): {len(df_cont):,}")

# Subsample for speed
if len(df_cont) > SAMPLE_N:
    df_sample = df_cont.sample(n=SAMPLE_N, random_state=42)
    print(f"  Subsampled to {SAMPLE_N:,} for VIF computation")
else:
    df_sample = df_cont

# ── 1. Spearman Correlation Matrix ────────────────────────────────────────────
print("\nComputing Spearman correlation matrix ...")
corr = df_sample[avail].corr(method="spearman")

fig, ax = plt.subplots(figsize=(max(8, len(avail)), max(6, len(avail) - 2)))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
    annot_kws={"size": 8}, ax=ax,
    cbar_kws={"shrink": 0.8, "label": "Spearman r"}
)
ax.set_title("Predictor Correlation Matrix (Spearman)", pad=12)
plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=150)
plt.close()
print(f"Saved: {heatmap_path}")

# Print high-correlation pairs
print(f"\nPairs with |r| > {CORR_THRESHOLD}:")
pairs = (
    corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
    .stack().reset_index()
)
pairs.columns = ["var1", "var2", "r"]
high = pairs[pairs["r"].abs() > CORR_THRESHOLD].sort_values("r", key=abs, ascending=False)
if len(high):
    print(high.to_string(index=False))
else:
    print("  None above threshold.")

# ── 2. VIF Scores ─────────────────────────────────────────────────────────────
print("\nComputing VIF scores ...")

def compute_vif(df_in, variables):
    X = df_in[variables].dropna().values
    vif_data = pd.DataFrame({
        "variable": variables,
        "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    }).sort_values("VIF", ascending=False)
    return vif_data

vif_all = compute_vif(df_sample, avail)
print("\nInitial VIF scores:")
print(vif_all.to_string(index=False))

vif_path = os.path.join(OUTPUT_DIR, "vif_scores_initial.csv")
vif_all.to_csv(vif_path, index=False)
print(f"\nSaved: {vif_path}")

# ── 3. Iterative VIF Pruning ─────────────────────────────────────────────────
print(f"\nIterative pruning (remove highest VIF > {VIF_THRESHOLD}) ...")
remaining = avail.copy()
removed_log = []
iteration = 0

while True:
    iteration += 1
    vif_iter = compute_vif(df_sample, remaining)
    max_vif   = vif_iter["VIF"].max()
    max_var   = vif_iter.loc[vif_iter["VIF"].idxmax(), "variable"]

    if max_vif <= VIF_THRESHOLD:
        print(f"  Iteration {iteration}: all VIF ≤ {VIF_THRESHOLD}. Done.")
        break
    print(f"  Iteration {iteration}: removing '{max_var}' (VIF={max_vif:.1f})")
    removed_log.append({"variable": max_var, "VIF_at_removal": max_vif, "iteration": iteration})
    remaining.remove(max_var)

    if len(remaining) < 2:
        print("  [WARNING] Too few variables remaining — stopping.")
        break

# Final VIF
vif_final = compute_vif(df_sample, remaining)
print("\nFinal VIF scores after pruning:")
print(vif_final.to_string(index=False))

vif_final_path = os.path.join(OUTPUT_DIR, "vif_scores_final.csv")
vif_final.to_csv(vif_final_path, index=False)

# ── 4. Write final variable list ──────────────────────────────────────────────
out_txt = os.path.join(OUTPUT_DIR, "vif_pruned_variables.txt")
with open(out_txt, "w") as f:
    f.write("# Variables retained after VIF pruning\n")
    f.write(f"# VIF threshold: {VIF_THRESHOLD}\n")
    f.write(f"# Retained: {len(remaining)}\n\n")
    for v in remaining:
        f.write(f"{v}\n")
    f.write("\n# Removed:\n")
    for r in removed_log:
        f.write(f"# {r['variable']:20s}  VIF={r['VIF_at_removal']:.1f}  (iteration {r['iteration']})\n")

print(f"\nSaved: {out_txt}")
print(f"\nRetained variables ({len(remaining)}): {remaining}")
if removed_log:
    removed_vars = [r["variable"] for r in removed_log]
    print(f"Removed  variables ({len(removed_vars)}): {removed_vars}")

print("\n✓ 05 complete. Proceed to 06_spatial_autocorrelation.py")
