"""
06_multicollinearity_vif.py
────────────────────────────
Multicollinearity screening:
  1. Spearman correlation matrix + heatmap
  2. Iterative VIF pruning (threshold = 10)
  3. Saves pruned variable list for downstream scripts
"""

import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

DATA_PATH = Path(__file__).parent / "outputs" / "predictor_flat.parquet"
OUT_DIR   = Path(__file__).parent / "outputs"
VIF_THRESH = 10.0
SAMPLE_N   = 50_000    # random sample for VIF (faster on large datasets)

print("=== 06 Multicollinearity + VIF ===")
df = pd.read_parquet(DATA_PATH)

# Drop spatial index and response variable
feature_cols = [c for c in df.columns if c not in ("row", "col", "subsidence_mm_yr")]
X = df[feature_cols].dropna()
print(f"  {len(X):,} complete rows, {len(feature_cols)} predictors")

# ── Spearman correlation ──────────────────────────────────────────────────────
corr = X.corr(method="spearman")
fig, ax = plt.subplots(figsize=(max(8, len(feature_cols)*0.8), max(6, len(feature_cols)*0.7)))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            linewidths=0.5, ax=ax, annot_kws={"size": 7})
ax.set_title("Spearman Correlation Matrix — Predictors", fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close()
print("  Saved correlation_heatmap.png")

# ── VIF pruning ───────────────────────────────────────────────────────────────
Xs = X.sample(min(SAMPLE_N, len(X)), random_state=42).copy()
remaining = list(feature_cols)
iteration = 0

while True:
    iteration += 1
    vif_data = pd.DataFrame({
        "variable": remaining,
        "VIF": [variance_inflation_factor(Xs[remaining].values, i)
                for i in range(len(remaining))]
    }).sort_values("VIF", ascending=False)

    max_vif = vif_data["VIF"].max()
    print(f"  Iteration {iteration}: max VIF = {max_vif:.2f}  ({len(remaining)} vars)")

    if max_vif <= VIF_THRESH:
        break

    drop_var = vif_data.iloc[0]["variable"]
    remaining.remove(drop_var)
    print(f"    Dropped: {drop_var}  (VIF={max_vif:.2f})")

print(f"\n  Retained {len(remaining)} variables after VIF pruning:")
for v in remaining:
    print(f"    {v}")

# Save results
vif_final = pd.DataFrame({
    "variable": remaining,
    "VIF": [variance_inflation_factor(Xs[remaining].values, i)
            for i in range(len(remaining))]
}).sort_values("VIF", ascending=False)
vif_final.to_csv(OUT_DIR / "vif_retained_variables.csv", index=False)

with open(OUT_DIR / "pruned_feature_list.json", "w") as f:
    json.dump(remaining, f, indent=2)

print("\n  Saved vif_retained_variables.csv and pruned_feature_list.json")
print("  Done. → Next: run 07_spatial_autocorrelation.py")
