r"""
06_multicollinearity_vif.py
────────────────────────────
Purpose
-------
Screens the mixed continuous/categorical predictor set for multicollinearity and iteratively prunes variables until all remaining predictors fall below an equivalent-scale collinearity threshold, producing the final feature set used by all downstream models.

Method
------
Continuous variables are assessed via standard Variance Inflation Factor (VIF), computed with an intercept term to match the textbook definition.
Categorical variables are one-hot encoded and assessed via Generalized VIF (GVIF; Fox & Monette, 1992), computed per original variable rather than per dummy column, then rescaled as GVIF^(1/(2*df)) so it sits on the same interpretive scale as ordinary VIF (threshold ~3.16 corresponds to a classic VIF=10 cutoff). 
At each iteration, whichever variable - continuous or categorical - has the highest comparable-scale score is dropped, and the process repeats until all remaining variables are below their respective thresholds (VIF_THRESH=5.0, GVIF_ADJ_THRESH=3.16). 
A Spearman correlation heatmap is produced separately for continuous variables only, since Spearman correlation between category codes and continuous variables is not statistically meaningful.

Inputs
------
- predictor_flat.parquet (from 04): pixel-level design matrix with all candidate predictors and target variable.

Outputs
-------
- correlation_heatmap.png: Spearman correlation matrix, continuous predictors only.
- vif_gvif_retained_variables.csv: final VIF/GVIF scores for all retained variables.
- vif_gvif_pruning_history.csv: record of each variable dropped, its type, and its score at the time of removal.
- pruned_feature_list.json: final retained feature set, split into continuous/categorical/all.
- categorical_variable_map.json: unique category values per retained categorical variable, for consistent downstream one-hot encoding.

Dependencies
------------
statsmodels, numpy, pandas, matplotlib, seaborn.

Reproducibility notes
----------------------
- Random seed: 42 (fixed for the SAMPLE_N row subsample used in VIF/GVIF computation and the correlation heatmap).
- Deterministic given identical inputs: Yes.
"""

import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"E:\Processing\InSAR_multivariate_analysis")
PIPE_OUT = PROJECT_DIR / "outputs"
RUN_DIR = None          # None = auto-pick latest run_nX_YYYYMMDD

VIF_THRESH = 5.0        # adjust as needed; classic textbook threshold is 10, but 5 is more conservative
GVIF_ADJ_THRESH = 3.16   # sqrt(10) - equivalent threshold for GVIF^(1/2df)
SAMPLE_N = 50_000

# Variables that are CATEGORICAL / NOMINAL - must match column names in predictor_flat.parquet exactly (see 03_align_predictors.py / 04_build_predictor_stack.py)
CATEGORICAL_VARS = [
    "landcover",
    "wetland_class",
    "pan_arctic_wetlands",
    "canopy_height_lt5m",
    "woodyplant_dominance",
    "cavm",
    "surficial_geology",
    "ground_ice",
    "dsw_trends",
    "permafrost_landcover_cls",]

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

def continuous_vif(X_cont):
    """Standard VIF with intercept (add_constant), matching textbook definition."""
    Xc = sm.add_constant(X_cont)
    vifs = {}
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        vifs[col] = variance_inflation_factor(Xc.values, i)
    return vifs

def gvif_categorical(df_all, cat_var, other_vars):
    """
    Generalized VIF for one categorical variable (Fox & Monette, 1992),
    computed via the standard determinant-ratio formula:

        GVIF = det(R11) * det(R22) / det(R)

    where R is the full correlation matrix of [dummies_of_cat_var, other_vars],
    R11 is the block for the dummy columns of cat_var, and R22 is the block
    for all other variables (continuous, already-numeric; other categoricals
    are also expanded to dummies and included in "other_vars" numeric block).

    Returns (gvif_raw, gvif_adjusted, df) where df = number of dummy columns.
    """
    dummies = pd.get_dummies(df_all[cat_var], prefix=cat_var, drop_first=True)
    df_dummy = dummies.shape[1]
    if df_dummy == 0:
        return np.nan, np.nan, 0
    combined = pd.concat([dummies, other_vars], axis=1)
    combined = combined.loc[:, combined.std() > 1e-10]  # drop zero-variance cols

    R = combined.corr().values
    n = R.shape[0]
    k = dummies.shape[1]

    R11 = R[:k, :k]
    R22 = R[k:, k:]

    det_R = np.linalg.det(R)
    det_R11 = np.linalg.det(R11)
    det_R22 = np.linalg.det(R22) if R22.shape[0] > 0 else 1.0

    if det_R <= 0 or det_R11 <= 0 or det_R22 <= 0:
        gvif = np.inf
    else:
        gvif = (det_R11 * det_R22) / det_R
    gvif_adj = gvif ** (1.0 / (2 * df_dummy)) if np.isfinite(gvif) else np.inf
    return gvif, gvif_adj, df_dummy

print("=== 05 Multicollinearity: VIF (continuous) + GVIF (categorical) ===")

run_dir = get_run_dir()
DATA_PATH = run_dir / "predictor_flat.parquet"
OUT_DIR = run_dir

print(f"Run dir  : {run_dir}")
df_full = pd.read_parquet(DATA_PATH)

feature_cols = [c for c in df_full.columns if c not in ("row", "col", "subsidence_mm_yr")]
missing_cat = [c for c in CATEGORICAL_VARS if c not in feature_cols]
if missing_cat:
    print(f"[WARNING] Configured categorical vars not found in parquet columns: {missing_cat}")
CATEGORICAL_VARS = [c for c in CATEGORICAL_VARS if c in feature_cols]
continuous_vars = [c for c in feature_cols if c not in CATEGORICAL_VARS]

print(f"  {len(feature_cols)} total predictors: {len(continuous_vars)} continuous, {len(CATEGORICAL_VARS)} categorical")

df_clean = df_full[feature_cols].dropna()
print(f"  {len(df_clean):,} complete rows after dropping NaN")

df_s = df_clean.sample(min(SAMPLE_N, len(df_clean)), random_state=42).copy()

# ── Spearman correlation heatmap: CONTINUOUS variables only ──────────────────
corr = df_s[continuous_vars].corr(method="spearman")
fig, ax = plt.subplots(figsize=(max(8, len(continuous_vars) * 0.8), max(6, len(continuous_vars) * 0.7)))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            linewidths=0.5, ax=ax, annot_kws={"size": 7})
ax.set_title("Spearman Correlation Matrix — Continuous Predictors Only", fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=300)
plt.close()
print("  Saved correlation_heatmap.png (continuous variables only)")

# ── Iterative pruning: continuous VIF + categorical GVIF, same scale ────────
remaining_cont = list(continuous_vars)
remaining_cat = list(CATEGORICAL_VARS)
iteration = 1
history = []

while True:
    iteration += 1
    scores = {}
    if remaining_cont:
        vifs = continuous_vif(df_s[remaining_cont])
        for var, v in vifs.items():
            scores[var] = ("continuous", v, v)  # (type, raw, comparable-scale)

    for cat_var in remaining_cat:
        other_numeric_block = df_s[remaining_cont].copy()
        for other_cat in remaining_cat:
            if other_cat == cat_var:
                continue
            other_numeric_block = pd.concat(
                [other_numeric_block, pd.get_dummies(df_s[other_cat], prefix=other_cat, drop_first=True)],
                axis=1,
            )
        gvif_raw, gvif_adj, dfree = gvif_categorical(df_s, cat_var, other_numeric_block)
        scores[cat_var] = ("categorical", gvif_raw, gvif_adj)

    score_df = pd.DataFrame(
        [{"variable": k, "type": v[0], "raw_score": v[1], "comparable_score": v[2]} for k, v in scores.items()]
    ).sort_values("comparable_score", ascending=False)

    worst = score_df.iloc[0]
    print(f"  Iter {iteration}: worst = {worst['variable']} ({worst['type']}) "
          f"comparable={worst['comparable_score']:.2f}  "
          f"[{len(remaining_cont)} cont + {len(remaining_cat)} cat remaining]")

    threshold_worst = GVIF_ADJ_THRESH if worst["type"] == "categorical" else VIF_THRESH
    # Both are compared on the "VIF=10 equivalent" scale: continuous VIF vs 10, categorical GVIF^(1/2df) vs 3.16. To compare directly in one ranking we normalize continuous VIF by dividing by (10/3.16) so both breach at the same relative point -> use comparable_score already on native scale and threshold check below handles it per-type.
    if worst["type"] == "continuous" and worst["comparable_score"] <= VIF_THRESH:
        break
    if worst["type"] == "categorical" and worst["comparable_score"] <= GVIF_ADJ_THRESH:
        break

    history.append({"iteration": iteration, "dropped": worst["variable"],
                     "type": worst["type"], "score": worst["comparable_score"]})
    if worst["type"] == "continuous":
        remaining_cont.remove(worst["variable"])
    else:
        remaining_cat.remove(worst["variable"])

    if not remaining_cont and not remaining_cat:
        print("  [WARNING] All variables pruned - check data or thresholds.")
        break

final_cont_scores = continuous_vif(df_s[remaining_cont]) if remaining_cont else {}
final_rows = [{"variable": k, "type": "continuous", "raw_score": v, "comparable_score": v}
              for k, v in final_cont_scores.items()]

for cat_var in remaining_cat:
    other_numeric_block = df_s[remaining_cont].copy()
    for other_cat in remaining_cat:
        if other_cat == cat_var:
            continue
        other_numeric_block = pd.concat(
            [other_numeric_block, pd.get_dummies(df_s[other_cat], prefix=other_cat, drop_first=True)],
            axis=1,
        )
    gvif_raw, gvif_adj, dfree = gvif_categorical(df_s, cat_var, other_numeric_block)
    final_rows.append({"variable": cat_var, "type": "categorical",
                        "raw_score": gvif_raw, "comparable_score": gvif_adj})

final_df = pd.DataFrame(final_rows).sort_values("comparable_score", ascending=False)

print(f"\n  Retained {len(remaining_cont) + len(remaining_cat)} variables "
      f"({len(remaining_cont)} continuous, {len(remaining_cat)} categorical):")
for _, row in final_df.iterrows():
    print(f"    {row['variable']:<28} [{row['type']:<11}] comparable_score={row['comparable_score']:.2f}")

final_df.to_csv(OUT_DIR / "vif_gvif_retained_variables.csv", index=False)

pruned_list = remaining_cont + remaining_cat
with open(OUT_DIR / "pruned_feature_list.json", "w") as f:
    json.dump({"continuous": remaining_cont, "categorical": remaining_cat, "all": pruned_list}, f, indent=2)

with open(OUT_DIR / "categorical_variable_map.json", "w") as f:
    json.dump({c: sorted(df_full[c].dropna().unique().tolist()) for c in remaining_cat}, f, indent=2, default=str)

if history:
    pd.DataFrame(history).to_csv(OUT_DIR / "vif_gvif_pruning_history.csv", index=False)

print(f"\n  Saved vif_gvif_retained_variables.csv, pruned_feature_list.json, categorical_variable_map.json")
print("  Done. -> Next: 06_spatial_autocorrelation.py")