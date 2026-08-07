r"""
10_consolidate_model_comparison.py

Purpose
-------
Builds the paper's central results table: a cross-model variable-importance agreement table answering "which variables explain InSAR subsidence, and do independent methods (linear, spatially-varying, nonlinear) agree on their importance, direction, and actual explanatory weight?". 
This is an explanatory (driver-attribution) synthesis, not a predictive-model leaderboard.
Also produces a secondary robustness table (explained variance + residual spatial autocorrelation per method) and a reproducibility appendix.

NOTE: There is no standalone Random Forest script/output in this pipeline - XGBoost is the sole tree-based (nonlinear) comparator alongside OLS, SEM, and GWR (linear / spatially-varying). RF is no longer referenced anywhere in this script.

Method
------
Reads each model's own saved results/importance files from the run directory (never hand-typed) and aligns them on a common variable name (aggregating one-hot dummy coefficient names from OLS/SEM back to their parent categorical variable, matching XGBoost's native-categorical column names), producing:

1. A ranked importance/coefficient-magnitude table per method (XGBoost permutation importance + mean |SHAP|, OLS/SEM standardized coefficient magnitude, GWR mean absolute local coefficient), each also expressed as a "variance_share_pct" - each method's per-variable importance normalized to sum to 100% within that method - so magnitude is comparable across methods with very different raw importance scales (e.g. SHAP values vs VIF-filtered regression coefficients).
2. A rank-agreement score (mean rank + pairwise Spearman correlation across methods) so agreement/disagreement is quantified, not just visually inspected.
3. A "reliable_driver" flag: a variable must satisfy BOTH a rank-agreement criterion (present in the top N of at least MIN_METHODS_IN_TOP_N methods) AND a numerical-weight criterion (mean variance_share_pct across methods >= MIN_VARIANCE_SHARE_PCT) to be labeled reliable. This directly guards against the failure mode where a variable ranks highly everywhere but explains a trivial fraction of variance in every method - agreement on an unimportant variable is not evidence of a real driver.
4. A robustness/explained-variance table flagging which methods were evaluated under spatial-fold robustness checks vs in-sample fit, so this distinction can never be silently lost in the manuscript.

Inputs
------
- xgb_permutation_importance.csv, xgb_mean_abs_shap.csv (XGBoost)
- ols_sem_standardized_coefficients.csv (OLS + SEM, one row per variable, already dummy-aggregated)
- gwr_local_coefficient_summary.csv (GWR, one row per variable, already dummy-aggregated)
- model_comparison.csv (model-level R2/AIC/Moran's I for OLS/SEM/GWR)
- xgb_cv_results.json, sem_results.json, gwr_results.json (robustness table)
All in the same run directory.

Outputs
-------
- variable_importance_agreement.csv: one row per variable, per-method importance, per-method variance_share_pct, per-method rank, mean rank, mean variance_share_pct, reliable_driver flag.
- variable_importance_rank_agreement_matrix.csv: pairwise Spearman rank correlation across methods.
- model_comparison_final.csv: one row per model, explained variance, validation method, residual Moran's I, comparability note.
- reproducibility_appendix.csv: one row per script, seed, key package versions, determinism note.

Dependencies
------------
pandas, numpy, scipy, json, pathlib.

Reproducibility notes
----------------------
- Random seed: not applicable (this script performs no stochastic computation).
- Deterministic given identical inputs: Yes.
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(r"E:\Processing\InSAR_multivariate_analysis")
PIPE_OUT = PROJECT_DIR / "outputs"
RUN_DIR = None

# Reliability thresholds (both must hold for reliable_driver = True)
TOP_N_FOR_AGREEMENT = 10
MIN_METHODS_IN_TOP_N = 3          # must be in top-10 in at least 3 of the available methods
MIN_VARIANCE_SHARE_PCT = 1.0      # mean variance share across methods must be >= 1%

# Spatial-control coordinate features (x_norm, y_norm, dist_to_center, etc.) are deliberately included in XGBoost as controls for residual spatial structure, but they are NOT physical/environmental drivers and were never fit in OLS/SEM/GWR (correctly, since they don't belong in an explanatory covariate set). 
# They must never be labeled reliable_driver=True regardless of how high they rank in XGBoost alone - a location coordinate outranking real predictors is evidence of unexplained spatial structure, not evidence of a driver mechanism.
SPATIAL_CONTROL_VARS = ["x_norm", "y_norm", "dist_to_center"]

# GWR uses a fundamentally different estimand (local, spatially-varying coefficients) from the three global methods (XGBoost, OLS, SEM) and is fit on a much smaller VIF-/modal-filtered subset of continuous-only predictors. 
# Averaging its rank/share directly into the same cross-model mean as global methods lets GWR-only outlier behavior (e.g. one variable's local coefficient spiking due to local collinearity) dominate the pooled importance score. 
# GWR is therefore reported as a separate column for context but EXCLUDED from mean_rank_across_methods / mean_variance_share_pct, which are computed only from methods sharing a comparable global-effect estimand.
GWR_METHOD_PREFIX = "GWR_"

def parse_run_key(path):
    m = re.match(r"run_n(\d+)_(\d{8})", path.name)
    return (-1, "00000000") if not m else (int(m.group(1)), m.group(2))

def get_run_dir():
    if RUN_DIR is not None:
        return Path(RUN_DIR)
    runs = sorted([p for p in PIPE_OUT.glob("run_n*_*") if p.is_dir()], key=parse_run_key)
    if not runs:
        raise RuntimeError(f"No run folders found in {PIPE_OUT}.")
    return runs[-1]

def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def load_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)

print("=== 10 Consolidate Model Comparison (explanatory synthesis) ===")
run_dir = get_run_dir()
print(f"Run dir: {run_dir}")

# --- 1. Robustness / explained-variance table (secondary, diagnostic) ------
rows = []

xgb_res = load_json(run_dir / "xgb_cv_results.json")
if xgb_res:
    oof_moran = xgb_res.get("oof_residual_spatial_autocorrelation", {})
    rows.append({
        "model": "XGBoost",
        "explained_variance": xgb_res.get("fold_r2_mean"),
        "explained_variance_std": xgb_res.get("fold_r2_std"),
        "validation_method": xgb_res.get("fold_method", "spatial grid-block fold split"),
        "residual_moran_I": oof_moran.get("moran_I"),
        "residual_moran_p": oof_moran.get("moran_p"),
        "validation_class": "spatial_fold_robustness_check",
    })

sem = load_json(run_dir / "sem_results.json")
if sem:
    rows.append({
        "model": "SEM (GM_Error)",
        "explained_variance": sem.get("pseudo_r2") or sem.get("pseudo_R2"),
        "explained_variance_std": None,
        "validation_method": "in-sample fit (no spatial fold split)",
        "residual_moran_I": sem.get("residual_moran_I"),
        "residual_moran_p": sem.get("residual_moran_p"),
        "validation_class": "in_sample_fit",
    })

gwr = load_json(run_dir / "gwr_results.json")
if gwr:
    rows.append({
        "model": "GWR",
        "explained_variance": gwr.get("R2"),
        "explained_variance_std": None,
        "validation_method": "in-sample fit (local coefficients, no spatial fold split)",
        "residual_moran_I": gwr.get("residual_moran_I"),
        "residual_moran_p": gwr.get("residual_moran_p"),
        "validation_class": "in_sample_fit",
    })

if (run_dir / "model_comparison.csv").exists():
    mc_df = pd.read_csv(run_dir / "model_comparison.csv")
    if "model" in mc_df.columns and "OLS" in mc_df["model"].values:
        ols_row = mc_df[mc_df["model"] == "OLS"].iloc[0]
        rows.append({
            "model": "OLS",
            "explained_variance": ols_row.get("R2"),
            "explained_variance_std": None,
            "validation_method": "in-sample fit (no spatial fold split)",
            "residual_moran_I": ols_row.get("residual_moran_I"),
            "residual_moran_p": ols_row.get("residual_moran_p"),
            "validation_class": "in_sample_fit",
        })

robustness_df = pd.DataFrame(rows)
if not robustness_df.empty:
    robustness_df["comparability_note"] = robustness_df["validation_class"].map({
        "spatial_fold_robustness_check": "Explained variance tested for stability across independent spatial folds.",
        "in_sample_fit": "Explained variance is an in-sample fit statistic, not a spatial-fold robustness check. "
                         "Not directly comparable to spatial_fold_robustness_check rows without caveat.",
    })
    print("\n Robustness / explained-variance table:")
    print(robustness_df.to_string(index=False))

robustness_df.to_csv(run_dir / "model_comparison_final.csv", index=False)
print(f"\n Saved model_comparison_final.csv ({len(robustness_df)} models)")

# --- 2. Cross-model variable-importance agreement table (PRIMARY output) ---
print("\n=== Building cross-model variable-importance agreement table ===")

importance_frames = {}

xgb_perm = load_csv(run_dir / "xgb_permutation_importance.csv")
if xgb_perm is not None:
    importance_frames["XGB_permutation"] = xgb_perm.set_index("feature")["perm_importance_mean"].clip(lower=0)

xgb_shap = load_csv(run_dir / "xgb_mean_abs_shap.csv")
if xgb_shap is not None:
    importance_frames["XGB_mean_abs_SHAP"] = xgb_shap.set_index("feature")["mean_abs_shap"]

ols_sem = load_csv(run_dir / "ols_sem_standardized_coefficients.csv")
if ols_sem is not None:
    ols_sem = ols_sem.set_index("variable")
    if "OLS_std_coef" in ols_sem.columns:
        importance_frames["OLS_std_coef_abs"] = ols_sem["OLS_std_coef"].abs()
    if "SEM_std_coef" in ols_sem.columns:
        importance_frames["SEM_std_coef_abs"] = ols_sem["SEM_std_coef"].abs()

gwr_coef_summary = load_csv(run_dir / "gwr_local_coefficient_summary.csv")
gwr_outlier_flags = None
if gwr_coef_summary is not None:
    gwr_coef_summary = gwr_coef_summary.set_index("variable")
    # Use median_abs_coef (robust to local-collinearity outlier pixels) as the primary GWR importance signal, rather than the raw mean, which can be inflated 10-30x by a handful of unstable pixels (see 09_spatial_model_gwr.py. Mean is still available in the raw file for inspection.
    if "median_abs_coef" in gwr_coef_summary.columns:
        importance_frames["GWR_median_abs_local_coef"] = gwr_coef_summary["median_abs_coef"]
    elif "mean_abs_coef" in gwr_coef_summary.columns:
        importance_frames["GWR_mean_abs_local_coef"] = gwr_coef_summary["mean_abs_coef"]
    if "outlier_flag" in gwr_coef_summary.columns:
        gwr_outlier_flags = gwr_coef_summary["outlier_flag"]

if importance_frames:
    combined = pd.DataFrame(importance_frames)
    method_cols = list(importance_frames.keys())
    gwr_cols = [c for c in method_cols if c.startswith(GWR_METHOD_PREFIX)]
    primary_cols = [c for c in method_cols if c not in gwr_cols]

    if gwr_outlier_flags is not None:
        combined["gwr_outlier_flag"] = gwr_outlier_flags

    # Variance share (%): each method's importance normalized to sum to 100% within that method, so raw-scale differences (SHAP vs regression coef) don't distort the numerical-weight comparison.
    share_cols = []
    for col in method_cols:
        share_col = f"{col}_share_pct"
        col_sum = combined[col].sum(skipna=True)
        combined[share_col] = (combined[col] / col_sum * 100) if col_sum and col_sum > 0 else np.nan
        share_cols.append(share_col)
    primary_share_cols = [f"{c}_share_pct" for c in primary_cols]
    gwr_share_cols = [f"{c}_share_pct" for c in gwr_cols]

    rank_cols = []
    for col in method_cols:
        rank_col = f"{col}_rank"
        combined[rank_col] = combined[col].rank(ascending=False, method="min")
        rank_cols.append(rank_col)
    primary_rank_cols = [f"{c}_rank" for c in primary_cols]
    gwr_rank_cols = [f"{c}_rank" for c in gwr_cols]

    # PRIMARY agreement metrics use only methods with a comparable global-effect estimand (XGBoost permutation, XGBoost SHAP, OLS, SEM). 
    # GWR is reported separately below - see GWR_METHOD_PREFIX note in CONFIG - so a single outlier local coefficient in GWR cannot dominate the pooled score.
    combined["mean_rank_across_methods"] = combined[primary_rank_cols].mean(axis=1)
    combined["mean_variance_share_pct"] = combined[primary_share_cols].mean(axis=1)
    combined["n_methods_present"] = combined[primary_cols].notna().sum(axis=1)
    combined["n_methods_in_top_n"] = (combined[primary_rank_cols] <= TOP_N_FOR_AGREEMENT).sum(axis=1)

    if gwr_cols:
        combined["gwr_mean_rank"] = combined[gwr_rank_cols].mean(axis=1)
        combined["gwr_mean_variance_share_pct"] = combined[gwr_share_cols].mean(axis=1)
        combined["gwr_in_top_n"] = (combined[gwr_rank_cols] <= TOP_N_FOR_AGREEMENT).any(axis=1)

    combined["is_spatial_control"] = combined.index.isin(SPATIAL_CONTROL_VARS)

    combined["reliable_driver"] = (
        (combined["n_methods_in_top_n"] >= MIN_METHODS_IN_TOP_N) &
        (combined["mean_variance_share_pct"] >= MIN_VARIANCE_SHARE_PCT) &
        (~combined["is_spatial_control"])
    )

    def _note(r):
        if r["is_spatial_control"]:
            return ("Spatial control coordinate, not a physical driver - excluded from "
                    "reliable_driver regardless of rank. High XGBoost rank indicates "
                    "unexplained residual spatial structure in subsidence, reported as a "
                    "separate finding (see model writeup), not attributed to this variable "
                    "as a mechanism.")
        if r["reliable_driver"]:
            note = "Reliable: agrees across global-effect methods (XGBoost/OLS/SEM) AND explains a non-trivial variance share."
        elif r["n_methods_in_top_n"] >= MIN_METHODS_IN_TOP_N:
            note = (f"Rank-agreement only: consistently ranked but mean variance share "
                    f"({r['mean_variance_share_pct']:.2f}%) is below the {MIN_VARIANCE_SHARE_PCT}% threshold - "
                    "likely not a meaningful driver despite agreement.")
        else:
            note = "Not consistently ranked highly across global-effect methods."
        if gwr_cols and pd.notna(r.get("gwr_mean_variance_share_pct", np.nan)) and r.get("gwr_in_top_n", False):
            note += (f" GWR (local, spatially-varying) separately ranks this in its own top-{TOP_N_FOR_AGREEMENT} "
                     f"with a mean local variance share of {r['gwr_mean_variance_share_pct']:.2f}% - "
                     "reported for context, not pooled into the reliability decision above.")
        if gwr_cols and bool(r.get("gwr_outlier_flag", False)):
            note += (" WARNING: GWR local coefficient flagged as outlier-pixel-driven "
                      "(mean/median ratio > 3x) - inspect coefficient map before citing GWR value.")
        return note

    combined["reliability_note"] = combined.apply(_note, axis=1)

    combined = combined.sort_values("mean_rank_across_methods")

    if len(rank_cols) >= 2:
        rank_matrix = combined[rank_cols].dropna()
        if len(rank_matrix) >= 3:
            corr_matrix = rank_matrix.corr(method="spearman")
            print("\n Pairwise Spearman rank-agreement across methods "
                  "(GWR included here for reference only - excluded from reliable_driver calc):")
            print(corr_matrix.round(2).to_string())
            corr_matrix.round(3).to_csv(run_dir / "variable_importance_rank_agreement_matrix.csv")

    combined.to_csv(run_dir / "variable_importance_agreement.csv")
    n_reliable = int(combined["reliable_driver"].sum())
    n_spatial_control = int(combined["is_spatial_control"].sum())
    print(f"\n Saved variable_importance_agreement.csv ({len(combined)} variables, "
          f"{len(importance_frames)} methods present: {method_cols})")
    print(f" Primary agreement computed from: {primary_cols} (GWR reported separately, see gwr_mean_rank / "
          f"gwr_mean_variance_share_pct columns)")
    print(f" {n_reliable} of {len(combined)} variables flagged reliable_driver=True "
          f"(>= {MIN_METHODS_IN_TOP_N} methods rank it top-{TOP_N_FOR_AGREEMENT} "
          f"AND mean variance share >= {MIN_VARIANCE_SHARE_PCT}%, excluding spatial controls)")
    print(f" {n_spatial_control} spatial-control coordinate(s) identified and excluded from reliable_driver "
          f"regardless of rank: {[v for v in SPATIAL_CONTROL_VARS if v in combined.index]}")
    display_cols = ["mean_rank_across_methods", "mean_variance_share_pct",
                     "n_methods_in_top_n", "is_spatial_control", "reliable_driver"] + method_cols
    print(combined[display_cols].head(15).to_string())
else:
    print(" [WARNING] No importance files found - check that XGBoost/GWR/OLS/SEM scripts "
          "have been run and saved their importance/coefficient outputs to this run directory.")

# --- 3. Reproducibility appendix -------------------------------------------
repro_rows = [
    {"script": "06_multicollinearity_vif.py", "seed": 42, "key_packages": "statsmodels", "determinism_note": "Deterministic"},
    {"script": "07_spatial_autocorrelation.py", "seed": "42 (semivariogram); unseeded (LISA permutations)", "key_packages": "libpysal, esda, skgstat", "determinism_note": "Moran's I / semivariogram deterministic; LISA p-values vary run-to-run (no seed param passed to Moran_Local)"},
    {"script": "08_xgboost_model.py", "seed": 42, "key_packages": "xgboost, shap, scikit-learn, libpysal, esda", "determinism_note": "Deterministic modulo XGBoost hist-based floating-point variance across hardware/thread count"},
    {"script": "09_spatial_model_gwr.py", "seed": "n/a (closed-form/iterative fits)", "key_packages": "mgwr, spreg, libpysal", "determinism_note": "Bandwidth golden-section search deterministic given fixed data and mgwr version; systematic (non-random) spatial thinning"},
    {"script": "10_consolidate_model_comparison.py", "seed": "n/a", "key_packages": "pandas, numpy, scipy", "determinism_note": "Deterministic - no stochastic computation"},
]
repro_df = pd.DataFrame(repro_rows)
repro_df.to_csv(run_dir / "reproducibility_appendix.csv", index=False)
print(f"\n Saved reproducibility_appendix.csv ({len(repro_df)} scripts)")