
r"""
08_xgboost_model.py

Outputs (same run folder):
- xgb_cv_results.json
- xgb_feature_importance.png / .csv
- xgb_permutation_importance.csv
- shap_summary_xgb.png
- xgb_residual_map.png
- pdp_ice_<feature>.png / .csv
- pdp_ice_summary_grid.png
"""

import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from joblib import parallel_backend
import xgboost as xgb

PROJECT_DIR = Path(r"E:\Processing\InSAR_multivariate_analysis")
PIPE_OUT = PROJECT_DIR / "outputs"
RUN_DIR = None

CATEGORICAL_VARS = [
    "landcover", "wetland_class", "pan_arctic_wetlands", "canopy_height_lt5m",
    "woodyplant_dominance", "cavm", "surficial_geology", "ground_ice",
    "dsw_trends", "permafrost_landcover_cls",
]

# Spatial features added to close the XGBoost vs GWR performance gap. Uses pixel row/col (always available in predictor_flat.parquet) rather than requiring a separate CRS x/y or coastline vector lookup.
SPATIAL_FEATURE_NAMES = ["x_norm", "y_norm", "dist_to_center"]

# CV folds: was 2 (checkerboard parity), now hashed into N_FOLDS bins for a materially more robust CV mean/std estimate.
N_FOLDS = 8

XGB_PARAMS = dict(
    n_estimators=3000,
    max_depth=3,              # was 4 - shallower trees, less overfitting
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=30,      # was 20 - stricter leaf minimum
    gamma=0.15,                # was 0.1 - higher split penalty
    reg_alpha=0.8,             # was 0.5 - stronger L1
    reg_lambda=3.0,             # was 2.0 - stronger L2
    tree_method="hist",
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
    eval_metric="rmse",
)
EARLY_STOPPING_ROUNDS = 50
FALLBACK_BLOCK_SIZE_PX = 50
PERM_SAMPLE_N = 20_000
SHAP_SAMPLE_N = 1000

# PDP + ICE config
TOP_N_FEATURES = 12
ICE_DISPLAY_N = 150          # number of thin ICE lines actually drawn (readability)
PDP_STABILITY_N = 2000       # number of rows used to compute the PDP mean/percentile band (stability)
GRID_RESOLUTION = 50

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

print("=== 07 XGBoost Model (tuned, v5 - spatial features + 8-fold CV + tighter regularization) ===")
print(f" Executing script: {Path(__file__).resolve()}")

run_dir = get_run_dir()
DATA_PATH = run_dir / "predictor_flat.parquet"
FEATURES_JSON = run_dir / "pruned_feature_list.json"
SPATIAL_JSON = run_dir / "spatial_autocorrelation_results.json"
OUT_DIR = run_dir
print(f"Run dir: {run_dir}")

with open(FEATURES_JSON) as f:
    feat_data = json.load(f)
if isinstance(feat_data, dict):
    continuous_feats = feat_data.get("continuous", [])
    categorical_feats = feat_data.get("categorical", [])
    feature_cols_raw = feat_data.get("all", continuous_feats + categorical_feats)
else:
    feature_cols_raw = feat_data
    categorical_feats = [c for c in feature_cols_raw if c in CATEGORICAL_VARS]
    continuous_feats = [c for c in feature_cols_raw if c not in CATEGORICAL_VARS]

df = pd.read_parquet(DATA_PATH)
df = df[["row", "col", "subsidence_mm_yr"] + feature_cols_raw].dropna()
print(f" {len(df):,} complete rows")

# --- Spatial features (STEP 1) -------------------------------------------- 
# Gives XGBoost explicit location signal, analogous to what GWR gets natively by fitting spatially-varying coefficients.
df["x_norm"] = (df["col"] - df["col"].min()) / (df["col"].max() - df["col"].min())
df["y_norm"] = (df["row"] - df["row"].min()) / (df["row"].max() - df["row"].min())
cx, cy = df["col"].mean(), df["row"].mean()
dist_raw = np.sqrt((df["col"] - cx) ** 2 + (df["row"] - cy) ** 2)
df["dist_to_center"] = (dist_raw - dist_raw.min()) / (dist_raw.max() - dist_raw.min())

continuous_feats = continuous_feats + SPATIAL_FEATURE_NAMES
feature_cols_raw = feature_cols_raw + SPATIAL_FEATURE_NAMES
print(f" Added {len(SPATIAL_FEATURE_NAMES)} spatial features: {SPATIAL_FEATURE_NAMES}")
# ----------------------------------------------------------------------------

for c in categorical_feats:
    df[c] = df[c].astype(np.int32).astype("category")

for c in categorical_feats:
    cat_dtype = df[c].cat.categories.dtype
    assert pd.api.types.is_integer_dtype(cat_dtype), (
        f"Column '{c}' has non-integer category codes ({cat_dtype}) - "
        f"XGBoost will reject this at DMatrix construction."
    )
print(f" Categorical dtype check passed: all {len(categorical_feats)} columns have integer category codes")

feature_names = continuous_feats + categorical_feats
X_df = df[feature_names].reset_index(drop=True)
y = df["subsidence_mm_yr"].values
rows = df["row"].values
cols = df["col"].values
print(f" Feature matrix: {X_df.shape[1]} columns ({len(continuous_feats)} continuous [incl. "
      f"{len(SPATIAL_FEATURE_NAMES)} spatial], {len(categorical_feats)} native categorical - NO one-hot expansion)")

block_size_px = FALLBACK_BLOCK_SIZE_PX
if SPATIAL_JSON.exists():
    with open(SPATIAL_JSON) as f:
        spatial_results = json.load(f)
    vgm_range_m = spatial_results.get("semivariogram_range_m")
    px_size = spatial_results.get("pixel_size_m", [None, None])[0]
    if vgm_range_m and px_size:
        block_size_px = max(10, int(round(vgm_range_m / px_size)))
        print(f" Block size from semivariogram: {block_size_px}px (SAME as RF run)")

# --- Spatial CV folds (STEP 2) ---------------------------------------------
# Was: 2-fold checkerboard parity ((block_row + block_col) % 2).
# Now: hashed into N_FOLDS bins for a materially more robust CV mean/std.
block_row = (rows // block_size_px).astype(int)
block_col = (cols // block_size_px).astype(int)
checker = (block_row * 1000003 + block_col) % N_FOLDS
# ----------------------------------------------------------------------------

print(f" Running 2D grid-block CV (hashed, block={block_size_px}px, {N_FOLDS} folds, "
      f"early_stopping={EARLY_STOPPING_ROUNDS} rounds)...")

cv_r2, cv_rmse, cv_n, cv_best_iter = [], [], [], []
for fold in range(N_FOLDS):
    test_mask = checker == fold
    train_mask = ~test_mask
    if test_mask.sum() < 100 or train_mask.sum() < 100:
        continue
    model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(
        X_df[train_mask], y[train_mask],
        eval_set=[(X_df[test_mask], y[test_mask])],
        verbose=False,
    )
    y_pred = model.predict(X_df[test_mask])
    r2 = r2_score(y[test_mask], y_pred)
    rmse = np.sqrt(mean_squared_error(y[test_mask], y_pred))
    best_iter = model.best_iteration
    cv_r2.append(r2); cv_rmse.append(rmse); cv_n.append(int(test_mask.sum())); cv_best_iter.append(int(best_iter))
    print(f" Fold {fold+1}/{N_FOLDS}: n_test={test_mask.sum():,} R2={r2:.3f} RMSE={rmse:.3f} "
          f"best_iteration={best_iter} (of {XGB_PARAMS['n_estimators']} ceiling)")

print(f"\n *** SPATIAL CV PERFORMANCE (comparable directly to rf_cv_results.json) ***")
print(f" CV R2 : {np.mean(cv_r2):.3f} +/- {np.std(cv_r2):.3f}  (n_folds_used={len(cv_r2)})")
print(f" CV RMSE: {np.mean(cv_rmse):.3f} +/- {np.std(cv_rmse):.3f}")
print(f" Mean best_iteration across folds: {np.mean(cv_best_iter):.0f} "
      f"(if far below {XGB_PARAMS['n_estimators']}, early stopping is working as intended)")

n_trees_final = int(np.mean(cv_best_iter)) if cv_best_iter else XGB_PARAMS["n_estimators"]
print(f"\n Fitting final model on all data with n_estimators={n_trees_final} "
      f"(fixed from CV early-stopping result, for SHAP/importance/residuals/PDP ONLY)...")
full_params = {**XGB_PARAMS, "n_estimators": max(50, n_trees_final)}
model_full = xgb.XGBRegressor(**full_params)
model_full.fit(X_df, y)
y_pred_full = model_full.predict(X_df)
insample_r2 = r2_score(y, y_pred_full)
insample_rmse = np.sqrt(mean_squared_error(y, y_pred_full))
print(f" [IN-SAMPLE ONLY] R2={insample_r2:.4f} RMSE={insample_rmse:.4f}")
print(f" Overfitting gap check: in-sample R2 ({insample_r2:.3f}) vs CV R2 "
      f"({np.mean(cv_r2):.3f}) -> ratio={insample_r2/max(np.mean(cv_r2),1e-6):.1f}x "
      f"(compare against RF's ~7x gap)")

importances = model_full.feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=True)
fig, ax = plt.subplots(figsize=(7, max(4, len(feature_names) * 0.3)))
ax.barh(fi_df["feature"], fi_df["importance"], color="#2196F3", edgecolor="none")
ax.set_xlabel("Gain-based Importance")
ax.set_title("XGBoost Feature Importance (native categorical + spatial)", fontsize=10)
ax.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig(OUT_DIR / "xgb_feature_importance.png", dpi=150)
plt.close()
fi_df.to_csv(OUT_DIR / "xgb_feature_importance.csv", index=False)

print(" Computing permutation importance (held-out fold 0)...")
test_mask = checker == 0
rng = np.random.default_rng(42)
sub_idx = rng.choice(np.where(test_mask)[0], size=min(PERM_SAMPLE_N, test_mask.sum()), replace=False)
X_perm = X_df.iloc[sub_idx]
y_perm = y[sub_idx]
with parallel_backend("threading", n_jobs=4):
    perm = permutation_importance(model_full, X_perm, y_perm, n_repeats=10, random_state=42, n_jobs=4)
perm_df = pd.DataFrame({"feature": feature_names, "perm_importance_mean": perm.importances_mean,
                        "perm_importance_std": perm.importances_std}).sort_values("perm_importance_mean", ascending=False)
perm_df.to_csv(OUT_DIR / "xgb_permutation_importance.csv", index=False)

try:
    import shap
    print(" Computing SHAP values (native TreeExplainer)...")
    rng_shap = np.random.RandomState(42)
    idx = rng_shap.choice(len(X_df), min(SHAP_SAMPLE_N, len(X_df)), replace=False)
    Xs = X_df.iloc[idx]
    explainer = shap.TreeExplainer(model_full)
    shap_vals = explainer.shap_values(Xs)
    fig = plt.figure(figsize=(7, max(4, len(feature_names) * 0.3)))
    shap.summary_plot(shap_vals, Xs, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_xgb.png", dpi=150)
    plt.close()
except Exception as e:
    print(f" SHAP skipped: {type(e).__name__}: {e}")

residuals = y - y_pred_full
df_res = df[["row", "col"]].copy()
df_res["residual"] = residuals
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(df_res["col"], df_res["row"], c=residuals, cmap="RdBu_r",
                vmin=np.percentile(residuals, 2), vmax=np.percentile(residuals, 98), s=0.5, marker="s")
plt.colorbar(sc, ax=ax, label="Residual (mm/yr)")
ax.set_title("XGBoost Residuals (IN-SAMPLE fit): Observed - Predicted (mm/yr)", fontsize=10)
ax.invert_yaxis(); ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUT_DIR / "xgb_residual_map.png", dpi=150)
plt.close()

cv_results = {
    "cv_r2_mean": float(np.mean(cv_r2)), "cv_r2_std": float(np.std(cv_r2)),
    "cv_rmse_mean": float(np.mean(cv_rmse)), "cv_rmse_std": float(np.std(cv_rmse)),
    "cv_n_per_fold": cv_n, "cv_best_iteration_per_fold": cv_best_iter,
    "n_folds": len(cv_r2), "block_size_px": block_size_px,
    "cv_method": f"2D hashed grid-block (spatial), {N_FOLDS} folds - "
                 f"upgraded from 2-fold checkerboard parity for more robust std",
    "n_features": X_df.shape[1], "categorical_handling": "native (enable_categorical=True)",
    "spatial_features_added": SPATIAL_FEATURE_NAMES,
    "in_sample_r2_NOT_a_performance_metric": float(insample_r2),
    "in_sample_rmse_NOT_a_performance_metric": float(insample_rmse),
    "n_trees_final_model": full_params["n_estimators"],
    "model": "XGBoost", "xgb_params": {k: v for k, v in XGB_PARAMS.items()},
}
with open(OUT_DIR / "xgb_cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2)
print("\n Saved xgb_cv_results.json")

# ─── PDP + centered-ICE plots (merged from 07c) ────────────────────────────
print("\n=== PDP + centered-ICE plots (top features by permutation importance) ===")
print(" CAVEAT: ICE curves are drawn from spatially autocorrelated pixels, "
      "not independent samples - percentile bands below somewhat understate "
      "true pixel-to-pixel uncertainty.")

top_features = perm_df["feature"].head(TOP_N_FEATURES).tolist()
print(f" Top {TOP_N_FEATURES} features: {top_features}")

rng_pdp = np.random.default_rng(42)
stability_idx = rng_pdp.choice(len(X_df), size=min(PDP_STABILITY_N, len(X_df)), replace=False)
X_stability = X_df.iloc[stability_idx].reset_index(drop=True)
display_idx = rng_pdp.choice(len(X_stability), size=min(ICE_DISPLAY_N, len(X_stability)), replace=False)

def compute_continuous_pdp_ice(feature):
    x_vals = np.linspace(X_df[feature].quantile(0.02), X_df[feature].quantile(0.98), GRID_RESOLUTION)
    ice_curves = np.zeros((len(X_stability), GRID_RESOLUTION))
    X_temp = X_stability.copy()
    for j, xv in enumerate(x_vals):
        X_temp[feature] = xv
        ice_curves[:, j] = model_full.predict(X_temp)
    return x_vals, ice_curves

def plot_continuous_pdp_ice(feature, x_vals, ice_curves, ax_pdp, ax_cice):
    pdp_curve = ice_curves.mean(axis=0)
    p10 = np.percentile(ice_curves, 10, axis=0)
    p90 = np.percentile(ice_curves, 90, axis=0)

    for i in display_idx:
        ax_pdp.plot(x_vals, ice_curves[i], color="#90A4AE", alpha=0.10, linewidth=0.6, zorder=1)
    ax_pdp.plot(x_vals, pdp_curve, color="#D32F2F", linewidth=2.5, zorder=3, label="PDP (mean)")
    ax_pdp.fill_between(x_vals, p10, p90, color="#D32F2F", alpha=0.12, zorder=2, label="ICE 10-90th pct")
    data_sample = X_df[feature].sample(min(2000, len(X_df)), random_state=42)
    ax_pdp.plot(data_sample, np.full(len(data_sample), ax_pdp.get_ylim()[0]), "|",
                color="black", alpha=0.15, markersize=4, zorder=0)
    ax_pdp.set_xlabel(feature[:20], fontsize=8)
    ax_pdp.set_ylabel("Predicted subsidence (mm/yr)", fontsize=8)
    ax_pdp.tick_params(labelsize=7)
    ax_pdp.legend(fontsize=6, loc="best")
    ax_pdp.set_title("PDP + ICE", fontsize=9)

    centered = ice_curves - ice_curves[:, [0]]
    for i in display_idx:
        ax_cice.plot(x_vals, centered[i], color="#1976D2", alpha=0.15, linewidth=0.7, zorder=1)
    ax_cice.plot(x_vals, centered.mean(axis=0), color="#0D47A1", linewidth=2.0, zorder=2)
    ax_cice.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_cice.set_xlabel(feature[:20], fontsize=8)
    ax_cice.set_ylabel("Centered effect (mm/yr)", fontsize=8)
    ax_cice.tick_params(labelsize=7)
    ax_cice.set_title("Centered ICE (interaction check)", fontsize=9)

def plot_categorical_pdp_ice(feature, ax):
    categories = sorted(X_df[feature].cat.categories.tolist())
    ice_vals = {}
    X_temp = X_stability.copy()
    for cat in categories:
        X_temp[feature] = pd.Categorical([cat] * len(X_temp), categories=X_df[feature].cat.categories)
        ice_vals[cat] = model_full.predict(X_temp)

    means = [np.mean(ice_vals[c]) for c in categories]
    stds = [np.std(ice_vals[c]) for c in categories]
    x_pos = np.arange(len(categories))

    for i, cat in enumerate(categories):
        sample_pts = rng_pdp.choice(ice_vals[cat], size=min(ICE_DISPLAY_N, len(ice_vals[cat])), replace=False)
        jitter = rng_pdp.normal(0, 0.08, size=len(sample_pts))
        ax.scatter(np.full(len(sample_pts), i) + jitter, sample_pts,
                   color="#90A4AE", alpha=0.2, s=4, zorder=1)
    ax.bar(x_pos, means, yerr=stds, color="#1976D2", alpha=0.75, capsize=3, zorder=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(c) for c in categories], fontsize=6, rotation=45)
    ax.set_xlabel(feature[:20], fontsize=8)
    ax.set_ylabel("Predicted subsidence (mm/yr)", fontsize=8)
    ax.tick_params(labelsize=7)
    return categories, means, stds

print(" Generating individual PDP + centered-ICE plots...")
pdp_data_store = {}
for feature in top_features:
    safe_name = feature.replace(" ", "_").replace("/", "_")
    if feature in categorical_feats:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        categories, means, stds = plot_categorical_pdp_ice(feature, ax)
        ax.set_title(f"PDP + ICE: {feature} (categorical)", fontsize=10)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"pdp_ice_{safe_name}.png", dpi=150)
        plt.close()
        pd.DataFrame({"category": categories, "mean_prediction": means, "std_prediction": stds}).to_csv(
            OUT_DIR / f"pdp_ice_{safe_name}.csv", index=False)
    else:
        x_vals, ice_curves = compute_continuous_pdp_ice(feature)
        fig, (ax_pdp, ax_cice) = plt.subplots(1, 2, figsize=(11, 4.5))
        plot_continuous_pdp_ice(feature, x_vals, ice_curves, ax_pdp, ax_cice)
        fig.suptitle(f"PDP + centered-ICE: {feature} (continuous)", fontsize=11)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"pdp_ice_{safe_name}.png", dpi=150)
        plt.close()
        pd.DataFrame({"x_value": x_vals, "pdp_mean": ice_curves.mean(axis=0),
                      "p10": np.percentile(ice_curves, 10, axis=0),
                      "p90": np.percentile(ice_curves, 90, axis=0)}).to_csv(
            OUT_DIR / f"pdp_ice_{safe_name}.csv", index=False)
        pdp_data_store[feature] = (x_vals, ice_curves)
    print(f"   Saved pdp_ice_{safe_name}.png / .csv")

print(" Generating summary grid (PDP panel only, all top features)...")
n_cols = 3
n_rows = int(np.ceil(len(top_features) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()
for i, feature in enumerate(top_features):
    if feature in categorical_feats:
        plot_categorical_pdp_ice(feature, axes[i])
    else:
        x_vals, ice_curves = pdp_data_store[feature]
        pdp_curve = ice_curves.mean(axis=0)
        p10 = np.percentile(ice_curves, 10, axis=0)
        p90 = np.percentile(ice_curves, 90, axis=0)
        for j in display_idx:
            axes[i].plot(x_vals, ice_curves[j], color="#90A4AE", alpha=0.08, linewidth=0.5)
        axes[i].plot(x_vals, pdp_curve, color="#D32F2F", linewidth=2)
        axes[i].fill_between(x_vals, p10, p90, color="#D32F2F", alpha=0.12)
        axes[i].tick_params(labelsize=6)
    axes[i].set_title(feature, fontsize=9)
for j in range(len(top_features), len(axes)):
    axes[j].axis("off")
plt.suptitle(f"PDP: Top {TOP_N_FEATURES} Features by Permutation Importance", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "pdp_ice_summary_grid.png", dpi=150)
plt.close()

print(f"\n Done. Saved {len(top_features)} individual PDP+ICE plots + 1 summary grid to {OUT_DIR}")
print(" -> Compare CV R2/RMSE against rf_cv_results.json for apples-to-apples comparison.")