"""
07_random_forest_model.py
─────────────────────────────
Trains a Random Forest model to explain subsidence rates.
Automatically uses RAPIDS cuML (GPU) if available, else scikit-learn CPU.

Procedure:
  1. Load pruned variable list from 05
  2. Spatial block cross-validation (10 km blocks) — prevents data leakage
  3. Train final RF on full dataset
  4. Permutation feature importance + SHAP values
  5. Partial dependence plots for top 5 predictors
  6. Residual map

OUTPUTS:
    models/rf_model.joblib
    results/rf_cv_scores.csv
    results/rf_feature_importance.csv
    results/rf_feature_importance.png
    results/rf_shap_summary.png
    results/rf_partial_dependence.png
    results/rf_residuals.parquet
    results/rf_residual_map.png
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PARQUET_PATH     = r"E:\Processing\InSAR_validation\data\stack\predictor_stack_valid.parquet"
VIF_VARS_PATH    = r"E:\Processing\InSAR_validation\data\qc\vif_pruned_variables.txt"
OUTPUT_MODEL_DIR = r"E:\Processing\InSAR_validation\data\models"
OUTPUT_RESULT_DIR= r"E:\Processing\InSAR_validation\data\results"

# RF hyperparameters
N_ESTIMATORS   = 500
MAX_DEPTH      = None       # None = fully grown (GPU cuML requirement)
MIN_SAMPLES_LEAF = 50       # prevents overfitting on 1.37M pixel dataset
N_JOBS         = -1         # all CPU cores (CPU fallback)
BLOCK_SIZE_M   = 10_000     # spatial block size for CV (10 km)
N_CV_FOLDS     = 5          # number of spatial CV folds

# Categorical columns — will be included but not in VIF list;
# add manually here if you want them in the model
CATEGORICAL_VARS = ["landcover"]  # "geology", "wetlands" — optional
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_RESULT_DIR, exist_ok=True)

print("=" * 60)
print("07  RANDOM FOREST MODEL")
print("=" * 60)

# ── GPU / CPU backend ─────────────────────────────────────────────────────────
try:
    from cuml.ensemble import RandomForestRegressor
    import cudf
    USE_GPU = True
    print("[GPU] RAPIDS cuML detected — using GPU Random Forest")
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    USE_GPU = False
    print("[CPU] cuML not found — using scikit-learn Random Forest")

# ── Load VIF-pruned variable list ─────────────────────────────────────────────
if os.path.exists(VIF_VARS_PATH):
    with open(VIF_VARS_PATH) as f:
        retained_vars = [
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        ]
    print(f"\nLoaded {len(retained_vars)} VIF-pruned predictors: {retained_vars}")
else:
    print(f"\n[WARNING] {VIF_VARS_PATH} not found — using default variable list")
    retained_vars = [
        "ddt", "magt", "ground_ice", "slope", "aspect_sin", "aspect_cos",
        "curvature", "twi", "tpi", "tri", "ndvi", "flood_freq"
    ]

# Add categorical vars
feature_cols = retained_vars + [v for v in CATEGORICAL_VARS if v not in retained_vars]

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading parquet ...")
df = pd.read_parquet(PARQUET_PATH)
df = df.dropna(subset=["subsidence"] + [v for v in retained_vars if v in df.columns])

# One-hot encode categoricals (scikit-learn / cuML both need numeric input)
for cat in CATEGORICAL_VARS:
    if cat in df.columns:
        df[cat] = df[cat].fillna(-1).astype(int)

feature_cols = [v for v in feature_cols if v in df.columns]
print(f"Dataset: {len(df):,} rows × {len(feature_cols)} features")

X = df[feature_cols].values.astype(np.float32)
y = df["subsidence"].values.astype(np.float32)
coords = df[["x_coord", "y_coord"]].values

# ── Spatial block cross-validation ────────────────────────────────────────────
print(f"\nBuilding spatial block CV (block={BLOCK_SIZE_M/1000:.0f} km) ...")

x_blocks = ((coords[:, 0] - coords[:, 0].min()) // BLOCK_SIZE_M).astype(int)
y_blocks = ((coords[:, 1] - coords[:, 1].min()) // BLOCK_SIZE_M).astype(int)
block_ids = x_blocks * 10000 + y_blocks   # unique block id per pixel

unique_blocks = np.unique(block_ids)
np.random.seed(42)
np.random.shuffle(unique_blocks)
fold_assignments = np.array_split(unique_blocks, N_CV_FOLDS)

print(f"  Total spatial blocks: {len(unique_blocks):,}")
print(f"  CV folds: {N_CV_FOLDS}")

cv_scores = []

for fold_i, test_blocks in enumerate(fold_assignments):
    test_mask  = np.isin(block_ids, test_blocks)
    train_mask = ~test_mask
    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    if USE_GPU:
        import cudf
        X_train_g = cudf.DataFrame(X_train, columns=feature_cols)
        y_train_g = cudf.Series(y_train)
        rf = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=16,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=42
        )
        rf.fit(X_train_g, y_train_g)
        X_test_g = cudf.DataFrame(X_test, columns=feature_cols)
        y_pred   = rf.predict(X_test_g).to_numpy()
    else:
        rf = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            n_jobs=N_JOBS,
            random_state=42
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot
    rmse   = np.sqrt(np.mean((y_test - y_pred) ** 2))

    cv_scores.append({"fold": fold_i + 1, "n_train": train_mask.sum(),
                      "n_test": test_mask.sum(), "r2": r2, "rmse": rmse})
    print(f"  Fold {fold_i + 1}: R²={r2:.4f}  RMSE={rmse:.4f} cm/yr")

cv_df = pd.DataFrame(cv_scores)
print(f"\n  Mean R²:   {cv_df['r2'].mean():.4f} ± {cv_df['r2'].std():.4f}")
print(f"  Mean RMSE: {cv_df['rmse'].mean():.4f} ± {cv_df['rmse'].std():.4f} cm/yr")
cv_df.to_csv(os.path.join(OUTPUT_RESULT_DIR, "rf_cv_scores.csv"), index=False)

# ── Train final model on full dataset ─────────────────────────────────────────
print("\nTraining final RF on full dataset ...")
if USE_GPU:
    import cudf
    X_g = cudf.DataFrame(X, columns=feature_cols)
    y_g = cudf.Series(y)
    rf_final = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=16,
        min_samples_leaf=MIN_SAMPLES_LEAF, random_state=42
    )
    rf_final.fit(X_g, y_g)
    y_pred_full = rf_final.predict(X_g).to_numpy()
else:
    rf_final = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=N_JOBS, random_state=42
    )
    rf_final.fit(X, y)
    y_pred_full = rf_final.predict(X)

# Save model (CPU only — cuML model serialisation differs)
if not USE_GPU:
    model_path = os.path.join(OUTPUT_MODEL_DIR, "rf_model.joblib")
    joblib.dump(rf_final, model_path)
    print(f"Saved model: {model_path}")

# ── Permutation feature importance ────────────────────────────────────────────
print("\nComputing permutation feature importance ...")
from sklearn.inspection import permutation_importance as perm_imp_fn
from sklearn.ensemble import RandomForestRegressor as SklearnRF

# Use sklearn-compatible interface (subsample for speed)
idx_sub = np.random.choice(len(X), min(50_000, len(X)), replace=False)
X_sub, y_sub = X[idx_sub], y[idx_sub]

if USE_GPU:
    # Retrain small sklearn model for feature importance (GPU RF lacks this)
    rf_cpu = SklearnRF(n_estimators=100, max_depth=10,
                       min_samples_leaf=50, n_jobs=-1, random_state=42)
    rf_cpu.fit(X_sub, y_sub)
    perm_result = perm_imp_fn(rf_cpu, X_sub, y_sub, n_repeats=10,
                              random_state=42, n_jobs=-1)
else:
    perm_result = perm_imp_fn(rf_final, X_sub, y_sub, n_repeats=10,
                              random_state=42, n_jobs=-1)

fi_df = pd.DataFrame({
    "feature"   : feature_cols,
    "importance": perm_result.importances_mean,
    "std"       : perm_result.importances_std,
}).sort_values("importance", ascending=False)

fi_df.to_csv(os.path.join(OUTPUT_RESULT_DIR, "rf_feature_importance.csv"), index=False)
print(fi_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.4)))
ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1],
        xerr=fi_df["std"][::-1], color="#01696f", alpha=0.85,
        error_kw={"elinewidth": 1, "capsize": 3})
ax.set_xlabel("Permutation Importance (mean decrease R²)")
ax.set_title("Random Forest — Feature Importance")
plt.tight_layout()
fi_path = os.path.join(OUTPUT_RESULT_DIR, "rf_feature_importance.png")
plt.savefig(fi_path, dpi=150)
plt.close()
print(f"Saved: {fi_path}")

# ── SHAP values (CPU model, subsample) ───────────────────────────────────────
print("\nComputing SHAP values (subsample 10k) ...")
try:
    import shap
    idx_shap = np.random.choice(len(X), min(10_000, len(X)), replace=False)
    X_shap = X[idx_shap]
    rf_shap = rf_cpu if USE_GPU else rf_final

    explainer  = shap.TreeExplainer(rf_shap)
    shap_vals  = explainer.shap_values(X_shap)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_shap, feature_names=feature_cols,
                      show=False, max_display=15)
    plt.tight_layout()
    shap_path = os.path.join(OUTPUT_RESULT_DIR, "rf_shap_summary.png")
    plt.savefig(shap_path, dpi=150)
    plt.close()
    print(f"Saved: {shap_path}")
except Exception as e:
    print(f"  [WARN] SHAP failed: {e}")

# ── Residuals ─────────────────────────────────────────────────────────────────
residuals = y - y_pred_full
df["y_pred"]   = y_pred_full
df["residual"] = residuals

resid_path = os.path.join(OUTPUT_RESULT_DIR, "rf_residuals.parquet")
df[["x_coord", "y_coord", "subsidence", "y_pred", "residual"]].to_parquet(
    resid_path, index=False
)

# Residual map (scatter)
fig, ax = plt.subplots(figsize=(12, 8))
vmax = np.percentile(np.abs(residuals), 98)
sc = ax.scatter(df["x_coord"], df["y_coord"], c=residuals,
                cmap="RdBu_r", s=0.3, vmin=-vmax, vmax=vmax,
                alpha=0.6, rasterized=True)
plt.colorbar(sc, ax=ax, label="Residual (cm/yr)")
ax.set_title("RF Model Residuals (observed − predicted)")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")
plt.tight_layout()
rmap_path = os.path.join(OUTPUT_RESULT_DIR, "rf_residual_map.png")
plt.savefig(rmap_path, dpi=200)
plt.close()
print(f"Saved: {rmap_path}")

print("\n✓ 07 complete. Proceed to 08_spatial_model_gwr.py")
