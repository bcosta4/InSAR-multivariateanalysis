"""
08_random_forest_model.py
──────────────────────────
Random Forest regression with:
  - Spatial block cross-validation (avoids spatial autocorrelation leakage)
  - GPU via cuML if available, else scikit-learn CPU with n_jobs=-1
  - SHAP feature importance
  - Permutation importance
  - Residual map (predicted vs. actual subsidence)
"""

import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

DATA_PATH    = Path(__file__).parent / "outputs" / "predictor_flat.parquet"
FEATURES_JSON = Path(__file__).parent / "outputs" / "pruned_feature_list.json"
OUT_DIR      = Path(__file__).parent / "outputs"

N_ESTIMATORS = 500
MAX_DEPTH    = None
N_SPATIAL_BLOCKS = 5   # spatial block CV folds
BLOCK_AXIS   = "row"   # split by row (NS direction) to create spatial blocks

print("=== 08 Random Forest Model ===")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
with open(FEATURES_JSON) as f:
    feature_cols = json.load(f)

df = df[["row", "col", "subsidence_mm_yr"] + feature_cols].dropna()
print(f"  {len(df):,} complete rows, {len(feature_cols)} features")

X = df[feature_cols].values
y = df["subsidence_mm_yr"].values
rows = df["row"].values

# ── GPU or CPU ────────────────────────────────────────────────────────────────
try:
    from cuml.ensemble import RandomForestRegressor
    print("  Backend: cuML GPU")
    gpu = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    print("  Backend: scikit-learn CPU (n_jobs=-1)")
    gpu = False

rf_kwargs = dict(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                 random_state=42)
if not gpu:
    rf_kwargs["n_jobs"] = -1

# ── Spatial block CV ──────────────────────────────────────────────────────────
print(f"  Running {N_SPATIAL_BLOCKS}-fold spatial block CV...")

block_edges = np.percentile(rows, np.linspace(0, 100, N_SPATIAL_BLOCKS + 1))
block_ids   = np.digitize(rows, block_edges[1:-1])

cv_r2, cv_rmse = [], []
for fold in range(N_SPATIAL_BLOCKS):
    test_mask  = block_ids == fold
    train_mask = ~test_mask
    if test_mask.sum() < 100:
        continue
    rf = RandomForestRegressor(**rf_kwargs)
    rf.fit(X[train_mask], y[train_mask])
    y_pred = rf.predict(X[test_mask])
    r2   = r2_score(y[test_mask], y_pred)
    rmse = np.sqrt(mean_squared_error(y[test_mask], y_pred))
    cv_r2.append(r2)
    cv_rmse.append(rmse)
    print(f"    Fold {fold+1}: R²={r2:.3f}  RMSE={rmse:.3f}")

print(f"\n  CV R²  : {np.mean(cv_r2):.3f} ± {np.std(cv_r2):.3f}")
print(f"  CV RMSE: {np.mean(cv_rmse):.3f} ± {np.std(cv_rmse):.3f}")

# ── Full model (all data) ─────────────────────────────────────────────────────
print("  Fitting full model on all data...")
rf_full = RandomForestRegressor(**rf_kwargs)
rf_full.fit(X, y)
y_pred_full = rf_full.predict(X)
print(f"  Full model R²   : {r2_score(y, y_pred_full):.4f}")
print(f"  Full model RMSE : {np.sqrt(mean_squared_error(y, y_pred_full)):.4f}")

# ── Feature importance ────────────────────────────────────────────────────────
importances = rf_full.feature_importances_
fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(7, max(4, len(feature_cols)*0.4)))
ax.barh(fi_df["feature"], fi_df["importance"], color="#2196F3", edgecolor="none")
ax.set_xlabel("Mean Decrease in Impurity")
ax.set_title("Random Forest Feature Importance", fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "rf_feature_importance.png", dpi=150)
plt.close()
fi_df.to_csv(OUT_DIR / "rf_feature_importance.csv", index=False)
print("  Saved rf_feature_importance.png")

# ── SHAP ──────────────────────────────────────────────────────────────────────
try:
    import shap
    print("  Computing SHAP values (sample of 5000)...")
    Xs = X[:5000] if len(X) > 5000 else X
    explainer = shap.TreeExplainer(rf_full)
    shap_vals = explainer.shap_values(Xs)

    fig, ax = plt.subplots(figsize=(7, max(4, len(feature_cols)*0.4)))
    shap.summary_plot(shap_vals, Xs, feature_names=feature_cols,
                      show=False, plot_size=(7, max(4, len(feature_cols)*0.4)))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=150)
    plt.close()
    print("  Saved shap_summary.png")
except Exception as e:
    print(f"  SHAP skipped: {e}")

# ── Residual map ──────────────────────────────────────────────────────────────
residuals = y - y_pred_full
df_res = df[["row", "col"]].copy()
df_res["residual"] = residuals

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(df_res["col"], df_res["row"],
                c=residuals, cmap="RdBu_r",
                vmin=np.percentile(residuals, 2),
                vmax=np.percentile(residuals, 98),
                s=0.5, marker="s")
plt.colorbar(sc, ax=ax, label="Residual (mm/yr)")
ax.set_title("RF Residuals: Observed − Predicted (mm/yr)", fontsize=10)
ax.invert_yaxis()
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUT_DIR / "rf_residual_map.png", dpi=150)
plt.close()
print("  Saved rf_residual_map.png")

# Save CV results
cv_results = {
    "cv_r2_mean": float(np.mean(cv_r2)),
    "cv_r2_std":  float(np.std(cv_r2)),
    "cv_rmse_mean": float(np.mean(cv_rmse)),
    "cv_rmse_std":  float(np.std(cv_rmse)),
    "n_folds": N_SPATIAL_BLOCKS,
    "n_estimators": N_ESTIMATORS,
}
with open(OUT_DIR / "rf_cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2)

print("  Done. → Next: run 09_spatial_model_gwr.py")
