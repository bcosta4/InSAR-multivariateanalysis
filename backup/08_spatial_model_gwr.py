"""
08_spatial_model_gwr.py
─────────────────────────────
Fits spatial regression models to explain subsidence:
  1. OLS baseline (spreg)
  2. Spatial Error Model — SEM (spreg ML_Error)
  3. Stratified spatial GWR on representative sample (~50k pixels)
     using mgwr library
  4. Maps local R² and local coefficients
  5. Moran's I on OLS residuals → confirms SEM need

OUTPUTS:
    results/ols_summary.txt
    results/sem_summary.txt
    results/gwr_local_r2_map.png
    results/gwr_coeff_<varname>.png  (one per predictor)
    results/gwr_results.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PARQUET_PATH     = r"E:\Processing\InSAR_validation\data\stack\predictor_stack_valid.parquet"
VIF_VARS_PATH    = r"E:\Processing\InSAR_validation\data\qc\vif_pruned_variables.txt"
OUTPUT_DIR       = r"E:\Processing\InSAR_validation\data\results"
QC_DIR           = r"E:\Processing\InSAR_validation\data\qc"

GWR_SAMPLE_N   = 50_000      # MGWR tractable at this size; increase if RAM allows
K_SPATIAL_W    = 8           # k-nearest neighbours for PySAL weights
OLS_SAMPLE_N   = 200_000     # OLS / SEM sample (full dataset for SEM if RAM allows)
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("08  SPATIAL REGRESSION + GWR")
print("=" * 60)

# ── Load VIF variable list ────────────────────────────────────────────────────
if os.path.exists(VIF_VARS_PATH):
    with open(VIF_VARS_PATH) as f:
        retained_vars = [l.strip() for l in f if l.strip() and not l.startswith("#")]
else:
    retained_vars = ["ddt", "magt", "ground_ice", "slope", "twi", "tpi", "ndvi"]

print(f"\nUsing {len(retained_vars)} predictors: {retained_vars}")

# ── Load data ─────────────────────────────────────────────────────────────────
df_full = pd.read_parquet(PARQUET_PATH)
df_full = df_full.dropna(subset=["subsidence"] + retained_vars)
print(f"Loaded {len(df_full):,} complete-case pixels")

# ─────────────────────────────────────────────────────────────────────────────
#  PART A: OLS + SPATIAL ERROR MODEL
# ─────────────────────────────────────────────────────────────────────────────
try:
    from libpysal.weights import KNN
    from spreg import OLS, ML_Error, ML_Lag
    from esda.moran import Moran
    SPREG_AVAILABLE = True
except ImportError:
    print("[WARN] spreg / libpysal not available. Skipping OLS/SEM.")
    SPREG_AVAILABLE = False

if SPREG_AVAILABLE:
    print(f"\nBuilding OLS/SEM sample ({min(OLS_SAMPLE_N, len(df_full)):,} pixels) ...")
    df_ols = df_full.sample(n=min(OLS_SAMPLE_N, len(df_full)), random_state=42)

    coords_ols = df_ols[["x_coord", "y_coord"]].values
    y_ols = df_ols["subsidence"].values.reshape(-1, 1)
    X_ols = df_ols[retained_vars].values

    print("Building KNN weights ...")
    w_ols = KNN.from_array(coords_ols, k=K_SPATIAL_W)
    w_ols.transform = "r"

    # OLS
    print("Fitting OLS ...")
    ols = OLS(y_ols, X_ols, w=w_ols,
              name_y="subsidence", name_x=retained_vars,
              spat_diag=True)

    ols_summary = str(ols.summary)
    ols_path = os.path.join(OUTPUT_DIR, "ols_summary.txt")
    with open(ols_path, "w") as f:
        f.write(ols_summary)
    print(f"OLS R²={ols.r2:.4f}  AIC={ols.aic:.2f}")
    print(f"Saved: {ols_path}")

    # Moran's I on OLS residuals
    mi_resid = Moran(ols.u.flatten(), w_ols, permutations=999)
    print(f"\nMoran's I on OLS residuals: I={mi_resid.I:.4f}  p={mi_resid.p_sim:.4f}")
    if mi_resid.p_sim < 0.05:
        print("  → Significant residual autocorrelation. SEM is justified.")

    # Spatial Error Model
    print("\nFitting Spatial Error Model (SEM) ...")
    sem = ML_Error(y_ols, X_ols, w=w_ols,
                   name_y="subsidence", name_x=retained_vars)
    sem_summary = str(sem.summary)
    sem_path = os.path.join(OUTPUT_DIR, "sem_summary.txt")
    with open(sem_path, "w") as f:
        f.write(sem_summary)
    print(f"SEM pseudo-R²={sem.pr2:.4f}  AIC={sem.aic:.2f}  λ={sem.lam:.4f}")
    print(f"Saved: {sem_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  PART B: MGWR on stratified spatial sample
# ─────────────────────────────────────────────────────────────────────────────
try:
    from mgwr.gwr import GWR, MGWR
    from mgwr.sel_bw import Sel_BW
    MGWR_AVAILABLE = True
except ImportError:
    print("[WARN] mgwr not installed. Install: pip install mgwr")
    MGWR_AVAILABLE = False

if MGWR_AVAILABLE:
    print(f"\n── GWR on stratified sample ({GWR_SAMPLE_N:,} pixels) ──")

    # Stratified spatial sample: take proportional samples from spatial blocks
    block_size = 5_000   # 5 km blocks for stratification
    df_full["_xb"] = ((df_full["x_coord"] - df_full["x_coord"].min()) // block_size).astype(int)
    df_full["_yb"] = ((df_full["y_coord"] - df_full["y_coord"].min()) // block_size).astype(int)
    df_full["_block"] = df_full["_xb"].astype(str) + "_" + df_full["_yb"].astype(str)

    n_blocks   = df_full["_block"].nunique()
    per_block  = max(1, GWR_SAMPLE_N // n_blocks)
    df_gwr     = (df_full.groupby("_block", group_keys=False)
                  .apply(lambda g: g.sample(min(per_block, len(g)), random_state=42))
                  .sample(frac=1, random_state=42).head(GWR_SAMPLE_N))

    print(f"  Stratified sample: {len(df_gwr):,} pixels from {n_blocks:,} blocks")

    coords_g   = df_gwr[["x_coord", "y_coord"]].values
    y_g        = df_gwr["subsidence"].values.reshape(-1, 1)
    X_g        = df_gwr[retained_vars].values

    # Normalise X for GWR stability
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_g_sc = scaler.fit_transform(X_g)

    # Bandwidth selection (AICc)
    print("  Selecting GWR bandwidth via AICc ...")
    try:
        bw_selector = Sel_BW(coords_g, y_g, X_g_sc)
        bw = bw_selector.search(criterion="AICc")
        print(f"  Optimal bandwidth: {bw:.1f} m")
    except Exception as e:
        bw = len(df_gwr) * 0.1    # fallback: 10% of observations
        print(f"  [WARN] Bandwidth selection failed ({e}). Using fallback: {bw:.0f}")

    # Fit GWR
    print("  Fitting GWR ...")
    gwr_model   = GWR(coords_g, y_g, X_g_sc, bw=bw)
    gwr_results = gwr_model.fit()

    print(f"  GWR AICc={gwr_results.aicc:.2f}  R²={gwr_results.R2:.4f}")

    # Local R² and coefficients
    local_r2     = gwr_results.localR2
    local_coeffs = gwr_results.params   # shape: (n_pixels, n_vars+1)

    gwr_out = df_gwr[["x_coord", "y_coord", "subsidence"]].copy()
    gwr_out["local_r2"] = local_r2

    for j, vname in enumerate(["intercept"] + retained_vars):
        gwr_out[f"coeff_{vname}"] = local_coeffs[:, j]

    gwr_csv = os.path.join(OUTPUT_DIR, "gwr_results.csv")
    gwr_out.to_csv(gwr_csv, index=False)
    print(f"  Saved: {gwr_csv}")

    # Local R² map
    fig, ax = plt.subplots(figsize=(11, 9))
    sc = ax.scatter(gwr_out["x_coord"], gwr_out["y_coord"],
                    c=gwr_out["local_r2"], cmap="YlOrRd",
                    s=2, vmin=0, vmax=1, alpha=0.7, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Local R²")
    ax.set_title(f"GWR Local R² — Subsidence Rate\n(n={len(df_gwr):,}, bw={bw:.0f} m)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    plt.tight_layout()
    lr2_path = os.path.join(OUTPUT_DIR, "gwr_local_r2_map.png")
    plt.savefig(lr2_path, dpi=150)
    plt.close()
    print(f"  Saved: {lr2_path}")

    # Local coefficient maps — top 3 predictors by RF importance (if available)
    fi_path = os.path.join(OUTPUT_DIR, "rf_feature_importance.csv")
    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path)
        top_vars = fi_df.head(3)["feature"].tolist()
    else:
        top_vars = retained_vars[:3]

    for vname in top_vars:
        coeff_col = f"coeff_{vname}"
        if coeff_col not in gwr_out.columns:
            continue
        vmax_c = np.percentile(np.abs(gwr_out[coeff_col]), 98)
        fig, ax = plt.subplots(figsize=(11, 9))
        sc = ax.scatter(gwr_out["x_coord"], gwr_out["y_coord"],
                        c=gwr_out[coeff_col], cmap="RdBu_r",
                        s=2, vmin=-vmax_c, vmax=vmax_c,
                        alpha=0.7, rasterized=True)
        plt.colorbar(sc, ax=ax, label=f"Local coeff — {vname}")
        ax.set_title(f"GWR Local Coefficient: {vname}")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        plt.tight_layout()
        coeff_path = os.path.join(OUTPUT_DIR, f"gwr_coeff_{vname}.png")
        plt.savefig(coeff_path, dpi=150)
        plt.close()
        print(f"  Saved: {coeff_path}")

print("\n✓ 08 complete. Proceed to 09_run_pipeline.py (master script)")
