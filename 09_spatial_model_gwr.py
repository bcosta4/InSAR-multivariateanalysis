"""
09_spatial_model_gwr.py
────────────────────────
Spatial statistical models in order of complexity:
  1. OLS — global baseline
  2. Spatial Error Model (SEM) — via spreg
  3. GWR / MGWR — geographically weighted regression

Outputs: local R² map, coefficient maps, model comparison CSV.
"""

import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH     = Path(__file__).parent / "outputs" / "predictor_flat.parquet"
FEATURES_JSON = Path(__file__).parent / "outputs" / "pruned_feature_list.json"
OUT_DIR       = Path(__file__).parent / "outputs"
SAMPLE_N      = 50_000

print("=== 09 Spatial Models: OLS → SEM → GWR ===")

df = pd.read_parquet(DATA_PATH)
with open(FEATURES_JSON) as f:
    feature_cols = json.load(f)

df = df[["row", "col", "subsidence_mm_yr"] + feature_cols].dropna()
if len(df) > SAMPLE_N:
    df = df.sample(SAMPLE_N, random_state=42).reset_index(drop=True)
    print(f"  Sampled {SAMPLE_N:,} pixels")

coords  = df[["col", "row"]].values.astype(np.float64)
y       = df["subsidence_mm_yr"].values.reshape(-1, 1)
X_cols  = feature_cols
X_raw   = df[X_cols].values

# ── 1. OLS ────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ols = LinearRegression().fit(X_raw, y.ravel())
y_ols = ols.predict(X_raw)
ols_r2 = r2_score(y.ravel(), y_ols)
print(f"\n  OLS global R² = {ols_r2:.4f}")

# ── 2. SEM (Spatial Error Model) ─────────────────────────────────────────────
try:
    import libpysal, spreg
    import numpy as np
    print("  Building spatial weights for SEM (KNN-8)...")
    w = libpysal.weights.KNN.from_array(coords, k=8)
    w.transform = "r"

    # Add constant
    X_sem = np.column_stack([np.ones(len(X_raw)), X_raw])
    sem = spreg.GM_Error(y, X_sem, w=w, name_y="subsidence",
                         name_x=["const"] + X_cols)
    print(f"  SEM pseudo-R²  = {sem.pr2:.4f}")
    print(f"  Lambda (spatial error): {sem.lam:.4f}")

    sem_results = {"pseudo_r2": float(sem.pr2), "lambda": float(sem.lam)}
    with open(OUT_DIR / "sem_results.json", "w") as f:
        json.dump(sem_results, f, indent=2)
except Exception as e:
    print(f"  SEM skipped: {e}")

# ── 3. GWR ────────────────────────────────────────────────────────────────────
try:
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW
    import numpy as np

    # Standardise X for GWR
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_gwr = scaler.fit_transform(X_raw)
    X_gwr = np.column_stack([np.ones(len(X_gwr)), X_gwr])

    print("\n  GWR: selecting bandwidth (golden section search)...")
    selector = Sel_BW(coords, y, X_gwr)
    bw = selector.search(bw_min=50, bw_max=500)
    print(f"  GWR optimal bandwidth: {bw:.1f}")

    print("  Fitting GWR...")
    gwr_model = GWR(coords, y, X_gwr, bw=bw).fit()
    print(f"  GWR  R²   = {gwr_model.R2:.4f}")
    print(f"  GWR AICc  = {gwr_model.aicc:.2f}")

    # ── Local R² map ─────────────────────────────────────────────────────────
    local_r2 = gwr_model.localR2.ravel()
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["col"], df["row"], c=local_r2,
                    cmap="viridis", vmin=0, vmax=1, s=0.8, marker="s")
    plt.colorbar(sc, ax=ax, label="Local R²")
    ax.set_title(f"GWR Local R²  (global R²={gwr_model.R2:.3f})", fontsize=10)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "gwr_local_r2_map.png", dpi=150)
    plt.close()
    print("  Saved gwr_local_r2_map.png")

    # ── Coefficient maps ─────────────────────────────────────────────────────
    print("  Saving GWR coefficient maps...")
    for i, col_name in enumerate(["const"] + X_cols):
        coeffs = gwr_model.params[:, i]
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(df["col"], df["row"], c=coeffs,
                        cmap="RdBu_r",
                        vmin=np.percentile(coeffs, 2),
                        vmax=np.percentile(coeffs, 98),
                        s=0.8, marker="s")
        plt.colorbar(sc, ax=ax, label="Coefficient")
        ax.set_title(f"GWR coefficient: {col_name}", fontsize=10)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        plt.tight_layout()
        fname = f"gwr_coeff_{col_name.replace(' ','_')}.png"
        plt.savefig(OUT_DIR / fname, dpi=150)
        plt.close()

    gwr_results = {
        "R2": float(gwr_model.R2),
        "AICc": float(gwr_model.aicc),
        "bandwidth": float(bw),
    }
    with open(OUT_DIR / "gwr_results.json", "w") as f:
        json.dump(gwr_results, f, indent=2)
    print("  Saved gwr_results.json")

except Exception as e:
    print(f"  GWR error: {e}")

print("\n  Done. → Next: run 10_run_pipeline.py to orchestrate all stages")
