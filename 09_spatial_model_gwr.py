r"""
08_spatial_model_gwr.py
──────────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import sys
import json
import inspect
import warnings
import traceback
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"E:\Processing\InSAR_multivariate_analysis")
PIPE_OUT = PROJECT_DIR / "outputs"
REF_PATH = PROJECT_DIR / "scripts" / "outputs" / "subsidence_clean.tif"
RUN_DIR = None


CATEGORICAL_VARS = [
    "landcover", "wetland_class", "pan_arctic_wetlands", "canopy_height_lt5m",
    "woodyplant_dominance", "cavm", "surficial_geology", "ground_ice",
    "dsw_trends", "permafrost_landcover_cls",
]
SAMPLE_TARGET = 8_000
K_NEIGHBORS = 8
FALLBACK_BW_RANGE_PX = (50, 500)
MIN_DUMMY_COUNT = 30
CORR_DROP_THRESHOLD = 0.95
COND_THRESHOLD = 1e8
GWR_MAX_FEATURES = 15
GWR_VIF_THRESHOLD = 8.0
MODAL_VALUE_SHARE_MAX = 0.15
GWR_BW_MIN_MULTIPLIER = 5
GWR_BW_MIN_FLOOR_PTS = 100
GWR_MAX_RETRY = 4
GWR_BW_WIDEN_FACTOR = 1.75
GWR_BW_CEILING_FRAC = 0.98     # NEW - flag if found bw is within 2% of bw_max
GWR_BW_MAX_WIDEN_FACTOR = 2.0  # NEW - how much to widen bw_max on ceiling-hit
GWR_MAX_CEILING_RETRY = 3      # NEW - separate retry budget for ceiling widening
# ──────────────────────────────────────────────────────────────────────────────
def log(msg):
    print(msg, flush=True)

@contextlib.contextmanager
def suppress_stderr():
    """Redirect the OS-level stderr file descriptor to os.devnull for the
    duration of the block, then restore it. This operates below Python's
    warnings module entirely (via os.dup2 on the raw fd), so it reliably
    silences LinAlgWarning output regardless of whether it's emitted via
    warnings.warn(), a compiled extension writing to stderr directly, or a
    subprocess (which inherits the redirected fd) - immune to the
    category-name/module-parsing issues that broke every prior attempt."""
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)

def call_with_optional_pool(func, **kwargs):
    """Call func(**kwargs), passing pool=None only if func's signature
    supports it - forces serial execution where possible."""
    sig_params = inspect.signature(func).parameters
    if "pool" in sig_params:
        kwargs["pool"] = None
    return func(**kwargs)

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

def morans_i_on_residuals(residuals, coords_px, k=K_NEIGHBORS):
    import libpysal
    import esda
    w = libpysal.weights.KNN.from_array(coords_px, k=k)
    w.transform = "r"
    mi = esda.Moran(residuals, w)
    return float(mi.I), float(mi.p_sim)

def compute_vif(Xz):
    """Compute VIF for every column of a standardized DataFrame."""
    vifs = {}
    for col in Xz.columns:
        other_cols = [c for c in Xz.columns if c != col]
        if not other_cols:
            vifs[col] = 1.0
            continue
        try:
            beta, *_ = np.linalg.lstsq(Xz[other_cols].values, Xz[col].values, rcond=None)
            resid = Xz[col].values - Xz[other_cols].values @ beta
            var_col = Xz[col].values.var()
            r2 = 1 - (resid.var() / var_col) if var_col > 0 else 0
            vifs[col] = 1 / (1 - r2) if r2 < 0.9999 else np.inf
        except Exception:
            vifs[col] = np.inf
    return vifs

def prune_design_matrix(X_df, min_dummy_count, corr_threshold, cond_threshold, is_dummy_mask):
    """Remove near-empty dummies, zero-variance cols, greedily prune high pairwise correlation, verify full rank, THEN iteratively drop the highest-VIF column until the condition number is acceptable."""
    dropped = []
    X_df = X_df.copy()

    for col in list(X_df.columns):
        if is_dummy_mask.get(col, False):
            nnz = int((X_df[col] != 0).sum())
            if nnz < min_dummy_count:
                dropped.append((col, f"near-empty dummy (n={nnz} < {min_dummy_count})"))
                X_df = X_df.drop(columns=[col])

    for col in list(X_df.columns):
        if X_df[col].std() == 0:
            dropped.append((col, "zero variance"))
            X_df = X_df.drop(columns=[col])

    corr = X_df.corr().abs()
    cols = list(X_df.columns)
    to_drop = set()
    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue
            if corr.loc[cols[i], cols[j]] > corr_threshold:
                dropped.append((cols[j], f"correlated {corr.loc[cols[i], cols[j]]:.3f} with {cols[i]}"))
                to_drop.add(cols[j])
    X_df = X_df.drop(columns=list(to_drop))

    while np.linalg.matrix_rank(X_df.values.astype(np.float64)) < len(X_df.columns) and len(X_df.columns) > 1:
        dropped_col = X_df.columns[-1]
        dropped.append((dropped_col, "removed to restore full column rank"))
        X_df = X_df.drop(columns=[dropped_col])

    Xz = (X_df - X_df.mean()) / X_df.std().replace(0, 1)
    while len(Xz.columns) > 1:
        cond = np.linalg.cond(Xz.values.astype(np.float64))
        if cond < cond_threshold:
            break
        vifs = compute_vif(Xz)
        worst_col = max(vifs, key=vifs.get)
        dropped.append((worst_col, f"dropped for conditioning (cond={cond:.2e} >= {cond_threshold:.0e}, VIF={vifs[worst_col]:.1f})"))
        X_df = X_df.drop(columns=[worst_col])
        Xz = Xz.drop(columns=[worst_col])

    return X_df, dropped

def modal_value_share(series):
    """Fraction of samples equal to the single most common value."""
    vc = series.value_counts(dropna=True)
    if len(vc) == 0:
        return 1.0
    return float(vc.iloc[0] / len(series))

def select_gwr_features(X_df, y, continuous_cols, max_features, vif_threshold, modal_share_max):
    """Stage 1: drop features with modal_value_share > modal_share_max.
    Stage 2: rank survivors by |corr with y|. Stage 3: iteratively drop
    highest-VIF feature until all remaining have VIF < vif_threshold."""
    diag_rows = []
    eligible_cols = []
    for c in continuous_cols:
        share = modal_value_share(X_df[c])
        ok = share <= modal_share_max
        diag_rows.append({"feature": c, "modal_value_share": share, "passed_modal_filter": ok})
        if ok:
            eligible_cols.append(c)
    diag_df = pd.DataFrame(diag_rows).sort_values("modal_value_share", ascending=False)

    if len(eligible_cols) == 0:
        return [], diag_df

    corr_with_y = X_df[eligible_cols].apply(
        lambda col: abs(np.corrcoef(col.values, y.ravel())[0, 1]))
    candidate_cols = corr_with_y.sort_values(ascending=False).index.tolist()[:max_features * 2]

    Xz = X_df[candidate_cols].copy()
    Xz = (Xz - Xz.mean()) / Xz.std().replace(0, 1)
    while len(Xz.columns) > 1:
        vifs = compute_vif(Xz)
        max_vif = max(vifs.values())
        if max_vif < vif_threshold and len(Xz.columns) <= max_features:
            break
        if max_vif >= vif_threshold:
            worst_col = max(vifs, key=vifs.get)
            Xz = Xz.drop(columns=[worst_col])
        else:
            lowest_corr_col = corr_with_y[Xz.columns].idxmin()
            Xz = Xz.drop(columns=[lowest_corr_col])


    selected = list(Xz.columns)
    return selected, diag_df

log("=== 08 Spatial Models: OLS -> SEM -> GWR (v13) ===")

run_dir = get_run_dir()
DATA_PATH = run_dir / "predictor_flat.parquet"
FEATURES_JSON = run_dir / "pruned_feature_list.json"
SPATIAL_JSON = run_dir / "spatial_autocorrelation_results.json"
OUT_DIR = run_dir
log(f"Run dir: {run_dir}")

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

log(f"  Pruned features: {len(continuous_feats)} continuous, {len(categorical_feats)} categorical")

df_full = pd.read_parquet(DATA_PATH)
df_full = df_full[["row", "col", "subsidence_mm_yr"] + feature_cols_raw].dropna()
n_full = len(df_full)
log(f"  Full complete-case dataset: {n_full:,} rows")

for c in categorical_feats:
    df_full[c] = df_full[c].astype(np.int32)

with rasterio.open(REF_PATH) as ref:
    px_size_x = abs(ref.transform.a)
    px_size_y = abs(ref.transform.e)

thin_step = max(1, int(np.sqrt(n_full / SAMPLE_TARGET)))
df = df_full[(df_full["row"] % thin_step == 0) & (df_full["col"] % thin_step == 0)].reset_index(drop=True)
log(f"  Systematic thinning step={thin_step} -> {len(df):,} rows for spatial models")

X_cont = df[continuous_feats].reset_index(drop=True)
X_cat_parts = []
is_dummy = {c: False for c in continuous_feats}
for c in categorical_feats:
    dummies = pd.get_dummies(df[c].reset_index(drop=True), prefix=c, drop_first=True)
    for dc in dummies.columns:
        is_dummy[dc] = True
    X_cat_parts.append(dummies)
X_cat = pd.concat(X_cat_parts, axis=1) if X_cat_parts else pd.DataFrame(index=X_cont.index)

X_df_raw = pd.concat([X_cont, X_cat], axis=1)
log(f"  Design matrix (pre-pruning): {X_df_raw.shape[1]} columns "
    f"({len(continuous_feats)} continuous + {X_cat.shape[1]} dummy columns)")

X_df, dropped_log = prune_design_matrix(
    X_df_raw, MIN_DUMMY_COUNT, CORR_DROP_THRESHOLD, COND_THRESHOLD, is_dummy)
feature_names = list(X_df.columns)
X_raw = X_df.values.astype(np.float64)
final_cond = np.linalg.cond((X_df - X_df.mean()).div(X_df.std().replace(0, 1)).values)
log(f"  Design matrix (post-pruning): {X_raw.shape[1]} columns "
    f"({len(dropped_log)} columns dropped, final cond(X)={final_cond:.2e})")

pd.DataFrame(dropped_log, columns=["column", "reason"]).to_csv(
    OUT_DIR / "design_matrix_pruning_log.csv", index=False)

coords = df[["col", "row"]].values.astype(np.float64)
y = df["subsidence_mm_yr"].values.reshape(-1, 1)

model_comparison_rows = []

# ── 1. OLS ────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ols = LinearRegression().fit(X_raw, y.ravel())
y_ols = ols.predict(X_raw)
ols_r2 = r2_score(y.ravel(), y_ols)
ols_resid = y.ravel() - y_ols
ols_moran_i, ols_moran_p = morans_i_on_residuals(ols_resid, coords)
n_params_ols = X_raw.shape[1] + 1
rss_ols = float(np.sum(ols_resid ** 2))
aic_ols = len(y) * np.log(rss_ols / len(y)) + 2 * n_params_ols

log(f"\n  [OLS] R2={ols_r2:.4f}  residual_Moran_I={ols_moran_i:.4f} (p={ols_moran_p:.4f})")

model_comparison_rows.append({
    "model": "OLS", "R2": ols_r2, "AIC_or_AICc": aic_ols,
    "residual_moran_I": ols_moran_i, "residual_moran_p": ols_moran_p,
})

# ── 2. SEM (Spatial Error Model) ─────────────────────────────────────────────
try:
    import libpysal
    import spreg

    log("\n  [SEM] Building KNN-8 spatial weights and fitting GM_Error...")
    w = libpysal.weights.KNN.from_array(coords, k=K_NEIGHBORS)
    w.transform = "r"

    sem = spreg.GM_Error(y, X_raw, w=w, name_y="subsidence", name_x=feature_names)

    sem_pred = sem.predy.ravel()
    sem_resid = y.ravel() - sem_pred
    sem_moran_i, sem_moran_p = morans_i_on_residuals(sem_resid, coords)

    sem_lambda = None
    if hasattr(sem, "lam"):
        sem_lambda = float(sem.lam)
    elif hasattr(sem, "betas") and hasattr(sem, "name_x") and sem.name_x and sem.name_x[-1] == "lambda":
        sem_lambda = float(sem.betas[-1][0])
    lambda_str = f"{sem_lambda:.4f}" if sem_lambda is not None else "N/A"

    log(f"  [SEM] pseudo_R2={sem.pr2:.4f}  lambda={lambda_str}  "
        f"residual_Moran_I={sem_moran_i:.4f} (p={sem_moran_p:.4f})")

    model_comparison_rows.append({
        "model": "SEM", "R2": float(sem.pr2), "AIC_or_AICc": np.nan,
        "residual_moran_I": sem_moran_i, "residual_moran_p": sem_moran_p,
    })

    sem_results = {"pseudo_r2": float(sem.pr2), "lambda": sem_lambda,
                   "residual_moran_I": sem_moran_i, "residual_moran_p": sem_moran_p}
    with open(OUT_DIR / "sem_results.json", "w") as f:
        json.dump(sem_results, f, indent=2)
except Exception as e:
    print(f"\n  !!! SEM FAILED: {type(e).__name__}: {e}", flush=True, file=sys.stderr)
    traceback.print_exc(file=sys.stderr)

# ── 3. GWR ────────────────────────────────────────────────────────────────────
try:
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW
    from sklearn.preprocessing import StandardScaler

    continuous_in_pruned = [c for c in X_df.columns if not is_dummy.get(c, False)]
    if len(continuous_in_pruned) == 0:
        raise RuntimeError("No continuous features survived pruning - cannot run GWR.")

    top_gwr_cols, modal_diag_df = select_gwr_features(
        X_df, y, continuous_in_pruned, GWR_MAX_FEATURES,
        GWR_VIF_THRESHOLD, MODAL_VALUE_SHARE_MAX)

    modal_diag_df.to_csv(OUT_DIR / "gwr_modal_value_diagnostic.csv", index=False)
    n_failed_modal = int((~modal_diag_df["passed_modal_filter"]).sum())
    log(f"\n  [GWR] Feature selection: {n_failed_modal} excluded (constant-region), "
        f"{len(top_gwr_cols)} selected via VIF<{GWR_VIF_THRESHOLD}: {top_gwr_cols}")

    if len(top_gwr_cols) < 2:
        raise RuntimeError(
            f"Only {len(top_gwr_cols)} continuous feature(s) survived filtering - "
            f"not enough for GWR. See gwr_modal_value_diagnostic.csv."
        )

    X_gwr_df = X_df[top_gwr_cols]
    scaler = StandardScaler()
    X_gwr_scaled = scaler.fit_transform(X_gwr_df.values.astype(np.float64))
    X_gwr = np.column_stack([np.ones(len(X_gwr_scaled)), X_gwr_scaled])

    min_pts_needed = max(GWR_BW_MIN_MULTIPLIER * (len(top_gwr_cols) + 1), GWR_BW_MIN_FLOOR_PTS)
    bw_min, bw_max = FALLBACK_BW_RANGE_PX
    bw_min = max(bw_min, min_pts_needed)
    if SPATIAL_JSON.exists():
        with open(SPATIAL_JSON) as f:
            spatial_results = json.load(f)
        vgm_range_m = spatial_results.get("semivariogram_range_m")
        if vgm_range_m:
            vgm_range_px = vgm_range_m / px_size_x
            bw_min = max(min_pts_needed, int(round(vgm_range_px * 0.2)))
            bw_max = max(bw_min * 2, int(round(vgm_range_px * 2.0)))
    log(f"  [GWR] Bandwidth search bounds: [{bw_min}, {bw_max}]")

    gwr_success = False
    gwr_model = None
    bw = None
    attempt = 0
    ceiling_retry = 0
    ceiling_hit_log = []

    while not gwr_success and attempt < GWR_MAX_RETRY:
        try:
            log(f"  [GWR] Attempt {attempt + 1}/{GWR_MAX_RETRY}: searching bandwidth "
                f"in [{bw_min}, {bw_max}]...")
            with suppress_stderr():
                selector = Sel_BW(coords, y, X_gwr)
                bw = call_with_optional_pool(selector.search, bw_min=bw_min, bw_max=bw_max)
                gwr_model = call_with_optional_pool(GWR(coords, y, X_gwr, bw=bw).fit)
            gwr_success = True
            log(f"  [GWR] Search converged: bandwidth={bw:.1f} (bw_max was {bw_max})")

            # NEW: ceiling-pinning check. A bandwidth found right at bw_max is not a converged optimum - it means AICc was still improving when the search ran out of room. Widen and re-search rather than accepting a boundary artifact as the final answer.
            while (bw >= bw_max * GWR_BW_CEILING_FRAC) and (ceiling_retry < GWR_MAX_CEILING_RETRY):
                ceiling_retry += 1
                old_bw_max = bw_max
                bw_max = int(round(bw_max * GWR_BW_MAX_WIDEN_FACTOR))
                ceiling_hit_log.append({
                    "retry": ceiling_retry, "bw_found": float(bw),
                    "old_bw_max": old_bw_max, "new_bw_max": bw_max,
                })
                log(f"  [GWR] Bandwidth {bw:.1f} is pinned at ceiling (bw_max={old_bw_max}) - "
                    f"this means the optimum may lie beyond the search range. "
                    f"Widening bw_max to {bw_max} and re-searching "
                    f"({ceiling_retry}/{GWR_MAX_CEILING_RETRY})...")
                with suppress_stderr():
                    selector = Sel_BW(coords, y, X_gwr)
                    bw = call_with_optional_pool(selector.search, bw_min=bw_min, bw_max=bw_max)
                    gwr_model = call_with_optional_pool(GWR(coords, y, X_gwr, bw=bw).fit)
                log(f"  [GWR] Re-search result: bandwidth={bw:.1f} (bw_max={bw_max})")

            if bw >= bw_max * GWR_BW_CEILING_FRAC:
                log(f"  [GWR] WARNING: bandwidth still pinned at ceiling after "
                    f"{ceiling_retry} widen-retries (final bw={bw:.1f}, bw_max={bw_max}). "
                    f"AICc may still be decreasing beyond this range - treat this "
                    f"bandwidth/R2 as a lower bound, not a confirmed optimum.")
            else:
                log(f"  [GWR] Bandwidth {bw:.1f} confirmed below ceiling "
                    f"(bw_max={bw_max}) - optimum located within search range.")

        except np.linalg.LinAlgError:
            attempt += 1
            bw_min = int(bw_min * GWR_BW_WIDEN_FACTOR)
            bw_max = max(bw_max, bw_min * 2)
            log(f"  [GWR] Singular matrix at this bandwidth range - widening bw_min to "
                f"{bw_min} and retrying ({attempt}/{GWR_MAX_RETRY})...")

    if not gwr_success:
        raise RuntimeError(
            f"GWR could not find a non-singular bandwidth after {GWR_MAX_RETRY} attempts "
            f"(final bw_min={bw_min}). Try lowering GWR_VIF_THRESHOLD or GWR_MAX_FEATURES."
        )

    gwr_resid = gwr_model.resid_response.ravel()
    gwr_moran_i, gwr_moran_p = morans_i_on_residuals(gwr_resid, coords)

    log(f"  [GWR] R2={gwr_model.R2:.4f}  AICc={gwr_model.aicc:.2f}  "
        f"residual_Moran_I={gwr_moran_i:.4f} (p={gwr_moran_p:.4f})")

    model_comparison_rows.append({
        "model": "GWR", "R2": float(gwr_model.R2), "AIC_or_AICc": float(gwr_model.aicc),
        "residual_moran_I": gwr_moran_i, "residual_moran_p": gwr_moran_p,
    })

    local_r2 = gwr_model.localR2.ravel()
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["col"], df["row"], c=local_r2, cmap="viridis",
                     vmin=0, vmax=1, s=3, marker="s")
    plt.colorbar(sc, ax=ax, label="Local R2")
    ax.set_title(f"GWR Local R2 (global R2={gwr_model.R2:.3f}, "
                 f"{len(top_gwr_cols)} VIF-filtered continuous features, bw={bw:.0f})", fontsize=10)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "gwr_local_r2_map.png", dpi=300)
    plt.close()

    gwr_col_names = ["const"] + top_gwr_cols
    coeff_dir = OUT_DIR / "gwr_coefficient_maps"
    coeff_dir.mkdir(exist_ok=True)
    for i, col_name in enumerate(gwr_col_names):
        coeffs = gwr_model.params[:, i]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sc = ax.scatter(df["col"], df["row"], c=coeffs, cmap="RdBu_r",
                         vmin=np.percentile(coeffs, 2), vmax=np.percentile(coeffs, 98),
                         s=3, marker="s")
        plt.colorbar(sc, ax=ax, label="Coefficient")
        ax.set_title(f"GWR coefficient: {col_name}", fontsize=9)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        plt.tight_layout()
        safe_name = col_name.replace(" ", "_").replace("/", "_")
        plt.savefig(coeff_dir / f"gwr_coeff_{safe_name}.png", dpi=300)
        plt.close()
    log(f"  [GWR] Saved local R2 map + {len(gwr_col_names)} coefficient maps to {coeff_dir}")

    gwr_results = {
        "R2": float(gwr_model.R2), "AICc": float(gwr_model.aicc), "bandwidth": float(bw),
        "residual_moran_I": gwr_moran_i, "residual_moran_p": gwr_moran_p,
        "bw_search_min": bw_min, "bw_search_max": bw_max,
        "features_used": top_gwr_cols,
        "n_features_used": len(top_gwr_cols),
        "n_features_excluded_modal_filter": n_failed_modal,
        "retry_attempts": attempt,
        "ceiling_retry_attempts": ceiling_retry,          # NEW
        "bandwidth_pinned_at_final_ceiling": bool(bw >= bw_max * GWR_BW_CEILING_FRAC),  # NEW
        "ceiling_widen_history": ceiling_hit_log,          # NEW
    }
    with open(OUT_DIR / "gwr_results.json", "w") as f:
        json.dump(gwr_results, f, indent=2)

except Exception as e:
    print(f"\n  !!! GWR FAILED: {type(e).__name__}: {e}", flush=True, file=sys.stderr)
    traceback.print_exc(file=sys.stderr)

# ── Model comparison table ───────────────────────────────────────────────────
comp_df = pd.DataFrame(model_comparison_rows)
comp_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)
log("\n  === MODEL COMPARISON ===")
log(comp_df.to_string(index=False))
log("\n  Done. -> Next: run 10_run_pipeline.py to orchestrate all stages")