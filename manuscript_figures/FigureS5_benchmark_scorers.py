"""
Single-pass scorer robustness on REAL counts (no simulator) — AUROC.
=============================================================================

Three scorers:

    1) FuzzyAverage + MI        -> spatialAttractor.FastMICalculateNumba().mi_scan
    2) FuzzyAverage + Spearman  -> scipy.stats.spearmanr   (one call per gene)
    3) bivariate Moran's I raw  -> esda.moran.Moran_BV      (raw degraded counts)

AUROC is reported.

Run:
    .venv_sa/bin/python code/benchmark_scorers_realcounts.py
"""

import os
import sys
import types
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import libpysal
from esda.moran import Moran_BV

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
REVISION = "./revision"
WU_DIR = os.path.join(REVISION, "wu")
OUTDIR = os.path.join(REVISION, "benchmark_scorers_realcounts")
SA_DIR = os.path.join(REVISION, "aCAF-immune-main/spatialAttractor/functions")

SAMPLES = ["CID4465", "CID4290", "1142243F", "CID4535", "CID44971", "1160920F"]
SEED_GENE = "COL11A1"
N_TRUE = 20              # size of the self-ranked ground-truth module (top-K)
N_BACKGROUND = 100       # extra candidate genes in the panel
N_REPS = 5
MIN_DETECT = 0.05
SEED = 1

DROPOUT_GRID = [0.0, 0.2, 0.4]
DEPTH_GRID = [1.0, 0.5, 0.25]
CONDITIONS = [(d, f) for d in DROPOUT_GRID for f in DEPTH_GRID]

METHODS = ["MI", "Spearman", "Moran_raw"]


# --------------------------------------------------------------------------
# Import the published spatialAttractor functions (FuzzyAverage + numba MI)
# --------------------------------------------------------------------------
def _load_sa(funcdir):
    for name in ("spatialdata_io", "spatialdata_io._constants"):
        sys.modules.setdefault(name, types.ModuleType(name))
    _c = types.ModuleType("spatialdata_io._constants._constants")
    _c.VisiumHDKeys = type("VisiumHDKeys", (), {})
    sys.modules.setdefault("spatialdata_io._constants._constants", _c)
    sys.path.insert(0, os.path.abspath(funcdir))
    from tools import FuzzyAverage, FastMICalculateNumba
    return FuzzyAverage, FastMICalculateNumba


FuzzyAverage, FastMICalculateNumba = _load_sa(SA_DIR)
print(f"Loaded published spatialAttractor functions from {SA_DIR}")


# --------------------------------------------------------------------------
# Reference + panel
# --------------------------------------------------------------------------
def load_reference(path):
    adata = sc.read_h5ad(path)
    assert "spatial" in adata.obsm
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    adata.obs["row"] = np.asarray(adata.obsm["spatial"][:, 1], dtype=float)
    adata.obs["col"] = np.asarray(adata.obsm["spatial"][:, 0], dtype=float)
    return adata


def seed_detection(adata, seed_gene):
    counts = adata.layers["counts"]
    counts = counts.tocsc() if sparse.issparse(counts) else sparse.csc_matrix(counts)
    j = adata.var_names.get_loc(seed_gene)
    col = counts[:, j].toarray().ravel()
    return float((col > 0).mean())


def build_gene_panel(adata, seed_gene, n_true, n_background, min_detect, rng):
    """Seed + (n_true + n_background) detected genes. The true/background split is
    irrelevant here (ground truth is self-ranked); only the panel list is used."""
    counts = adata.layers["counts"]
    counts = counts.tocsc() if sparse.issparse(counts) else sparse.csc_matrix(counts)
    detect = np.asarray((counts > 0).mean(axis=0)).ravel()
    expressed = set(adata.var_names[detect >= min_detect])
    assert seed_gene in adata.var_names, f"seed gene {seed_gene} not in reference"
    expressed.discard(seed_gene)
    pool = sorted(expressed)
    n_needed = n_true + n_background
    assert len(pool) >= n_needed, f"only {len(pool)} genes pass detection; need {n_needed}"
    chosen = list(rng.choice(pool, size=n_needed, replace=False))
    panel = [seed_gene] + chosen
    return panel


def panel_counts(adata_panel):
    X = adata_panel.layers["counts"]
    X = np.asarray(X.todense()) if sparse.issparse(X) else np.asarray(X)
    return X.astype(np.float32)


# --------------------------------------------------------------------------
# Dropout / library-size degradation applied to REAL counts
# --------------------------------------------------------------------------
def apply_dropout_depth(counts, dropout_p, depth_frac, rng):
    c = np.rint(np.asarray(counts)).astype(np.int64)
    c[c < 0] = 0
    if depth_frac < 1.0:
        c = rng.binomial(c, depth_frac)
    if dropout_p > 0.0:
        c = c * (rng.random(c.shape) >= dropout_p)
    return c.astype(np.float32)


# --------------------------------------------------------------------------
# FuzzyAverage smoothing , graph precomputed once per sample
# --------------------------------------------------------------------------
def build_smoothing_graph(adata_panel):
    template = adata_panel.copy()
    FuzzyAverage().fuzzy_average(template, spatial_key="spatial", n_neighs=6,
                                include_self=True, coord_type="grid", delaunay=True,
                                use_existing_W=False)
    return template.obsp["spatial_connectivities"].tocsr()


def smooth_counts(counts, coords, W):
    ad = sc.AnnData(X=np.asarray(counts, dtype=np.float32))
    ad.obsm["spatial"] = coords
    ad.obsp["spatial_connectivities"] = W
    Xf = FuzzyAverage().fuzzy_average(ad, spatial_key="spatial", n_neighs=6,
                                      include_self=True, use_existing_W=True)
    return np.asarray(Xf)


# --------------------------------------------------------------------------
# Library-based metric scans
# --------------------------------------------------------------------------
def build_libpysal_W(conn):
    """Row-standardised libpysal W from a squidpy/FuzzyAverage neighbour graph."""
    W = libpysal.weights.WSP(conn.tocsr().astype(float)).to_W(silence_warnings=True)
    W.transform = "r"
    return W


def mi_scan_fuzzy(seed_fuzzy, X_fuzzy, mi_calc):
    """FuzzyAverage + MI via FastMICalculateNumba.mi_scan."""
    return np.asarray(mi_calc.mi_scan(seed_fuzzy, X_fuzzy, num_bins=6, spline_order=3,
                                      normalize=True, negateMI=True))


def spearman_scan_scipy(seed_vec, X):
    """Fuzzy + Spearman via scipy.stats.spearmanr, per gene."""
    seed_vec = np.asarray(seed_vec, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    out = np.empty(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        out[j] = spearmanr(seed_vec, X[:, j]).statistic
    out[~np.isfinite(out)] = -np.inf
    return out


def moran_scan_esda(seed_vec, X, W):
    """Bivariate Moran's I via esda.moran.Moran_BV, per gene."""
    seed_vec = np.asarray(seed_vec, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    out = np.empty(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        try:
            out[j] = Moran_BV(seed_vec, X[:, j], W, permutations=0).I
        except Exception:
            out[j] = -np.inf
    out[~np.isfinite(out)] = -np.inf
    return out


# --------------------------------------------------------------------------
# Ground truth + AUROC evaluation
# --------------------------------------------------------------------------
def ground_truth_module(ref_scores, gene_names, seed_gene, k):
    """Top-k non-seed genes by the clean-data reference score = the module."""
    keep = [i for i, g in enumerate(gene_names) if g != seed_gene]
    genes = [gene_names[i] for i in keep]
    vals = np.array([ref_scores[i] for i in keep], dtype=np.float64)
    vals[~np.isfinite(vals)] = -np.inf
    order = np.argsort(-vals)
    return set(genes[i] for i in order[:k])


def evaluate_auroc(scores, gene_names, seed_gene, true_set):
    keep = [i for i, g in enumerate(gene_names) if g != seed_gene]
    genes = [gene_names[i] for i in keep]
    vals = np.array([scores[i] for i in keep], dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    fill = (finite.min() - 1.0) if finite.size else 0.0
    vals = np.where(np.isfinite(vals), vals, fill)
    labels = np.array([g in true_set for g in genes], dtype=int)
    if not (0 < labels.sum() < len(labels)):
        return np.nan
    return roc_auc_score(labels, vals)


# --------------------------------------------------------------------------
# Per-sample benchmark
# --------------------------------------------------------------------------
def run_sample(sample):
    path = os.path.join(WU_DIR, f"{sample}.h5ad")
    rng = np.random.default_rng(SEED)
    adata = load_reference(path)
    det = seed_detection(adata, SEED_GENE)
    print(f"\n===== {sample}: {adata.n_obs} spots x {adata.n_vars} genes | "
          f"{SEED_GENE} detected in {det*100:.1f}% of spots =====")

    panel = build_gene_panel(adata, SEED_GENE, N_TRUE, N_BACKGROUND, MIN_DETECT, rng)
    adata_panel = adata[:, panel].copy()
    coords = adata_panel.obsm["spatial"]
    gene_names = list(adata_panel.var_names)
    seed_idx = gene_names.index(SEED_GENE)

    conn = build_smoothing_graph(adata_panel)       # FuzzyAverage neighbour graph
    W = build_libpysal_W(conn)                      # row-standardised libpysal W
    mi_calc = FastMICalculateNumba()

    real_counts = panel_counts(adata_panel)         # clean, real observed counts

    # ---- self-ranked ground-truth module from CLEAN real counts ----
    clean_smoothed = smooth_counts(real_counts, coords, conn)
    ref_scores = spearman_scan_scipy(clean_smoothed[:, seed_idx].astype(np.float64),
                                     clean_smoothed)
    true_set = ground_truth_module(ref_scores, gene_names, SEED_GENE, N_TRUE)

    rows = []
    for dropout_p, depth_frac in CONDITIONS:
        for rep in range(N_REPS):
            degraded = apply_dropout_depth(real_counts, dropout_p, depth_frac, rng)
            smoothed = smooth_counts(degraded, coords, conn)

            seed_fuzzy = smoothed[:, seed_idx].astype(np.float64)
            seed_raw = degraded[:, seed_idx].astype(np.float64)

            scans = {
                "MI": mi_scan_fuzzy(seed_fuzzy, smoothed, mi_calc),
                "Spearman": spearman_scan_scipy(seed_fuzzy, smoothed),
                "Moran_raw": moran_scan_esda(seed_raw, degraded, W),
            }
            for method, scores in scans.items():
                auroc = evaluate_auroc(scores, gene_names, SEED_GENE, true_set)
                rows.append({"sample": sample, "method": method,
                             "dropout_p": dropout_p, "depth_frac": depth_frac,
                             "rep": rep, "auroc": auroc})
        print(f"  done dropout_p={dropout_p}, depth_frac={depth_frac}")

    df = pd.DataFrame(rows)
    sdir = os.path.join(OUTDIR, sample)
    os.makedirs(sdir, exist_ok=True)
    df.to_csv(os.path.join(sdir, "benchmark_results_raw.csv"), index=False)
    return df, det


def run_benchmark():
    os.makedirs(OUTDIR, exist_ok=True)
    all_raw = []
    detections = {}
    for s in SAMPLES:
        df, det = run_sample(s)
        all_raw.append(df)
        detections[s] = det

    raw = pd.concat(all_raw, ignore_index=True)
    raw.to_csv(os.path.join(OUTDIR, "ALLSAMPLES_raw.csv"), index=False)

    pd.DataFrame({"sample": list(detections), "detection": list(detections.values())}) \
        .to_csv(os.path.join(OUTDIR, "seed_detection.csv"), index=False)

    summary = (raw.groupby(["sample", "method", "dropout_p", "depth_frac"])["auroc"]
               .agg(["mean", "std"]).reset_index())
    summary.to_csv(os.path.join(OUTDIR, "ALLSAMPLES_summary.csv"), index=False)

    overall = (raw.groupby(["sample", "method"])["auroc"].mean().reset_index())
    overall["detection"] = overall["sample"].map(detections)
    overall = overall.sort_values(["sample", "method"])
    overall.to_csv(os.path.join(OUTDIR, "ALLSAMPLES_overall_by_method.csv"), index=False)

    print("\n=== Overall mean AUROC over all stress conditions (per sample x method) ===")
    with pd.option_context("display.width", 200):
        print(overall.round(4).to_string(index=False))
    print(f"\nSaved raw/summary/overall CSVs under {OUTDIR}")


# ==========================================================================
# Figures — AUROC only
# ==========================================================================
SIM = "real counts + dropout/library-size stress; self-ranked clean-data module; library metrics"
COL = {"MI": "#2a78d6", "Spearman": "#008300", "Moran_raw": "#e87ba4"}
LAB = {"MI": "MI (FuzzyAverage + MI scan)", "Spearman": "Spearman (scipy)",
       "Moran_raw": "Moran's I — raw (esda BV)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
CHANCE = 0.5


def apply_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
        "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "axes.edgecolor": INK2, "axes.linewidth": 0.8, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
        "svg.fonttype": "none"})


def _sem(df, group, val):
    g = df.groupby(group)[val]
    return g.mean(), g.std() / np.sqrt(g.size())


def load_detection():
    df = pd.read_csv(f"{OUTDIR}/seed_detection.csv")
    return dict(zip(df["sample"], df["detection"]))


def robustness_grid(raws, samples, det):
    nrow = len(samples)
    fig, axes = plt.subplots(nrow, 2, figsize=(10, 2.6 * nrow), sharey=True)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.95, bottom=0.05,
                        wspace=0.10, hspace=0.28)
    specs = [("dropout_p", "Dropout probability", [0.0, 0.2, 0.4],
              "AUROC vs dropout"),
             ("depth_frac", "Library-size fraction", [0.25, 0.5, 1.0],
              "AUROC vs library size")]
    for c, (gv, xl, xt, coltitle) in enumerate(specs):
        for r, s in enumerate(samples):
            ax = axes[r, c]
            raw = raws[s]
            for m in METHODS:
                mu, se = _sem(raw[raw.method == m], gv, "auroc")
                ax.errorbar(mu.index, mu.values, yerr=se.values, color=COL[m],
                            marker="o", ms=4.5, lw=1.8, capsize=2.2,
                            elinewidth=0.8, zorder=3)
            ax.axhline(CHANCE, color=INK2, lw=0.7, ls=":", zorder=1)
            ax.set_xticks(xt)
            ax.set_ylim(0.0, 1.02)          # FULL y-axis, not clipped
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.grid(color=GRID, lw=0.7, zorder=0)
            ax.set_axisbelow(True)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(coltitle, fontsize=14, fontweight="bold")
            if r == nrow - 1:
                ax.set_xlabel(xl, fontsize=13)
            if c == 0:
                ax.set_ylabel(f"{s}\n({SEED_GENE} {det[s]*100:.0f}%)",
                              fontsize=13, fontweight="bold")
    handles = [Line2D([0], [0], color=COL[m], lw=2, marker="o", ms=5, label=LAB[m])
               for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.998), fontsize=12)
    out = f"{OUTDIR}/{SEED_GENE}_realcounts_AUROC_robustness_grid"
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=170)
    plt.close(fig)
    print("wrote", out + ".pdf/.png")


def crosssample_summary(raws, samples, det):
    means = {(s, m): raws[s][raws[s].method == m]["auroc"].mean()
             for s in samples for m in METHODS}
    xt = [f"{s}\n({SEED_GENE} {det[s]*100:.0f}%)" for s in samples]
    x = np.arange(len(samples))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.12)
    for i, m in enumerate(METHODS):
        vals = [means[(s, m)] for s in samples]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=COL[m], label=LAB[m], zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, color=INK2, rotation=90)
    ax.axhline(CHANCE, color=INK2, lw=0.8, ls=":", zorder=1)
    ax.text(len(samples) - 0.5, CHANCE + 0.01, "chance", fontsize=11, color=INK2, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(xt, fontsize=12)
    ax.set_ylabel("AUROC (mean over all stress conditions)", fontsize=14)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=13)
    out = f"{OUTDIR}/{SEED_GENE}_realcounts_AUROC_ALLSAMPLES"
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=180)
    plt.close(fig)
    print("wrote", out + ".pdf/.png")

    rows = [{"sample": s, "detection": round(det[s], 4), "method": m,
             "auroc": round(means[(s, m)], 4)}
            for s in samples for m in METHODS]
    pd.DataFrame(rows).to_csv(f"{OUTDIR}/{SEED_GENE}_realcounts_AUROC_ALLSAMPLES_table.csv",
                              index=False)
    print("wrote master table csv")


def make_figures():
    apply_style()
    det = load_detection()
    samples = sorted(det, key=lambda s: -det[s])
    raws = {s: pd.read_csv(f"{OUTDIR}/{s}/benchmark_results_raw.csv") for s in samples}
    robustness_grid(raws, samples, det)
    crosssample_summary(raws, samples, det)


def main():
    run_benchmark()
    make_figures()


if __name__ == "__main__":
    main()
