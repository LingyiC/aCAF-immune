"""
Scorers of associations.

    1) fuzzy average + MI                 -> spatialAttractor.FastMICalculateNumba
    2) fuzzy average + Spearman           -> scipy.stats.spearmanr
    3) bivariate Moran's I                -> esda.moran.Moran_BV

"""
import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

import libpysal
from esda.moran import Moran_BV

import spatialAttractor

METHODS = ["mi", "spearman", "moran_raw"]

def spearman_scan_scipy(seed_vec, X):
    """fuzzy+Spearman via scipy.stats.spearmanr, one call per gene.
    """
    seed_vec = np.asarray(seed_vec, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    g = X.shape[1]
    out = np.empty(g, dtype=np.float64)
    for j in range(g):
        out[j] = spearmanr(seed_vec, X[:, j]).statistic
    out[~np.isfinite(out)] = -np.inf
    return out


def build_libpysal_W(adata):
    """Row-standardized libpysal W from the squidpy neighbor graph."""
    Wsp = adata.obsp["spatial_connectivities"].tocsr().astype(float)
    W = libpysal.weights.WSP(Wsp).to_W(silence_warnings=True)
    W.transform = "r"
    return W


def moran_scan_esda(seed_vec, X, W):
    """bivariate Moran's I via esda.moran.Moran_BV, one call per gene.
    """
    seed_vec = np.asarray(seed_vec, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    g = X.shape[1]
    out = np.empty(g, dtype=np.float64)
    for j in range(g):
        try:
            out[j] = Moran_BV(seed_vec, X[:, j], W, permutations=0).I
        except Exception:
            out[j] = -np.inf          # constant gene / degenerate -> rank last
    out[~np.isfinite(out)] = -np.inf
    return out


def score_sample_lib(adata, seed_gene="COL11A1"):

    X_fuzzy = adata.layers["X_fuzzy"]
    X_raw = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)

    seed_idx = adata.var_names.get_loc(seed_gene)
    seed_fuzzy = X_fuzzy[:, seed_idx]
    seed_raw = X_raw[:, seed_idx]

    # (1) fuzzy average + MI  (unchanged package method)
    mi_calc = spatialAttractor.FastMICalculateNumba()
    mi_scores = mi_calc.mi_scan(seed_fuzzy, X_fuzzy,
                                num_bins=6, spline_order=3,
                                normalize=True, negateMI=True)

    # (2) fuzzy average + Spearman  (scipy)
    spearman_scores = spearman_scan_scipy(seed_fuzzy, X_fuzzy)

    # (3,4) bivariate Moran's I  (esda)
    W = build_libpysal_W(adata)
    moran_raw_scores = moran_scan_esda(seed_raw, X_raw, W)

    return adata.var_names, {
        "mi": mi_scores,
        "spearman": spearman_scores,
        "moran_raw": moran_raw_scores,
    }


def main():
    """Full sweep with the library scorers.
    """
    df_manuscript = pd.read_excel("dataset_in_manuscript.xlsx")
    df_filter = df_manuscript[(df_manuscript["COL11A1_pct_gt1"] >= 10)]

    out_dir = "./milist"
    new_dir = os.path.join(out_dir, "mi_spearman_moran_lib")   # separate subfolder
    os.makedirs(new_dir, exist_ok=True)

    combined_rank = {m: pd.DataFrame() for m in METHODS}

    for idx, row in df_filter.iterrows():
        dataset_name = row["dataset_name"]
        sampleID = row["SampleID"]
        print(f"\n===== Processing dataset: {dataset_name} | sample: {sampleID} =====")

        sample_dir = sampleID[:-5] if sampleID.endswith(".h5ad") else sampleID
        folder = f"./spatial_acaf/{dataset_name}/raw/{sample_dir}"

        proc = spatialAttractor.spatialTool()
        adata = proc.dataprocess(folder, skip_qc=False, verbose=False, cluster=False)

        fuzzyA = spatialAttractor.FuzzyAverage()
        fuzzyA.fuzzy_average(adata, layer_out="X_fuzzy",
                             n_neighs=6, coord_type="grid", include_self=True)

        var_names, scores = score_sample_lib(adata, seed_gene="COL11A1")

        col_name = dataset_name + "_" + sampleID
        for method in METHODS:
            sorted_idx = np.argsort(-scores[method])
            ranked_series = pd.Series(var_names[sorted_idx], name=col_name)
            combined_rank[method] = pd.concat([combined_rank[method], ranked_series], axis=1)

    for method in METHODS:
        df = combined_rank[method]
        method_dir = os.path.join(new_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        rank_out = f"{method_dir}/COL11A1_pct10_ranked_seedCOL11A1.csv"
        top_out = f"{method_dir}/COL11A1_pct10_ranked_seedCOL11A1_TOP10000_RANKED.csv"
        bottom_out = f"{method_dir}/COL11A1_pct10_ranked_seedCOL11A1_BOTTOM10000_RANKED.csv"
        df.to_csv(rank_out, index=False)
        df.head(10000).to_csv(top_out, index=False)
        df.tail(10000).iloc[::-1].to_csv(bottom_out, index=False)
        print(f"\n [{method}] saved to {rank_out}")

    print("\n Completed! Results under:")
    print(new_dir)


if __name__ == "__main__":
    main()
