from __future__ import annotations
import json

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import sparse
import os
np.random.seed(42)  
from anndata import AnnData
import os
from PIL import Image
import json
import matplotlib.image as mpimg
from spatialdata_io._constants._constants import VisiumHDKeys
from pathlib import Path
import spatialdata_io
class spatialTool:
    def generate_lowres_image(self, spatial_path):
        """
        Generate a low-resolution image from high-resolution image for Visium data
        using the scale factors from scalefactors_json.json
        
        The scale factor to convert hires → lowres is calculated as:
        scale_factor = tissue_lowres_scalef / tissue_hires_scalef
        
        Parameters:
        spatial_path: path to the spatial folder containing images
        """
        # spatial_path = "/Volumes/T7/datasets/Abalo_skin/raw/V10F24-015_B1/spatial"
        # proc.generate_lowres_image(spatial_path)
        
        # Check if low-res image already exists
        lowres_path = os.path.join(spatial_path, "tissue_lowres_image.png")
        hires_path = os.path.join(spatial_path, "tissue_hires_image.png")
        scalefactors_path = os.path.join(spatial_path, "scalefactors_json.json")
        
        if os.path.exists(lowres_path):
            print(f"Low-res image already exists at {lowres_path}")
            return
        
        if not os.path.exists(hires_path):
            print(f"High-res image not found at {hires_path}")
            return
        
        if not os.path.exists(scalefactors_path):
            print(f"Scalefactors file not found at {scalefactors_path}")
            return
        
        # Read scale factors from scalefactors_json.json
        try:
            with open(scalefactors_path, 'r') as f:
                scalefactors = json.load(f)
            
            # Get both scale factors
            if 'tissue_lowres_scalef' not in scalefactors or 'tissue_hires_scalef' not in scalefactors:
                print("Required scale factors not found in scalefactors_json.json")
                print(f"Available keys: {list(scalefactors.keys())}")
                return
            
            tissue_lowres_scalef = scalefactors['tissue_lowres_scalef']
            tissue_hires_scalef = scalefactors['tissue_hires_scalef']
            
            # Calculate the scale factor to convert hires → lowres
            scale_factor = tissue_lowres_scalef / tissue_hires_scalef
            
            print(f"tissue_hires_scalef: {tissue_hires_scalef}")
            print(f"tissue_lowres_scalef: {tissue_lowres_scalef}")
            print(f"Calculated scale factor (lowres/hires): {scale_factor}")
            
        except Exception as e:
            print(f"Error reading scalefactors_json.json: {e}")
            return
        
        # Load high-res image
        print(f"Loading high-res image from {hires_path}")
        hires_img = Image.open(hires_path)
        
        # Calculate new dimensions
        width, height = hires_img.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Create low-res image
        lowres_img = hires_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save low-res image
        lowres_img.save(lowres_path)
        print(f"Low-res image saved to {lowres_path}")
        print(f"Original hires size: {width}x{height}, New lowres size: {new_width}x{new_height}")
        
        return lowres_img


    def read_visium_any(self, path, load_images=True, source="spaceranger", raw = True):
        """
        Read spatial transcriptomics data from a .h5ad file or a Space Ranger outs/ folder.
        Priority:
        1. If path points to a .h5ad file -> load AnnData directly.
        2. If path/filtered_feature_bc_matrix.h5 exists -> read with Scanpy.
        3. Else fall back to matrix.mtx.gz + barcodes.tsv.gz + features.tsv.gz.

        Parameters
        ----------
        path : str
            Path to either an .h5ad file or a Space Ranger 'outs' directory.
        load_images : bool, optional
            Whether to load spatial images if available (default: True).
        source : {"spaceranger", "visium_sge"}, optional
            Which software produced the output (default: "spaceranger").

        Returns
        -------
        adata : AnnData
            Annotated data matrix with spatial information.
        """

        # Case 1: Already an AnnData object on disk
        if path.endswith(".h5ad") and os.path.isfile(path):
            adata = sc.read_h5ad(path)
            if raw:
                # Restore raw counts matrix
                if adata.raw is not None:
                    adata = adata.raw.to_adata()
                elif "counts" in adata.layers:
                    adata.X = adata.layers["counts"].copy()
                
                # Keep only minimal Cell Ranger-like structure
                # Preserve only basic obs columns and spatial info
                essential_obs_cols = {"in_tissue", "array_row", "array_col"}
                cols_to_keep = [col for col in adata.obs.columns if col in essential_obs_cols]
                if cols_to_keep:
                    adata.obs = adata.obs[cols_to_keep].copy()
                else:
                    # If no essential columns, keep minimal obs
                    adata.obs = adata.obs[[]]
                
                # Clear all layers
                adata.layers = {}
                adata.raw = None
                
                # Keep only spatial information in uns and obsm
                spatial_keys = {"spatial"}  # Keep spatial metadata
                spatial_obsm_keys = {"spatial", "X_spatial"}  # Keep spatial coordinates
                
                # Clear uns except spatial info
                new_uns = {}
                for k in adata.uns.keys():
                    if k in spatial_keys:
                        new_uns[k] = adata.uns[k]
                adata.uns = new_uns
                
                # Clear obsm except spatial coordinates
                new_obsm = {}
                for k in adata.obsm.keys():
                    if k in spatial_obsm_keys:
                        new_obsm[k] = adata.obsm[k]
                adata.obsm = new_obsm
                
                # Clear all other analysis results
                adata.obsp = {}
                adata.varm = {}
                adata.var = adata.var[[]]  # Keep var index but clear all columns

            return adata

        # Case 2: H5 feature matrix exists
        h5_file = os.path.join(path, "filtered_feature_bc_matrix.h5")
        hires_path = os.path.join(path, "spatial", "tissue_hires_image.png")
        lowres_path = os.path.join(path, "spatial", "tissue_lowres_image.png")
        if os.path.isfile(h5_file):
            if os.path.isfile(hires_path):
                adata = sc.read_visium(path=path, count_file="filtered_feature_bc_matrix.h5",
                                    load_images=load_images, library_id = "library_id")
                
                print(f"Loaded filtered_feature_bc_matrix.h5 from {path}")
                
                # ---- add microns_per_pixel and spatial_um for hires case ----
                scalefactors_file = os.path.join(path, "spatial", "scalefactors_json.json")
                if os.path.isfile(scalefactors_file):
                    with open(scalefactors_file, "r") as f:
                        scalefactors = json.load(f)
                    
                    # Compute microns per pixel
                    if "spot_diameter_fullres" in scalefactors:
                        microns_per_pixel = 55.0 / float(scalefactors["spot_diameter_fullres"])
                        adata.uns["spatial"]["library_id"]["scalefactors"]["microns_per_pixel"] = microns_per_pixel
                        print(f"→ Computed microns_per_pixel = {microns_per_pixel:.4f} µm/pixel")
                        
                        # Create spatial_um in micrometers
                        if "spatial" in adata.obsm:
                            adata.obsm["spatial_um"] = adata.obsm["spatial"] * microns_per_pixel
                            print("Added adata.obsm['spatial_um'] with coordinates in micrometers.")
            else:
                # load without images, then add lowres manually
                adata = sc.read_visium(
                    path=path,
                    count_file="filtered_feature_bc_matrix.h5",
                    load_images=False,
                    library_id="library_id"
                )
                print(f"Loaded filtered_feature_bc_matrix.h5 from {path} (no hires image)")

                # ---- add lowres image ----
                if os.path.isfile(lowres_path):
                    lowres_img = mpimg.imread(lowres_path)
                    adata.uns["spatial"]["library_id"]["images"] = {"lowres": lowres_img}

                    # ---- add scalefactors ----
                    scalefactors_file = os.path.join(path, "spatial", "scalefactors_json.json")
                    microns_per_pixel = None  # >>> added <<<
                    if os.path.isfile(scalefactors_file):
                        with open(scalefactors_file, "r") as f:
                            scalefactors = json.load(f)
                        adata.uns["spatial"]["library_id"]["scalefactors"] = scalefactors

                        # >>> added: compute microns per pixel <<<
                        if "spot_diameter_fullres" in scalefactors:
                            microns_per_pixel = 55.0 / float(scalefactors["spot_diameter_fullres"])
                            adata.uns["spatial"]["library_id"]["scalefactors"]["microns_per_pixel"] = microns_per_pixel
                            print(f"→ Computed microns_per_pixel = {microns_per_pixel:.4f} µm/pixel")
                    else:
                        print("scalefactors_json.json not found.")

                    # ---- ensure spatial coordinates are present ----
                    tissue_positions_file = os.path.join(path, "spatial", "tissue_positions.csv")
                    if not os.path.isfile(tissue_positions_file):
                        tissue_positions_file = os.path.join(path, "spatial", "tissue_positions_list.csv")

                    if os.path.isfile(tissue_positions_file):
                        coords = pd.read_csv(tissue_positions_file, header=None)

                        # detect format
                        if coords.shape[1] == 6:  # new format
                            coords.columns = [
                                "barcode", "in_tissue", "array_row", "array_col",
                                "pxl_col_in_fullres", "pxl_row_in_fullres"
                            ]
                        elif coords.shape[1] == 5:  # old format (rare)
                            coords.columns = [
                                "barcode", "in_tissue", "array_row", "array_col",
                                "pxl_row_in_fullres"
                            ]

                        coords.set_index("barcode", inplace=True)

                        # attach metadata
                        adata.obs = adata.obs.join(coords[["in_tissue", "array_row", "array_col"]])

                        # ensure correct axis order: [x, y] = [col, row]
                        if {"pxl_col_in_fullres", "pxl_row_in_fullres"}.issubset(coords.columns):
                            adata.obsm["spatial"] = coords.loc[
                                adata.obs_names, ["pxl_col_in_fullres", "pxl_row_in_fullres"]
                            ].values.astype(float)
                        # >>> added: create spatial_um in micrometers <<<
                        if microns_per_pixel is not None:
                            adata.obsm["spatial_um"] = adata.obsm["spatial"] * microns_per_pixel
                            print("Added adata.obsm['spatial_um'] with coordinates in micrometers.")

                        # filter to on-tissue only
                        if "in_tissue" in adata.obs:
                            before = adata.n_obs
                            adata = adata[adata.obs["in_tissue"] == 1].copy()
                            after = adata.n_obs
                            print(f"Filtered off-tissue spots: kept {after} / {before}")

                    else:
                        print("No tissue_positions.csv found → spatial coords missing.")

                    print("Added lowres image, scalefactors, and spatial coords manually.")
                else:
                    print("⚠️ No lowres image found.")



        return adata


    def dataprocess(self, base, skip_qc=False, barcode = None, verbose = True):
        spatial = f"{base}/spatial"
        adata = self.read_visium_any(base)

        # ---- quick QC & preprocess ----

        adata.var_names_make_unique()
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
        if verbose: 
            sns.histplot(adata.obs["total_counts"], kde=False)

        # We applied light, slide-robust QC to Visium data. For each slide, we annotated mitochondrial genes (prefix “MT-”) and computed per-spot QC metrics with `scanpy.pp.calculate_qc_metrics`. 
        # We retained **on-tissue** spots (`in_tissue == 1`), removed spots with **low complexity** (<200 detected genes) or **low library size** (<1,000 UMIs), excluded spots with **high mitochondrial fraction** (≥20% of counts), and capped **high-count outliers** by discarding spots above the **99th percentile** of total counts. 
        # These thresholds balance noise removal with preservation of biologically relevant tumor heterogeneity.

        if not skip_qc:
            lo = max(1000, np.percentile(adata.obs["total_counts"], 1))
            hi = max(35000, np.percentile(adata.obs["total_counts"], 99)) # some samples have very high count 

            mask = (
                (adata.obs.get("in_tissue", 1) == 1) & # on tissue 
                (adata.obs["n_genes_by_counts"] >= 200) & # minimal complexity check 
                (adata.obs["total_counts"].between(lo, hi)) & # low library size
                (adata.obs["pct_counts_mt"] < 20) # kee low mitochondrial fraction
            )

            adata = adata[mask].copy()
        else:
            if barcode:
                mask = adata.obs_names.isin(barcode)
                adata = adata[mask].copy()
            
        # normalize & log
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)

        # HVGs / PCA / neighbors / UMAP / clustering
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(
            adata, key_added="clusters", flavor="igraph", directed=False, n_iterations=2
        )

        # ---- plots ----
        if verbose: 
            # img_key="lowres"
            sc.pl.spatial(adata, img_key=None, color="clusters", size=1.5)
        # sc.pl.embedding(adata,  basis='X_spatial', show=False, color='clusters')
        
        # Fix spatial coordinate naming for squidpy compatibility
        if "X_spatial" in adata.obsm and "spatial" not in adata.obsm:
            adata.obsm["spatial"] = adata.obsm["X_spatial"].copy()
        
        return adata

    def visiumHDreader(self, outs_path, bin_size):
        # Using Tai-Hsien's Visium HD reader
        outs_path = Path(outs_path)
        bin_size = bin_size
        path_bin = outs_path / bin_size
        path_bin_spatial = path_bin / VisiumHDKeys.SPATIAL

        # Load gene expression
        # Determine counts file based on bin_size
        if bin_size == "segmented_outputs":
            counts_file = "raw_feature_cell_matrix.h5"
        elif "square_" in bin_size and bin_size.endswith("um"):
            counts_file = "raw_feature_bc_matrix.h5"
        else:
            counts_file = "raw_feature_cell_matrix.h5"  # default fallback
        adata = sc.read_10x_h5(path_bin / counts_file, gex_only=False)
        adata.var_names_make_unique()

        # Load scalefactors and images
        with open(path_bin_spatial / VisiumHDKeys.SCALEFACTORS_FILE) as f:
            scalefactors = json.load(f)

        hires_img = np.array(Image.open(path_bin_spatial / "tissue_hires_image.png"))
        lowres_img = np.array(Image.open(path_bin_spatial / "tissue_lowres_image.png"))

        library_id = "visium_hd_segmentation"
        adata.uns["spatial"] = {
            library_id: {
                "images": {
                    "hires": hires_img,
                    "lowres": lowres_img,
                },
                "scalefactors": scalefactors,
                "metadata": {
                    "source_image_path": "tissue_hires_image.png"
                }
            }
        }

        # Handle spatial coordinates based on bin_size type
        if "square_" in bin_size and bin_size.endswith("um"):
            # For square bins, load tissue_positions.parquet
            tissue_pos_file = path_bin_spatial / "tissue_positions.parquet"
            tissue_pos = pd.read_parquet(tissue_pos_file)
            
            # Set barcode as index if it's a column
            if "barcode" in tissue_pos.columns:
                tissue_pos = tissue_pos.set_index("barcode")
            
            # Align with adata barcodes
            tissue_pos = tissue_pos.loc[adata.obs_names]
            
            # Extract spatial coordinates (pxl_col_in_fullres, pxl_row_in_fullres)
            adata.obsm["spatial"] = tissue_pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values
            
            # Use spot_diameter_fullres from scalefactors if available
            if "spot_diameter_fullres" not in scalefactors:
                # For square bins, estimate based on bin size
                # Extract bin size number (e.g., "square_008um" -> 8)
                bin_size_num = int(bin_size.replace("square_", "").replace("um", ""))
                scalefactors["spot_diameter_fullres"] = bin_size_num
        else:
            # For segmented outputs, use geojson
            shapes_name = "cell_segmentation"
            adata.obs[VisiumHDKeys.INSTANCE_KEY] = np.arange(len(adata))
            adata.obs[VisiumHDKeys.REGION_KEY] = shapes_name
            adata.obs[VisiumHDKeys.REGION_KEY] = adata.obs[VisiumHDKeys.REGION_KEY].astype("category")

            # Load cell shapes
            shapes = spatialdata_io.geojson(
                outs_path / "segmented_outputs" / "cell_segmentations.geojson",
                coordinate_system=""
            )

            # Match shape order to adata
            centroids = shapes.geometry.centroid
            adata.obsm["spatial"] = np.vstack([centroids.x.values, centroids.y.values]).T

            # Estimate spot diameter
            areas = shapes.geometry.area
            mean_area = np.mean(areas)
            diameter_pixels = 2 * np.sqrt(mean_area / np.pi)
            adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = diameter_pixels

        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        
        # Read clustering results and filter cells without cluster assignments
        clusters_df = pd.read_csv(outs_path / bin_size / "analysis" / "clustering" / "gene_expression_graphclust" / "clusters.csv")
        
        # Remove rows with NA cluster assignments
        clusters_df_valid = clusters_df[clusters_df['Cluster'].notna()].copy()
        
        # Create cluster mapping only for cells with valid cluster assignments
        cluster_mapping = dict(zip(clusters_df_valid['Barcode'], clusters_df_valid['Cluster']))
        
        # Count cells before filtering
        n_cells_before = adata.n_obs
        
        # Filter adata to keep only cells that have cluster assignments
        cells_with_clusters = adata.obs.index.isin(cluster_mapping.keys())
        adata = adata[cells_with_clusters].copy()
        
        # Now assign clusters (should have no NAs)
        adata.obs['clusters'] = adata.obs.index.map(cluster_mapping)
        adata.obs['clusters'] = adata.obs['clusters'].astype('category')
        
        # Report filtering
        n_cells_after = adata.n_obs
        n_cells_removed = n_cells_before - n_cells_after
        print(f"Filtered cells without cluster assignments: {n_cells_removed} removed, {n_cells_after} retained (out of {n_cells_before} total)")
        
        # Add spatial coordinates in micrometers (if scalefactors has microns_per_pixel)
        # For Visium HD, typically 1 pixel ≈ 2.0 micrometers at bin size 8
        if "microns_per_pixel" in scalefactors:
            microns_per_pixel = scalefactors["microns_per_pixel"]
            adata.obsm["spatial_um"] = adata.obsm["spatial"] * microns_per_pixel
        # elif "spot_diameter_fullres" in scalefactors:
        #     # Estimate from spot diameter if available (assuming 55 micron spots)
        #     microns_per_pixel = 55.0 / scalefactors["spot_diameter_fullres"]
        #     adata.obsm["spatial_um"] = adata.obsm["spatial"] * microns_per_pixel
        # else:
        #     # Default fallback for HD segmentation (commonly ~2 microns per pixel)
        #     microns_per_pixel = 2.0
        #     adata.obsm["spatial_um"] = adata.obsm["spatial"] * microns_per_pixel
        #     print(f"Warning: Using default microns_per_pixel = {microns_per_pixel}")

        return adata


class FuzzyAverage:
    def fuzzy_average(
        self,
        adata,
        *,
        spatial_key: str = "spatial",
        n_neighs: int = 6,
        include_self: bool = True,
        use_existing_W: bool = True,
        coord_type: str = "generic",
        delaunay: bool = True,
        layer_out: str | None = None,
        copy: bool = False,
    ):
        """
        Build a row-normalized (average) smoothing operator and apply fuzzy averaging to adata.X.
        Optionally stores results in a layer and/or returns a copy.
        Returns X_fuzzy (ndarray) unless `copy=True` and `layer_out` is provided,
        in which case the modified AnnData copy is returned.
        """
        # Build or retrieve W
        if use_existing_W and "spatial_connectivities" in adata.obsp:
            W = adata.obsp["spatial_connectivities"].tocsr()
        else:
            try:
                import squidpy as sq
            except ImportError as e:
                raise ImportError("squidpy is required to build the spatial graph.") from e
            if spatial_key not in adata.obsm:
                raise KeyError(f"`{spatial_key}` not found in adata.obsm")
            sq.gr.spatial_neighbors(
                adata,
                coord_type=coord_type,
                delaunay=delaunay,
                n_neighs=n_neighs,
                spatial_key=spatial_key,
            )
            W = adata.obsp["spatial_connectivities"].tocsr()

        # Build smoothing operator S_norm
        S = W + sparse.eye(W.shape[0], format="csr") if include_self else W
        row_sums = np.asarray(S.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        S_norm = sparse.diags(1.0 / row_sums, format="csr") @ S

        # Apply fuzzy smoothing
        X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        X_fuzzy = S_norm @ X

        # Store output layer if requested
        if layer_out is not None:
            if copy:
                ad = adata.copy()
                ad.layers[layer_out] = X_fuzzy
                return ad
            adata.layers[layer_out] = X_fuzzy

        return X_fuzzy


# ============================================================
# numba MI calculation functions - cafr-C parity
# Based on CAFR and Tai-Hsien's code 
# ============================================================
import numpy as np
from numba import njit

_LOG2 = np.log(2.0)

# ---------- Helpers ----------

@njit(cache=True)
def _log2_like_c(x: float) -> float:
    # C: log(x)/log(2)
    return np.log(x) / _LOG2

@njit(cache=True)
def _build_knots(num_bins: int, spline_order: int):
    k = np.zeros(num_bins + spline_order)
    n_internal = num_bins - spline_order
    for i in range(spline_order, spline_order + n_internal):
        k[i] = (i - spline_order + 1.0) / (n_internal + 1.0)
    for i in range(spline_order + n_internal, num_bins + spline_order):
        k[i] = 1.0
    return k

@njit(cache=True)
def _basis_all_order1(t: float, knots, num_bins: int):
    # Base (order==1) with special right-edge inclusion for the last bin (C parity)
    N = np.zeros(num_bins)
    for i in range(num_bins):
        left = knots[i]
        right = knots[i + 1]
        if (t >= left and t < right and left < right) or (abs(t - right) < 1e-10 and (i + 1 == num_bins)):
            N[i] = 1.0
    return N

@njit(cache=True)
def _basis_all(t: float, knots, num_bins: int, spline_order: int):
    # Cox–de Boor recursion; spline_order==1 is base; clamp tiny negatives to 0 (C parity)
    N = _basis_all_order1(t, knots, num_bins)
    for q in range(2, spline_order + 1):
        Nq = np.zeros(num_bins)
        for i in range(num_bins):
            # term 1
            d1 = knots[i + q - 1] - knots[i]
            e1 = 0.0
            if d1 > 1e-12:
                e1 = (t - knots[i]) / d1 * N[i]

            # term 2
            e2 = 0.0
            if i + 1 < num_bins:
                d2 = knots[i + q] - knots[i + 1]
                if d2 > 1e-12:
                    e2 = (knots[i + q] - t) / d2 * N[i + 1]

            val = e1 + e2
            # C code: if (e1 + e2 < 0) return 0
            Nq[i] = val if val > 0.0 else 0.0
        N = Nq
    return N

@njit(cache=True)
def _normalize_to_unit(x):
    xmin = x.min()
    xmax = x.max()
    out = np.empty_like(x)
    rng = xmax - xmin
    if rng <= 0.0:
        for i in range(x.shape[0]):
            out[i] = 0.0
        return out
    inv = 1.0 / rng
    for i in range(x.shape[0]):
        out[i] = (x[i] - xmin) * inv
    return out

@njit(cache=True)
def _entropy_from_counts(counts, n_samples: int):
    H = 0.0
    inv_n = 1.0 / n_samples
    for i in range(counts.shape[0]):
        p = counts[i] * inv_n
        if p > 0.0:
            H -= p * _log2_like_c(p)   # C parity
    return H

@njit(cache=True)
def _entropy2d_from_counts(counts2d, n_samples: int):
    H = 0.0
    inv_n = 1.0 / n_samples
    for i in range(counts2d.shape[0]):
        for j in range(counts2d.shape[1]):
            p = counts2d[i, j] * inv_n
            if p > 0.0:
                H -= p * _log2_like_c(p)  # C parity
    return H

@njit(cache=True)
def _product_moment(x, y):
    """
    C parity:
      sumXY = Σ(x_i * y_i)
      sumX  = Σ(x_i)
      sumY  = Σ(y_i)
      productMoment = n*sumXY - sumX*sumY
    """
    n = x.shape[0]
    sumX = 0.0
    sumY = 0.0
    sumXY = 0.0
    for i in range(n):
        xi = x[i]
        yi = y[i]
        sumX += xi
        sumY += yi
        sumXY += xi * yi
    return n * sumXY - sumX * sumY

# ---------- Core MI (C parity) ----------

@njit(cache=True)
def mi_2d_fast(x, y, num_bins=6, spline_order=3, normalize=False, negateMI=False):
    n = x.shape[0]
    knots = _build_knots(num_bins, spline_order)

    xn = _normalize_to_unit(x)
    yn = _normalize_to_unit(y)

    px = np.zeros(num_bins)
    py = np.zeros(num_bins)
    pxy = np.zeros((num_bins, num_bins))

    for s in range(n):
        bx = _basis_all(xn[s], knots, num_bins, spline_order)
        by = _basis_all(yn[s], knots, num_bins, spline_order)

        # marginals
        for i in range(num_bins):
            px[i] += bx[i]
            py[i] += by[i]

        # joint
        for i in range(num_bins):
            bix = bx[i]
            if bix == 0.0:
                continue
            for j in range(num_bins):
                pxy[i, j] += bix * by[j]

    Hx = _entropy_from_counts(px, n)
    Hy = _entropy_from_counts(py, n)
    Hxy = _entropy2d_from_counts(pxy, n)
    mi = Hx + Hy - Hxy  # raw MI

    if normalize:
        # C parity normalization
        pxx = np.zeros((num_bins, num_bins))
        for s in range(n):
            bx = _basis_all(xn[s], knots, num_bins, spline_order)
            for i in range(num_bins):
                bi = bx[i]
                if bi == 0.0:
                    continue
                for j in range(num_bins):
                    pxx[i, j] += bi * bx[j]
        Hxx = _entropy2d_from_counts(pxx, n)

        pyy = np.zeros((num_bins, num_bins))
        for s in range(n):
            by = _basis_all(yn[s], knots, num_bins, spline_order)
            for i in range(num_bins):
                bi = by[i]
                if bi == 0.0:
                    continue
                for j in range(num_bins):
                    pyy[i, j] += bi * by[j]
        Hyy = _entropy2d_from_counts(pyy, n)

        mix = 2.0 * Hx - Hxx
        miy = 2.0 * Hy - Hyy
        larger = mix if mix > miy else miy
        if larger == 0.0:  # EXACT C behavior
            larger = 1.0
        mi = mi / larger

    # Optional signed MI, C parity
    if negateMI:
        if _product_moment(x, y) < 0.0:
            mi = -mi

    return mi

@njit(cache=True)
def _precompute_basis_matrix(vec, num_bins=6, spline_order=3):
    n = vec.shape[0]
    knots = _build_knots(num_bins, spline_order)
    vn = _normalize_to_unit(vec)
    B = np.zeros((n, num_bins))
    for s in range(n):
        B[s, :] = _basis_all(vn[s], knots, num_bins, spline_order)
    return B

@njit(cache=True)
def mi_scan_against_matrix(seed_vec, X_fuzzy, num_bins=6, spline_order=3, normalize=True, negateMI=False):
    """
    Compute MI(seed_vec, X_fuzzy[:, j]) for all columns j.
    Matches C getAllMIWz(): normalization and optional signed MI.
    """
    n, g = X_fuzzy.shape
    knots = _build_knots(num_bins, spline_order)

    # Seed basis & entropies
    B_seed = _precompute_basis_matrix(seed_vec, num_bins, spline_order)

    px = np.zeros(num_bins)
    for s in range(n):
        for i in range(num_bins):
            px[i] += B_seed[s, i]
    Hx = _entropy_from_counts(px, n)

    pxx = np.zeros((num_bins, num_bins))
    for s in range(n):
        for i in range(num_bins):
            bi = B_seed[s, i]
            if bi == 0.0:
                continue
            for j in range(num_bins):
                pxx[i, j] += bi * B_seed[s, j]
    Hxx = _entropy2d_from_counts(pxx, n)
    mix = 2.0 * Hx - Hxx  # for normalization branch (seed side)

    mi_scores = np.empty(g)

    for j in range(g):
        y = X_fuzzy[:, j]
        yn = _normalize_to_unit(y)

        py = np.zeros(num_bins)
        pxy = np.zeros((num_bins, num_bins))

        for s in range(n):
            by = _basis_all(yn[s], knots, num_bins, spline_order)

            # py
            for i in range(num_bins):
                py[i] += by[i]

            # joint with seed
            for i in range(num_bins):
                bix = B_seed[s, i]
                if bix == 0.0:
                    continue
                for k in range(num_bins):
                    pxy[i, k] += bix * by[k]

        Hy = _entropy_from_counts(py, n)
        Hxy = _entropy2d_from_counts(pxy, n)
        mi = Hx + Hy - Hxy

        if normalize:
            # Hyy for this gene
            pyy = np.zeros((num_bins, num_bins))
            for s in range(n):
                by = _basis_all(yn[s], knots, num_bins, spline_order)
                for a in range(num_bins):
                    bia = by[a]
                    if bia == 0.0:
                        continue
                    for b in range(num_bins):
                        pyy[a, b] += bia * by[b]
            Hyy = _entropy2d_from_counts(pyy, n)

            miy = 2.0 * Hy - Hyy
            larger = mix if mix > miy else miy
            if larger == 0.0:  # EXACT C behavior
                larger = 1.0
            mi = mi / larger

        if negateMI:
            if _product_moment(seed_vec, y) < 0.0:
                mi = -mi

        mi_scores[j] = mi

    return mi_scores

# ---------- Rank utilities (outside numba) ----------
def _rank_1d(a: np.ndarray) -> np.ndarray:
    """
    R parity for rank(x): average ranks for ties (like base R default).
    """
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)

    # handle ties -> average of ranks
    # find groups of equal values in sorted order
    sorted_vals = a[order]
    i = 0
    n = a.size
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            # average of [i+1, ..., j] in 1-based ranks
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks

def _rank_rows_inplace(X: np.ndarray) -> None:
    """
    Rank each row (R's `apply(..., rank)` in-place). Leaves X as float64.
    """
    for i in range(X.shape[0]):
        X[i, :] = _rank_1d(X[i, :])

# ---------- Diff-bins core (Numba) ----------
@njit(cache=True)
def mi_2d_diffbins_fast(x, y, binx=6, biny=6, sox=3, soy=3, normalize=True, negateMI=False):
    """
    C parity of mi2DiffBins():
      - separate knots and orders for X and Y
      - normalization uses MI(X,X) with binx,sox and MI(Y,Y) with biny,soy
    """
    n = x.shape[0]

    knots_x = _build_knots(binx, sox)
    knots_y = _build_knots(biny, soy)

    xn = _normalize_to_unit(x)
    yn = _normalize_to_unit(y)

    px = np.zeros(binx)
    py = np.zeros(biny)
    pxy = np.zeros((binx, biny))

    for s in range(n):
        bx = _basis_all(xn[s], knots_x, binx, sox)
        by = _basis_all(yn[s], knots_y, biny, soy)

        # marginals
        for i in range(binx):
            px[i] += bx[i]
        for j in range(biny):
            py[j] += by[j]

        # joint
        for i in range(binx):
            bi = bx[i]
            if bi == 0.0:
                continue
            for j in range(biny):
                pxy[i, j] += bi * by[j]

    Hx = _entropy_from_counts(px, n)
    Hy = _entropy_from_counts(py, n)
    Hxy = _entropy2d_from_counts(pxy, n)
    mi = Hx + Hy - Hxy

    if normalize:
        # MI(X,X) with (binx,sox)
        pxx = np.zeros((binx, binx))
        for s in range(n):
            bx = _basis_all(xn[s], knots_x, binx, sox)
            for i in range(binx):
                bi = bx[i]
                if bi == 0.0:
                    continue
                for j in range(binx):
                    pxx[i, j] += bi * bx[j]
        Hxx = _entropy2d_from_counts(pxx, n)
        mix = 2.0 * Hx - Hxx

        # MI(Y,Y) with (biny,soy)
        pyy = np.zeros((biny, biny))
        for s in range(n):
            by = _basis_all(yn[s], knots_y, biny, soy)
            for i in range(biny):
                bi = by[i]
                if bi == 0.0:
                    continue
                for j in range(biny):
                    pyy[i, j] += bi * by[j]
        Hyy = _entropy2d_from_counts(pyy, n)
        miy = 2.0 * Hy - Hyy

        larger = mix if mix > miy else miy
        if larger == 0.0:
            larger = 1.0
        mi = mi / larger

    if negateMI and _product_moment(x, y) < 0.0:
        mi = -mi

    return mi


# ---------- Public compatible wrapper ----------
class FastMICalculateNumba:
    def mi_2d(self, x, y,
              num_bins=6, spline_order=3,
              normalize=True, negateMI=True,
              rank_based=False):
        """
        R getMI() parity for continuous x,y:
          - rank_based => rank-transform both
          - normalize, negateMI as in C
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if rank_based:
            x = _rank_1d(x)
            y = _rank_1d(y)

        return float(mi_2d_fast(x, y, num_bins, spline_order, normalize, negateMI))

    def mi_2d_categorical(self, x, y_labels,
                          num_bins=6, spline_order=3,
                          normalize=True, negateMI=False,
                          ordered=False):
        """
        R getMI() parity when y is a factor:
          - Map factor levels to 1..L (float64)
          - soy = 3 if ordered, else 1 (exact R logic)
          - binx=num_bins, sox=spline_order
          - biny = number of levels
        """
        x = np.asarray(x, dtype=np.float64)

        # Map labels -> [1..L]
        y_labels = np.asarray(y_labels)
        # stable level order: unique in encounter order
        # (R uses levels() ordering; if you need a custom order, pass y_labels already coded)
        _, inv = np.unique(y_labels, return_inverse=True)
        y_num = inv.astype(np.float64) + 1.0  # 1..L for readability; values are arbitrary continuous here

        binx = int(num_bins)
        sox = int(spline_order)
        biny = int(np.unique(y_num).size)
        soy = 3 if ordered else 1

        return float(mi_2d_diffbins_fast(x, y_num, binx, biny, sox, soy, normalize, negateMI))

    def mi_2d_diffbins(self, x, y,
                       binx=6, biny=6, sox=3, soy=3,
                       normalize=True, negateMI=False,
                       rank_based=False):
        """
        Direct exposure of C's mi2DiffBins().
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if rank_based:
            x = _rank_1d(x)
            y = _rank_1d(y)

        return float(mi_2d_diffbins_fast(x, y, binx, biny, sox, soy, normalize, negateMI))

    def mi_scan(self, seed_vec, X_fuzzy,
                num_bins=6, spline_order=3,
                normalize=True, negateMI=True,
                rank_based=False):
        """
        R getAllMIWz() parity:
          - If rank_based, rank each row of X_fuzzy and the seed (like R branch)
          - Uses same (num_bins, spline_order) for all columns (R parity)
        """
        seed_vec = np.asarray(seed_vec, dtype=np.float64)
        X_fuzzy = np.asarray(X_fuzzy, dtype=np.float64)

        if rank_based:
            # rank each row of X_fuzzy, and the seed
            X_rank = X_fuzzy.copy()
            _rank_rows_inplace(X_rank)
            seed_rank = _rank_1d(seed_vec)
            return mi_scan_against_matrix(seed_rank, X_rank,
                                          num_bins, spline_order, normalize, negateMI)
        else:
            return mi_scan_against_matrix(seed_vec, X_fuzzy,
                                          num_bins, spline_order, normalize, negateMI)

    def find_attractor(self, data, vec, gene_names,
                       a=5, max_iter=100, epsilon=1e-14,
                       num_bins=6, spline_order=3,
                       rank_based=False, negate_mi=True,
                       verbose=True, 
                       rankdata = None):
        """
        Python equivalent of R findAttractor()
        Returns: (sorted_mi, sorted_gene_indices)
        """
        data = np.asarray(data, dtype=np.float64)
        vec = np.asarray(vec, dtype=np.float64)
        m, n = data.shape

        # Ranking (R parity)
        if rank_based:
            vec = rankdata(vec)
            data_in = np.zeros_like(data)
            for i in range(m):
                data_in[i, :] = rankdata(data[i, :])
        else:
            data_in = data.copy()

        # Initial MI
        mi = self.mi_scan(vec, data_in.T,
                          num_bins=num_bins, spline_order=spline_order,
                          normalize=False, negateMI=negate_mi)
        premi = mi.copy()

        # Compute weights
        w = np.abs(mi) ** a
        w_sum = np.sum(w)
        if w_sum == 0:
            raise ValueError("All MI weights are zero.")
        w = w / w_sum
        w[mi < 0] = 0.0

        metagene = data.T @ w
        if rank_based:
            metagene = rankdata(metagene)

        # Iteration loop
        c = 0
        while c < max_iter:
            mi = self.mi_scan(metagene, data_in.T,
                              num_bins=num_bins, spline_order=spline_order,
                              normalize=False, negateMI=negate_mi)
            delta = np.sum((mi - premi) ** 2)

            if verbose:
                print(f"Iteration {c+1}\tDelta = {delta:.3e}")
                top_idx = np.argsort(mi)[::-1][:20]
                top_vals = mi[top_idx]
                print("Top 20 MI values:")
                for rank, (idx, val) in enumerate(zip(top_idx, top_vals), start=1):
                    gene_name = gene_names[idx] if idx < len(gene_names) else f"Gene_{idx}"
                    print(f"{rank:2d}. {gene_name:<15} (idx:{idx:4d})  weight={val:.6f}")
                print("-" * 40)

            if delta < epsilon:
                break

            premi = mi.copy()
            mi[mi < 0] = 0.0
            w = np.abs(mi) ** a
            w_sum = np.sum(w)
            if w_sum == 0:
                break
            w = w / w_sum
            w[mi < 0] = 0.0
            metagene = data.T @ w
            if rank_based:
                metagene = rankdata(metagene)

            c += 1

        if c >= max_iter:
            if verbose:
                print("Max iterations reached without convergence.")
            return None, None

        sorted_idx = np.argsort(mi)[::-1]
        weights = mi[sorted_idx]
        idx =  sorted_idx
        genes = gene_names[idx]
        gene_weight_pairs = list(zip(genes, weights))
        gene_weight_pairs_sorted = sorted(gene_weight_pairs, key=lambda x: x[1], reverse=True)
        return gene_weight_pairs_sorted





# ---------- Public wrapper ----------

# class FastMICalculateNumba:
#     def mi_2d(self, x, y, num_bins=6, spline_order=3, normalize=True, negateMI=True):
#         return float(mi_2d_fast(np.asarray(x, dtype=np.float64),
#                                 np.asarray(y, dtype=np.float64),
#                                 num_bins, spline_order, normalize, negateMI))

#     def mi_scan(self, seed_vec, X_fuzzy, num_bins=6, spline_order=3, normalize=True, negateMI=True):
#         return mi_scan_against_matrix(np.asarray(seed_vec, dtype=np.float64),
#                                       np.asarray(X_fuzzy, dtype=np.float64),
#                                       num_bins, spline_order, normalize, negateMI)


# class MICalculate:
#     # Author: MI python version by Tai-Hsien 
#     # from __future__ import division  # For floating number division in python2
#     import numpy as np
#     import pandas as pd
#     import time

#     # Cox-de Boor recursion formula
#     def basis_function(self, i, p, t, k_vector, num_bins):
#         if p == 1:
#             if (t >= k_vector[i]) and (t < k_vector[i + 1]) and (k_vector[i] < k_vector[i + 1]):
#                 return 1
#             elif abs(t - k_vector[i + 1]) < 1e-10 and (i + 1 == num_bins):
#                 return 1
#             else:
#                 return 0
#         else:
#             d1 = k_vector[i + p - 1] - k_vector[i]
#             n1 = t - k_vector[i]
#             d2 = k_vector[i + p] - k_vector[i + 1]
#             n2 = k_vector[i + p] - t

#             if d1 < 1e-10 and d2 < 1e-10:
#                 e1 = 0
#                 e2 = 0
#             elif d1 < 1e-10:
#                 e1 = 0
#                 e2 = n2 / d2 * self.basis_function(i + 1, p - 1, t, k_vector, num_bins)
#             elif d2 < 1e-10:
#                 e1 = n1 / d1 * self.basis_function(i, p - 1, t, k_vector, num_bins)
#                 e2 = 0
#             else:
#                 e1 = n1 / d1 * self.basis_function(i, p - 1, t, k_vector, num_bins)
#                 e2 = n2 / d2 * self.basis_function(i + 1, p - 1, t, k_vector, num_bins)

#             if e1 + e2 < 0:
#                 return 0

#         return e1 + e2

#     # Convert the values into B-Spline weights
#     def find_weights(self, x, knots, num_bins, spline_order):
#         num_samples = len(x)
#         weights = np.zeros(num_samples * num_bins)

#         xmax = max(x)
#         xmin = min(x)
#         x = (x - xmin) / (xmax - xmin)

#         for i in range(num_samples):
#             for j in range(num_bins):
#                 weights[j * num_samples + i] = self.basis_function(j, spline_order, x[i], knots, num_bins)

#         return weights

#     # 1-D Entropy
#     def entropy_1d(self, weights, num_samples, num_bins):
#         H = 0
#         for i in range(num_bins):
#             p = np.sum(weights[i * num_samples:(i + 1) * num_samples])
#             if p > 0:
#                 H -= (p / num_samples) * np.log2(p / num_samples)
#         return H

#     # 2-D Entropy
#     def entropy_2d(self, weights_x, weights_y, num_samples, num_bins):
#         H = 0
#         for ix in range(num_bins):
#             for iy in range(num_bins):
#                 p = np.sum(weights_x[ix * num_samples: (ix + 1) * num_samples] *
#                         weights_y[iy * num_samples: (iy + 1) * num_samples])
#                 if p > 0:
#                     H -= (p / num_samples) * np.log2(p / num_samples)
#         return H

#     # Mutual Information
#     def mi_2d(self, x, y, num_bins=6, spline_order=3, normalize=False):
#         num_samples = len(x)
#         knots = np.zeros(num_bins + spline_order)
#         n_internal_points = num_bins - spline_order

#         for i in range(spline_order, spline_order + n_internal_points):
#             knots[i] = (i - spline_order + 1) / (n_internal_points + 1)
#         for i in range(spline_order + n_internal_points, num_bins + spline_order):
#             knots[i] = 1

#         wx = self.find_weights(x, knots, num_bins, spline_order)
#         wy = self.find_weights(y, knots, num_bins, spline_order)

#         Hx = self.entropy_1d(wx, num_samples, num_bins)
#         Hy = self.entropy_1d(wy, num_samples, num_bins)
#         Hxy = self.entropy_2d(wx, wy, num_samples, num_bins)

#         mi = Hx + Hy - Hxy

#         # --- normalization (match C code from CAFR) ---
#         if normalize:
#             Hxx = self.entropy_2d(wx, wx, num_samples, num_bins)
#             Hyy = self.entropy_2d(wy, wy, num_samples, num_bins)
#             mix = 2 * Hx - Hxx
#             miy = 2 * Hy - Hyy
#             largerMI = max(mix, miy)
#             if largerMI == 0:
#                 largerMI = 1
#             mi /= largerMI
#         # -----------------------------------

#         return mi


#     def find_attractor(self, df, vec, a=5, bin=6, so=3, max_iter=100, epsilon=1E-14, verbose=True):
#         c = 0
#         mi = np.zeros(len(df.columns))

#         for idx in range(len(df.columns)):
#             mi[idx] = self.mi_2d(df.iloc[:, idx].values, vec, bin, so)

#         premi = mi.copy()
#         w = np.abs(mi) ** a / np.sum(np.abs(mi) ** a)
#         w[mi < 0] = 0
#         metagene = np.dot(df, w)

#         while c < max_iter:
#             for idx in range(len(df.columns)):
#                 mi[idx] = self.mi_2d(df.iloc[:, idx].values, metagene, bin, so)

#             delta = np.sum((mi - premi) ** 2)

#             if verbose:
#                 print("\r{}, {}.".format(c, np.round(delta, 4)), end="", flush=True)
#                 time.sleep(0.1)

#             if delta < epsilon:
#                 break

#             premi = mi.copy()
#             w = np.abs(mi) ** a / np.sum(np.abs(mi) ** a)
#             w[mi < 0] = 0
#             metagene = np.dot(df, w)
#             c += 1

#         if c >= max_iter:
#             return np.nan
#         return mi

# class MICalculate:
#     def get_all_miwz_R(
#         self,
#         expr_matrix,
#         gene: str,
#         gene_names,
#         bin: int = 6,
#         so: int = 3,
#         rankBased: bool = False,
#         normalize: bool = True,
#         sorting: bool = True,
#         negateMI: bool = True,
#         use_cafr: bool = True,
#         r_source: str | None = None,
#     ):
#         """
#         Call the R function getAllMIWz(data, x, ...) through rpy2 and return a pandas Series.

#         Parameters
#         ----------
#         expr_matrix : array-like (cells x genes)
#             Expression matrix, rows are cells, columns are genes.
#         gene : str
#             Target gene symbol present in gene_names.
#         gene_names : list-like
#             Gene names corresponding to columns in expr_matrix.
#         bin, so, rankBased, normalize, sorting, negateMI : R function arguments.
#         use_cafr : bool
#             If True, runs `library(cafr)`. If False, or if cafr not installed,
#             you can supply `r_source="path/to/getAllMIWz.R"` to source the function.
#         r_source : str | None
#             Path to a local R file that defines getAllMIWz (used if cafr is unavailable).

#         Returns
#         -------
#         pandas.Series
#             Named by gene, values are MI to the target `gene`.
#         """
#         import numpy as np
#         import pandas as pd
#         import rpy2.robjects as ro
#         from rpy2.robjects import numpy2ri
#         from rpy2.robjects.conversion import localconverter
#         from rpy2.robjects.packages import importr

#         # --- prep R side: load cafr or source file ---
#         r = ro.r
#         have_func = False
#         if use_cafr:
#             try:
#                 importr("cafr")
#                 r("library(cafr)")
#                 have_func = bool(r('exists("getAllMIWz")'))
#             except Exception:
#                 have_func = False
#         if not have_func:
#             if not r_source:
#                 raise RuntimeError(
#                     "R function getAllMIWz is not available. "
#                     "Install/load 'cafr' or provide r_source='path/to/getAllMIWz.R'."
#                 )
#             r(f'source("{r_source}")')
#             if not bool(r('exists("getAllMIWz")')):
#                 raise RuntimeError("After sourcing, getAllMIWz is still not found in R.")

#         # --- check inputs ---
#         gene_names = list(gene_names)
#         if gene not in gene_names:
#             raise ValueError(f"Gene '{gene}' not found in gene_names")
#         gene_idx = gene_names.index(gene)

#         # --- extract vectors/matrix ---
#         expr_matrix = np.asarray(expr_matrix)
#         x = expr_matrix[:, gene_idx].ravel()
#         data = expr_matrix.T  # genes × cells

#         # --- push numpy -> R ---
#         from rpy2.robjects.vectors import StrVector
#         with localconverter(ro.default_converter + numpy2ri.converter):
#             ro.globalenv["data_mat"] = data
#             ro.globalenv["x_vec"] = x
#         ro.globalenv["rn"] = StrVector(gene_names)
#         r("rownames(data_mat) <- rn")

#         # --- call R getAllMIWz ---
#         call = (
#             "getAllMIWz("
#             "data_mat, x_vec, "
#             f"bin={int(bin)}, so={int(so)}, "
#             f"rankBased={'TRUE' if rankBased else 'FALSE'}, "
#             f"normalize={'TRUE' if normalize else 'FALSE'}, "
#             f"sorting={'TRUE' if sorting else 'FALSE'}, "
#             f"negateMI={'TRUE' if negateMI else 'FALSE'})"
#         )
#         res = r(call)

#         # --- convert result back ---
#         with localconverter(ro.default_converter + numpy2ri.converter):
#             mi_values = np.array(res, dtype=float)
#         mi_names = list(res.names) if res.names is not None else gene_names
#         return pd.Series(mi_values, index=mi_names, name="MI")

#     def get_pairwise_mi(
#         self,
#         x_vec,
#         y_vec,
#         bin: int = 6,
#         so: int = 3,
#         rankBased: bool = False,
#         normalize: bool = True,
#         negateMI: bool = True,
#         use_cafr: bool = True,
#         r_source: str | None = None,
#     ):
#         """
#         Compute mutual information (MI) between two vectors using R's getMI function.

#         Parameters
#         ----------
#         x_vec, y_vec : array-like
#             Two numeric vectors of equal length.
#         bin, so, rankBased, normalize, negateMI : R function arguments.
#         use_cafr : bool
#             If True, loads `cafr` library. If False, or cafr not installed,
#             you can supply `r_source="path/to/getMI.R"` to source the function.
#         r_source : str | None
#             Path to a local R file that defines getMI (used if cafr is unavailable).

#         Returns
#         -------
#         float
#             Mutual information value.
#         """
#         import numpy as np
#         import rpy2.robjects as ro
#         from rpy2.robjects import numpy2ri
#         from rpy2.robjects.conversion import localconverter
#         from rpy2.robjects.packages import importr

#         # --- prep R side: load cafr or source file ---
#         r = ro.r
#         have_func = False
#         if use_cafr:
#             try:
#                 importr("cafr")
#                 r("library(cafr)")
#                 have_func = bool(r('exists("getMI")'))
#             except Exception:
#                 have_func = False
#         if not have_func:
#             if not r_source:
#                 raise RuntimeError(
#                     "R function getMI is not available. "
#                     "Install/load 'cafr' or provide r_source='path/to/getMI.R'."
#                 )
#             r(f'source("{r_source}")')
#             if not bool(r('exists("getMI")')):
#                 raise RuntimeError("After sourcing, getMI is still not found in R.")

#         # --- check vectors ---
#         x_vec = np.asarray(x_vec, dtype=float).ravel()
#         y_vec = np.asarray(y_vec, dtype=float).ravel()
#         if x_vec.shape[0] != y_vec.shape[0]:
#             raise ValueError("x_vec and y_vec must be the same length")

#         # --- push numpy -> R ---
#         with localconverter(ro.default_converter + numpy2ri.converter):
#             ro.globalenv["x_vec"] = x_vec
#             ro.globalenv["y_vec"] = y_vec

#         # --- call R getMI ---
#         call = (
#             "getMI("
#             "x_vec, y_vec, "
#             f"bin={int(bin)}, so={int(so)}, "
#             f"rankBased={'TRUE' if rankBased else 'FALSE'}, "
#             f"normalize={'TRUE' if normalize else 'FALSE'}, "
#             f"negateMI={'TRUE' if negateMI else 'FALSE'})"
#         )
#         res = r(call)[0]  # scalar

#         return float(res)



# ARCHIVEED 

    # def build_fuzzy_operator(
    #     self,
    #     adata,
    #     *,
    #     spatial_key: str = "spatial",
    #     n_neighs: int = 6,
    #     delaunay: bool = True,
    #     include_self: bool = True,
    #     use_existing_W: bool = False,
    #     coord_type: str = "generic",
    # ):
    #     """
    #     Build a row-normalized spatial smoothing operator S_norm from an AnnData.
    #     Returns (S_norm, W) as CSR matrices.
    #     """
    #     # Get or build adjacency W
    #     if use_existing_W and "spatial_connectivities" in adata.obsp:
    #         W = adata.obsp["spatial_connectivities"].tocsr()
    #     else:
    #         try:
    #             import squidpy as sq
    #         except ImportError as e:
    #             raise ImportError(
    #                 "squidpy is required to build the spatial graph when W is not present."
    #             ) from e
    #         if spatial_key not in adata.obsm:
    #             raise KeyError(f"`{spatial_key}` not found in adata.obsm")
    #         sq.gr.spatial_neighbors(
    #             adata,
    #             coord_type=coord_type,      
    #             delaunay=delaunay,
    #             n_neighs=n_neighs,
    #             spatial_key=spatial_key,
    #         )
    #         W = adata.obsp["spatial_connectivities"].tocsr()

    #     S = W
    #     if include_self:
    #         S = S + sparse.eye(S.shape[0], format="csr")

    #     # Row-normalize: S_norm = D^{-1} S, 1 to 1/number of neighbors 
    #     row_sums = np.asarray(S.sum(axis=1)).ravel()
    #     row_sums[row_sums == 0] = 1.0
    #     Dinv = sparse.diags(1.0 / row_sums, format="csr")
    #     S_norm = Dinv @ S
    #     return S_norm, W


    # def fuzzy_average_adata(
    #     self,
    #     adata,
    #     *,
    #     S_norm=None,
    #     spatial_key: str = "spatial",
    #     n_neighs: int = 6,
    #     delaunay: bool = True,
    #     include_self: bool = True,
    #     layer_out: str | None = None,
    #     copy: bool = False,
    #     coord_type: str = "generic",
    #     use_existing_W: bool = True,   # default to reuse if present
    # ):
    #     """
    #     Apply fuzzy averaging to adata.X and optionally store to a layer.
    #     Returns X_fuzzy (ndarray) unless copy=True and layer_out is provided,
    #     in which case returns the modified AnnData copy.
    #     """
    #     if S_norm is None:
    #         S_norm, _ = self.build_fuzzy_operator(
    #             adata,
    #             spatial_key=spatial_key,
    #             n_neighs=n_neighs,
    #             delaunay=delaunay,
    #             include_self=include_self,
    #             use_existing_W=use_existing_W,
    #             coord_type=coord_type,
    #         )

    #     # average expression matrix
    #     X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    #     X_fuzzy = S_norm @ X

    #     if layer_out is not None:
    #         if copy:
    #             ad = adata.copy()
    #             ad.layers[layer_out] = X_fuzzy
    #             return ad
    #         else:
    #             adata.layers[layer_out] = X_fuzzy
    #     return X_fuzzy


    # def fuzzy_average_gene(
    #     self,
    #     adata,
    #     gene: str,
    #     *,
    #     S_norm=None,
    #     spatial_key: str = "spatial",
    #     n_neighs: int = 6,
    #     delaunay: bool = True,
    #     include_self: bool = True,
    #     coord_type: str = "generic",
    #     use_existing_W: bool = True,
    # ):
    #     """
    #     Convenience: fuzzy-average a single gene from `adata` and return a 1D vector.
    #     """
    #     if S_norm is None:
    #         S_norm, _ = self.build_fuzzy_operator(
    #             adata,
    #             spatial_key=spatial_key,
    #             n_neighs=n_neighs,
    #             delaunay=delaunay,
    #             include_self=include_self,
    #             use_existing_W=use_existing_W,
    #             coord_type=coord_type,
    #         )
    #     if gene not in adata.var_names:
    #         raise KeyError(f"Gene '{gene}' not found in adata.var_names")
    #     vec = adata[:, gene].X
    #     vec = vec.toarray().ravel() if sparse.issparse(vec) else np.asarray(vec).ravel()
    #     return S_norm @ vec


    # def plot_fuzzy_vs_raw(
    #     self,
    #     adata,
    #     gene: str = "COL11A1",
    #     *,
    #     spatial_key: str = "spatial",
    #     S_norm=None,                   # pass from build_fuzzy_operator(...) if you have it
    #     layer_fuzzy: str | None = None,  # e.g. "X_fuzzy" if you stored it
    #     n_neighs: int = 6,
    #     delaunay: bool = True,
    #     include_self: bool = True,
    #     coord_type: str = "generic",
    #     use_existing_W: bool = True,
    # ):
    #     """
    #     Plot a gene on the original matrix vs its fuzzy-averaged version.
    #     If S_norm is None and layer_fuzzy is None, this will build S_norm.
    #     """
    #     if gene not in adata.var_names:
    #         raise KeyError(f"Gene '{gene}' not found in adata.var_names")

    #     # raw vector
    #     raw = adata[:, gene].X
    #     raw = raw.toarray().ravel() if sparse.issparse(raw) else np.asarray(raw).ravel()

    #     # fuzzy vector
    #     if layer_fuzzy is not None:
    #         if layer_fuzzy not in adata.layers:
    #             raise KeyError(f"Layer '{layer_fuzzy}' not found in adata.layers")
    #         Xf = adata.layers[layer_fuzzy]
    #         Xf = Xf.toarray() if sparse.issparse(Xf) else np.asarray(Xf)
    #         j = int(np.where(adata.var_names == gene)[0][0])
    #         fuzzy = Xf[:, j]
    #     else:
    #         if S_norm is None:
    #             S_norm, _ = self.build_fuzzy_operator(
    #                 adata,
    #                 spatial_key=spatial_key,
    #                 n_neighs=n_neighs,
    #                 delaunay=delaunay,
    #                 include_self=include_self,
    #                 use_existing_W=use_existing_W,
    #                 coord_type=coord_type,
    #             )
    #         fuzzy = S_norm @ raw

    #     # coords
    #     if spatial_key not in adata.obsm:
    #         raise KeyError(f"`{spatial_key}` not found in adata.obsm")
    #     coords = adata.obsm[spatial_key]

    #     # shared color scale
    #     vmin = float(min(raw.min(), fuzzy.min()))
    #     vmax = float(max(raw.max(), fuzzy.max()))

    #     # plot
    #     fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    #     s1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=raw, s=15, cmap="viridis",
    #                         vmin=vmin, vmax=vmax, alpha=0.9)
    #     axes[0].set_title(f"{gene} (raw)")
    #     axes[0].set_xlabel("Array Col")
    #     axes[0].set_ylabel("Array Row")
    #     plt.colorbar(s1, ax=axes[0], label="expression")

    #     s2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=fuzzy, s=15, cmap="viridis",
    #                         vmin=vmin, vmax=vmax, alpha=0.9)
    #     axes[1].set_title(f"{gene} (fuzzy)")
    #     axes[1].set_xlabel("Array Col")
    #     axes[1].set_ylabel("Array Row")
    #     plt.colorbar(s2, ax=axes[1], label="expression")

    #     axes[0].invert_yaxis()
    #     axes[1].invert_yaxis()
    #     plt.show()
