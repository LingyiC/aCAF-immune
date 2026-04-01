# SpatialAttractor

**SpatialAttractor** discovers spatial gene attractors using neighbor-smoothed expression and mutual information.

## What it does
- Builds neighbor-smoothed (fuzzy) expression from spatial transcriptomics spots.
- Computes mutual information (MI) between a seed gene and all genes.
- Finds attractor gene programs with iterative MI-based weighting.
- Supports practical workflows for Visium/Visium-like datasets.

## Repository layout
- `tutorial.ipynb` — end-to-end tutorial (load data → fuzzy expression → MI scan → attractor search).
- `functions/` — helper functions and core methods.

## Quick start
1. Open `tutorial.ipynb`.
2. Set dataset fields in the load-data section:
   - `dataset_name = "HTAN_WUSTL/scRNA_vali/raw_spatial"`
   - `folder = "./datasets/" + dataset_name + "/"`
   - `sampleID = "HT232P1H2A2"`
3. Run all cells in order.

### Directory structure
```
├── /path/to/dataset/
│   └── raw/
│       ├── {sampleID1}/
│       │   ├── filtered_feature_bc_matrix.h5
│       │   ├── filtered_feature_bc_matrix/
│       │   │   ├── barcodes.tsv.gz
│       │   │   ├── features.tsv.gz
│       │   │   └── matrix.mtx.gz
│       │   └── spatial/
│       │       ├── tissue_positions_list.csv
│       │       ├── scalefactors_json.json
│       │       └── tissue_lowres_image.png
│       ├── {sampleID2}/
│       │   ├── filtered_feature_bc_matrix.h5
│       │   ├── filtered_feature_bc_matrix/
│       │   │   ├── barcodes.tsv.gz
│       │   │   ├── features.tsv.gz
│       │   │   └── matrix.mtx.gz
│       │   └── spatial/
│       │       ├── tissue_positions_list.csv
│       │       ├── scalefactors_json.json
│       │       └── tissue_lowres_image.png
```

### Required files

**`filtered_feature_bc_matrix.h5`** — HDF5-formatted gene expression matrix 

**`filtered_feature_bc_matrix/`** — Gene expression matrix (Optional, 10x Genomics format, fallback if .h5 unavailable)
- `barcodes.tsv.gz` — Spot/cell barcodes
- `features.tsv.gz` — Gene names and IDs
- `matrix.mtx.gz` — Sparse matrix of counts

**`spatial/`** — Spatial coordinate and image data
- `tissue_positions_list.csv` — Barcode-to-coordinate mapping (columns: `barcode, in_tissue, array_row, array_col, pxl_row, pxl_col`)
- `scalefactors_json.json` — Image scale factors for visualization
- `tissue_lowres_image.png` — Low-resolution H&E or IF image

### Example
For a sample with ID `HT232P1H2A2`, your directory would look like:
```
./datasets/HTAN_WUSTL/scRNA_vali/raw_spatial/raw/HT232P1H2A2/
├── filtered_feature_bc_matrix.h5
├── filtered_feature_bc_matrix/
│   ├── barcodes.tsv.gz
│   ├── features.tsv.gz
│   └── matrix.mtx.gz
└── spatial/
    ├── tissue_positions_list.csv
    ├── scalefactors_json.json
    └── tissue_lowres_image.png
```

## Core analysis flow
1. **Load data** with `spatialTool().dataprocess(...)`
2. **Fuzzy smoothing** with `FuzzyAverage().fuzzy_average(...)`
3. **MI scan** with `FastMICalculateNumba().mi_scan(...)`
4. **Attractor search** with `FastMICalculateNumba().find_attractor(...)`

## Notes
- Keep preprocessing and fuzzy-neighbor settings consistent across datasets for comparable MI rankings.
- Small parameter changes (e.g., `delaunay`, QC filters, seed gene) can change attractor results.
