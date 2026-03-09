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
   - `folder = "/Volumes/T7/datasets/" + dataset_name + "/"`
   - `sampleID = "HT232P1H2A2"`
3. Run all cells in order.

## Core analysis flow
1. **Load data** with `spatialTool().dataprocess(...)`
2. **Fuzzy smoothing** with `FuzzyAverage().fuzzy_average(...)`
3. **MI scan** with `FastMICalculateNumba().mi_scan(...)`
4. **Attractor search** with `FastMICalculateNumba().find_attractor(...)`

## Notes
- Keep preprocessing and fuzzy-neighbor settings consistent across datasets for comparable MI rankings.
- Small parameter changes (e.g., `delaunay`, QC filters, seed gene) can change attractor results.
