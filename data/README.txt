Data Source:

BRCA (10x Visium breast cancer)
  -> prepared Visium inputs under data/BRCA/Breast_Caner/
  -> ground-truth annotation: spatial/brca_ground_truth_annotation.csv
  -> related references in the SHARP paper (breast-cancer / SRT background):
     Yoosuf et al., Breast Cancer Research, 2020;
     Dong & Zhang, Nature Communications, 2022.

Mouse E18.5 Brain
  -> data/E18.5_mouse_brain/E18_adata_rna.h5ad
  -> annotation field: Combined_Clusters
  -> used in the SHARP paper as the Mouse E18.5 benchmark.

Stereo-seq Adult Brain (Stereo_AB)
  -> data/Stereo/Stereo_AB.h5ad
  -> annotation field: Annotation

Stereo-seq DPI (Stereo_DPI)
  -> data/Stereo/Stereo_DPI.h5ad
  -> annotation field: ground_truth or Annotation

Please cite the SHARP paper when using these prepared benchmark inputs:

Liu, J. et al. Decoding Spatial Heterophily: A Spectral Graph Contrastive Method
for Spatially Resolved Transcriptomics. IEEE BIBM, 2026.

Distribution note:
  - BRCA files are stored in this repository under data/BRCA/
  - E18_adata_rna.h5ad / Stereo_AB.h5ad / Stereo_DPI.h5ad are provided in
    GitHub Release datasets-v1 (use: bash scripts/download_data.sh)
