Data download sources for the four SHARP benchmarks
====================================================

Raw datasets are NOT redistributed in this repository.
Please download them from the original public sources below, then place the
files into the directory layout shown in README.md.

1) BRCA (10x Visium / Visium CytAssist human breast cancer)
   - Public demo / GEO-style Visium CytAssist breast-cancer inputs, e.g.
     https://www.10xgenomics.com/datasets
     (search: Visium CytAssist Human Breast Cancer)
   - Related GEO example with filtered_feature_bc_matrix.h5:
     https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7782699
   - Place into:
     data/BRCA/Breast_Caner/filtered_feature_bc_matrix.h5
     data/BRCA/Breast_Caner/spatial/tissue_positions_list.csv
     data/BRCA/Breast_Caner/spatial/brca_ground_truth_annotation.csv
     data/BRCA/Breast_Caner/spatial/scalefactors_json.json
   - Note: ground-truth annotation CSV should provide spot-level labels used
     for ARI evaluation in the paper.

2) Mouse E18.5 Brain (MISAR-seq RNA)
   - Original study: Jiang et al., Nature Methods, 2023
     “Simultaneous profiling of spatial gene expression and chromatin
     accessibility during mouse brain development”
     https://www.nature.com/articles/s41592-023-01884-1
   - Data portal / project page:
     https://www.biosino.org/node/project/detail/OEP003285
   - Code / processed release companion:
     https://github.com/gpenglab/MISAR-seq
     https://zenodo.org/record/7714382
   - Place the RNA AnnData as:
     data/E18.5_mouse_brain/E18_adata_rna.h5ad
   - Required annotation field: Combined_Clusters

3) Stereo_AB (ARTISTA Adult telencephalon)
   - ARTISTA download page:
     https://db.cngb.org/stomics/artista/download/
   - Download Adult.h5ad and rename/copy to:
     data/Stereo/Stereo_AB.h5ad
   - Dataset page:
     https://db.cngb.org/stomics/datasets/STDS0000056
   - Required annotation field: Annotation

4) Stereo_DPI (ARTISTA 2DPI regeneration section)
   - ARTISTA download page:
     https://db.cngb.org/stomics/artista/download/
   - Download the corresponding 2DPI section h5ad (e.g. 2DPI_1.h5ad) and
     rename/copy to:
     data/Stereo/Stereo_DPI.h5ad
   - Required annotation field: ground_truth or Annotation

Please cite the original data papers together with the SHARP paper when using
these benchmarks.
