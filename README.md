# SHARP

SHARP is an unsupervised spatial transcriptomics graph-learning method for recovering spatial domains and tissue boundaries in low-homophily spatial graphs. The public release contains the final model implementation and reproducibility commands for BRCA, E18.5 Mouse Brain, Stereo_AB, and Stereo_DPI.

## Environment

Use a Linux environment with an NVIDIA GPU and a CUDA-compatible PyTorch/DGL installation. Python 3.9 was used for the original experiments.

```bash
conda create -n sharp python=3.9 -y
conda activate sharp
pip install -r requirements.txt
conda install -c conda-forge r-mclust rpy2 -y
```

Install PyTorch, DGL, and PyTorch Geometric using the wheels matching your CUDA version when the generic `pip` packages are not compatible with your system.

## Data layout

Raw datasets are not redistributed in this repository. Place the four prepared inputs below before running the commands.

```text
SHARP-code/
├── data/
│   ├── BRCA/Breast_Caner/
│   │   ├── filtered_feature_bc_matrix.h5
│   │   └── spatial/
│   │       ├── tissue_positions_list.csv
│   │       └── brca_ground_truth_annotation.csv
│   ├── E18.5_mouse_brain/
│   │   └── E18_adata_rna.h5ad
│   └── Stereo/
│       ├── Stereo_AB.h5ad
│       └── Stereo_DPI.h5ad
└── results/
```

The supplied `*.h5ad` files must include spatial coordinates and benchmark annotations: `Combined_Clusters` for E18.5 Mouse Brain, `Annotation` for Stereo_AB, and either `ground_truth` or `Annotation` for Stereo_DPI. BRCA annotations are read from `brca_ground_truth_annotation.csv`.

## Reproduction commands

```bash
# BRCA
python main.py -dataset BRCA -k 10 -refine_k 5 -margin_hom 0.6 -margin_het 0.4 -alpha 0.1 -temp_disc 0.7 -output_dir results/BRCA

# E18.5 Mouse Brain
python main.py -dataset E18_mouse_brain -k 10 -refine_k 5 -margin_hom 0.3 -margin_het 0.7 -output_dir results/E18_mouse_brain

# Stereo_AB
python main.py -dataset Stereo_AB -k 12 -refine_k 15 -margin_hom 0.6 -margin_het 0.4 -alpha 0.2 -temp_disc 0.7 -output_dir results/Stereo_AB

# Stereo_DPI
python main.py -dataset Stereo_DPI -k 20 -refine_k 15 -margin_hom 0.5 -margin_het 0.5 -alpha 0.15 -temp_disc 1.0 -lr_gcl 0.0002 -lr_disc 0.0002 -output_dir results/Stereo_DPI
```

Each command writes two files:

```text
<dataset>_best_spatial.png
<dataset>_best_umap.png
```

## Notes

- The clustering step uses the R package `mclust` with the number of components set to the benchmark annotation class count, solely for standardized ARI evaluation.
- The graph uses spatial k-nearest neighbours (`-k`); `-refine_k` controls majority-vote spatial label refinement.
- Structural encodings are cached in `data/se/` after the first run and are reused in later runs.
