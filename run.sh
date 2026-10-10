#!/usr/bin/env bash
set -euo pipefail

python main.py -dataset BRCA -k 10 -refine_k 5 -margin_hom 0.6 -margin_het 0.4 -alpha 0.1 -temp_disc 0.7 -output_dir results/BRCA
python main.py -dataset E18_mouse_brain -k 10 -refine_k 5 -margin_hom 0.3 -margin_het 0.7 -output_dir results/E18_mouse_brain
python main.py -dataset Stereo_AB -k 12 -refine_k 15 -margin_hom 0.6 -margin_het 0.4 -alpha 0.2 -temp_disc 0.7 -output_dir results/Stereo_AB
python main.py -dataset Stereo_DPI -k 20 -refine_k 15 -margin_hom 0.5 -margin_het 0.5 -alpha 0.15 -temp_disc 1.0 -lr_gcl 0.0002 -lr_disc 0.0002 -output_dir results/Stereo_DPI
