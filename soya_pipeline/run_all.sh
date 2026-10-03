#!/usr/bin/env bash
# Usage: DS=leaf SOYA_ROOT=/path/to/folder_containing_both_datasets bash run_all.sh
set -euo pipefail
DS=${DS:-leaf}
A="--dataset $DS"
python 01_build_manifest.py $A      # validate, dedupe, leakage-safe split
python 02_cache_images.py   $A      # decode+resize once -> .npy
python 03_eda.py            $A      # EDA (train only)
python 04_extract_features.py $A    # hand-crafted + frozen-CNN embeddings
python 05_train_classical.py $A     # seconds-minutes, CPU
python 06_finetune_cnn.py   $A      # minutes on a GPU
python 07_evaluate_test.py  $A --tta   # ONE final look at the test set
