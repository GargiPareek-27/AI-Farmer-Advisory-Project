# Soybean Disease Recognition from Leaf Photos and UAV Frames

A leakage-aware machine-learning pipeline for the **MH-SoyaHealthVision** dataset (Indian UAV and leaf images for crop health assessment, Mendeley Data).

It cleans and splits the data, builds hand-crafted and frozen-CNN features, trains classical models and a fine-tuned CNN, and evaluates once on a held-out test split.

| Set | Task | Images after cleaning | Classes |
|---|---|---|---|
| Leaf | Close-up leaf photos | 2,756 (of 2,782) | 6: Healthy, Rust, Mosaic, Septoria brown spot, Frog-eye leaf spot, Caterpillar/Semilooper pest attack |
| UAV | Top-down drone frames (3840×2160) | 2,831 (of 2,842) | 4: Healthy, Rust, Mosaic, Semilooper pest attack |

> **Scope of this README.** It documents the code that is committed in this repository: a **single stratified, cluster-aware train / val / test split** with model selection on val and one final test evaluation.
>
> The results in `soya_run.ipynb` (5-fold leaf CV, UAV grouped CV, audit, ensembles) were produced by an **extended version of the pipeline that is not in this repo**. See [What is not in this repo](#what-is-not-in-this-repo) before trying to run the notebook.

---

## Pipeline

```
raw images
   │ 01_build_manifest.py   validate, de-duplicate, cluster near-duplicates, split
   ▼
manifest.csv ── classes.json ── dropped.csv
   │ 02_cache_images.py     decode + resize once → uint8 .npy per split
   │ 03_eda.py              tables and figures (content stats on TRAIN only)
   │ 04_extract_features.py hand-crafted (90) + frozen-CNN embeddings
   │ 05_train_classical.py  LogReg / SVC / RF / HistGB, pick best on VAL
   │ 06_finetune_cnn.py     transfer-learning fine-tune, best epoch on VAL
   ▼
07_evaluate_test.py         one-shot TEST evaluation (full metric suite)
08_predict.py               inference on new images / folders / UAV tiles
```

Every script takes `--dataset {leaf,uav}`, `--data-root` (or env `SOYA_ROOT`), `--artifacts`, `--img-size` (default 224), `--seed` (default 42) and `--n-jobs`.

---

## Repository layout

```
soya_pipeline/
  common.py                 CLI arguments, artifact paths, seeding, manifest/cache/feature loaders
  features.py               90 hand-crafted colour / vegetation-index / texture features
  models.py                 torchvision backbones, embeddings, batched inference, checkpoints
  01_build_manifest.py      validate, de-duplicate, cluster, split
  02_cache_images.py        decode + resize every image once
  03_eda.py                 EDA tables and figures
  04_extract_features.py    hand-crafted + frozen-CNN features for every split
  05_train_classical.py     classical models on features
  06_finetune_cnn.py        CNN fine-tuning
  07_evaluate_test.py       metrics, confusion matrices, bootstrap CI, test log
  08_predict.py             inference (file, folder, tiled UAV frame)
  run_all.sh                end-to-end driver
  make_dummy_data.py        tiny synthetic dataset for smoke tests
  requirements.txt
  docs/ENGINEERING_DECISIONS.md
```

The notebook `soya_run.ipynb` sits at the repository root.

All outputs go to `artifacts/<dataset>/` (git-ignored). See [docs/ENGINEERING_DECISIONS.md](docs/ENGINEERING_DECISIONS.md) for the layout and design rationale.

---

## Quick start

### 1. Get the data

Download the dataset from Mendeley Data (DOI [10.17632/hkbgh5s3b7.1](https://doi.org/10.17632/hkbgh5s3b7.1)) and unzip so these folders sit side by side:

```
Soyabean_Leaf_Image_Dataset/
Soyabean_UAV-Based_Image_Dataset/
```

The class label is the **name of each image's parent folder**.

### 2. Install and point to the data

```bash
pip install -r requirements.txt
export SOYA_ROOT=/path/to/folder/containing/both/datasets
```

### 3. Run the leaf pipeline

```bash
python 01_build_manifest.py   --dataset leaf --img-size 320
python 02_cache_images.py     --dataset leaf --img-size 320
python 03_eda.py              --dataset leaf --img-size 320
python 04_extract_features.py --dataset leaf --img-size 320
python 05_train_classical.py  --dataset leaf --img-size 320
python 06_finetune_cnn.py     --dataset leaf --img-size 320 \
    --model efficientnet_b0 --epochs 30 --patience 8
python 07_evaluate_test.py    --dataset leaf --img-size 320 --tta   # ONE look at the test set
```

`run_all.sh` chains the same steps with default settings (`DS=leaf` or `DS=uav`, 224 px).

### 4. Predict on new images

```bash
# single image or folder, top-3 classes
python 08_predict.py --dataset leaf --input leaf.jpg

# large UAV frame cut into an N×N grid; prints the share of tiles per class
python 08_predict.py --dataset uav --input frame.jpg --tile-grid 4 --out preds.csv
```

The checkpoint defaults to `artifacts/<dataset>/models/cnn_best.pt`; override with `--ckpt`. Image size and normalisation come from the checkpoint.

### 5. Smoke test without the real data

```bash
python make_dummy_data.py --out ./dummy_data
export SOYA_ROOT=./dummy_data
DS=leaf bash run_all.sh
```

The dummy set includes exact and flipped duplicates so the de-duplication guards can be seen working. Steps that need PyTorch (`04` deep features, `06`, `07 --which cnn`, `08`) run on CPU, slowly.

---

## What each script does

| Script | Key options | Output |
|---|---|---|
| `01_build_manifest.py` | `--test-size 0.20`, `--val-size 0.10`, `--hash-thr 6`, `--no-dihedral` | `manifest.csv`, `classes.json`, `dropped.csv` |
| `02_cache_images.py` | `--force` | `cache/{train,val,test}_x<size>.npy`, `cache/stats.json` (train-only mean/std) |
| `03_eda.py` | – | `eda/*.png`, `eda/eda_summary.json`, `eda/color_stats_by_class.csv` |
| `04_extract_features.py` | `--backbone`, `--batch-size`, `--no-pretrained`, `--skip-hand`, `--skip-deep` | `features/{split}_hand.npy`, `features/{split}_deep_<backbone>.npy` |
| `05_train_classical.py` | `--feature-sets hand deep both`, `--models logreg svc rf hgb`, `--cv N`, `--pca N` | `models/classical_best.joblib`, `reports/classical_results.csv` |
| `06_finetune_cnn.py` | `--model`, `--epochs 15`, `--warmup-epochs 2`, `--lr-head 2e-3`, `--lr-backbone 3e-4`, `--patience 4`, `--label-smoothing 0.1`, `--no-class-weights` | `models/cnn_best.pt`, `reports/cnn_history.csv`, `reports/cnn_curves.png` |
| `07_evaluate_test.py` | `--split {test,val}`, `--which {cnn,classical,both}`, `--tta`, `--bootstrap 500` | `reports/<tag>_{metrics.json,report.txt,confusion.png,misclassified.csv}`, `reports/test_eval_log.txt` |
| `08_predict.py` | `--input`, `--ckpt`, `--topk 3`, `--tile-grid N`, `--out` | console output, optional CSV |

Supported backbones: `mobilenet_v3_large`, `efficientnet_b0`, `resnet18`, `resnet50` (ImageNet-pretrained torchvision weights).

### Leakage safeguards in the code

- Corrupt files are dropped; exact duplicates (MD5) are dropped; identical bytes under different labels are all dropped.
- Near-duplicates are found with a perceptual hash (pHash on a 64 px thumbnail), compared across the 8 flip/rotation variants, and merged into clusters with union-find (default Hamming threshold 6 of 64 bits).
- The split is `StratifiedGroupKFold` over those clusters, decided once on file metadata only. Hard assertions stop the run if a cluster or an MD5 crosses splits.
- Scalers and PCA live inside sklearn pipelines fitted on train only; class weights come from train labels only.
- Model selection (classical and CNN) uses **val**. `05` and `06` never load the test split.
- `07` logs every test evaluation in `reports/test_eval_log.txt` and prints a warning on repeat use.
- Confidence intervals resample **clusters**, not single images.

---

## Metrics reported by `07_evaluate_test.py`

Accuracy, balanced accuracy, macro and weighted F1, macro precision/recall, MCC, ROC-AUC (one-vs-rest), log-loss, expected calibration error (15 bins), per-image latency, a cluster-bootstrap 95 % CI on macro-F1, per-class report, count and row-normalised confusion matrices, and a CSV of misclassified files sorted by confidence.

`--tta` averages logits over the original, horizontal flip and vertical flip.

---

## Results recorded in the notebook

These numbers are in the outputs saved in `soya_run.ipynb`. They come from the extended pipeline (grouped folds, ensembles), **not** from the scripts committed here, so the committed code will not reproduce them as-is.

**Leaf: 5-fold stratified CV, pooled out-of-fold, n = 2,756** (95 % CI from cluster bootstrap)

| Model | Macro-F1 | 95 % CI | Accuracy |
|---|---|---|---|
| Classical (hand + frozen MobileNetV3, best config per fold) | 0.690 | 0.671 – 0.709 | 0.724 |
| DINOv2-S/14 frozen + SVC | 0.684 | 0.665 – 0.702 | – |
| EfficientNet-B0, 320 px, flip TTA | 0.733 | 0.716 – 0.749 | 0.747 |
| B0 + B0/CutMix (ensemble of 2) | 0.742 | 0.726 – 0.760 | 0.756 |
| B0 + B0/CutMix + ConvNeXt-Tiny (ensemble of 3) | **0.761** | 0.744 – 0.778 | 0.775 |

The best model was chosen after comparing variants on the same folds, so treat 0.761 as an experimental benchmark, not an independently selected estimate. The weakest classes are Frog-eye (F1 0.564) and Septoria brown spot (0.623); Healthy is 0.998.

**UAV: grouped 3-fold CV by video, n = 2,831**

| Evaluation | Macro-F1 |
|---|---|
| CNN (MobileNetV3-Large, 224 px) | 0.876 (CI 0.737 – 0.959) |
| Classical features + HistGradientBoosting | 0.909 (CI 0.834 – 0.957) |
| Training-free 1-NN on frozen features | 0.873 (mean of folds: 0.867 / 0.921 / 0.831) |
| Mean-RGB-only probe | 0.647 – 0.718 (chance 0.25) |

A training-free 1-NN lookup matches the trained CNN, and leave-one-date-out recall is unstable (three largest date blocks 0.67 – 0.76; smaller blocks 0.00 – 1.00). **The UAV scores do not show generalisation to new fields, dates or seasons.** UAV classes are strongly tied to recording dates; only 1 of 8 parseable dates is shared by two classes.

---

## What is not in this repo

The notebook and the results above rely on features that the committed scripts do not have:

- `05_audit_split.py` (source-group overlap, nearest-train similarity, 1-NN probe, mean-RGB probe, leave-one-date-out)
- `09_cv_summary.py` and `metrics.py` (pooled out-of-fold metrics, cluster-bootstrap helper)
- `10_predict.py` with multi-checkpoint ensembling
- `--fold` / `--folds` options, fixed CV folds in the manifest, and the per-fold artifact folders
- video/source-group parsing from UAV file names (here, clusters come from pHash only)
- `--cutmix` and the `convnext_tiny` backbone, and ensembling of several checkpoints
- script names: the notebook calls `06_train_classical.py`, `07_finetune_cnn.py`, `08_evaluate.py`; here these are `05_train_classical.py`, `06_finetune_cnn.py`, `07_evaluate_test.py`

Also, the notebook downloads its code from a private Drive zip (`soya_pipeline.zip`), not from this repository.

---

## Known limitations of the committed code

- **UAV leakage is not prevented.** Frames from one video are grouped only if their pHash is close, which the project's own earlier analysis found leaves most evaluation frames sharing a video with training. Do not report a UAV score from a single split of this code.
- **One checkpoint slot.** `06` always writes `models/cnn_best.pt`, so a second training run overwrites the first. `07` and `08` read only that file; there is no ensembling.
- **Stale caches.** `02` skips existing files and `common.load_xy` only checks the row count. If you rebuild the manifest (new seed or threshold) without `02_cache_images.py --force`, rows can be silently misaligned. Features have the same problem (file names do not include the image size).
- **Aspect ratio.** All images, including 16:9 UAV frames, are resized to a square.
- **Augmentation.** `06` applies one random 90° rotation per batch, not per image.
- **Internal splits.** The HistGradientBoosting early-stopping split and the SVC's Platt-scaling folds are random, not cluster-aware (inside the training set only; held-out scores are unaffected).
- **Trusted checkpoints only.** `load_checkpoint` uses `torch.load(..., weights_only=False)`.
- **No tests**, unpinned `requirements.txt`, and GPU training is not bit-exact (`cudnn.benchmark`, mixed precision).
- This is an image-recognition pipeline, not a validated agronomic or treatment advisory system.

---

## Data and credits

**Dataset:** *MH-SoyaHealthVision: An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment* by Sayali Shinde and Dr. Vahida Attar, Mendeley Data, Version 1, DOI [10.17632/hkbgh5s3b7.1](https://doi.org/10.17632/hkbgh5s3b7.1), licensed CC BY 4.0. Refer to the original record for full terms and attribution requirements.

## Licence

The code in this repository is released under the MIT License. The dataset is not covered by it and remains under its original CC BY 4.0 licence.
