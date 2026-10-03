# Engineering Decisions: What Each File Does and Why

This document describes the code **as committed** in this repository: a single stratified, cluster-aware train / val / test split, model selection on val, and one final test evaluation.

Features seen in `soya_run.ipynb` (grouped CV folds, split audit, CutMix, ConvNeXt, ensembles) belong to an extended version that is not committed. They are described in [section 7](#7-how-the-notebook-relates-to-this-code).

The recurring theme: this dataset makes it easy to report a score that means nothing, so the pipeline is built to prevent leakage and to make repeated test-set use visible.

---

## 1. Dataset facts that drove the design

Measured with this pipeline (counts from the notebook's saved output).

| Fact | Consequence |
|---|---|
| Leaf: 2,782 files, 26 dropped (duplicates / corrupt / conflicting labels) → 2,756 | Cleaning is part of the pipeline, not a manual step |
| Leaf: 5× class imbalance (845 Rust vs 168 Frog-eye) | Macro-F1 as the main metric; class-weighted loss |
| Leaf: no meaningful near-duplicates (every image forms its own cluster) | A stratified split is reasonably safe for leaf |
| UAV: 2,842 files, 11 dropped → 2,831 frames, from only 9 / 9 / 13 videos for Rust / Mosaic / Semilooper plus 49 healthy source blocks | Frames in a video are near-identical, so frame-level splitting leaks |
| UAV: classes are tied to recording dates (only 1 of 8 parseable dates is shared by two classes) | Class is partly confounded with the recording session |
| UAV frames are 3840×2160 | Draft-mode JPEG decoding and a one-time resize cache are needed |

The committed code reacts to the first four rows with de-duplication, pHash clusters and class weights. It has **no** video/source grouping for UAV (see section 6).

---

## 2. Data flow

```
raw images ──01──> manifest.csv, classes.json, dropped.csv
                       │
                       ├──02──> cache/{train,val,test}_x<size>.npy, cache/stats.json
                       │            │
                       │            ├──03 EDA  (train content stats only)
                       │            ├──04──> features/{split}_hand.npy
                       │            │        features/{split}_deep_<backbone>.npy
                       │            │              │
                       │            │              └──05 classical models ──> classical_best.joblib
                       │            │
                       │            └──06 fine-tune CNN ──────────────────> cnn_best.pt
                       │
                       └──07 evaluate (test) <── cnn_best.pt, classical_best.joblib
                                │
                                └──> reports/*, test_eval_log.txt
                                
new images ──08 predict <── cnn_best.pt
```

**Core idea:** the manifest is the single source of truth. It stores `cluster` and `split` once, and every later script selects rows by `split`. No later step creates its own split.

### Artifact layout

```
artifacts/<dataset>/
  manifest.csv          relpath, label, label_id, width, height, bytes, md5, cluster, split
  classes.json          sorted class names (index = label_id)
  dropped.csv           every dropped file with the reason

  cache/
    train_x<size>.npy   uint8 (N, size, size, 3), rows in manifest order for that split
    val_x<size>.npy
    test_x<size>.npy
    stats.json          train-only per-channel mean/std

  features/
    {train,val,test}_hand.npy
    {train,val,test}_deep_<backbone>.npy
    hand_feature_names.json

  models/
    classical_best.joblib
    cnn_best.pt

  reports/
    classical_results.csv
    cnn_history.csv, cnn_curves.png
    <tag>_metrics.json, <tag>_report.txt, <tag>_confusion.png, <tag>_misclassified.csv
    test_eval_log.txt

  eda/
    class_distribution.png, image_geometry.png, cluster_sizes.png,
    color_stats.png, color_stats_by_class.csv, samples.png, mean_images.png,
    eda_summary.json
```

The artifact folder depends on the dataset but **not** on the image size, so runs at different sizes share `features/`, `models/` and `reports/`.

---

## 3. File-by-file reference

### `common.py`: shared plumbing

- `common_args` adds `--dataset`, `--data-root` (env `SOYA_ROOT`), `--artifacts`, `--img-size`, `--seed`, `--n-jobs`; each script adds its own options.
- `Paths` computes every artifact path and creates the folders.
- `load_manifest`, `split_df`, `load_xy` return cached images, labels and the matching manifest rows. `load_xy` asserts that the cache length equals the split length.
- `load_resized` uses JPEG draft mode (decode at reduced size), EXIF transpose, then bilinear resize to a square.
- `load_features` concatenates hand and/or deep features for a split.
- `set_seed` seeds `random` and NumPy; it seeds torch only if the calling script has already imported it.

### `features.py`: 90 hand-crafted features

| Group | Count |
|---|---|
| RGB, HSV, LAB channel mean and std | 18 |
| HSV histograms (12 hue, 8 sat, 8 value bins) | 28 |
| Vegetation indices ExG, ExR, VARI, GLI (mean, std, p10, p90 each) | 16 |
| Colour fractions (green, yellow, brown, dark) | 4 |
| GLCM (6 properties × 2 distances, averaged over 4 angles, 32 grey levels) | 12 |
| Uniform LBP histogram (P=8, R=1) | 10 |
| Laplacian variance (log) and Canny edge density | 2 |

NaN and infinite values are replaced by 0. The set is CPU-friendly and interpretable.

### `models.py`: PyTorch helpers

- Backbones: `mobilenet_v3_large`, `efficientnet_b0`, `resnet18`, `resnet50` (torchvision, ImageNet weights).
- `build_model` swaps the last classifier layer; `build_embedder` replaces it with `Identity` for frozen embeddings.
- `split_backbone_head` returns backbone parameters, head parameters and backbone modules (used for separate learning rates and frozen warm-up).
- `forward_batches` runs batched inference on uint8 arrays with optional AMP and optional flip TTA (mean of logits for original, horizontal flip, vertical flip).
- `save_checkpoint` / `load_checkpoint` store the architecture, class list, image size and normalisation with the weights.

### `01_build_manifest.py`: validate, de-duplicate, cluster, split

1. Scan the dataset folder and compute, in parallel, size, MD5 and pHash for each file. Unreadable files are dropped.
2. Drop **all** copies of identical bytes that appear under different labels.
3. Drop remaining exact duplicates (MD5), keeping the first.
4. Compute a pHash on a 64 px thumbnail for each of the 8 flip/rotation variants (`--no-dihedral` hashes only the original).
5. Union-find cluster images whose best pairwise Hamming distance is ≤ `--hash-thr` (default 6 of 64 bits).
6. Split with `StratifiedGroupKFold` over clusters: first a ~20 % test fold, then ~10 % of the total as val from the rest.
7. Assert that no cluster and no MD5 spans several splits; warn if one cluster holds more than 5 % of the data.
8. Write `manifest.csv`, `classes.json`, `dropped.csv`.

The split is decided here, on metadata only, before any scaling, augmentation, feature selection or class-weight computation.

### `02_cache_images.py`

Decodes and resizes every image once per split into a uint8 array. The train-only channel mean/std are stored in `stats.json` (used only when training without ImageNet weights). Existing files are skipped unless `--force`.

### `03_eda.py`

Class balance per split, image geometry and file size, near-duplicate cluster sizes, colour/brightness/ExG statistics, sample grids and mean images. Split sizes use all splits; every **content** statistic uses train only.

### `04_extract_features.py`

Computes hand-crafted features and frozen-CNN embeddings for every split. Nothing is fitted here (no scaler, PCA or selection), so there is no leakage; scaling happens later inside sklearn pipelines fitted on train.

### `05_train_classical.py`

Trains each (feature set × model) combination as `StandardScaler [+ PCA] + classifier`, fitted on train only.

| Model | Settings |
|---|---|
| Logistic regression | C=1, balanced class weights, max_iter 3000 |
| RBF SVC | C=10, balanced, probability=True |
| Random forest | 300 trees, balanced_subsample |
| HistGradientBoosting | max_iter 300, early stopping, balanced |

The best combination by **val macro-F1** is saved. `--cv N` adds a cluster-grouped CV inside train for information. The test split is never loaded.

### `06_finetune_cnn.py`: transfer learning

- Whole train set held on the GPU as uint8; augmentation runs on the GPU (flips, 90° rotation, ±20 % brightness/contrast, random zoom-crop 0.75–1.0 with reflection padding). No hue jitter, because colour is a disease cue.
- AMP, `channels_last`, AdamW (weight decay 1e-4), OneCycle schedule.
- Warm-up: the first `--warmup-epochs` epochs train the head only, with backbone BatchNorm kept in eval mode. Backbone and head use separate learning rates.
- Class-weighted cross-entropy with label smoothing 0.1.
- Per epoch: val loss, accuracy, macro-F1. The best epoch is chosen by val macro-F1 (ties broken by val loss). Early stopping counts only after warm-up.
- The test split is never loaded.

### `07_evaluate_test.py`: one-shot final evaluation

- Loads `cnn_best.pt` and/or `classical_best.joblib` and evaluates on `--split test` (or `val` for debugging).
- Reports accuracy, balanced accuracy, macro/weighted F1, macro precision/recall, MCC, ROC-AUC (OvR), log-loss, ECE, latency, per-class report, confusion matrices and misclassified files.
- Macro-F1 has a 95 % CI from a cluster bootstrap (`--bootstrap`, default 500).
- Every test evaluation is appended to `reports/test_eval_log.txt`; later runs print a warning that repeated test use leaks information. The log has one line per model, so evaluating CNN and classical together counts as two entries.

### `08_predict.py`

Predicts one image or a folder with the checkpoint's own image size and normalisation. `--tile-grid N` cuts each image into N×N tiles, classifies each, and reports the share of tiles per class (intended for large UAV frames).

### `run_all.sh`, `make_dummy_data.py`, `requirements.txt`

- `run_all.sh` runs steps 01–07 with default settings (224 px; `DS` selects the dataset).
- `make_dummy_data.py` writes a small synthetic dataset in the real folder layout, including exact and flipped duplicates, for smoke tests.
- `requirements.txt` lists lower-bounded (not pinned) dependencies.

---

## 4. Engineering decisions and their reasons

| # | Decision | Why |
|---|---|---|
| 1 | Split once, in the manifest, on metadata only | No later step can silently create a different split |
| 2 | Drop all copies of identical bytes under different labels | Such samples cannot consistently represent two classes |
| 3 | Hash flipped/rotated versions | Augmented copies appear mirrored or rotated |
| 4 | Cluster near-duplicates and split by cluster | Near-copies must not straddle train and test |
| 5 | Stratified group split | Keeps class balance and keeps clusters together |
| 6 | Hard leakage assertions | Leakage should stop the run, not appear later as a high score |
| 7 | One cache per split in manifest order | Decode once; row order is tied to the manifest |
| 8 | Draft-mode JPEG decoding | Big UAV frames decode about 5× faster |
| 9 | Frozen embeddings with no fitted transforms | Features cannot leak; scaling/PCA are fitted on train only |
| 10 | Model selection on val, test untouched until `07` | Prevents direct test-set tuning |
| 11 | Macro-F1 as the primary metric | Leaf has about 5× class imbalance |
| 12 | Class-weighted loss and label smoothing | Handles imbalance and visually similar classes |
| 13 | Bootstrap over clusters | Images in a cluster are not independent |
| 14 | GPU-resident data and GPU augmentation | Removes the CPU data-loading bottleneck on Colab |
| 15 | No hue jitter | Colour carries disease information |
| 16 | Frozen warm-up, separate backbone/head learning rates | Protects pretrained features early in training |
| 17 | Early stopping on val macro-F1 | Accuracy hides minority-class failures |
| 18 | Log every test evaluation | Makes repeated test use visible |
| 19 | Checkpoint stores arch, classes, size, mean/std | Inference needs no extra configuration |
| 20 | Flip TTA | Cheap; leaf and UAV images have no canonical orientation |

---

## 5. Experiments recorded in the notebook

These come from the extended pipeline (see section 7), not from the committed scripts.

| Experiment | Outcome |
|---|---|
| UAV, first random frame split (original version) | ≈0.99 macro-F1; leakage, discarded |
| UAV, pHash-only grouping | Still left most evaluation frames sharing a video with training |
| UAV, video-grouped 3-fold CV | CNN 0.876, classical 0.909, training-free 1-NN 0.873 |
| UAV, mean-RGB-only probe | 0.647 – 0.718 macro-F1 per fold (chance 0.25) |
| UAV, leave-one-date-out recall | Three largest date blocks 0.67 – 0.76; smaller blocks 0.00 – 1.00 |
| Leaf, classical (5-fold) | 0.690 pooled macro-F1 |
| Leaf, DINOv2-S/14 frozen + SVC | 0.684 |
| Leaf, EfficientNet-B0 at 320 px | 0.733 |
| Leaf, 50/50 blend of B0 and DINOv2 probabilities | 0.749 |
| Leaf, B0 + CutMix B0 ensemble | 0.742 |
| Leaf, B0 + CutMix B0 + ConvNeXt-Tiny ensemble | 0.761 (accuracy 0.775) |

Several variants were compared on the same folds, so the best leaf number is an experimental benchmark, not an independently selected estimate.

---

## 6. Known limitations of the committed code

- **UAV leakage is not prevented.** Clusters come from pHash only; there is no parsing of video/source IDs from file names. A single split of the UAV data with this code would likely reproduce the inflated scores the project discarded.
- **One checkpoint slot.** `06` writes `models/cnn_best.pt` every time, so the next run overwrites it. `07` and `08` read only that file; there is no ensembling.
- **Stale caches and features.** `02` skips existing files and `load_xy` only checks the row count. Rebuilding the manifest without `02 --force` can silently misalign rows when split sizes happen to match. Feature file names do not include the image size.
- **Square resize.** 16:9 UAV frames are squashed to a square; at 224 px small lesions may vanish. `08 --tile-grid` helps at inference only.
- **Per-batch rotation.** The 90° rotation in `gpu_augment` is drawn once per batch, not per image.
- **Non-grouped internal splits.** HistGradientBoosting's early-stopping holdout and the SVC's Platt-scaling folds are random. They sit inside the training set, so held-out scores are unaffected.
- **Single split, small val.** One val/test split gives a noisy estimate, especially for UAV where val can hold only a few dozen images.
- **`weights_only=False`** in `load_checkpoint`: load only checkpoints you trust.
- No tests, unpinned requirements, and GPU training is not bit-exact.
- Recognition only: there is no validated treatment or pesticide advice.

---

## 7. How the notebook relates to this code

`soya_run.ipynb` runs an extended version of this pipeline: fixed CV folds stored in the manifest (with video/source and date parsing for UAV), a split audit, a pooled cross-validation summary with cluster-bootstrap confidence intervals, CutMix and ConvNeXt-Tiny training, and multi-checkpoint ensembling. It loads that code from `soya_pipeline.zip` on Google Drive. The DINOv2 embeddings, the SVC grid search and the probability blending are written inline in the notebook.

The committed scripts are the single-split baseline of the same design, with slightly different numbering (`05_train_classical.py`, `06_finetune_cnn.py`, `07_evaluate_test.py` instead of `06_`, `07_`, `08_`). Committing the extended scripts would make the notebook runnable from this repository.

---

## Final interpretation

The committed code is a clean, leakage-aware **single-split baseline** with strong evaluation hygiene. The notebook's results show a leaf CV score of 0.761 macro-F1 and UAV scores of 0.88 – 0.91 that do not demonstrate generalisation, because a training-free nearest-neighbour lookup matches the CNN and recording date is confounded with class.
