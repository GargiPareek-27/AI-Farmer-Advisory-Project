# Engineering Documentation: what each file does and why

This document explains every script, the artifacts it produces, and the engineering decisions behind it.
The recurring theme: **this dataset makes it very easy to report a score that means nothing**, so the
pipeline is built to prevent that and to measure how far the numbers can be trusted.

---

## 1. Dataset facts that drove the design

| Fact (measured) | Consequence |
|---|---|
| Leaf: 2,782 files, 26 dropped (duplicates / corrupt / conflicting labels, see `dropped.csv`) -> 2,756 | Cleaning is part of the pipeline, not manual |
| Leaf: no near-duplicates; nearest-train similarity median 0.87; 1-NN macro-F1 0.59 | A random stratified split is safe; CV gives a tight estimate |
| Leaf: 5x class imbalance (845 rust vs 168 Frog-eye) | Macro-F1, class-weighted loss |
| UAV: 2,842 files, 11 dropped -> 2,831 frames from only **9 / 9 / 13** videos for rust / mosaic / semilooper, 49 blocks for healthy | Frames within a video are near-identical: group by video |
| UAV: first random split gave nearest-train similarity 0.98 and 1-NN F1 0.997 | The first 0.99 results were leakage |
| UAV: only 1 of 8 parseable capture dates is shared by two classes | Class is partly confounded with recording session |
| UAV healthy: numbered stills (`image_012.jpg`), no timestamp | Different source than the diseased classes |

---

## 2. Data flow

```
raw images --01--> manifest.csv (clean, grouped, split, folds)
                       |
                       +--02--> cache/all_x<size>.npy  (one uint8 array, rows = manifest order)
                                   |
                                   +--04--> features/all_hand.npy, all_deep_<backbone>.npy
                                   |            |
                                   |            +--05 audit, 06 classical models
                                   +--07 fine-tune CNN (reads train/val rows only)
                                                |
                                   08 evaluate (val or test) --> *_pred.csv --> 09 CV summary
                                                |
                                   10 predict on new images
```

Core idea: **the manifest is the single source of truth.** It stores `group`, `cluster`, `split` and `fold` once.
Every later script only *selects rows* from the cache and feature arrays; nothing ever re-splits.

### Artifact layout

```
artifacts/<dataset>/
  manifest.csv   classes.json   dropped.csv
  cache/all_x<size>.npy
  features/all_hand.npy  all_deep_<backbone>.npy  hand_feature_names.json
  eda/           tables and figures
  models/<main|fold0..>/   cnn_<model>_<size>[_cutmix].pt + .json sidecar, classical_best.joblib
  reports/<main|fold0..>/  *_metrics.json  *_report.txt  *_pred.csv  *_confusion.png
                           *_misclassified.csv  cnn_*_history.csv  cnn_*_curves.png
                           classical_results.csv  test_eval_log.txt
  reports/cv/    pooled cross-validation summaries
```

`manifest.csv` columns: `relpath, label, label_id, width, height, bytes, md5, group, date, cluster, split, fold`.

---

## 3. File-by-file reference

### `common.py`: shared plumbing
* **`common_args`**: one argument parser shared by all scripts (`--dataset`, `--img-size`, `--fold`, `--seed`, `--backbone`, ...) so options mean the same everywhere.
* **`Paths`**: computes every artifact path from `(dataset, size, fold)`. Models and reports are stored per split (`main`, `fold0`, ...), so runs never overwrite each other.
* **`load_manifest` / `_fold_split`**: with `--fold k`, rows of fold *k* become **test**; from the remaining rows, about 1/6 of each class's clusters become **validation** (at least one per class, seeded), the rest **train**.
* **`load_xy`, `load_features`, `load_resized`**: select rows from the one big cache; `load_resized` uses JPEG draft mode (about 5x faster decoding of the 3840x2160 UAV frames).

### `metrics.py`
* `softmax`, `ece_score` (expected calibration error, 15 bins), and **`cluster_bootstrap_f1`**: 95 % CI of macro-F1 obtained by resampling *whole clusters*, not single images (images in a cluster are not independent).

### `features.py`
* 90 hand-crafted features per image: colour statistics, vegetation-index and texture descriptors (names stored in `hand_feature_names.json`). Fast, CPU-only, interpretable baseline.

### `models.py`
* torchvision backbones with ImageNet weights: MobileNetV3-Large, EfficientNet-B0/B2, ConvNeXt-Tiny, ResNet-18/50.
* `build_model` swaps the last layer; `build_embedder` removes it to get frozen embeddings; `split_backbone_head` separates parameters for different learning rates.
* `forward_batches`: batched GPU inference with optional flip TTA and mixed precision; checkpoint save/load stores architecture, classes, image size and normalisation together.

### `01_build_manifest.py`: validate, de-duplicate, group, split
1. Scans images; unreadable files are dropped.
2. **Exact duplicates** (MD5): keep the first. **Same bytes under different labels**: drop all (label noise).
3. **Near-duplicates**: 64-bit DCT perceptual hash of a 64 px thumbnail, computed for all **8 flips/rotations** and compared with the best match; pairs within `--hash-thr` (default 6 bits) are linked.
4. **Source groups (UAV)** parsed from file names: `DJI_<timestamp>_<id>_D_<frame>` and `DJI_<id>_<frame>` -> the video; `image_<n>` -> bursts of consecutive numbers (`--burst`, default blocks of 10).
5. A **union-find** merges the pHash links and the source groups into final `cluster` ids.
6. `StratifiedGroupKFold` makes a 70/10/20 split (`split`) and K folds (`fold`; 5 for leaf, 3 for UAV), all on metadata only.
7. **Hard assertions**: no cluster, group or MD5 spans two splits or two folds. Warnings if a cluster exceeds 40 % of a class, if a class has fewer groups than folds, or if a class has fewer than 15 independent sources.
8. `--max-per-group N` can thin redundant frames. Re-running clears stale cache/feature files, because row order changes.

### `02_cache_images.py`
Decodes and resizes every image once (EXIF-rotated, bilinear, squashed to a square) into a single uint8 array in manifest order.

### `03_eda.py`
Prints the tables that decided the design (images per class and split, independent sources per class, capture dates per class, imbalance) and saves plots; the sample grid shows **one frame per source group** so one video cannot fill it.

### `04_extract_features.py`
Computes hand-crafted features and frozen-backbone embeddings for **all rows at once**. Nothing is fitted here, so computing it for all rows cannot leak; scaling and PCA happen later inside sklearn pipelines fitted on train only.

### `05_audit_split.py`: the trust check
For the active split, reports:
* **(a)** share of val/test images whose source group is in train (must be 0 %);
* **(b)** cosine similarity of each val/test image to its nearest train image, and the macro-F1 of a **1-NN lookup** (a model that only memorises);
* **(c)** macro-F1 using only the 3 mean-RGB values;
* **(d)** leave-one-date-out recall for classes recorded on 2+ dates.
Reading it: if a trivial lookup matches your trained model, the task is solved by scene recognition, not disease recognition.

### `06_train_classical.py`
Pipelines of `StandardScaler` (optional PCA) plus Logistic Regression, RBF-SVC, Random Forest or HistGradientBoosting (class-balanced), on hand / deep / both feature sets. Selected by **validation** macro-F1; test is never loaded.

### `07_finetune_cnn.py`: transfer learning, tuned for a free Colab GPU
* The whole train set sits on the GPU as uint8; **augmentation runs on the GPU** (flips, 90-degree rotations, +/-20 % brightness/contrast, random zoom-in crop 0.75-1.0). No CPU data-loader bottleneck.
* Mixed precision, channels-last memory format, AdamW.
* **Two-stage schedule:** backbone frozen for `--warmup-epochs` (BatchNorm stays in eval mode), then unfrozen.
* **LR schedule:** head ramps up over half an epoch; backbone LR ramps linearly over one epoch after unfreezing; both decay with cosine.
* Class-weighted cross-entropy, label smoothing 0.1, optional **CutMix**.
* Early stopping and best-checkpoint on **validation macro-F1** (ties broken by val loss).
* Checkpoint is named `cnn_<model>_<size>[_cutmix]` with a JSON sidecar holding its validation score.

### `08_evaluate.py`
Accuracy, balanced accuracy, macro/weighted F1, macro precision/recall, MCC, ROC-AUC (one-vs-rest), log-loss, ECE, latency per image, cluster-bootstrap CI, per-class report, confusion matrices, misclassified list, and per-image probabilities.
* `--cnn best` picks the best-val checkpoint at that image size; `--cnn all` **averages probabilities** of all of them (ensemble); `--tta` averages original, horizontal-flip and vertical-flip predictions.
* Every **test** evaluation is logged with a timestamp, and a warning is printed from the second look onward.

### `09_cv_summary.py`
Collects the per-fold prediction files and reports mean +/- sd of fold macro-F1, **pooled out-of-fold macro-F1 with cluster-bootstrap CI**, per-class precision/recall, and the pooled confusion matrix.

### `10_predict.py`
Predicts a file or folder. Several `--ckpt` files give a probability-averaged ensemble. `--tile-grid N` cuts a large UAV frame into N x N tiles and returns the share of each predicted class.

### `run_all.sh`, `make_dummy_data.py`, `requirements.txt`
`run_all.sh` chains the steps (single split, or CV with `K=`). `make_dummy_data.py` builds a small synthetic dataset with UAV-style file names so the whole pipeline can be smoke-tested in minutes.

---

## 4. Engineering decisions and their reasons

| # | Decision | Why |
|---|---|---|
| 1 | Split once, in the manifest, on metadata only | Prevents any later step from re-splitting differently or peeking at labels/pixels |
| 2 | Merge perceptual-hash clusters with parsed video ids | pHash missed near-identical frames of one video; filenames carry the true source |
| 3 | Hash all 8 flips/rotations | Duplicates are often rotated or mirrored copies |
| 4 | Drop files with identical bytes but different labels | Cannot both be right; keeping them poisons training and evaluation |
| 5 | Stratified **group** K-fold, folds fixed in the manifest | Class balance preserved; groups never leak; all scripts agree on the folds |
| 6 | Cross-validation for reporting | A single 20 % test set of 552 images has CI +/- 0.04; 5-fold pooled over 2,756 images has +/- 0.017 |
| 7 | Validation carved from the training groups per fold, test fold untouched | Model selection never sees test data |
| 8 | Macro-F1 as primary metric | 5x class imbalance; accuracy would reward ignoring small classes |
| 9 | Bootstrap CI over clusters, not images | Images in a cluster are not independent; image-level CIs are too narrow |
| 10 | One cache + one feature file, row-indexed | Any split/fold is just row selection: no per-split recomputation, no row-order bugs |
| 11 | Features for all rows, scalers inside pipelines | Unfitted transforms cannot leak; fitted ones are refit per fold on train only |
| 12 | GPU-resident data and GPU-side augmentation | Colab provides about 2 CPU cores; CPU data loading was the bottleneck |
| 13 | No hue jitter in augmentation | Colour is a disease cue (yellowing in rust/mosaic) |
| 14 | Frozen warm-up and separate backbone/head learning rates | Protects pretrained features while the new head is random |
| 15 | New LR schedule with backbone ramp | The earlier one-cycle schedule peaked exactly when the backbone unfroze and UAV validation F1 collapsed at that epoch |
| 16 | Class-weighted loss and label smoothing | Imbalance and noisy visually-similar classes |
| 17 | 320 px for leaf, 224 px for UAV | Leaf at 320 improved over 224; UAV frames are huge and fold runs must stay cheap (limitation: lesions may be lost) |
| 18 | Early stopping on val macro-F1, but report CV | Validation sets are small (about 276 leaf images; 1-2 videos per UAV class) and noisy |
| 19 | Flip TTA | Cheap, small consistent gain; images have no canonical orientation |
| 20 | Ensemble of different architectures | Different models make different errors; this gave the largest gain (+3 points) |
| 21 | Never score all fold models together on the dataset | Every image was in the training set of most fold models: inflated score |
| 22 | Audit before training (step 5) | A 1-NN lookup and date-holdout tell you what a score actually proves |
| 23 | No single-split headline for UAV | With 9-13 videos per class, a single split can leave a class empty in val or test (it did) |
| 24 | Test-evaluation log | Makes repeated test looks visible and discourages tuning on test |
| 25 | Checkpoint name includes `_cutmix`, JSON sidecar | A variant once overwrote the baseline checkpoint; sidecar lets step 8 pick the best-val model without loading weights |
| 26 | Own pHash implementation (OpenCV DCT) | Removes the `imagehash` dependency |
| 27 | Hard assertions and warnings in step 1 | Leakage should stop the pipeline, not appear later as a good score |
| 28 | Copy `artifacts/` to Drive **without** `cache/` | The cache (about 1.3 GB) is rebuilt in minutes; models, reports and features are what a Colab reset destroys |

---

## 5. Experiments that were tried (so nothing is hidden)

| Experiment | Outcome |
|---|---|
| Random frame split on UAV (v1) | 0.99 macro-F1: leakage; discarded |
| pHash-only grouping (UAV) | Still left about 90 % of val/test frames sharing a video with train |
| Video-grouped 3-fold CV (UAV) | CNN 0.876, classical 0.909; 1-NN lookup 0.873: no evidence of learning beyond similarity |
| Leave-one-date-out (UAV) | Recall 0.00-1.00 by block; about 0.7 for large blocks |
| EfficientNet-B0 at 320 px (leaf) | 0.733 pooled 5-fold |
| + CutMix, ConvNeXt-Tiny, ensembling (leaf) | 0.742 (2 models), 0.761 (3 models) |
| DINOv2-S/14 frozen features + SVC (leaf, notebook-only experiment) | 0.684 alone; 0.749 when averaged with the baseline CNN; superseded by the 3-CNN ensemble |

Because several variants were compared on the same folds, treat the best number as slightly optimistic.

---

## 6. Known limitations and how to extend

* **More UAV data is the only real fix for the UAV confound**: several videos per class, recorded on the *same days and fields*.
* **UAV resolution:** tile the frame (e.g. 3 x 3 crops) and train at tile level, keeping all tiles of a video in one fold. Not implemented as a training option; `10_predict.py --tile-grid` does tiling at inference only.
* **Tune on validation only.** If you try more variants, compare them on validation or in a fresh nested CV, not on the pooled test folds.
* **Deployment cost:** the full ensemble is 15 checkpoints (5 folds x 3 models). Pick one fold's three models, or the best-validation fold, for a lighter service.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `cache / manifest mismatch` | Re-run `02_cache_images.py --force` after changing the manifest |
| `SOYA_ROOT` / "No images found" | Point `--data-root` or `SOYA_ROOT` at the folder that **contains** both dataset folders; unzip the class zips |
| CUDA out of memory (ConvNeXt, 384 px) | Add `--batch-size 32` |
| Class count warning in step 1 | Check that sub-folders are the class folders and nothing extra was unzipped |
| `09_cv_summary.py` finds 0 or several files | Pass a longer or more exact `--model-tag` (e.g. `cnn_ensemble3`) |
| A fold's validation score is far from its test score (UAV) | Expected: validation holds 1-2 videos per class |
