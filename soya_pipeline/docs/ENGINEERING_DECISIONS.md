# Engineering documentation: what each file does and why

The recurring theme: **it is very easy to report a score that means nothing on this kind of data**, so the pipeline is built to prevent that and to
measure how far its numbers can be trusted.

**Scope.** This repository covers the **leaf-photo** half of MH-SoyaHealthVision only (6 classes). The UAV half was explored earlier and is out of scope:
its frames are consecutive video frames, the first random split reported a meaningless 0.99 macro-F1 (leakage), and even with video-grouped folds the
classes were recorded on different dates, so those scores could not be shown to generalise. That experience is why the leakage guards below exist.
The UAV code is not included in this version of the project.

**Numbers.** Where this document quotes results, they are the pooled out-of-fold values of the recorded Colab run of `leaf_ensemble3` (see `RESULTS.md`): 3-CNN ensemble macro-F1 0.7602 (95 % CI 0.7438 - 0.7770), accuracy 0.7747.

---

## 1. Dataset facts that drove the design

| Fact (measured) | Consequence |
|---|---|
| Leaf: 2,782 files, 26 dropped (duplicates / corrupt / conflicting labels, see `dropped.csv`) -> 2,756 | Cleaning is part of the pipeline, not manual |
| Leaf: no near-duplicates (2,756 images form 2,756 clusters at the default 6-bit threshold); nearest-train similarity median 0.871; 1-NN macro-F1 0.588 (validation) / 0.594 (test) | A random stratified split is safe; CV gives a tight estimate |
| Leaf: 5x class imbalance (845 rust vs 168 Frog-eye) | Macro-F1, class-weighted loss |
| Leaf: no source / plant / session identifiers in the file names | Every image is its own group; independence between images cannot be verified (see limitations); `--group-regex` is there if you can recover ids |

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

reproduce.py  = a recipe (configs/recipes.json) expanded into exactly these commands, plus a reproduction report
```

The **manifest is the single source of truth.** It stores `group`, `cluster`, `split` and `fold` once. Every later script only selects rows from the cache
and feature arrays; nothing ever re-splits. Cache and feature files carry a **fingerprint** (md5 of the manifest they were built from) and are refused if the manifest has changed.

### Artifact layout

```
artifacts/<dataset>/
  manifest.csv  classes.json  dropped.csv  mixed_label_clusters.csv (only if any)
  cache/all_x<size>.npy (+ .meta.json)
  features/all_hand.npy  all_deep_<backbone>.npy (+ .meta.json)  hand_feature_names.json
  eda/
  models/<main|fold0..>/   cnn_<model>_<size>[_cutmix].pt + .json sidecar, classical_best[_<backbone>].joblib
  reports/<main|fold0..>/  *_metrics.json  *_report.txt  *_pred.csv  *_confusion.png  *_misclassified.csv
                           cnn_*_history.csv  cnn_*_curves.png  classical_results.csv  test_eval_log.txt
  reports/cv/              <tag>_cv_summary.json  <tag>_cv_confusion.csv  reproduction_<recipe>.json
```

`manifest.csv` columns: `relpath, label, label_id, width, height, bytes, md5, group, cluster, split, fold`.

---

## 3. File-by-file reference

### `common.py`: shared plumbing
* **`common_args`**: one argument parser shared by all scripts (`--dataset`, `--img-size`, `--fold`, `--seed`, `--backbone`, `--artifacts`, ...).
* **`Paths`**: every artifact path from `(dataset, size, fold)`. Models and reports are stored per split (`main`, `fold0`, ...), so runs never overwrite each other.
* **`load_manifest` / `_fold_split`**: with `--fold k`, rows of fold *k* become **test**; about 1/6 of each class's remaining clusters (at least one) become **validation**
  (seeded); the rest **train**. A guard stops the run if any class would be missing from train, val or test.
* **`check_fold_coverage`**: used by step 1 to fail early when a class has fewer than 2 independent clusters outside any test fold.
* **`write_meta` / `meta_state` / `check_meta`**: manifest fingerprints for cache and feature files (`ok` / `missing` / `stale`).
* **`load_xy`, `load_features`, `load_resized`**: select rows from the one big cache; `load_resized` uses JPEG draft mode to speed up decoding of large photos.
* **`classical_bundle`, `classical_tag`**: naming of classical models per embedding backbone.

### `metrics.py`
`softmax`, `ece_score` (expected calibration error, 15 bins) and **`cluster_bootstrap_f1`**: 95 % CI of macro-F1 obtained by resampling *whole clusters*, not single images.

### `schedules.py`
`lr_lambdas`: the two-stage learning-rate schedule (frozen warm-up, one-epoch ramp of the backbone LR, cosine decay). Pure Python so it is unit-tested without PyTorch.

### `features.py`
90 hand-crafted features per image (colour statistics, vegetation indices, texture descriptors); names in `hand_feature_names.json`. Fast, CPU-only, interpretable baseline.

### `models.py`
torchvision backbones with ImageNet weights (MobileNetV3-Large, EfficientNet-B0/B2, ConvNeXt-Tiny, ResNet-18/50) plus DINOv2 as a frozen embedder.
`build_model` swaps the last layer; `build_embedder` removes it; `split_backbone_head` separates parameters for different learning rates; `forward_batches` does batched GPU inference
with optional flip TTA and mixed precision; checkpoints store architecture, classes, image size and normalisation together.

### `01_build_manifest.py`: validate, de-duplicate, group, split
1. Unreadable files are dropped.
2. **Exact duplicates** (MD5): keep the first. **Same bytes under different labels**: drop all (label noise).
3. **Near-duplicates**: 64-bit DCT perceptual hash of a 64 px thumbnail for all **8 flips/rotations**; pairs within `--hash-thr` (default 6 bits) are linked.
4. **Source groups**: every image is its own group unless `--group-regex` is given (group 1 of the regex is the source id; files with the same id are never split).
5. A **union-find** merges the hash links and the source groups into the final `cluster` ids.
6. `StratifiedGroupKFold` makes a 70/10/20 split (`split`) and K folds (`fold`; 5 by default), all on metadata only.
7. **Hard assertions**: no cluster, group or MD5 spans two splits or two folds; every class has >= 2 clusters outside each test fold. **Warnings**: a cluster holds > 40 % of a class,
   a class has fewer groups than folds, a class has fewer than 15 independent sources, clusters mix classes (`mixed_label_clusters.csv`), a class is missing from the single split.
8. Re-running clears stale cache/feature files, because row order changes.

### `02_cache_images.py` / `04_extract_features.py`
Decode and resize once; compute hand-crafted and frozen-backbone features for **all rows**. Nothing is fitted here, so computing it for all rows cannot leak.
Both write the manifest fingerprint; `02` rebuilds a cache whose fingerprint is stale or missing.

### `03_eda.py`
The tables that decided the design (images per class and split, source groups per class, imbalance) and plots; the sample grid shows **one image per source group**.

### `05_audit_split.py`: the trust check
(a) share of val/test images whose source group is in train (0 % by construction, a sanity check); (b) cosine similarity to the nearest train image and the macro-F1 of a **1-NN lookup**;
(c) macro-F1 from the 3 mean-RGB values only. If a trivial lookup matches your trained model, the task is solved by similarity, not disease recognition.

### `06_train_classical.py`
`StandardScaler` (optional PCA) plus Logistic Regression, RBF-SVC, Random Forest or HistGradientBoosting (class-balanced) on hand / deep / both features. Selected by **validation** macro-F1; test is never loaded.

### `07_finetune_cnn.py`
Whole train set on the GPU as uint8; **GPU-side augmentation** (flips, 90-degree rotation of the batch, +/-20 % brightness/contrast, random zoom-in crop 0.75-1.0; no hue jitter); mixed precision;
channels-last; AdamW. Backbone frozen for `--warmup-epochs` (BatchNorm in eval mode), class-weighted cross-entropy with label smoothing 0.1, optional **CutMix**. Early stopping and the best checkpoint
use **validation** macro-F1. Checkpoints are named `cnn_<model>_<size>[_cutmix]` with a JSON sidecar.

### `08_evaluate.py`
Accuracy, balanced accuracy, macro/weighted F1, MCC, ROC-AUC, log-loss, ECE, latency, cluster-bootstrap CI, per-class report, confusion matrices, misclassified list, per-image probabilities.
`--cnn best` picks the best-val checkpoint at that size, `--cnn all` **averages probabilities** of all of them, `--cnn <stem>` evaluates one; `--tta` averages original, horizontal and vertical flips.
Every **test** evaluation is logged with a timestamp.

### `09_cv_summary.py`
Per-fold mean +/- sd, **pooled out-of-fold macro-F1 with cluster-bootstrap CI**, per-class precision/recall and the pooled confusion matrix. `--model-tag` must equal a prediction-file stem or be its prefix up to an underscore.

### `10_predict.py`
Predicts a file or folder. Several `--ckpt` give a probability-averaged ensemble; `--out` writes a CSV.

### `reproduce.py` + `configs/recipes.json`
Expands a recipe into exactly the documented commands (`--dry-run` prints them), runs them, and writes `reproduction_<recipe>.json` with the source (zip checksum or commit), library versions, GPU and per-model CV summaries.
`--no-torch` skips every PyTorch step so the data pipeline can be smoke-tested on CPU (this is what CI does). `--resume` skips finished steps, `--sync-dir` backs results up after every step and restores them at start, and a pre-flight check stops early when the dataset folder, packages or GPU are missing. The final ensemble is evaluated with an **explicit member list**, so stray checkpoints in a fold folder cannot change it.

### `tests/`
24 tests: unit tests (metrics, schedule, source-group parsing, union-find, fold-coverage guard, wildcard expansion, checkpoint selection, resume markers) and end-to-end tests of the real scripts on synthetic data (leakage invariants, duplicate and flip handling, fold coverage,
every image tested once, stale-cache refusal, cache equals direct decoding, saved audit, recipe command list, resume + sync after a lost runtime, preflight failure).

### `make_dummy_data.py`
Synthetic leaf dataset with the same folder layout as the real one, including one exact copy and one flipped copy per class so de-duplication and clustering are exercised.

---

## 4. Engineering decisions and their reasons

| # | Decision | Why |
|---|---|---|
| 1 | Split once, in the manifest, on metadata only | Prevents any later step from re-splitting differently or peeking at labels/pixels |
| 2 | Merge perceptual-hash clusters with optional `--group-regex` source ids | pHash can miss near-identical shots of one source; file names, when they carry a source id, are the ground truth |
| 3 | Hash all 8 flips/rotations | Duplicates are often rotated or mirrored copies (covered by a test) |
| 4 | Drop files with identical bytes but different labels | Cannot both be right; keeping them poisons training and evaluation |
| 5 | Stratified **group** K-fold, folds fixed in the manifest | Class balance preserved; groups never leak; all scripts agree on the folds |
| 6 | Cross-validation for reporting | A single 20 % test set of 552 images has CI about +/-0.04; 5-fold pooled over 2,756 images about +/-0.017 (recorded run: 0.7438 - 0.7770 around 0.7602) |
| 7 | Validation carved from the training groups per fold, test fold untouched | Model selection never sees test data |
| 8 | Macro-F1 as primary metric | 5x class imbalance; accuracy would reward ignoring small classes |
| 9 | Bootstrap CI over clusters, not images | Images in a cluster are not independent |
| 10 | One cache + one feature file, row-indexed | Any split/fold is just row selection; no per-split recomputation, no row-order bugs |
| 11 | Manifest fingerprint on cache and feature files | A cache built from an old manifest would silently misalign labels and images; now refused |
| 12 | Features for all rows, scalers inside pipelines | Unfitted transforms cannot leak; fitted ones are refit per fold on train only |
| 13 | GPU-resident data and GPU-side augmentation | Colab provides about 2 CPU cores; CPU data loading was the bottleneck |
| 14 | No hue jitter in augmentation | Colour is a disease cue (yellowing in rust/mosaic) |
| 15 | Frozen warm-up and separate backbone/head learning rates | Protects pretrained features while the new head is random |
| 16 | LR schedule with backbone ramp (`schedules.py`) | The earlier one-cycle schedule peaked exactly when the backbone unfroze and validation F1 collapsed at that epoch |
| 17 | Class-weighted loss and label smoothing | Imbalance and noisy, visually similar classes |
| 18 | 320 px input | Small lesions are easier to see at higher resolution. An early single-split comparison (MobileNetV3-Large at 224 px vs EfficientNet-B0 at 320 px) favoured 320 px, but the two runs also differed in architecture, so this is a design choice, not an isolated measured effect; 224 px was not re-tested in the recorded run |
| 19 | Early stopping on val macro-F1, but report CV | Validation sets are small (about 276 leaf images) and noisy |
| 20 | Flip TTA (original, horizontal and vertical flips; outputs averaged before the softmax) | Cheap, and the images have no canonical orientation. All CNN results in the recorded run use TTA; its effect was not ablated |
| 21 | Ensemble of different architectures | Different models make different errors. In the recorded run the 3-CNN ensemble (0.7602) is +0.0262 over EfficientNet-B0 (0.7340) and +0.0091 over the best single model, ConvNeXt-Tiny (0.7511); it won on all five folds against EfficientNet-B0 and on four of five against ConvNeXt-Tiny. The intervals of the ensemble and ConvNeXt-Tiny overlap, so the gain over the best member is modest. The CutMix member alone (0.7290) did not beat the baseline; its contribution to the ensemble was not ablated |
| 22 | Never score all fold models together on the dataset | Every image was in the training set of most fold models: inflated score |
| 23 | Audit before training (step 5) | A 1-NN lookup and a colour-only probe tell you what a score actually proves |
| 24 | CV, not the single split, is the headline; step 1 warns when a class is missing from the single split | A single 20 % test set is noisy (CI about +/-0.04) and can lose a rare class |
| 25 | Fold-coverage guard (found by the end-to-end tests) | With few groups per class a fold could leave a class without training images; the classifier then returned fewer probability columns than classes and evaluation crashed. It now fails early with a clear message |
| 26 | Test-evaluation log | Makes repeated test looks visible and discourages tuning on test |
| 27 | Checkpoint name includes `_cutmix`, JSON sidecar | A variant once overwrote the baseline checkpoint; the sidecar lets step 8 pick the best-val model without loading weights |
| 28 | Own pHash implementation (OpenCV DCT) | Removes the `imagehash` dependency |
| 29 | Recipes + `reproduce.py` instead of notebook-only commands | The ensemble result first existed only as notebook cells; a recipe is versioned, reviewable and prints the exact commands |
| 30 | `--no-torch` mode and CPU tests with synthetic data | Lets CI exercise the data pipeline, leakage guarantees and CV bookkeeping on every push without a GPU |
| 31 | Reproduction report with source, commit (if any), versions and GPU | Makes "reproduced" a checkable statement; GPU training is not bit-exact so differences are recorded, not hidden |
| 32 | Training code left numerically unchanged during the engineering refactor | The reported ensemble result (0.7602) must stay attributable to the code that produced it; only `lr_lambdas` moved (same function) |
| 33 | Notebook runs from the project zip kept in Google Drive and records the zip's md5 | The exact code behind a result is identifiable without GitHub, and the zip survives Colab resets |
| 33b | Dataset unpacking skips UAV archives by their own name, not by the whole path | The archive's top folder is called "...UAV and Leaf...", so a whole-path test skipped the leaf archives too |
| 34 | Streamed, memory-mapped cache build (`02`), written to a `.partial.npy` and renamed when complete | Peak memory is one image, not two copies of an 850 MB array; a crash can never leave a truncated file under the final name |
| 35 | `07` deletes its completion marker (`cnn_<run>.json`) at start and writes it last | The marker means "training finished"; `--resume` and `08 --cnn all` rely on it, so a crashed rerun must not leave an old one behind |
| 36 | `08 --cnn` validates names and accepts `a,b,c` | A typo failed deep inside `torch.load`; an explicit list makes the recipe's ensemble independent of whatever else is in the folder |
| 37 | `10_predict.py` expands wildcards itself and searches all `models/*` folders | PowerShell does not expand `*`; fold-only runs have no `main/` checkpoints |
| 38 | Pre-flight check, `--resume`, `--sync-dir` in `reproduce.py` | Colab sessions drop during 70-90 minute runs and wipe local files; recovery must not mean starting over |
| 39 | `forward_batches` copies each memmap slice | Read-only arrays gave a PyTorch warning about undefined behaviour |
| 40 | Copy `artifacts/` to Drive **without** `cache/` | The cache (about 1.3 GB) is rebuilt in minutes; models, reports and features are what a Colab reset destroys |

---

## 5. Experiments that were tried

| Experiment | Outcome |
|---|---|
| UAV subset (earlier, now out of scope; code not included) | Random frame split gave a leaked 0.99; video-grouped CV gave 0.88-0.91 but matched a training-free lookup and was confounded with recording date |
| Classical features + SVC / LogReg / HistGB (leaf) | 0.6904 pooled 5-fold |
| EfficientNet-B0 at 320 px + TTA (leaf) | 0.7340 pooled 5-fold |
| EfficientNet-B0 + CutMix at 320 px + TTA (leaf) | 0.7290 pooled 5-fold (not better than the baseline) |
| ConvNeXt-Tiny at 320 px + TTA (leaf) | 0.7511 pooled 5-fold (best single model) |
| 3-CNN ensemble + TTA (leaf) | **0.7602** pooled 5-fold, 95 % CI 0.7438 - 0.7770 (best) |
| DINOv2-S/14 frozen features + SVC (leaf) | 0.6837 alone; 0.7514 averaged 50/50 with the EfficientNet-B0 baseline; superseded by the 3-CNN ensemble |

Because several variants were compared on the same folds, treat the best number as slightly optimistic.

---

## 6. What is and is not verified

| Claim | Evidence |
|---|---|
| Split, grouping, duplicate handling, fold bookkeeping, CV summaries, cache fingerprints, resume/sync behave as documented | 24 automated tests on synthetic data (CPU) |
| CNN training / evaluation code produces the reported numbers | One recorded full execution of `leaf_ensemble3` on Colab (Tesla T4, PyTorch 2.11.0+cu130, 55 steps, about 110.9 min, 0 steps skipped; 24 tests and the GPU smoke test passed in the same session); **the PyTorch paths are not covered by CI** |
| The project reproduces the reported numbers from a clean run | **Pending**: the recorded run used the unpacked project folder in Google Drive (no zip md5, no commit hash), so it is not an independent clean-ZIP reproduction; fill the log in `RESULTS.md` section 12 |
| `--backbone dinov2_*` in step 4 (the option inside the pipeline) | **Untested on a GPU**: the reported DINOv2 numbers came from the notebook cells in `soya_reproduce.ipynb` |
| Pinned library versions | The versions of the recorded run are listed in `RESULTS.md` section 1; **pending**: `requirements-lock.txt` is not yet part of the project |

## 7. Known limitations and how to extend

* **Unknown source overlap in leaf photos.** If several photos come from the same plant or field, near-identical conditions can sit on both sides of a split without being detected (pHash only catches near-copies). Source ids from the dataset authors would allow true grouped folds.
* **Tune on validation only.** More variants should be compared on validation or in a nested CV, not on the pooled test folds. Several variants were already compared on the same folds, so the ensemble figure may be somewhat optimistic.
* **No external validation dataset.** All results come from cross-validation on one dataset.
* **GPU training is not bit-exact;** reruns can vary by roughly +/-0.01 macro-F1.
* **Deployment cost:** the full ensemble is 15 checkpoints (5 folds x 3 models). Pick one fold's three models, or the best-validation fold, for a lighter service; add a confidence threshold and a "not sure" output first.
* **Test coverage gap:** PyTorch code paths (`models.py`, `07`, `08` with CNNs, `10`) have no automated tests; a small CPU test with a tiny model would close it.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `built from a different manifest` | Re-run `02_cache_images.py --force` and `04_extract_features.py` after changing the manifest |
| `SOYA_ROOT` / "No images found" | Point `--data-root` or `SOYA_ROOT` at the folder that **contains** `Soyabean_Leaf_Image_Dataset/`; unzip the class zips |
| `too few independent clusters outside the test fold` | Use fewer `--folds`, or relax `--hash-thr` / `--group-regex` |
| CUDA out of memory (ConvNeXt, 384 px) | Add `--batch-size 32` to the training step |
| Class count warning in step 1 | Check that sub-folders are the class folders and nothing extra was unzipped |
| `09_cv_summary.py` finds 0 or several files | Pass a more exact `--model-tag` (e.g. `cnn_ensemble3_tta`) |
