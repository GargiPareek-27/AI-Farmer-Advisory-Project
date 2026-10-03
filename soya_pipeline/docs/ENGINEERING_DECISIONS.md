# Engineering Documentation: What Each File Does and Why

This document explains every script, the artifacts it produces, and the engineering decisions behind it.

The recurring theme: this dataset makes it very easy to report a score that means nothing, so the pipeline is built to prevent that and to measure how far the numbers can be trusted.

---

## 1. Dataset facts that drove the design

| Fact (measured) | Consequence |
|---|---|
| Leaf: 2,782 files, 26 dropped (duplicates / corrupt / conflicting labels) → 2,756 | Cleaning is part of the pipeline, not manual |
| Leaf: no meaningful near-duplicates; nearest-train similarity median ≈ 0.87; 1-NN macro-F1 ≈ 0.59 | A random stratified split is substantially safer than the UAV case; CV gives a tighter estimate |
| Leaf: 5× class imbalance (845 Rust vs 168 Frog-eye) | Macro-F1 and class-weighted training are important |
| UAV: 2,842 files, 11 dropped → 2,831 frames from only 9 / 9 / 13 videos for Rust / Mosaic / Semilooper, plus 49 healthy source blocks | Frames within a video are highly correlated: group by video/source |
| UAV: first random split gave nearest-train similarity ≈ 0.98 and 1-NN F1 ≈ 0.997 | The first ≈0.99 result was leakage and was discarded |
| UAV: only 1 of 8 parseable capture dates is shared by two classes | Class is partly confounded with recording session/date |
| UAV healthy images use numbered stills such as `image_012.jpg` and do not contain the same timestamp structure as the diseased videos | Healthy imagery has a different acquisition/source structure |

---

## 2. Data flow

```
raw images --01--> manifest.csv (clean, grouped, split, folds)
                       |
                       +--02--> cache/all_x<size>.npy
                       |           (one uint8 array, rows = manifest order)
                       |
                       +--04--> features/all_hand.npy
                       |           all_deep_<backbone>.npy
                       |                |
                       |                +--05 audit
                       |                +--06 classical models
                       |
                       +--07 fine-tune CNN
                                  |
                                  +--08 evaluate
                                         |
                                         +-- *_pred.csv
                                               |
                                               +--09 CV summary
                                                        |
                                                        +--10 predict
```

**Core idea:** the manifest is the single source of truth. It stores `group`, `cluster`, `split` and `fold` once.

Every later script selects rows from the cache/features according to that manifest rather than creating a new independent split.

### Artifact layout

```
artifacts/<dataset>/
  manifest.csv
  classes.json
  dropped.csv

  cache/
    all_x<size>.npy

  features/
    all_hand.npy
    all_deep_<backbone>.npy
    hand_feature_names.json

  eda/
    tables and figures

  models/
    main/
    fold0/
    fold1/
    ...

  reports/
    main/
    fold0/
    fold1/
    ...

  reports/cv/
    pooled cross-validation summaries
```

`manifest.csv` contains the core metadata used downstream, including:

- `relpath`
- `label`
- `label_id`
- `width`
- `height`
- `bytes`
- `md5`
- `group`
- `date`
- `cluster`
- `split`
- `fold`

---

## 3. File-by-file reference

### `common.py`: shared plumbing

- Shared command-line argument parsing for dataset, image size, fold, seed and model configuration.
- `Paths` computes artifact paths from dataset/image-size/fold so different experiments do not overwrite each other.
- Shared cache and feature loaders keep row ordering tied to `manifest.csv`.
- Fold handling ensures that downstream scripts select the same rows defined by the manifest.

### `metrics.py`: evaluation utilities

- `softmax`
- expected calibration error
- cluster-bootstrap confidence intervals
- macro-F1 confidence intervals obtained by resampling clusters rather than individual images

Cluster-level bootstrap is used because images belonging to the same source/video/cluster are not independent observations.

### `features.py`

Approximately 90 hand-crafted features per image, including:

- colour statistics
- RGB / HSV / LAB descriptors
- vegetation indices
- colour fractions
- texture descriptors
- GLCM features
- LBP
- Laplacian variance
- edge-density measurements

The resulting representation provides a fast, CPU-friendly and interpretable baseline.

### `models.py`

The model utilities support ImageNet-pretrained torchvision architectures used in the experiments, including:

- MobileNetV3-Large
- EfficientNet-B0
- EfficientNet-B2
- ConvNeXt-Tiny
- ResNet-18
- ResNet-50

The module provides:

- classification model construction
- frozen embedding extraction
- backbone/head separation
- batched inference
- optional test-time augmentation
- mixed-precision inference
- checkpoint saving/loading

### `01_build_manifest.py`: validate, de-duplicate, group, split

1. Scans the dataset and drops unreadable/corrupt files.
2. Detects exact duplicates using MD5.
3. Drops identical bytes appearing under conflicting labels.
4. Detects perceptual near-duplicates using DCT-based perceptual hashing.
5. Applies hashing across transformed versions including flips/rotations.
6. Parses UAV source/video information from file names.
7. Groups consecutive frames from the same source/video.
8. Merges perceptual-hash links and source groups into final clusters.
9. Creates the main train/validation/test split.
10. Creates fixed cross-validation folds.
11. Uses group-aware splitting so correlated source groups do not cross boundaries.
12. Performs hard leakage assertions.

The manifest is generated before feature/model training so that no later experiment silently creates a different split.

### `02_cache_images.py`

Decodes and resizes every image once into a single uint8 array in manifest order.

The clean notebook run used:

- 320 × 320 for leaf experiments
- 224 × 224 for UAV experiments

This avoids repeatedly decoding the original images during each fold/model run.

### `03_eda.py`

Produces the dataset diagnostics that drove the experimental design:

- images per class
- images per split
- class imbalance
- independent source groups
- capture dates
- representative samples
- dataset-level tables and figures

For UAV data, source-aware inspection is particularly important because one video can otherwise dominate a random sample grid.

### `04_extract_features.py`

Computes:

- handcrafted features
- frozen CNN embeddings

The notebook produced:

```
Leaf handcrafted features: (2756, 90)
```

The frozen embeddings are generated without fitting a classifier, so later scaling/model fitting can still be restricted to training data.

### `05_audit_split.py`: the trust check

For the active split/fold, the audit reports:

- **(a)** percentage of validation/test images whose source group appears in training
- **(b)** nearest-train feature similarity and training-free 1-NN macro-F1
- **(c)** macro-F1 from a mean-RGB-only probe
- **(d)** leave-one-date-out recall where enough dates exist

The interpretation is deliberate:

> If a trivial nearest-neighbour lookup performs similarly to the trained model, the benchmark may be dominated by visual/source similarity rather than disease recognition.

For UAV, this audit exposed the original random-split problem and remained part of the final evaluation rather than being hidden after the score dropped.

### `06_train_classical.py`

Trains classical models on:

- handcrafted features
- frozen deep features
- combined feature representations

Models explored include:

- Logistic Regression
- RBF-SVC
- Random Forest
- HistGradientBoosting

Scaling/PCA, where used, are fitted within the training portion.

For the leaf CV experiment, the selected classical configuration produced approximately:

```
pooled macro-F1 = 0.690
pooled accuracy = 0.724
```

For UAV grouped CV, the classical HistGradientBoosting result reached:

```
pooled macro-F1 = 0.909
pooled accuracy = 0.902
```

The UAV number is intentionally reported together with the audit because it is strongly affected by the dataset's acquisition structure.

### `07_finetune_cnn.py`: transfer learning

The CNN experiments were designed for a Colab T4 GPU and limited compute.

The training pipeline uses:

- ImageNet-pretrained weights
- GPU-resident image data
- GPU-side augmentation
- mixed precision
- AdamW
- class-weighted loss
- label smoothing
- optional CutMix
- frozen-backbone warm-up
- backbone/head learning-rate separation
- early stopping
- validation macro-F1 for checkpoint selection

Leaf experiments used 320 px.

UAV experiments used 224 px to keep grouped 3-fold experimentation computationally practical.

The main leaf CNN progression was:

```
EfficientNet-B0 + flip TTA
        ↓
CutMix EfficientNet-B0 ensemble
        ↓
EfficientNet-B0 + CutMix + ConvNeXt-Tiny ensemble
```

### `08_evaluate.py`

Reports:

- accuracy
- balanced accuracy
- macro/weighted F1
- macro precision/recall
- MCC
- ROC-AUC where applicable
- log loss
- ECE
- latency
- confidence intervals
- per-class reports
- confusion matrices
- misclassified examples
- per-image probabilities

It also supports:

- test-time augmentation
- selecting the best validation checkpoint
- probability averaging across ensemble members

The notebook used flip TTA and probability averaging for the CNN ensemble experiments.

### `09_cv_summary.py`

Collects the per-fold prediction files and reports:

- per-fold macro-F1
- mean and standard deviation across folds
- pooled out-of-fold macro-F1
- cluster-bootstrap confidence interval
- pooled accuracy
- per-class precision/recall/F1
- pooled confusion matrix

This is the source of the final leaf headline result:

```
3-CNN ensemble

Pooled macro-F1: 0.7606
95% CI:           0.7439 - 0.7777
Accuracy:         0.7747
```

### `10_predict.py`

Provides inference on new images.

The notebook/documented workflow supports:

- single-image inference
- folder inference
- probability-averaged checkpoints
- UAV frame tiling using an `N × N` grid

Fold checkpoints should be averaged only for new images.

They should not be jointly scored on the same CV dataset because each image participated in training for most of the fold models.

### `run_all.sh`, `make_dummy_data.py`, `requirements.txt`

- `run_all.sh` provides an end-to-end driver for the pipeline.
- `make_dummy_data.py` generates a small synthetic dataset with leaf/UAV-style file naming so that the pipeline can be smoke-tested without downloading the real dataset.
- `requirements.txt` contains the Python dependencies used by the pipeline.

---

## 4. Engineering decisions and their reasons

| # | Decision | Why |
|---|---|---|
| 1 | Split once, in the manifest, on metadata only | Prevents later steps from silently creating different splits |
| 2 | Merge perceptual-hash clusters with parsed video/source groups | pHash alone did not capture all correlated UAV frames |
| 3 | Hash transformed versions | Duplicates can appear rotated or mirrored |
| 4 | Drop identical bytes under conflicting labels | Such samples cannot consistently represent two classes |
| 5 | Stratified group K-fold with fixed folds | Preserves class balance while keeping correlated groups together |
| 6 | Cross-validation for the main leaf result | Gives a tighter estimate than relying on one 20 % split |
| 7 | Keep each test fold untouched during model selection | Prevents direct test-fold tuning |
| 8 | Macro-F1 as the primary metric | Leaf data has approximately 5× class imbalance |
| 9 | Bootstrap over clusters | Images within the same cluster are not independent |
| 10 | One cache + one feature file | Prevents row-order/split inconsistencies |
| 11 | Fit transforms only on training data | Prevents preprocessing leakage |
| 12 | GPU-resident data and GPU augmentation | Reduces the CPU data-loading bottleneck on Colab |
| 13 | Avoid hue jitter | Colour changes can remove or alter disease-relevant cues |
| 14 | Frozen warm-up + separate backbone/head learning rates | Protects pretrained representations during initial adaptation |
| 15 | Use a controlled fine-tuning schedule | Earlier scheduling experiments showed instability around backbone unfreezing |
| 16 | Class-weighted loss + label smoothing | Handles imbalance and visually similar/noisy classes |
| 17 | 320 px leaf / 224 px UAV | Leaf benefited from higher resolution; UAV runs needed to remain computationally practical |
| 18 | Early stopping on validation macro-F1 | Small validation sets can be noisy |
| 19 | Flip TTA | Cheap augmentation at inference with no canonical image orientation |
| 20 | Ensemble different architectures | Different models make complementary errors |
| 21 | Never score all fold models together on the same CV data | Most images were seen during training by most fold models |
| 22 | Audit before trusting the headline score | 1-NN and date-holdout reveal whether the task is dominated by similarity/session |
| 23 | No single-split headline for UAV | Few independent videos can make a single split unstable or unrepresentative |
| 24 | Test-evaluation logging | Makes repeated test evaluation visible |
| 25 | Separate CutMix checkpoint names | Prevents variants from overwriting baseline checkpoints |
| 26 | Own DCT-based pHash implementation | Keeps duplicate detection under project control |
| 27 | Hard leakage assertions | Leakage should stop the pipeline rather than appear later as a high score |
| 28 | Save artifacts without the large cache when backing up | The cache can be rebuilt; models/reports/features are more expensive to lose |

---

## 5. Experiments that were tried (so nothing is hidden)

| Experiment | Outcome |
|---|---|
| Random frame split on UAV (v1) | ≈0.99 macro-F1: leakage; discarded |
| pHash-only grouping (UAV) | Still left roughly 90 % of evaluation frames sharing a video with training |
| Video-grouped 3-fold CV (UAV) | CNN 0.876, classical 0.909; 1-NN 0.873 |
| Mean-RGB-only UAV probe | 0.65–0.72 macro-F1, above chance |
| Leave-one-date-out UAV analysis | Recall varied from 0.00–1.00 by block; large blocks were around 0.67–0.76 |
| EfficientNet-B0 at 320 px (leaf) | 0.733 pooled 5-fold macro-F1 |
| EfficientNet-B0 + CutMix ensemble | 0.742 pooled 5-fold macro-F1 |
| DINOv2-S/14 frozen features + SVC | 0.684 pooled macro-F1 |
| 50/50 CNN + DINOv2 probability blend | 0.749 pooled macro-F1 |
| EfficientNet-B0 + CutMix + ConvNeXt-Tiny | 0.761 pooled macro-F1 |
| 3-model leaf ensemble | 0.775 accuracy / 0.761 macro-F1 |

The DINOv2 experiment was retained as a documented comparison but was superseded by the 3-CNN ensemble.

Because several model variants were compared on the same folds, the final best number should be treated as an experimental benchmark rather than a completely independently selected estimate.

---

## 6. Known limitations and how to extend

- **UAV data volume and independence:** the available UAV data contains only a small number of independent videos/source groups. More videos from the same fields and dates are needed to establish robust disease-level generalisation.
- **UAV acquisition confounding:** classes are strongly associated with recording dates/sessions. Grouping by video removes obvious frame leakage but does not eliminate session-level confounding.
- **UAV nearest-neighbour performance:** a training-free 1-NN lookup reaches a score close to the trained CNN, showing that visual/source similarity explains a substantial part of the measured performance.
- **UAV resolution:** 224 px processing may lose small lesions or subtle disease structures.
- **Leaf class imbalance:** Frog-eye and Septoria remain substantially harder than Healthy and Mosaic.
- **External validation:** the strongest next experiment is evaluation on an independent soybean disease dataset or newly collected field data.
- **UAV tiling:** `10_predict.py --tile-grid` supports tiled inference, but a fully trained multi-scale/tile-level UAV training strategy is future work.
- **Interpretability:** Grad-CAM or related methods can be added to check whether CNN predictions rely on disease-relevant regions.
- **Hyperparameter search:** broader sweeps were avoided because of the available compute budget.
- **Advisory layer:** the current work performs image-based recognition; it does not yet provide validated agronomic treatment or pesticide recommendations.

The most important scientific extension for UAV data is not simply a larger model. It is collecting multiple videos per class across shared fields and dates, followed by field/flight/date-level evaluation.

---

## Final project interpretation

The leaf experiments provide the strongest evidence of disease-recognition capability in the current dataset.

The UAV experiments are intentionally presented differently. Their purpose is partly to demonstrate how much apparent performance changes when evaluation respects video grouping and acquisition structure.

Therefore:

**Leaf:**

```
cleaned data
    ↓
5-fold CV
    ↓
model comparison
    ↓
3-CNN ensemble
    ↓
0.761 macro-F1
```

**UAV:**

```
video grouping
    ↓
3-fold CV
    ↓
CNN / classical / 1-NN comparison
    ↓
date-holdout + RGB audit
    ↓
high score, but limited generalisation evidence
```

That distinction is a deliberate part of the project's methodology rather than a weakness hidden from the results.
