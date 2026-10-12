# Soybean Leaf Disease Recognition

[![CI](https://github.com/GargiPareek-27/AI-Farmer-Advisory-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/GargiPareek-27/AI-Farmer-Advisory-Project/actions/workflows/ci.yml)

A leakage-aware, reproducible machine-learning pipeline that classifies soybean **leaf photos** into 6 classes, built on the leaf half of the
**MH-SoyaHealthVision** dataset (Mendeley Data). It is the machine-learning module of the AI Farmer Advisory Project; it is an image classifier,
not an advisory or treatment system.

| | |
|---|---|
| Task | Close-up leaf photo -> one of 6 classes: Healthy, Rust, Mosaic, Septoria brown spot, Frog-eye leaf spot, Caterpillar / Semilooper pest attack |
| Data | 2,782 files found, **2,756 images kept** after cleaning (26 dropped) |
| Headline result | **0.7602 macro-F1** (95 % CI 0.7438 - 0.7770, about 0.76), accuracy 0.7747: pooled out-of-fold, 5-fold cross-validation, 3-CNN ensemble with flip TTA |
| Recorded execution | Google Colab, NVIDIA Tesla T4, PyTorch 2.11.0+cu130, October 2026; `leaf_ensemble3`, 55 steps, about 110.9 minutes, 0 steps skipped |
| Status | Results come from one recorded Colab execution run from the unpacked project folder in Drive. An independent clean-ZIP reproduction is **pending** (see [`docs/RESULTS.md`](docs/RESULTS.md) section 12) |

> **Scope.** Leaf photos only. The UAV half of the dataset was explored earlier and is deliberately excluded: its frames are consecutive video
> frames, and its scores could not be shown to generalise (see [`docs/RESULTS.md`](docs/RESULTS.md) section 11). No advisory reasoning, severity
> estimation, API or application is implemented here.

---

## 1. Data

| Class | Images | Share |
|---|---:|---:|
| Soyabean_Rust | 845 | 30.7 % |
| Soyabean_Mosaic | 694 | 25.2 % |
| Caterpillar and Semilooper Pest Attack | 577 | 20.9 % |
| Soyabean_Spectoria_Brown_Spot | 268 | 9.7 % |
| Healthy_Soyabean | 204 | 7.4 % |
| Soyabean_Frog_Leaf_Eye | 168 | 6.1 % |
| **Total** | **2,756** | |

The class imbalance (largest / smallest) is 5.0x. Class = name of the image's parent folder.

**Cleaning (step 1).** Unreadable files are dropped; exact duplicates (MD5) are reduced to one copy; files with identical bytes under *different* labels are
dropped as label noise. 26 of the 2,782 files were removed (they are listed in `artifacts/leaf/dropped.csv` with a reason).

---

## 2. Leakage prevention and split strategy

An earlier UAV experiment reported a macro-F1 of about 0.99 that was wrong (consecutive video frames on both sides of the split). The lesson shaped the pipeline:

* **Split once, on metadata only**, in step 1, stored in `manifest.csv`. Later steps only select rows; nothing re-splits.
* **Near-duplicate clusters.** A 64-bit perceptual hash is computed for all 8 flips / 90-degree rotations of each image; pairs within 6 bits are linked
  (union-find). A cluster never spans two splits or folds, and this is asserted. On the leaf data **no near-duplicate pair was found**: the 2,756 images form
  2,756 clusters (largest cluster: 1 image).
* **Optional source groups.** `--group-regex` merges files that share a source id parsed from the file name. The leaf file names carry no such id, so every
  image is its own group.
* **Stratified group 5-fold cross-validation.** Folds are fixed in the manifest (551-552 images each). For fold *k*, fold *k* is the **test** set, about 1/6 of
  each class's remaining clusters (at least one) are **validation**, the rest are **training**. Every image is tested exactly once.
* **A single 70 / 10 / 20 split** (1,928 / 276 / 552) is also stored and used for the audit; the reported results use the 5 folds.
* **Audit before training (step 5):** share of validation / test images whose source group is in train (0.0 %); cosine similarity to the nearest train image
  (median 0.871) and the macro-F1 of a plain 1-NN lookup (0.588 validation, 0.594 test); a colour-only probe using just the three mean-RGB values
  (macro-F1 0.277 against 0.17 chance). A trivial lookup is far below the trained models.

---

## 3. Models

All metrics below are pooled out-of-fold macro-F1 over the 5 folds.

**Hand-crafted features (90).** Colour statistics, vegetation indices and texture descriptors per image (`features.py`).

**Frozen deep features (960).** Global-pooled embeddings of an ImageNet-pretrained MobileNetV3-Large, computed for all images once (nothing is fitted, so this cannot leak).
The combined set has 1,050 features.

**Classical baselines (step 6).** `StandardScaler` plus Logistic Regression, RBF-SVC, Random Forest or HistGradientBoosting (class-balanced) on hand / deep / both
feature sets, selected per fold by validation macro-F1. RBF-SVC won on every fold (deep features on folds 0 and 3, combined features on folds 1, 2 and 4).
Pooled result: **0.6904**.

**Transfer learning (step 7).** ImageNet-pretrained torchvision CNNs fine-tuned at **320 px**:

* **EfficientNet-B0** and **ConvNeXt-Tiny** (the recorded ensemble members); MobileNetV3-Large, EfficientNet-B2 and ResNet-18/50 are also supported.
* Whole training set held on the GPU as uint8; **GPU-side augmentation** (flips, 90-degree rotation, +/-20 % brightness / contrast, random zoom-in crop; no hue jitter,
  because colour is a disease cue); mixed precision; AdamW (weight decay 1e-4).
* **Two-stage schedule:** backbone frozen for 2 epochs (head LR 2e-3), then the backbone LR ramps up over one epoch (3e-4; 1e-4 for ConvNeXt-Tiny) and both decay with cosine.
* **Class-weighted cross-entropy** (weights from training labels only) with **label smoothing 0.1**.
* **CutMix** (Beta alpha 1.0, applied to half of the batches) in one of the three ensemble members.
* Up to 30 epochs, batch size 64, **early stopping on validation macro-F1** (patience 8); the best-validation checkpoint is kept.

**Test-time augmentation.** At evaluation the model outputs for the original, the horizontally flipped and the vertically flipped image are averaged before the softmax.

**Ensemble.** The three CNNs' predicted probabilities are averaged. The members are named explicitly in the evaluation command
(`--cnn efficientnet_b0_320,efficientnet_b0_320_cutmix,convnext_tiny_320`), so other checkpoints in the folder cannot change it.

**DINOv2 experiment (notebook only).** Frozen DINOv2-S/14 features with an SVC reached 0.6837 (95 % CI 0.665 - 0.702), below the fine-tuned CNNs. Averaging its
probabilities 50/50 with the single EfficientNet-B0 gave 0.7514 (0.735 - 0.768), still below the 3-CNN ensemble. It is not part of the final model.

---

## 4. Evaluation methodology

* **Metric:** macro-F1 (5x imbalance makes accuracy misleading), with accuracy, per-class precision / recall / F1, MCC, ROC-AUC, log-loss and calibration error also computed per fold.
* **Pooled out-of-fold estimate:** predictions from the 5 test folds are pooled, so all 2,756 images are scored exactly once by a model that never saw them.
* **Confidence intervals:** 95 % bootstrap interval over *clusters* (1,000 resamples), not over single images.
* **Model selection uses validation only;** the test fold is never loaded during training or classical-model selection, and every test evaluation is logged.
* **Never average all five fold models and score them on the dataset:** each image was in the training set of four of the five models.

---

## 5. Results

Recorded Colab run, `leaf_ensemble3`. Full tables, per-fold values, the confusion matrix and provenance: [`docs/RESULTS.md`](docs/RESULTS.md).

| Model | Macro-F1 | 95 % CI | Accuracy |
|---|---:|---:|---:|
| Classical (hand-crafted + frozen MobileNetV3 features; best of LogReg / SVC / HistGB per fold) | 0.6904 | 0.6709 - 0.7085 | 0.7242 |
| EfficientNet-B0, 320 px, flip TTA | 0.7340 | 0.7183 - 0.7505 | 0.7485 |
| EfficientNet-B0 + CutMix, 320 px, flip TTA | 0.7290 | 0.7124 - 0.7462 | 0.7395 |
| ConvNeXt-Tiny, 320 px, flip TTA | 0.7511 | 0.7344 - 0.7687 | 0.7656 |
| **3-CNN ensemble + TTA** | **0.7602** | **0.7438 - 0.7770** | **0.7747** |

Macro-F1 per fold of the ensemble: 0.7647, 0.7675, 0.7552, 0.7642, 0.7456 (mean 0.7594, sd 0.0090). The ensemble beat the EfficientNet-B0 baseline and the CutMix
variant on all five folds and ConvNeXt-Tiny on four of five (it was behind on fold 2: 0.7552 vs 0.7909). Its advantage over the best single model is +0.0091 and the
confidence intervals overlap, so it is a modest, not a conclusive, gain. CutMix alone did not improve on the baseline in this run (0.7290 vs 0.7340); its contribution to the ensemble was not ablated.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Caterpillar and Semilooper Pest Attack | 0.761 | 0.763 | **0.762** | 577 |
| Healthy_Soyabean | 0.995 | 1.000 | **0.998** | 204 |
| Soyabean_Frog_Leaf_Eye | 0.535 | 0.589 | **0.561** | 168 |
| Soyabean_Mosaic | 0.882 | 0.798 | **0.838** | 694 |
| Soyabean_Rust | 0.804 | 0.761 | **0.782** | 845 |
| Soyabean_Spectoria_Brown_Spot | 0.542 | 0.728 | **0.621** | 268 |
| macro average | 0.753 | 0.773 | **0.760** | 2,756 |

Weighted F1 0.779; accuracy 0.775. Healthy leaves are recognised almost perfectly. The errors are mostly between look-alike spot diseases: the largest confusion is
rust predicted as Septoria (76 of 845), followed by mosaic predicted as caterpillar / semilooper (68 of 694); 38 of 168 Frog-eye leaves are predicted as rust.

---

## 6. Quick start

### Colab (recommended, T4 GPU), from the project in Google Drive

1. Put `AI-Farmer-Advisory-Project_leaf_only.zip` (or the unpacked project folder) anywhere in **My Drive**.
2. Open `notebooks/soya_reproduce.ipynb` in Colab and choose **Runtime > Change runtime type > T4 GPU**.
3. Run it top to bottom. It mounts Drive, finds the project, unpacks it to `/content/project`, installs the dependencies, downloads the dataset, runs the
   tests and a GPU smoke test, and reproduces the results with `reproduce.py`. No GitHub access is needed.

When the project is a zip, the setup cell records the zip's md5 so each result can be traced to the exact version of the code. To change the code:
edit it, make a new zip, upload it to Drive, and re-run the notebook.

### Command line

```bash
pip install -r requirements.txt                  # full environment (PyTorch included)
export SOYA_ROOT=/path/to/folder/containing/Soyabean_Leaf_Image_Dataset

python reproduce.py --list                       # available recipes (configs/recipes.json)
python reproduce.py leaf_ensemble3 --dry-run     # print every command without running it
python reproduce.py leaf_ensemble3               # 5 folds x 3 CNNs (the recorded run took about 111 min on a T4)
python reproduce.py leaf_baseline                # single EfficientNet-B0
```

Each recipe finishes by writing `artifacts/leaf/reports/cv/reproduction_<recipe>.json` (source, commit hash if any, library versions, GPU, per-model CV
summaries). Compare it with `docs/RESULTS.md`.

**Before starting**, `reproduce.py` checks that the dataset folder exists, that the packages are installed and (for CNN steps) whether a GPU is
visible, and stops with a readable message instead of failing an hour in.

**Surviving a Colab disconnect.** Add `--resume` to continue (finished steps are skipped; a CNN counts as finished only when training completed)
and `--sync-dir` to back results up to Google Drive after every step and restore them automatically:

```bash
python reproduce.py leaf_ensemble3 --resume --sync-dir /content/drive/MyDrive/soya_artifacts
```

Use `--resume` only to continue the *same* run; for another recipe, fold count or image size use a fresh `--artifacts` folder (a mismatching fold
count is detected and refused). The image cache is never synced; it is rebuilt in about two minutes.

### Without a GPU or the real data (about 2 minutes)

```bash
pip install -r requirements-dev.txt              # no PyTorch needed
python -m pytest                                 # 24 tests on synthetic data
python make_dummy_data.py --out ./dummy_data
SOYA_ROOT=./dummy_data python reproduce.py leaf_baseline --no-torch --img-size 64 --folds 3
```

### Single steps

Every step is a script that can be run on its own (`python <script> --help`):

| # | Script | What it does |
|---|--------|--------------|
| 1 | `01_build_manifest.py` | validate, drop duplicates, cluster, split, make CV folds |
| 2 | `02_cache_images.py` | decode and resize every image once into one array |
| 3 | `03_eda.py` | tables and figures about the data |
| 4 | `04_extract_features.py` | hand-crafted and frozen-CNN features for all images |
| 5 | `05_audit_split.py` | leakage / shortcut audit of the active split (also saved as `reports/<split>/audit.txt`) |
| 6 | `06_train_classical.py` | LogReg / SVC / RF / HistGB on features |
| 7 | `07_finetune_cnn.py` | transfer-learning fine-tune of a CNN |
| 8 | `08_evaluate.py` | metrics on val or test; ensembles (`--cnn all` or `--cnn a,b,c`); flip TTA (`--tta`) |
| 9 | `09_cv_summary.py` | pooled out-of-fold results over all folds |
| 10 | `10_predict.py` | inference on new images (single model or ensemble) |

`--fold k` (steps 5-8) makes fold *k* the test set; validation is about 1/6 of each class's remaining clusters; training is the rest.

---

## 7. Using the trained models

```bash
# one image or a folder; several checkpoints are averaged (wildcards are expanded by the script, also on Windows)
python 10_predict.py --input leaf.jpg \
  --ckpt artifacts/leaf/models/fold0/cnn_efficientnet_b0_320.pt artifacts/leaf/models/fold0/cnn_convnext_tiny_320.pt
```

Do not score the five fold models together on the dataset itself: each image was in the training set of four of the five models, so the score would
be inflated. Use that average only for new images. Checkpoints are not stored in Git (`*.pt` is ignored); publish them as release assets.
The classifier always returns one of the six classes and has no "not soybean / not sure" output; add a confidence threshold before any farmer-facing use.

---

## 8. Testing

`python -m pytest` runs 24 tests on synthetic data (no GPU, no PyTorch, about 2 minutes; 24 passed in 191 s on the Colab runtime of the recorded session):

* metrics, LR schedule, source-group parsing, union-find clustering, fold-coverage guard (unit tests);
* the real scripts end to end through `reproduce.py --no-torch`: no cluster / group / file crosses a split or fold, exact duplicates are removed,
  flipped copies share a cluster, every image is tested exactly once, stale caches are refused, the cache equals direct decoding, the audit is saved;
* `--resume` / `--sync-dir`: a lost runtime is restored from the backup and gives an identical result; a different fold count is refused; a missing
  dataset fails fast;
* explicit ensemble lists (`--cnn a,b,c`), checkpoint-name validation, wildcard expansion for `10_predict.py`.

CI (`.github/workflows/ci.yml`) runs the same suite, a lint pass (`ruff`, errors only) and byte-compiles every script. **The PyTorch code (07, 08 with
CNNs, 10, `models.py`) is not covered by CI**; the notebook's GPU smoke test and the full reproduction exercise it.

---

## 9. Repository layout

```
soya_pipeline/
├── README.md
├── reproduce.py               runs a recipe end to end and writes a reproduction report
├── configs/recipes.json       leaf_ensemble3, leaf_baseline
├── common.py  metrics.py  features.py  models.py  schedules.py
├── 01_build_manifest.py ... 10_predict.py
├── make_dummy_data.py         synthetic dataset for smoke tests
├── tests/                     unit + end-to-end tests
├── notebooks/soya_reproduce.ipynb
├── docs/
│   ├── ENGINEERING_DECISIONS.md   what every file does and why
│   └── RESULTS.md                 numbers, provenance and the reproduction log
├── requirements.txt  requirements-core.txt  requirements-dev.txt
└── pytest.ini  ruff.toml
```

Outputs go to `artifacts/leaf/` (manifest, cache, features, models, reports); this folder is git-ignored.

---

## 10. Limitations

* **Source overlap cannot be ruled out.** The leaf file names carry no plant / field / session id, so every image is treated as its own group.
  Near-copies are caught by perceptual hashing (none were found), but several photos of the same plant under similar conditions may still sit on both sides of a split;
  that would make the scores somewhat optimistic. The audit (1-NN macro-F1 0.59) shows the task is not solved by lookup, not that overlap is absent.
* **No external validation dataset.** Single dataset, single source; labels were not independently re-audited.
* **The ensemble result may be somewhat optimistic.** Several variants (single models, a CutMix variant, an ensemble, a DINOv2 experiment and a blend) were compared on the
  same folds. The ensemble was a planned experiment, not picked after looking at test results, but the selection effect cannot be ruled out.
* **Weak classes.** Frog-eye leaf spot (F1 0.561) and Septoria brown spot (0.621) are not reliable enough for unsupervised use. The model has no "not soybean / not sure" output.
* **Not a diagnostic or advisory tool.** Predictions are research output and should be confirmed by an agronomist before any treatment decision. Severity estimation,
  advisory reasoning and treatment recommendations are not implemented.
* **Test coverage.** The PyTorch training / evaluation paths are not covered by CI.
* **GPU training is not bit-exact;** expect about +/-0.01 macro-F1 between reruns.

## 11. Reproducibility statement

* Seeds are fixed (`--seed 42`) and folds are stored in `manifest.csv`. GPU training is not bit-exact (`cudnn.benchmark`, mixed precision): expect about
  +/-0.01 macro-F1 between reruns.
* **What was executed:** one full run of `leaf_ensemble3` on Colab (Tesla T4, PyTorch 2.11.0+cu130; 55 steps; about 110.9 minutes; 0 steps skipped by `--resume`),
  with 24 tests passing and the GPU smoke test passing in the same session. All numbers in this README come from that run's outputs.
* **What is not yet established:** that run used the unpacked project folder in Google Drive, so no zip md5 was recorded and no commit hash exists (`commit None`).
  It is therefore **not** an independent clean-ZIP reproduction; that is pending and is logged in `docs/RESULTS.md` section 12 once done.
* Library versions of the run (from `pip freeze`): numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, scikit-image 0.25.2, opencv-python-headless 5.0.0.93, pillow 11.3.0,
  joblib 1.6.0, matplotlib 3.10.0, torch 2.11.0+cu130, torchvision 0.26.0+cu130. `requirements*.txt` give minimum versions only; the notebook writes the exact versions to
  `requirements-lock.txt`, which is not yet part of the project.
* Every test-set evaluation is appended to `reports/<split>/test_eval_log.txt`; tuning on test shows up there.
* Pretrained weights are torchvision ImageNet weights (downloaded automatically) and, for the experiment, DINOv2 from `torch.hub`.

## Data and credits

Dataset: Shinde, S., Attar, V. (2024). *MH-SoyaHealthVision: An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment*. Mendeley Data, V1.
[doi:10.17632/hkbgh5s3b7.1](https://doi.org/10.17632/hkbgh5s3b7.1). Dataset paper: *An Indian UAV and leaf image dataset for integrated crop health assessment of
soybean crop*, Data in Brief. The dataset has its own licence terms; check them on the Mendeley page and cite the dataset if you use this work.
The dataset is not redistributed here; the notebook downloads it from Mendeley.

## Licence

The code in this repository is released under the MIT licence, see [`../LICENSE`](../LICENSE).
