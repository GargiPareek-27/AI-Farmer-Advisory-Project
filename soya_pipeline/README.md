# Soybean Disease Recognition from Leaf Photos and UAV Frames

Leakage-safe, reproducible machine-learning pipeline for the **MH-SoyaHealthVision** dataset
(an Indian UAV and leaf image dataset for crop health assessment, Mendeley Data).
It trains and honestly evaluates classifiers on two image sets:

| Set | Task | Images (after cleaning) | Classes |
|---|---|---|---|
| **Leaf** | Close-up leaf photos | 2,756 | 6: Healthy, Rust, Mosaic, Septoria brown spot, Frog-eye leaf spot, Caterpillar/Semilooper pest attack |
| **UAV** | Top-down drone frames (3840x2160) | 2,831 | 4: Healthy, Rust, Mosaic, Semilooper pest attack |

> **Honest summary:** Leaf recognition reaches **0.76 macro-F1** under a leakage-checked 5-fold
> cross-validation. UAV scores look higher (0.88-0.91) but are **not** evidence of generalisation to
> new fields or seasons, because classes were recorded on different dates (see [Limitations](#limitations)).
> The pipeline exists largely to make that difference measurable.

---

## Results

### Leaf (5-fold stratified cross-validation, every image tested exactly once, n = 2,756)

Pooled out-of-fold macro-F1 with a 95 % confidence interval (bootstrap over near-duplicate clusters).

| Model | Macro-F1 | 95 % CI | Accuracy |
|---|---|---|---|
| Classical (hand-crafted + frozen MobileNetV3 features; best of LogReg / SVC / HistGB per fold) | 0.690 | 0.671 - 0.709 | 0.724 |
| DINOv2-S/14 frozen features + SVC | 0.684 | 0.665 - 0.702 | - |
| EfficientNet-B0, 320 px, flip TTA | 0.733 | 0.716 - 0.749 | 0.747 |
| + CutMix variant (ensemble of 2) | 0.742 | 0.726 - 0.760 | 0.756 |
| EfficientNet-B0 + B0/CutMix + ConvNeXt-Tiny (ensemble of 3) | **0.761** | **0.744 - 0.778** | **0.775** |

The 3-model ensemble beat the single EfficientNet-B0 on all five folds (0.748-0.766 vs 0.721-0.747).

Per-class F1 of the 3-model ensemble:

| Class | F1 | Comment |
|---|---|---|
| Healthy | 0.998 | essentially perfect |
| Mosaic | 0.836 | |
| Rust | 0.782 | most common confusion target |
| Caterpillar / Semilooper | 0.761 | confused with mosaic and rust |
| Septoria brown spot | 0.623 | over-predicted (precision 0.55, recall 0.72) |
| Frog-eye leaf spot | 0.564 | 38 of 168 predicted as rust |

Remaining errors are mostly between visually similar spot diseases (rust, Septoria, Frog-eye).

### UAV (grouped 3-fold cross-validation by video, n = 2,831)

| Evaluation | Macro-F1 | Note |
|---|---|---|
| CNN (MobileNetV3-Large, 224 px) | 0.876 (CI 0.74 - 0.96) | folds 0.91 / 0.95 / 0.75 |
| Classical features + HistGB | 0.909 (CI 0.83 - 0.96) | folds 0.92 / 0.93 / 0.85 |
| **Training-free 1-NN lookup on frozen features** | 0.873 (mean of folds) | folds 0.87 / 0.92 / 0.83 |
| Mean-RGB-only probe (3 numbers per image) | 0.65 - 0.72 | chance = 0.25 |
| Leave-one-date-out recall (large blocks) | 0.67 - 0.76 | deep features; small blocks fall as low as 0.00 - 0.08 |

**Reading this table:** a plain nearest-neighbour lookup matches the trained CNN, and recall drops when a
whole recording date is withheld. UAV scores therefore describe performance on *new videos of
previously seen field conditions*, not on new fields or seasons.

---

## Why this pipeline exists: leakage

The first version of this project reported UAV macro-F1 of about 0.99. That number was wrong:

* UAV frames are **consecutive frames of a few videos** (9-13 videos per class). Splitting frames at random puts
  near-identical frames on both sides of the split (about 90 % of validation/test frames shared a video with
  training; nearest-train cosine similarity was 0.98 and a 1-NN lookup scored 0.997).
* Perceptual hashing alone did not catch this. The final pipeline merges **perceptual-hash clusters with
  video / flight identifiers parsed from file names**, so a whole video always stays in one split or fold.
* Even after fixing that, **classes were recorded on largely different dates** (only 1 of 8 capture dates is
  shared by two classes), so a model can score well by recognising the session. The audit step quantifies this.

Leaf images showed no near-duplicates (median nearest-train similarity 0.87, 1-NN macro-F1 0.59), and
its results are the project's headline.

---

## Repository layout

```
soya_pipeline/
  common.py                 CLI arguments, artifact paths, fold logic, cache/feature loaders
  metrics.py                softmax, ECE, cluster-bootstrap confidence interval
  features.py               90 hand-crafted colour / vegetation-index / texture features
  models.py                 torchvision backbones, embedding extraction, checkpoints
  01_build_manifest.py      validate, de-duplicate, group, split, make CV folds
  02_cache_images.py        decode + resize every image once into one uint8 array
  03_eda.py                 tables and figures about the data
  04_extract_features.py    hand-crafted + frozen-CNN features for all images
  05_audit_split.py         leakage / shortcut audit of the active split
  06_train_classical.py     LogReg / SVC / RF / HistGB on features
  07_finetune_cnn.py        fast transfer-learning fine-tune of a CNN
  08_evaluate.py            full metric suite on val or test; ensembles; TTA
  09_cv_summary.py          pooled out-of-fold results over all folds
  10_predict.py             inference on new images (single model or ensemble)
  run_all.sh                end-to-end driver
  make_dummy_data.py        tiny synthetic dataset for smoke tests
  requirements.txt
soya_run.ipynb              Colab notebook that calls the scripts
docs/ENGINEERING_DECISIONS.md   what every file does and why it was built that way
```

All outputs go to `artifacts/<dataset>/` (manifest, cache, features, models, reports). See the docs for the layout.

---

## Quick start (Google Colab, T4 GPU)

1. Download the dataset from Mendeley Data and unzip so that these two folders exist side by side:
   `Soyabean_Leaf_Image_Dataset/` and `Soyabean_UAV-Based_Image_Dataset/` (class = parent folder name).
2. Upload `soya_pipeline.zip` and `soya_run.ipynb`, open the notebook, and run top to bottom.

### Command line

```bash
pip install -r requirements.txt
export SOYA_ROOT=/path/to/folder/containing/both/datasets

# Leaf: 5-fold cross-validation with EfficientNet-B0 at 320 px
python 01_build_manifest.py   --dataset leaf --img-size 320
python 02_cache_images.py     --dataset leaf --img-size 320
python 03_eda.py              --dataset leaf --img-size 320
python 04_extract_features.py --dataset leaf --img-size 320
python 05_audit_split.py      --dataset leaf --img-size 320
for f in 0 1 2 3 4; do
  python 06_train_classical.py --dataset leaf --img-size 320 --fold $f
  python 07_finetune_cnn.py    --dataset leaf --img-size 320 --fold $f --model efficientnet_b0 --epochs 30 --patience 8
  python 08_evaluate.py        --dataset leaf --img-size 320 --fold $f --tta
done
python 09_cv_summary.py --dataset leaf --img-size 320 --model-tag cnn_efficientnet_b0_320_tta
```

Add the ensemble members (per fold), then evaluate their average:

```bash
python 07_finetune_cnn.py --dataset leaf --img-size 320 --fold $f --model efficientnet_b0 --epochs 30 --patience 8 --cutmix 1.0
python 07_finetune_cnn.py --dataset leaf --img-size 320 --fold $f --model convnext_tiny   --epochs 30 --patience 8 --lr-backbone 1e-4
python 08_evaluate.py     --dataset leaf --img-size 320 --fold $f --which cnn --cnn all --tta
python 09_cv_summary.py   --dataset leaf --img-size 320 --model-tag cnn_ensemble3
```

UAV (use cross-validation, never the single split, and read the audit first):

```bash
python 01_build_manifest.py --dataset uav --img-size 224 --folds 3
python 02_cache_images.py --dataset uav --img-size 224
python 04_extract_features.py --dataset uav --img-size 224
for f in 0 1 2; do python 05_audit_split.py --dataset uav --img-size 224 --fold $f; done
for f in 0 1 2; do
  python 06_train_classical.py --dataset uav --img-size 224 --fold $f
  python 07_finetune_cnn.py    --dataset uav --img-size 224 --fold $f --epochs 20 --warmup-epochs 3 --lr-backbone 1e-4 --patience 6
  python 08_evaluate.py        --dataset uav --img-size 224 --fold $f --tta
done
python 09_cv_summary.py --dataset uav --img-size 224 --model-tag cnn
python 09_cv_summary.py --dataset uav --img-size 224 --model-tag classical
```

Smoke test without the real data:

```bash
python make_dummy_data.py --out ./dummy_data && export SOYA_ROOT=./dummy_data
DS=leaf K=3 SIZE=64 bash run_all.sh
```

**Run time (Colab T4, approximate):** manifest + cache + features 5-8 min per dataset; EfficientNet-B0 at 320 px
about 4-5 min per fold; ConvNeXt-Tiny about 6-7 min per fold; UAV cross-validation about 15-25 min.

---

## Using the trained models

```bash
# one image or a folder; several checkpoints are averaged
python 10_predict.py --dataset leaf --input leaf.jpg \
  --ckpt artifacts/leaf/models/fold*/cnn_efficientnet_b0_320.pt artifacts/leaf/models/fold*/cnn_convnext_tiny_320.pt

# big UAV frame cut into an NxN grid; reports the share of tiles per class
python 10_predict.py --dataset uav --input frame.jpg --tile-grid 4 --ckpt artifacts/uav/models/fold0/cnn_*.pt
```

Do not score the five fold models together on the dataset itself: each image was in the training set of
four of the five models, so the score would be inflated. Use that average only for new images.

---

## Limitations

* **UAV confound.** Classes were recorded on different dates; healthy comes from a different file source
  (numbered stills with no timestamp) and is recognised almost perfectly. Disease recall fell to roughly 0.7
  when a whole date was withheld. Generalisation to new fields or seasons is **not established**.
* **Few independent UAV sources.** 9-13 videos per class; per-fold scores vary (0.75-0.95) and intervals are wide.
* **Leaf ensemble is a selected variant.** Several variants (DINOv2, blends, two ensembles) were compared on the
  same folds, so 0.761 is probably slightly optimistic. The ensemble was a planned experiment, not picked after
  looking at test results.
* **Single dataset, single source.** No external test set; labels were not independently re-audited.
* **UAV images are resized to 224 x 224** (aspect ratio not preserved), which may hide small lesions.
* **Not agronomic advice.** Weak classes (Frog-eye, Septoria) are not reliable enough for unsupervised use.

---

## Reproducibility notes

* Seeds are fixed (`--seed 42`) and folds are stored in `manifest.csv`, but GPU training is not bit-exact
  (`cudnn.benchmark`, mixed precision), so expect about +/-0.01 macro-F1 between reruns.
* Every test-set evaluation is appended to `reports/<split>/test_eval_log.txt`; tuning on test shows up there.
* Pretrained weights are torchvision ImageNet weights, downloaded automatically.

## Data and credits

Dataset: *MH-SoyaHealthVision: An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment*, Mendeley Data
([dataset authors and licence: add here]). Please cite the dataset if you use this work.

## Licence

[Add your licence here. The dataset has its own licence terms.]
