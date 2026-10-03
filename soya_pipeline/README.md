# Soybean Disease Recognition from Leaf Photos and UAV Frames

Leakage-safe, reproducible machine-learning pipeline for the **MH-SoyaHealthVision** dataset (an Indian UAV and leaf image dataset for crop health assessment, Mendeley Data).

It trains and honestly evaluates classifiers on two image sets:

| Set | Task | Images (after cleaning) | Classes |
|---|---|---|---|
| Leaf | Close-up leaf photos | 2,756 | 6: Healthy, Rust, Mosaic, Septoria brown spot, Frog-eye leaf spot, Caterpillar/Semilooper pest attack |
| UAV | Top-down drone frames (3840×2160) | 2,831 | 4: Healthy, Rust, Mosaic, Semilooper pest attack |

> **Honest summary:** Leaf recognition reaches **0.76 macro-F1** under leakage-checked 5-fold cross-validation. UAV scores are higher (0.88–0.91) but are **not** evidence of generalisation to new fields or seasons, because the classes are strongly associated with recording dates/sessions and the available UAV data contains only a small number of independent videos. The audit measures how far those numbers can be trusted.

---

## Results

### Leaf (5-fold stratified cross-validation, every image tested exactly once, n = 2,756)

Pooled out-of-fold macro-F1 with a 95 % confidence interval obtained using bootstrap resampling over clusters.

| Model | Macro-F1 | 95 % CI | Accuracy |
|---|---|---|---|
| Classical (hand-crafted + frozen MobileNetV3 features; best feature/model configuration per fold) | 0.690 | 0.671 - 0.709 | 0.724 |
| DINOv2-S/14 frozen features + SVC | 0.684 | 0.665 - 0.702 | — |
| EfficientNet-B0, 320 px, flip TTA | 0.733 | 0.716 - 0.749 | 0.747 |
| + CutMix variant (ensemble of 2) | 0.742 | 0.726 - 0.760 | 0.756 |
| EfficientNet-B0 + B0/CutMix + ConvNeXt-Tiny (ensemble of 3) | **0.761** | 0.744 - 0.778 | 0.775 |

The 3-model ensemble improved over the single EfficientNet-B0 on all five folds:

- Ensemble: 0.748 - 0.766
- EfficientNet-B0: 0.721 - 0.747

The final pooled result is **macro-F1 = 0.7606** and **accuracy = 0.7747**.

Per-class F1 of the 3-model ensemble:

| Class | F1 | Comment |
|---|---|---|
| Healthy | 0.998 | essentially perfect |
| Mosaic | 0.836 | |
| Rust | 0.782 | frequent confusion target |
| Caterpillar / Semilooper | 0.761 | confused with mosaic and rust |
| Septoria brown spot | 0.623 | precision 0.55, recall 0.72 |
| Frog-eye leaf spot | 0.564 | 38 of 168 predicted as rust |

Remaining errors are concentrated among visually similar disease categories, especially Rust, Septoria brown spot and Frog-eye leaf spot.

### UAV (grouped 3-fold cross-validation by video, n = 2,831)

| Evaluation | Macro-F1 | Note |
|---|---|---|
| CNN (MobileNetV3-Large, 224 px) | 0.876 (CI 0.74 - 0.96) | folds 0.91 / 0.95 / 0.75 |
| Classical features + HistGradientBoosting | 0.909 (CI 0.83 - 0.96) | folds 0.92 / 0.93 / 0.85 |
| Training-free 1-NN lookup on frozen features | 0.873 (mean of folds) | folds 0.87 / 0.92 / 0.83 |
| Mean-RGB-only probe (3 numbers per image) | 0.65 - 0.72 | chance = 0.25 |
| Leave-one-date-out recall (large blocks) | 0.67 - 0.76 | deep features; small blocks can fall to 0.00 - 0.08 |

**Reading this table:** the training-free nearest-neighbour lookup performs similarly to the trained CNN, and performance drops when an entire recording date is withheld. The UAV results therefore describe performance on new videos under similar field/acquisition conditions, not demonstrated generalisation to unseen fields or seasons.

---

## Why this pipeline exists: leakage

The first version of this project reported UAV macro-F1 of about 0.99. That result was discarded after investigating the data.

- UAV frames are consecutive frames from a small number of videos. Random frame-level splitting allowed near-identical frames from the same video to appear in both training and evaluation. In the original audit, about 90 % of validation/test frames shared a video with training, nearest-train similarity was about 0.98, and a training-free 1-NN lookup reached about 0.997 macro-F1.
- Perceptual hashing alone did not completely solve the problem. The final pipeline combines perceptual-hash similarity with video/flight identifiers parsed from file names, keeping correlated frames from the same source together.
- Even after video-aware grouping, the UAV classes remain strongly associated with recording sessions/dates. Only a small amount of date overlap exists between classes, so a model can partially exploit acquisition/session characteristics rather than disease appearance.
- The audit therefore includes nearest-neighbour similarity, a mean-RGB probe and leave-one-date-out analysis.
- Leaf images showed no meaningful near-duplicate problem under the same analysis, and the leakage-checked 5-fold leaf result is used as the project's main headline result.

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
  07_finetune_cnn.py        transfer-learning fine-tune of a CNN
  08_evaluate.py            full metric suite on val or test; ensembles; TTA
  09_cv_summary.py          pooled out-of-fold results over all folds
  10_predict.py             inference on new images (single model or ensemble)
  run_all.sh                end-to-end driver
  make_dummy_data.py        tiny synthetic dataset for smoke tests
  requirements.txt
  soya_run.ipynb            Colab notebook that calls the scripts
  docs/ENGINEERING_DECISIONS.md
                            what every file does and why it was built that way
```

All outputs go to `artifacts/<dataset>/` (manifest, cache, features, models, reports).

See `docs/ENGINEERING_DECISIONS.md` for the artifact layout and engineering rationale.

---

## Quick start (Google Colab, T4 GPU)

1. Download the dataset from Mendeley Data and unzip so that these two folders exist side by side:

   `Soyabean_Leaf_Image_Dataset/` and `Soyabean_UAV-Based_Image_Dataset/`

   The class label is taken from the parent folder name.

2. Upload `soya_pipeline.zip` and `soya_run.ipynb`, open the notebook, and run top to bottom.

The notebook records the clean experimental run used for the reported results.

### Command line

```bash
pip install -r requirements.txt
export SOYA_ROOT=/path/to/folder/containing/both/datasets
```

#### Leaf

The main leaf experiment uses 5-fold cross-validation with EfficientNet-B0 at 320 px.

```bash
python 01_build_manifest.py   --dataset leaf --img-size 320
python 02_cache_images.py     --dataset leaf --img-size 320
python 03_eda.py              --dataset leaf --img-size 320
python 04_extract_features.py --dataset leaf --img-size 320
python 05_audit_split.py      --dataset leaf --img-size 320

for f in 0 1 2 3 4; do
  python 06_train_classical.py --dataset leaf --img-size 320 --fold $f
  python 07_finetune_cnn.py    --dataset leaf --img-size 320 --fold $f \
      --model efficientnet_b0 --epochs 30 --patience 8
  python 08_evaluate.py        --dataset leaf --img-size 320 --fold $f --tta
done

python 09_cv_summary.py \
    --dataset leaf \
    --img-size 320 \
    --model-tag cnn_efficientnet_b0_320_tta
```

The final ensemble experiments extend this by adding the CutMix EfficientNet-B0 variant and ConvNeXt-Tiny:

```bash
for f in 0 1 2 3 4; do
  python 07_finetune_cnn.py \
      --dataset leaf --img-size 320 --fold $f \
      --model efficientnet_b0 --epochs 30 --patience 8 --cutmix 1.0

  python 07_finetune_cnn.py \
      --dataset leaf --img-size 320 --fold $f \
      --model convnext_tiny --epochs 30 --patience 8 \
      --lr-backbone 1e-4

  python 08_evaluate.py \
      --dataset leaf --img-size 320 --fold $f \
      --which cnn --cnn all --tta
done

python 09_cv_summary.py \
    --dataset leaf \
    --img-size 320 \
    --model-tag cnn_ensemble3
```

#### UAV

UAV evaluation uses grouped 3-fold cross-validation and should be read together with the split audit.

```bash
python 01_build_manifest.py --dataset uav --img-size 224 --folds 3
python 02_cache_images.py   --dataset uav --img-size 224
python 04_extract_features.py --dataset uav --img-size 224

for f in 0 1 2; do
  python 05_audit_split.py --dataset uav --img-size 224 --fold $f
done

for f in 0 1 2; do
  python 06_train_classical.py \
      --dataset uav --img-size 224 --fold $f

  python 07_finetune_cnn.py \
      --dataset uav --img-size 224 --fold $f \
      --epochs 20 --warmup-epochs 3 \
      --lr-backbone 1e-4 --patience 6

  python 08_evaluate.py \
      --dataset uav --img-size 224 --fold $f --tta
done

python 09_cv_summary.py \
    --dataset uav --img-size 224 --model-tag cnn

python 09_cv_summary.py \
    --dataset uav --img-size 224 --model-tag classical
```

### Additional leaf experiments

The notebook also contains:

- DINOv2-S/14 frozen embeddings
- SVC with inner cross-validation for `C`
- DINOv2/CNN probability blending
- EfficientNet-B0 + CutMix ensemble
- EfficientNet-B0 + CutMix + ConvNeXt-Tiny ensemble

The DINOv2 experiment produced:

```
DINOv2 only       pooled macro-F1 = 0.6837
CNN only          pooled macro-F1 = 0.7325
50/50 CNN + DINO  pooled macro-F1 = 0.7488
```

The DINOv2 blend was subsequently superseded by the 3-CNN ensemble.

### Smoke test without the real data

```bash
python make_dummy_data.py --out ./dummy_data
export SOYA_ROOT=./dummy_data

DS=leaf K=3 SIZE=64 bash run_all.sh
```

**Run time (Colab T4, approximate):** manifest + cache + features 5–8 min per dataset; EfficientNet-B0 at 320 px about 4–5 min per fold; ConvNeXt-Tiny about 6–7 min per fold; UAV cross-validation about 15–25 min.

---

## Using the trained models

```bash
# One image or a folder; several checkpoints can be probability-averaged.
python 10_predict.py --dataset leaf --input leaf.jpg \
  --ckpt artifacts/leaf/models/fold*/cnn_efficientnet_b0_320.pt \
         artifacts/leaf/models/fold*/cnn_convnext_tiny_320.pt

# Large UAV frame cut into an NxN grid; reports the share of tiles per class.
python 10_predict.py --dataset uav --input frame.jpg \
  --tile-grid 4 \
  --ckpt artifacts/uav/models/fold0/cnn_*.pt
```

Do not score all five fold models together on the same leaf dataset and call that an unbiased evaluation. Each image was used for training by four of the five fold models. Averaging fold checkpoints is appropriate for new images, not for estimating performance on the same CV data.

---

## Limitations and Future Work

This project was developed under limited local GPU/compute resources, so the experiments focused on models and evaluation protocols that were practical to run within the available hardware budget. As a result, the current study does not include large-scale hyperparameter sweeps, extensive multi-GPU training, or very large foundation models.

The main limitations of the current work are:

- The leaf dataset is relatively limited in size and class balance, particularly for difficult classes such as Frog-eye and Septoria.
- The best leaf result was obtained after comparing multiple model variants on the same folds, so the final number should be treated as a strong experimental result rather than a fully independently selected benchmark.
- UAV evaluation is based on a limited number of independent flight/video groups, so broader field- and season-level generalisation requires further validation.
- UAV class labels are strongly associated with recording dates/sessions, making acquisition confounding a major limitation even after video-aware grouping.
- UAV images were processed at a constrained resolution for computational feasibility, which may limit recognition of small or subtle disease symptoms.
- The current evaluation primarily uses the available dataset; external-dataset validation would provide a stronger test of cross-domain generalisation.
- The training-free 1-NN UAV result being close to the CNN result is evidence that visual/session similarity explains a substantial portion of the observed performance.
- The current project is an image-recognition pipeline, not a validated agronomic recommendation system.

Future work will focus on:

- External validation on additional soybean disease datasets.
- Field-, flight-, and date-level cross-validation for stronger generalization estimates.
- Higher-resolution and multi-scale UAV inference using tiled crops.
- Targeted improvement of difficult disease classes through hard-example mining and class-balanced training.
- Grad-CAM or related interpretability methods to verify that predictions rely on disease-relevant visual regions.
- Larger hyperparameter/model searches when additional compute resources are available.
- More UAV videos for each class recorded across the same fields and dates, which is the most important step for separating disease recognition from acquisition/session effects.

---

## Reproducibility notes

- Seeds are fixed (`--seed 42`) and folds are stored in `manifest.csv`, but GPU training is not bit-exact (`cudnn.benchmark`, mixed precision), so small run-to-run variation is expected.
- The notebook's clean run was performed on a Google Colab Tesla T4 GPU.
- Every test-set evaluation is logged so repeated test looks can be identified.
- Pretrained torchvision weights are downloaded automatically.
- DINOv2 weights were downloaded separately during the notebook-only DINOv2 experiment.
- The notebook saves the final artifacts to Drive while excluding the large image cache, which can be rebuilt.

## Data and Credits

**Dataset:** *MH-SoyaHealthVision: An Indian UAV and Leaf Image Dataset for Integrated Crop Health Assessment* by Sayali Shinde and Dr. Vahida Attar, Mendeley Data, Version 1, DOI: [10.17632/hkbgh5s3b7.1](https://doi.org/10.17632/hkbgh5s3b7.1).

The dataset is made available under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence. Please refer to the original dataset record for the complete licence terms and attribution requirements.

## Licence

The code in this repository is released under the **MIT License**.

The dataset used by this project is not covered by the MIT License and remains subject to its original CC BY 4.0 licence and attribution requirements.
