# AI Farmer Advisory Project

An AI assistant for crop-health advice, built in stages. **What exists today is the machine-learning module for soybean leaf-disease
recognition.** Farmer-advisory reasoning, treatment recommendations, severity estimation, an API and an application are planned and are
**not implemented** in this repository.

| Layer | Status | Where |
|---|---|---|
| Soybean leaf-disease image classifier (6 classes) | **Implemented and cross-validated; one full end-to-end Colab execution recorded** | [`soya_pipeline/`](soya_pipeline/) |
| Farmer-advisory reasoning / treatment recommendations | Planned, not implemented | - |
| Severity estimation | Planned, not implemented | - |
| API, web or mobile application | Planned, not implemented | - |

---

## Best executed ML result

From the recorded Colab run of the `leaf_ensemble3` recipe (NVIDIA Tesla T4, PyTorch 2.11.0+cu130, 55 pipeline steps, about 110.9 minutes):

| | |
|---|---|
| Task | One close-up soybean leaf photo -> one of 6 classes: Healthy, Rust, Mosaic, Septoria brown spot, Frog-eye leaf spot, Caterpillar / Semilooper pest attack |
| Data | 2,782 leaf photos found, **2,756 kept** after cleaning (26 files dropped) |
| Evaluation | Stratified 5-fold cross-validation; every one of the 2,756 images is tested exactly once (pooled out-of-fold) |
| Model | EfficientNet-B0 + EfficientNet-B0 with CutMix + ConvNeXt-Tiny, 320 px, flip test-time augmentation, probabilities averaged |
| **Pooled macro-F1** | **0.7602** (95 % CI 0.7438 - 0.7770), about 0.76 |
| Accuracy | 0.7747 (about 0.775) |
| Macro-F1 per fold | 0.7647, 0.7675, 0.7552, 0.7642, 0.7456 (mean 0.7594) |

| Model (pooled out-of-fold, 5 folds) | Macro-F1 | 95 % CI | Accuracy |
|---|---:|---:|---:|
| Classical (hand-crafted + frozen MobileNetV3 features) | 0.6904 | 0.6709 - 0.7085 | 0.7242 |
| EfficientNet-B0, 320 px, flip TTA | 0.7340 | 0.7183 - 0.7505 | 0.7485 |
| EfficientNet-B0 + CutMix, 320 px, flip TTA | 0.7290 | 0.7124 - 0.7462 | 0.7395 |
| ConvNeXt-Tiny, 320 px, flip TTA | 0.7511 | 0.7344 - 0.7687 | 0.7656 |
| **3-CNN probability ensemble, flip TTA** | **0.7602** | **0.7438 - 0.7770** | **0.7747** |

Per-class F1 of the ensemble: Healthy 0.998, Mosaic 0.838, Rust 0.782, Caterpillar / Semilooper 0.762, Septoria brown spot 0.621, Frog-eye leaf spot 0.561.
Frog-eye and Septoria are the weak classes. All tables, the confusion matrix and provenance: [`soya_pipeline/docs/RESULTS.md`](soya_pipeline/docs/RESULTS.md).

---

## How the ML module is built

* **Data validation and cleaning:** unreadable files dropped, exact duplicates handled, files with identical bytes under different labels dropped.
* **Leakage-aware splitting:** perceptual-hash clustering (robust to flips and 90-degree rotations) so near-copies never straddle a split; stratified
  group 5-fold cross-validation fixed once in a manifest.
* **Features and baselines:** 90 hand-crafted colour / vegetation-index / texture features, 960 frozen MobileNetV3 features, classical models (LogReg, SVC, HistGB).
* **Transfer learning:** ImageNet-pretrained EfficientNet-B0 and ConvNeXt-Tiny fine-tuned at 320 px with GPU-side augmentation, CutMix (one variant),
  class-weighted loss, label smoothing, frozen warm-up and early stopping on validation macro-F1.
* **Inference-time:** flip test-time augmentation and a 3-model probability ensemble.
* **Evaluation:** pooled out-of-fold macro-F1 with cluster-bootstrap confidence intervals, per-class precision / recall / F1, confusion matrix.
* **Trust checks:** a split audit (nearest-train similarity, 1-NN lookup, colour-only probe) and a test-evaluation log.
* **Engineering:** 24 automated tests on synthetic data, a recipe runner (`reproduce.py`) with pre-flight checks, resume and Drive backup, and a Colab notebook.

---

## Scope and limitations

* **Leaf photos only.** The UAV half of the source dataset was explored earlier and is excluded; its scores could not be shown to generalise.
* **Not a diagnostic or advisory authority.** The classifier outputs a class label. It has no "not soybean / not sure" output, no severity estimate and
  no treatment logic. Predictions should be confirmed by an agronomist.
* **Independence between photos cannot be verified.** Leaf file names carry no plant, field or session id, so every image is its own group.
  Perceptual hashing found no near-duplicate pairs, but that cannot guarantee that photos of the same plant do not sit on both sides of a split.
* **No external validation dataset.** All results come from one dataset, evaluated by cross-validation.
* **Weaker classes:** Frog-eye leaf spot (F1 0.561) and Septoria brown spot (0.621).
* **The ensemble figure may be somewhat optimistic:** several variants (single models, an ensemble, a DINOv2 experiment and a blend) were compared on the same folds.
* **GPU training is not bit-exact:** expect about +/-0.01 macro-F1 between reruns.
* **Test coverage:** the PyTorch training / evaluation paths are not covered by CI; the notebook's GPU smoke test and the full run exercise them.
* **Reproducibility status:** the numbers above come from one recorded Colab execution run from the unpacked project folder in Google Drive. An independent
  clean-ZIP reproduction (with the zip's md5 recorded) is **pending**; see `soya_pipeline/docs/RESULTS.md`.

---

## Reproduce / run

Full instructions are in [`soya_pipeline/README.md`](soya_pipeline/README.md). In short:

* **Colab (T4 GPU):** put the project (zip or unpacked folder) in Google Drive, open `soya_pipeline/notebooks/soya_reproduce.ipynb`, run it top to bottom.
* **Command line:**
  ```bash
  cd soya_pipeline
  pip install -r requirements.txt
  export SOYA_ROOT=/path/to/folder/containing/Soyabean_Leaf_Image_Dataset
  python reproduce.py leaf_ensemble3 --dry-run   # print every command
  python reproduce.py leaf_ensemble3             # 5 folds x 3 CNNs; the recorded run took about 111 minutes on a T4
  ```
* **Tests without a GPU:** `pip install -r requirements-dev.txt && python -m pytest`.

The dataset (MH-SoyaHealthVision, Mendeley Data) is not redistributed; the notebook downloads it.

---

## Repository layout

```
.
├── README.md                      this file
├── LICENSE
├── .gitignore
├── .github/workflows/ci.yml       lint + tests on every push
└── soya_pipeline/                 ML module (see its README)
```
