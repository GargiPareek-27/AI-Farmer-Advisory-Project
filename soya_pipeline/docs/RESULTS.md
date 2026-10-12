# Results and provenance

This is the authoritative record of the numbers. Unless stated otherwise, every number comes from the **executed outputs of one recorded Colab run** of the
`leaf_ensemble3` recipe (section 1) and is a **pooled out-of-fold macro-F1** over 5 stratified folds in which every image is tested exactly once.
95 % confidence intervals are bootstrap intervals over near-duplicate *clusters* (1,000 resamples). Exact values are given to four decimals; prose may round
(for example 0.7602 is "about 0.76").

## 1. Provenance of the recorded run

| Item | Value |
|---|---|
| Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| PyTorch | 2.11.0+cu130 (torchvision 0.26.0+cu130) |
| Other libraries (`pip freeze`) | numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, scikit-image 0.25.2, opencv-python-headless 5.0.0.93, pillow 11.3.0, joblib 1.6.0, matplotlib 3.10.0 |
| When | October 2026 |
| Recipe | `leaf_ensemble3` (`python reproduce.py leaf_ensemble3 --resume --sync-dir ...`), 55 steps, 5 folds, 320 px |
| Duration | about 110.9 minutes |
| Steps skipped by `--resume` | 0 (everything was executed in this run) |
| Code source | the unpacked project folder in Google Drive; **no zip md5 recorded, no git commit (`commit None`)** |
| Tests in the same session | 24 passed (191 s); GPU smoke test passed |
| Seed | `--seed 42` (default); GPU training is not bit-exact |
| Pretrained weights | torchvision ImageNet weights; DINOv2-S/14 from `torch.hub` (experiment only) |

Training settings of the CNN members: up to 30 epochs with early stopping on validation macro-F1 (patience 8), batch size 64, AdamW (weight decay 1e-4),
backbone frozen for 2 epochs, head LR 2e-3, backbone LR 3e-4 (1e-4 for ConvNeXt-Tiny), class-weighted cross-entropy, label smoothing 0.1, mixed precision,
GPU-side augmentation, CutMix (Beta alpha 1.0, probability 0.5) in one member.

## 2. Data and splits

| Item | Value |
|---|---|
| Leaf images found | 2,782 |
| Files dropped by cleaning | 26 (reasons in `dropped.csv`) |
| Images retained | **2,756** |
| Classes | 6 |
| Near-duplicate clusters | 2,756 clusters for 2,756 images (largest cluster: 1): no near-duplicate pair at the default 6-bit threshold |
| Source groups | 2,756 (every image is its own group; file names carry no source id) |
| Leakage checks (step 1) | passed: no cluster, source group or identical file crosses a split or fold |
| Folds | 5, stratified; fold sizes 552 / 551 / 551 / 551 / 551; every image tested exactly once (pooled N = 2,756) |
| Main split (used for the audit) | train 1,928 / validation 276 / test 552 |

| Class | Images | Main split train / val / test |
|---|---:|---|
| Caterpillar and Semilooper Pest Attack | 577 | 404 / 57 / 116 |
| Healthy_Soyabean | 204 | 142 / 21 / 41 |
| Soyabean_Frog_Leaf_Eye | 168 | 118 / 17 / 33 |
| Soyabean_Mosaic | 694 | 487 / 69 / 138 |
| Soyabean_Rust | 845 | 590 / 85 / 170 |
| Soyabean_Spectoria_Brown_Spot | 268 | 187 / 27 / 54 |
| **Total** | **2,756** | **1,928 / 276 / 552** |

Class imbalance (largest / smallest in the training split): 5.00x.

## 3. Main results: model comparison

| Model | Macro-F1 | 95 % CI | Accuracy |
|---|---:|---:|---:|
| Classical (hand-crafted + frozen MobileNetV3 features) | **0.6904** | 0.6709 - 0.7085 | **0.7242** |
| EfficientNet-B0, 320 px + TTA | **0.7340** | 0.7183 - 0.7505 | **0.7485** |
| EfficientNet-B0, 320 px + CutMix + TTA | **0.7290** | 0.7124 - 0.7462 | **0.7395** |
| ConvNeXt-Tiny, 320 px + TTA | **0.7511** | 0.7344 - 0.7687 | **0.7656** |
| **3-CNN ensemble + TTA** (EfficientNet-B0 + EfficientNet-B0/CutMix + ConvNeXt-Tiny) | **0.7602** | **0.7438 - 0.7770** | **0.7747** |

Reading the comparison:

* The ensemble is +0.0262 over the EfficientNet-B0 baseline, +0.0091 over the best single model (ConvNeXt-Tiny) and +0.0698 over the classical pipeline.
* It beat the EfficientNet-B0 baseline, the CutMix variant and the classical pipeline on **all five folds**, and ConvNeXt-Tiny on **four of five**
  (fold 2: ensemble 0.7552, ConvNeXt-Tiny 0.7909).
* The ensemble's interval (0.7438 - 0.7770) overlaps ConvNeXt-Tiny's (0.7344 - 0.7687): its edge over the best single model is modest and not conclusive.
* CutMix alone did not improve on the baseline in this run (0.7290 vs 0.7340; better on 1 of 5 folds). Its contribution to the ensemble was not ablated.
* Headline: **0.7602 pooled macro-F1 (95 % CI 0.7438 - 0.7770), accuracy 0.7747** (about 0.760, 0.744 - 0.777 and 0.775 in prose).

## 4. Cross-validation details

Macro-F1 per test fold (5 folds, N = 2,756):

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | SD | Pooled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical | 0.6709 | 0.6983 | 0.7004 | 0.6972 | 0.6822 | 0.6898 | 0.0128 | 0.6904 |
| EfficientNet-B0 + TTA | 0.7206 | 0.7407 | 0.7424 | 0.7413 | 0.7221 | 0.7334 | 0.0110 | 0.7340 |
| EfficientNet-B0 + CutMix + TTA | 0.7389 | 0.7399 | 0.7114 | 0.7388 | 0.7143 | 0.7287 | 0.0145 | 0.7290 |
| ConvNeXt-Tiny + TTA | 0.7483 | 0.7467 | 0.7909 | 0.7416 | 0.7244 | 0.7504 | 0.0245 | 0.7511 |
| **3-CNN ensemble + TTA** | **0.7647** | **0.7675** | **0.7552** | **0.7642** | **0.7456** | **0.7594** | **0.0090** | **0.7602** |

Ensemble (`cnn_ensemble3_tta`), exact: mean fold macro-F1 0.7594196, pooled macro-F1 0.7602121, 95 % CI 0.7438344 - 0.7769567, pooled accuracy 0.7746734.
The ensemble also has the smallest spread across folds (SD 0.0090).

## 5. Per-class results

3-CNN ensemble + TTA, pooled out-of-fold (N = 2,756):

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Caterpillar and Semilooper Pest Attack | 0.761 | 0.763 | **0.762** | 577 |
| Healthy_Soyabean | 0.995 | 1.000 | **0.998** | 204 |
| Soyabean_Frog_Leaf_Eye | 0.535 | 0.589 | **0.561** | 168 |
| Soyabean_Mosaic | 0.882 | 0.798 | **0.838** | 694 |
| Soyabean_Rust | 0.804 | 0.761 | **0.782** | 845 |
| Soyabean_Spectoria_Brown_Spot | 0.542 | 0.728 | **0.621** | 268 |

Overall: macro precision 0.753, macro recall (= balanced accuracy) 0.773, **macro-F1 0.760**, weighted F1 0.779, **accuracy 0.775**.

Per-class F1 of all models (pooled out-of-fold):

| Class | Classical | EfficientNet-B0 | EfficientNet-B0 + CutMix | ConvNeXt-Tiny | 3-CNN ensemble |
|---|---:|---:|---:|---:|---:|
| Caterpillar and Semilooper Pest Attack | 0.691 | 0.717 | 0.716 | 0.744 | **0.762** |
| Healthy_Soyabean | 0.985 | 0.998 | 0.995 | 0.995 | **0.998** |
| Soyabean_Frog_Leaf_Eye | 0.475 | 0.532 | 0.538 | 0.529 | **0.561** |
| Soyabean_Mosaic | 0.786 | 0.823 | 0.821 | 0.833 | **0.838** |
| Soyabean_Rust | 0.742 | 0.760 | 0.748 | 0.774 | **0.782** |
| Soyabean_Spectoria_Brown_Spot | 0.463 | 0.575 | 0.555 | **0.632** | 0.621 |

Frog-eye leaf spot and Septoria brown spot are the weak classes for every model. ConvNeXt-Tiny is slightly better than the ensemble on Septoria (0.632 vs 0.621).

## 6. Confusion matrix

3-CNN ensemble + TTA, pooled out-of-fold (rows = true class, columns = predicted class):

```text
                                        Caterpilla  Healthy_So  Soyabean_F  Soyabean_M  Soyabean_R  Soyabean_S
Caterpillar and Semilooper Pest Attack         440           0          16          42          46          33
Healthy_Soyabean                                 0         204           0           0           0           0
Soyabean_Frog_Leaf_Eye                           9           0          99           2          38          20
Soyabean_Mosaic                                 68           1           6         554          29          36
Soyabean_Rust                                   52           0          52          22         643          76
Soyabean_Spectoria_Brown_Spot                    9           0          12           8          44         195
```

Column abbreviations: Caterpilla = Caterpillar and Semilooper Pest Attack; Healthy_So = Healthy_Soyabean; Soyabean_F = Frog-eye; Soyabean_M = Mosaic; Soyabean_R = Rust;
Soyabean_S = Septoria brown spot.

* All 204 healthy leaves are classified correctly; one mosaic leaf is predicted healthy.
* Largest confusions: rust -> Septoria (76 of 845), mosaic -> caterpillar / semilooper (68 of 694), rust -> Frog-eye (52), rust -> caterpillar / semilooper (52),
  caterpillar / semilooper -> rust (46), Septoria -> rust (44 of 268), Frog-eye -> rust (38 of 168).
* The diagonal sums to 2,135 of 2,756 images (accuracy 0.7747).

## 7. Classical baselines

Step 6 evaluated Logistic Regression, RBF-SVC and HistGradientBoosting (class-balanced) on three feature sets, selecting per fold by validation macro-F1:

| Feature set | Features |
|---|---:|
| Hand-crafted (colour, vegetation indices, texture) | 90 |
| Frozen MobileNetV3-Large embedding | 960 |
| Combined | 1,050 |

RBF-SVC was selected on every fold: deep features on folds 0 and 3, combined features on folds 1, 2 and 4. On fold 0 the validation macro-F1 ranged from 0.5603 (hand-crafted +
HistGB) to 0.7095 (deep + SVC). Pooled result: macro-F1 **0.6904** (95 % CI 0.6709 - 0.7085), accuracy **0.7242**; per-class F1: 0.691, 0.985, 0.475, 0.786, 0.742, 0.463
(order as in section 5).

## 8. Audit of the split (main split, step 5)

| Check | Result |
|---|---|
| (a) share of validation / test images whose source group is in train | 0.0 % / 0.0 % (by construction; every image is its own group) |
| (b) nearest-train cosine similarity, 5/25/50/75/95th percentile (test) | 0.821, 0.852, 0.871, 0.890, 0.915 |
| (b) 1-NN macro-F1 (plain lookup on frozen features) | 0.588 (validation), 0.594 (test) |
| (c) mean-RGB-only probe, test macro-F1 | 0.277 (chance 0.17) |

A plain nearest-neighbour lookup scores far below the fine-tuned models, so the task is not solved by lookup. This does **not** prove that photos of the same plant are
absent from both sides of a split (see section 13).

## 9. DINOv2 experiment (notebook only, not part of the final model)

Frozen DINOv2-S/14 features (384-dimensional, images resized to 224 px) with an SVC (C chosen by an inner grouped 3-fold search; C = 1 on every fold):

| Model | Pooled macro-F1 | 95 % CI |
|---|---:|---|
| DINOv2-S/14 frozen features + SVC | **0.6837** | 0.665 - 0.702 |
| CNN baseline only (EfficientNet-B0 + TTA) | 0.7340 | 0.716 - 0.751 |
| 50/50 probability blend: EfficientNet-B0 + DINOv2 SVC | 0.7514 | 0.735 - 0.768 |

Per-fold DINOv2 macro-F1: 0.6556, 0.6981, 0.6982, 0.6856, 0.6806 (mean 0.6836). DINOv2 alone is below every fine-tuned CNN, and the blend is below the 3-CNN ensemble
(0.7602). DINOv2 is **not** the best model. The blend uses fixed 50/50 weights (not tuned on the test folds).

## 10. Earlier interactive runs (superseded)

Before the recorded run, the same commands (now encoded as recipes) were executed interactively in Colab and documented at three decimals. They are kept only to show run-to-run variation:

| Item | Earlier interactive run | Recorded run (this document) |
|---|---|---|
| EfficientNet-B0, 320 px + TTA, pooled macro-F1 | 0.733 (CI 0.716 - 0.749) | 0.7340 (CI 0.7183 - 0.7505) |
| 3-CNN ensemble + TTA, pooled macro-F1 | 0.761 (CI 0.744 - 0.778), accuracy 0.775 | 0.7602 (CI 0.7438 - 0.7770), accuracy 0.7747 |
| EfficientNet-B0 + CutMix, two-model ensemble | 0.742 (CI 0.726 - 0.760) | not part of the recorded run |
| Single 70/10/20 split, EfficientNet-B0 + TTA, test n = 552 | macro-F1 0.742 (CI 0.704 - 0.780), accuracy 0.746, ECE 0.073, ROC-AUC 0.937 | not part of the recorded run |

The two executions agree to within 0.002 pooled macro-F1 for the baseline and the ensemble (the earlier values were documented at three decimals), consistent with the +/-0.01 rerun variation expected from GPU non-determinism.
The training scripts were refactored between the two executions without changing the training logic. The 0.761 above is the earlier run's rounded value, not the exact value of the recorded run.

## 11. Out of scope: UAV subset

The UAV half of MH-SoyaHealthVision was explored earlier and is **not part of this repository's claims**. Its code is not included in this version of the project. It is kept here only as a record of why the
leakage guards exist.

| Experiment | Result | Why it does not count |
|---|---|---|
| Random frame split | macro-F1 about 0.99 | consecutive frames of one video on both sides of the split |
| pHash-only grouping | about 90 % of val/test frames still shared a video with train | pHash does not link every pair of frames of one video |
| Video-grouped 3-fold CV | 0.876 (CNN), 0.909 (classical); a training-free 1-NN lookup scored 0.873 | classes were recorded on different dates, so a model can score well by recognising the session |

## 12. Reproducibility status

| Question | Status |
|---|---|
| Was the pipeline actually executed end to end on the real data? | **Yes.** One recorded Colab run: Tesla T4, PyTorch 2.11.0+cu130, 55 steps, about 110.9 minutes, 0 steps skipped; 24 tests and the GPU smoke test passed in the same session. |
| Do the numbers in this document come from that run? | **Yes**, from its printed outputs (sections 2 - 9). |
| Is the code version identifiable? | **Not exactly.** The run used the unpacked project folder in Google Drive; no zip md5 and no commit hash were recorded. |
| Has an independent clean-ZIP reproduction been done? | **No, pending.** It would run `notebooks/soya_reproduce.ipynb` from a fresh runtime on the submitted zip and record its md5. |
| Pinned library versions | The versions of the recorded run are listed in section 1; `requirements-lock.txt` is not yet part of the project. |
| Expected variation between reruns | about +/-0.01 macro-F1 (GPU non-determinism); the two executions in section 10 differed by less than 0.002. |

Clean-ZIP reproduction log (add one row per run; leave blank until a run exists):

| Date | Source (zip md5) | GPU / torch | Recipe | Pooled macro-F1 | 95 % CI | Delta vs recorded run | Tests passed |
|---|---|---|---|---|---|---|---|
| | | | `leaf_baseline` | | | (recorded 0.7340) | |
| | | | `leaf_ensemble3` | | | (recorded 0.7602) | |

## 13. Limitations

* **Source overlap cannot be ruled out.** Leaf file names carry no plant / field / session id, so every image is its own group. Perceptual hashing found no near-duplicate pairs, but
  near-duplicate detection cannot guarantee plant-level independence; photos of the same plant could sit on both sides of a split and make the scores somewhat optimistic.
* **No external validation dataset.** All numbers come from cross-validation on one dataset.
* **Weaker classes.** Frog-eye leaf spot (F1 0.561) and Septoria brown spot (0.621) are not reliable enough for unsupervised use.
* **The ensemble result may be somewhat optimistic.** Several variants were compared on the same folds (three single models, an ensemble, a DINOv2 experiment and a blend).
  The ensemble was a planned experiment, not picked after looking at the test folds, but a selection effect cannot be ruled out.
* **Not a diagnostic or advisory authority.** The model outputs a class label only: no confidence-based rejection, severity estimate or treatment logic.
* **Test coverage.** The PyTorch training / evaluation paths are not covered by CI.
* **GPU training is not bit-exact;** reruns can vary by roughly +/-0.01 macro-F1.
