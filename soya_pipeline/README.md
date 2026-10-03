# MH-SoyaHealthVision - leakage-safe, fast ML pipeline

Works for both parts of the dataset: `--dataset leaf` (6 classes) or `--dataset uav` (4 classes).

## Setup
```bash
pip install -r requirements.txt
export SOYA_ROOT=/path/to/folder   # contains Soyabean_Leaf_Image_Dataset/ and Soyabean_UAV-Based_Image_Dataset/
# no data yet? make a fake one to test the pipeline:
python make_dummy_data.py --out ./dummy_data && export SOYA_ROOT=./dummy_data
```
Class label = name of the image's parent folder. Step 1 prints the classes it found - check them.

## Run order
| # | script | what it does |
|---|--------|--------------|
| 1 | `01_build_manifest.py` | validate files, drop exact duplicates, cluster near-duplicates, grouped+stratified 70/10/20 split |
| 2 | `02_cache_images.py` | decode + resize once to `.npy` (fast draft-mode JPEG decoding); train-only mean/std |
| 3 | `03_eda.py` | class balance, resolutions, colour/ExG stats, samples, mean images (train only) |
| 4 | `04_extract_features.py` | 90 hand-crafted features (colour, HSV hist, ExG/VARI/GLI, GLCM, LBP, edges) + frozen CNN embeddings |
| 5 | `05_train_classical.py` | LogReg / SVC / RF / HistGB on hand / deep / both; picks best on VAL |
| 6 | `06_finetune_cnn.py` | pretrained MobileNetV3 / EfficientNet-B0 / ResNet; AMP, GPU augmentation, early stopping |
| 7 | `07_evaluate_test.py` | final test metrics, CIs, confusion matrices, misclassified list |
| 8 | `08_predict.py` | inference on new images; `--tile-grid 3` gives class shares for big UAV frames |

## How data leakage is prevented
- Exact duplicates removed; same bytes under two labels removed.
- Near-duplicates (pHash, invariant to flips/90-degree rotations, so augmented copies and
  consecutive drone frames are caught) are clustered; a cluster never spans train/val/test.
- Split happens once, on metadata, before any scaling / augmentation / class weights.
- Scaler / PCA live inside sklearn Pipelines fitted on train only; CV (optional) is group-aware.
- Augmentation is train-only; early stopping and model selection use VAL only.
- Scripts 05 and 06 never load the test split. Step 7 logs every test evaluation.
- Test CI is a cluster bootstrap, so correlated images don't inflate certainty.

UAV caution: if several images come from the same flight/plot, near-duplicate hashing may not
group them. If the filenames carry a flight/plot id, merge those ids into the `cluster` column in
`01_build_manifest.py` (union them with the pHash clusters) so a whole flight stays in one split.
Tune `--hash-thr` (default 6/64) if the clustering report looks off.

## How training time is kept low
Cache once, no JPEG decode afterwards; whole train set on GPU with GPU-side augmentation (no
DataLoader workers); frozen-backbone embeddings + classical models need one forward pass per
image; fine-tune uses AMP, channels_last, OneCycle, 2 head-only warm-up epochs, early stopping.
Use `--cv 0` (default) in step 5 for speed or `--cv 5` for a grouped-CV estimate;
`--img-size 160` makes everything ~2x faster (re-run step 2 onwards).
