"""Shared helpers: CLI args, artifact paths, seeding, manifest / cache / feature loading."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DATASET_DIRS = {"leaf": "Soyabean_Leaf_Image_Dataset"}      # leaf photos only (the UAV half of the dataset is out of scope)
EXPECTED_CLASSES = {"leaf": 6}
DEFAULT_FOLDS = {"leaf": 5}
BACKBONES = ["mobilenet_v3_large", "efficientnet_b0", "efficientnet_b2", "convnext_tiny",
             "resnet18", "resnet50"]                                   # can be fine-tuned (step 7)
EMBED_BACKBONES = BACKBONES + ["dinov2_vits14", "dinov2_vitb14"]       # can be used as frozen features (steps 4-6, 8)
DEFAULT_EMBED = "mobilenet_v3_large"


def common_args(description: str, extra=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASET_DIRS), default="leaf",
                   help="only 'leaf' is supported; the flag keeps the artifacts/<dataset>/ layout")
    p.add_argument("--data-root", default=os.environ.get("SOYA_ROOT", "./data"),
                   help="Folder that CONTAINS Soyabean_Leaf_Image_Dataset/ (or set env SOYA_ROOT)")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--fold", type=int, default=-1,
                   help="CV fold used as TEST (steps 5-10). -1 = the single 70/10/20 split")
    p.add_argument("--backbone", default=DEFAULT_EMBED, choices=EMBED_BACKBONES,
                   help="frozen-embedding backbone (steps 4-6)")
    if extra:
        extra(p)
    return p.parse_args()


class Paths:
    def __init__(self, args):
        self.dataset, self.size = args.dataset, args.img_size
        self.fold, self.seed = args.fold, args.seed
        self.tag = "main" if self.fold < 0 else f"fold{self.fold}"
        self.img_dir = Path(args.data_root) / DATASET_DIRS[args.dataset]
        self.root = Path(args.artifacts) / args.dataset
        self.cache, self.features, self.eda = (self.root / d for d in ("cache", "features", "eda"))
        self.models = self.root / "models" / self.tag      # per split / fold, so runs never overwrite
        self.reports = self.root / "reports" / self.tag
        for d in (self.cache, self.features, self.eda, self.models, self.reports):
            d.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.csv"
        self.classes_json = self.root / "classes.json"

    @property
    def x_path(self) -> Path:                              # ONE cache, rows in manifest order
        return self.cache / f"all_x{self.size}.npy"


def classical_bundle(P: Paths, backbone: str) -> Path:
    """Fitted classical model of one embedding backbone (the default keeps its original file name)."""
    return P.models / ("classical_best.joblib" if backbone == DEFAULT_EMBED else f"classical_best_{backbone}.joblib")


def classical_tag(bundle: dict) -> str:
    """Prediction-file stem. Default backbone: classical_<features>_<model>; others: classical-<backbone>."""
    if bundle["backbone"] == DEFAULT_EMBED:
        return f"classical_{bundle['feature_set']}_{bundle['model']}"
    return f"classical-{bundle['backbone']}"


def expand_paths(patterns) -> list[str]:
    """Expand wildcards ourselves (Windows shells do not), keep order, drop duplicates, fail on a pattern that matches nothing."""
    out: list[str] = []
    for pat in patterns:
        hits = sorted(glob.glob(pat)) if any(ch in pat for ch in "*?[") else [pat]
        if not hits:
            raise SystemExit(f"no file matches '{pat}'")
        out += [h for h in hits if h not in out]
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = sys.modules.get("torch")                       # only seed torch if already imported
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def check_fold_coverage(df: pd.DataFrame, folds: int | None = None) -> None:
    """Every class needs at least 2 clusters outside each test fold (one for train, one for val)."""
    folds = int(df.fold.max()) + 1 if folds is None else folds
    for f in range(folds):
        outside = df[df.fold != f].groupby("label").cluster.nunique()
        short = [f"{lab} ({int(outside.get(lab, 0))})" for lab in sorted(df.label.unique()) if outside.get(lab, 0) < 2]
        if short:
            raise SystemExit(f"fold {f}: too few independent clusters outside the test fold for: {', '.join(short)}. "
                             "Use fewer --folds, or relax --hash-thr / --group-regex so classes have more clusters.")


def _fold_split(df: pd.DataFrame, fold: int, seed: int) -> pd.DataFrame:
    """test = fold; val = ~1/6 of each class's remaining groups (at least one, so every class is
    present in val); train = the rest. Deterministic given (seed, fold)."""
    if "fold" not in df:
        raise SystemExit("manifest has no 'fold' column - rerun 01_build_manifest.py")
    k = int(df.fold.max()) + 1
    if not 0 <= fold < k:
        raise SystemExit(f"--fold must be in 0..{k - 1}")
    test = (df.fold == fold).to_numpy()
    rng = np.random.default_rng(seed * 1000 + fold)
    val_clusters = []
    for _, g in df[~test].groupby("label_id"):
        cl = np.sort(g.cluster.unique())
        rng.shuffle(cl)
        val_clusters += list(cl[:max(1, round(len(cl) / 6))])
    split = np.where(test, "test", np.where(df.cluster.isin(val_clusters), "val", "train"))
    counts = pd.crosstab(df.label, split)
    if counts.shape[1] != 3 or (counts == 0).any().any():
        raise SystemExit(f"fold {fold}: a class is missing from train, val or test:\n{counts}")
    return df.assign(split=split)


def manifest_md5(P: Paths) -> str:
    return hashlib.md5(P.manifest.read_bytes()).hexdigest()


def _meta_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def write_meta(P: Paths, path: Path, **extra) -> None:
    """Fingerprint a cache / feature file with the manifest it was built from."""
    _meta_path(path).write_text(json.dumps({"manifest_md5": manifest_md5(P), **extra}))


def meta_state(P: Paths, path: Path) -> str:
    """'ok' | 'missing' (built before fingerprints existed) | 'stale' (manifest changed since)."""
    mp = _meta_path(path)
    if not mp.exists():
        return "missing"
    return "ok" if json.loads(mp.read_text())["manifest_md5"] == manifest_md5(P) else "stale"


def check_meta(P: Paths, path: Path) -> None:
    state = meta_state(P, path)
    if state == "stale":
        raise SystemExit(f"{path.name} was built from a different manifest - rerun steps 2 and 4 (--force for 2)")
    if state == "missing":
        warnings.warn(f"{path.name} has no manifest fingerprint; rebuild it to enable the staleness check")


def load_manifest(P: Paths) -> pd.DataFrame:
    if not P.manifest.exists():
        raise SystemExit(f"{P.manifest} not found - run 01_build_manifest.py first")
    df = pd.read_csv(P.manifest, keep_default_na=False, na_values=[])
    df["row"] = np.arange(len(df))                         # position in the cache / feature arrays
    return df if P.fold < 0 else _fold_split(df, P.fold, P.seed)


def load_classes(P: Paths) -> list[str]:
    return json.loads(P.classes_json.read_text())


def split_df(P: Paths, split: str) -> pd.DataFrame:
    m = load_manifest(P)
    return m[m.split == split].reset_index(drop=True)


def load_xy(P: Paths, split: str):
    """uint8 images (N,H,W,3), int labels and the manifest rows of one split (same order)."""
    if not P.x_path.exists():
        raise SystemExit(f"{P.x_path} not found - run 02_cache_images.py --img-size {P.size}")
    check_meta(P, P.x_path)
    X = np.load(P.x_path, mmap_mode="r")
    df = split_df(P, split)
    assert len(X) == len(load_manifest(P)), "cache / manifest mismatch - rerun 02_cache_images.py --force"
    return np.ascontiguousarray(X[df.row.to_numpy()]), df.label_id.to_numpy(np.int64), df


def load_resized(path, size: int) -> np.ndarray:
    """Fast, consistent image loader (JPEG draft mode speeds up decoding of large photos)."""
    with Image.open(path) as im:
        im.draft("RGB", (size * 2, size * 2))
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def _load_feature_file(P: Paths, path: Path) -> np.ndarray:
    A = np.load(path)                                    # FileNotFoundError is handled by the callers
    check_meta(P, path)
    n = len(load_manifest(P))
    if len(A) != n:
        raise SystemExit(f"{path.name} has {len(A)} rows but the manifest has {n} - rerun step 4")
    return A


def load_features(P: Paths, split: str, feature_set: str, backbone: str) -> np.ndarray:
    rows = split_df(P, split).row.to_numpy()
    parts = []
    if feature_set in ("hand", "both"):
        parts.append(_load_feature_file(P, P.features / "all_hand.npy")[rows])
    if feature_set in ("deep", "both"):
        parts.append(_load_feature_file(P, P.features / f"all_deep_{backbone}.npy")[rows])
    return np.concatenate(parts, axis=1).astype(np.float32)
