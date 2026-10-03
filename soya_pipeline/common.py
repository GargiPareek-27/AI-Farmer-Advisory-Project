"""Shared helpers: CLI args, artifact paths, seeding, manifest / cache / feature loading."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DATASET_DIRS = {
    "leaf": "Soyabean_Leaf_Image_Dataset",
    "uav": "Soyabean_UAV-Based_Image_Dataset",
}
EXPECTED_CLASSES = {"leaf": 6, "uav": 4}


def common_args(description: str, extra=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--dataset", choices=sorted(DATASET_DIRS), required=True)
    p.add_argument(
        "--data-root",
        default=os.environ.get("SOYA_ROOT", "./data"),
        help="Folder that CONTAINS the two dataset folders (or set env SOYA_ROOT)",
    )
    p.add_argument("--artifacts", default="artifacts", help="Output folder for everything")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    if extra:
        extra(p)
    return p.parse_args()


class Paths:
    def __init__(self, args):
        self.dataset = args.dataset
        self.size = args.img_size
        self.img_dir = Path(args.data_root) / DATASET_DIRS[args.dataset]
        self.root = Path(args.artifacts) / args.dataset
        self.cache = self.root / "cache"
        self.features = self.root / "features"
        self.models = self.root / "models"
        self.reports = self.root / "reports"
        self.eda = self.root / "eda"
        for d in (self.cache, self.features, self.models, self.reports, self.eda):
            d.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.csv"
        self.classes_json = self.root / "classes.json"
        self.stats_json = self.cache / "stats.json"

    def x_path(self, split: str) -> Path:
        return self.cache / f"{split}_x{self.size}.npy"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import sys

    torch = sys.modules.get("torch")  # only seed torch if the calling script already imported it
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_manifest(P: Paths) -> pd.DataFrame:
    if not P.manifest.exists():
        raise SystemExit(f"{P.manifest} not found - run 01_build_manifest.py first")
    return pd.read_csv(P.manifest)


def load_classes(P: Paths) -> list[str]:
    return json.loads(P.classes_json.read_text())


def split_df(P: Paths, split: str) -> pd.DataFrame:
    m = load_manifest(P)
    return m[m.split == split].reset_index(drop=True)


def load_xy(P: Paths, split: str, mmap: bool = True):
    """Cached uint8 images (N,H,W,3), int labels, and the manifest rows (same order)."""
    df = split_df(P, split)
    if not P.x_path(split).exists():
        raise SystemExit(f"{P.x_path(split)} not found - run 02_cache_images.py first")
    X = np.load(P.x_path(split), mmap_mode="r" if mmap else None)
    assert len(X) == len(df), "cache / manifest mismatch - rerun 02_cache_images.py --force"
    return X, df.label_id.to_numpy(np.int64), df


def load_resized(path, size: int) -> np.ndarray:
    """Fast, consistent image loader (JPEG draft-mode decoding makes big UAV images ~5x faster)."""
    with Image.open(path) as im:
        im.draft("RGB", (size * 2, size * 2))
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def load_features(P: Paths, split: str, feature_set: str, backbone: str) -> np.ndarray:
    hand = P.features / f"{split}_hand.npy"
    deep = P.features / f"{split}_deep_{backbone}.npy"
    parts = []
    if feature_set in ("hand", "both"):
        parts.append(np.load(hand))
    if feature_set in ("deep", "both"):
        parts.append(np.load(deep))
    return np.concatenate(parts, axis=1).astype(np.float32)
