#!/usr/bin/env python
"""Tiny fake leaf dataset (same folder layout as the real one) to smoke-test the pipeline.
Includes one exact copy and one flipped copy per class, to exercise de-duplication and clustering."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--out", default="./dummy_data")
p.add_argument("--per-class", type=int, default=40)
a = p.parse_args()

LEAF = ["Healthy", "Rust", "Mosaic_Virus", "Septoria_Brown_Spot", "Frogeye_Leaf_Spot", "Pest_Attack"]
rng = np.random.default_rng(0)


def base_img(ci):
    base = np.array([40 + 20 * ci, 120 - 8 * ci, 40 + 10 * ci], dtype=float)
    smooth = np.kron(rng.random((4, 5)), np.ones((30, 32)))[:, :, None]      # random low-frequency layout
    img = np.clip(base + rng.normal(0, 12, (120, 160, 3)) + 90 * (smooth - 0.5) + 30 * rng.random((1, 1, 3)), 0, 255)
    for _ in range(ci + 1):
        y, x = rng.integers(0, 100), rng.integers(0, 140)
        img[y:y + 12, x:x + 12] = (150, 90, 40)
    return img


def save(img, path):
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path, quality=92)


for ci, c in enumerate(LEAF):
    d = Path(a.out, "Soyabean_Leaf_Image_Dataset", c); d.mkdir(parents=True, exist_ok=True)
    for i in range(a.per_class):
        save(base_img(ci), d / f"img_{i:03d}.jpg")
    (d / "img_exact_copy.jpg").write_bytes((d / "img_000.jpg").read_bytes())
    img = np.asarray(Image.open(d / "img_001.jpg"))[:, ::-1]
    save(img, d / "img_flipped_copy.jpg")

print("done ->", a.out)
