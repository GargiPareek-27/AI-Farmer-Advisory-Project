#!/usr/bin/env python
"""Create a tiny fake dataset (same folder layout) to smoke-test the pipeline in a minute.
Includes a few exact and flipped duplicates so you can see the leakage guards working."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--out", default="./dummy_data")
p.add_argument("--per-class", type=int, default=40)
a = p.parse_args()

SPEC = {"Soyabean_Leaf_Image_Dataset": ["Healthy", "Rust", "Mosaic_Virus", "Septoria_Brown_Spot", "Frogeye_Leaf_Spot", "Pest_Attack"],
        "Soyabean_UAV-Based_Image_Dataset": ["Healthy", "Rust", "Mosaic_Virus", "Pest_Attack"]}
rng = np.random.default_rng(0)
for ds, classes in SPEC.items():
    for ci, c in enumerate(classes):
        d = Path(a.out) / ds / c
        d.mkdir(parents=True, exist_ok=True)
        base = np.array([40 + 20 * ci, 120 - 8 * ci, 40 + 10 * ci], dtype=float)
        for i in range(a.per_class):
            img = np.clip(base + rng.normal(0, 25, (240, 320, 3)) + 30 * rng.random((1, 1, 3)), 0, 255).astype(np.uint8)
            for _ in range(ci + 1):  # class-specific "lesions"
                y, x = rng.integers(0, 200), rng.integers(0, 280)
                img[y:y + 20, x:x + 20] = (150, 90, 40)
            Image.fromarray(img).save(d / f"img_{i:03d}.jpg", quality=92)
        Image.fromarray(img[:, ::-1]).save(d / "img_flipped_copy.jpg", quality=92)   # near-dup (flip)
        (d / "img_exact_copy.jpg").write_bytes((d / "img_000.jpg").read_bytes())      # exact dup
print("done ->", a.out)
