#!/usr/bin/env python
"""Step 8 - Predict with the fine-tuned CNN on new images (file or folder).

--tile-grid N cuts each image into an NxN grid and classifies every tile: useful for big UAV
frames, gives the share of each class (healthy / disease / pest) across the field.
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from common import IMG_EXT, Paths, common_args, load_resized
from models import forward_batches, get_device, load_checkpoint
from pathlib import Path


def tiles_of(path, grid, size):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        out = []
        for r in range(grid):
            for c in range(grid):
                box = (c * w // grid, r * h // grid, (c + 1) * w // grid, (r + 1) * h // grid)
                out.append(np.asarray(im.crop(box).resize((size, size), Image.Resampling.BILINEAR)))
        return np.stack(out)


def main():
    def extra(p):
        p.add_argument("--input", required=True, help="image file or folder")
        p.add_argument("--ckpt", default=None, help="default: <artifacts>/<dataset>/models/cnn_best.pt")
        p.add_argument("--topk", type=int, default=3)
        p.add_argument("--tile-grid", type=int, default=0)
        p.add_argument("--out", default=None, help="optional CSV with predictions")

    args = common_args(__doc__.splitlines()[0], extra)
    P = Paths(args)
    device = get_device()
    model, ck = load_checkpoint(args.ckpt or P.models / "cnn_best.pt", device)
    classes, size = ck["classes"], ck["img_size"]

    src = Path(args.input)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    rows = []
    for f in files:
        if args.tile_grid:
            X = tiles_of(f, args.tile_grid, size)
            pred = forward_batches(model, X, device, ck["mean"], ck["std"]).argmax(1)
            share = np.bincount(pred, minlength=len(classes)) / len(pred)
            print(f"{f.name}: " + ", ".join(f"{classes[k]} {share[k]:.0%}" for k in np.argsort(-share) if share[k] > 0))
            rows.append({"file": str(f), **{c: share[i] for i, c in enumerate(classes)}})
        else:
            X = load_resized(f, size)[None]
            logits = forward_batches(model, X, device, ck["mean"], ck["std"])[0]
            p = np.exp(logits - logits.max()); p /= p.sum()
            top = np.argsort(-p)[:args.topk]
            print(f"{f.name}: " + ", ".join(f"{classes[k]} {p[k]:.1%}" for k in top))
            rows.append({"file": str(f), "pred": classes[top[0]], "confidence": float(p[top[0]])})
    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
