#!/usr/bin/env python
"""Step 10 - Predict new images (file or folder) with one or several fine-tuned CNNs.

--ckpt accepts several files (e.g. the CV fold models, artifacts/leaf/models/fold*/cnn_*.pt):
their probabilities are averaged, which is the cheapest ensemble you can get from a CV run.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import IMG_EXT, Paths, common_args, expand_paths, load_resized
from metrics import softmax
from models import forward_batches, get_device, load_checkpoint


def main():
    def extra(p):
        p.add_argument("--input", required=True, help="image file or folder")
        p.add_argument("--ckpt", nargs="+", default=None, help="checkpoint file(s); wildcards allowed; default: the best-validation CNN found under artifacts/<dataset>/models")
        p.add_argument("--topk", type=int, default=3)
        p.add_argument("--out", default=None, help="optional CSV with predictions")

    args = common_args(__doc__.splitlines()[0], extra)
    P = Paths(args)
    device = get_device()
    if not Path(args.input).exists():
        raise SystemExit(f"--input {args.input} does not exist")
    if args.ckpt is None:
        metas = {f: json.loads(f.read_text()) for f in sorted((P.root / "models").glob("*/cnn_*.json"))
                 if f.with_suffix(".pt").exists()}
        if not metas:
            raise SystemExit(f"no CNN checkpoint under {P.root / 'models'}; train one or pass --ckpt")
        best = max(metas, key=lambda f: metas[f]["val_macro_f1"])
        args.ckpt = [str(best.with_suffix(".pt"))]
        print(f"using {args.ckpt[0]} (validation macro-F1 {metas[best]['val_macro_f1']:.4f}); pass several --ckpt to ensemble")
    else:
        args.ckpt = expand_paths(args.ckpt)
    loaded = [load_checkpoint(c, device) for c in args.ckpt]
    classes, size = loaded[0][1]["classes"], loaded[0][1]["img_size"]
    assert all(ck["classes"] == classes and ck["img_size"] == size for _, ck in loaded), "checkpoints disagree"

    def proba(X):
        return np.mean([softmax(forward_batches(m, X, device, ck["mean"], ck["std"])) for m, ck in loaded], 0)

    src = Path(args.input)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    rows = []
    for f in files:
        p = proba(load_resized(f, size)[None])[0]
        top = np.argsort(-p)[:args.topk]
        print(f"{f.name}: " + ", ".join(f"{classes[k]} {p[k]:.1%}" for k in top))
        rows.append({"file": str(f), "pred": classes[top[0]], "confidence": float(p[top[0]])})
    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
