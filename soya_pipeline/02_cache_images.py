#!/usr/bin/env python
"""Step 2 - Decode + resize every image ONCE and cache as uint8 .npy (big training-time saver).

After this step no script touches the original JPEGs again, so each epoch / experiment
skips JPEG decoding entirely. Normalisation statistics are computed on TRAIN ONLY.
"""
import json

import numpy as np
from joblib import Parallel, delayed

from common import Paths, common_args, load_resized, split_df


def channel_stats(X):
    s, ss, n = np.zeros(3), np.zeros(3), 0
    for i in range(0, len(X), 256):
        c = np.asarray(X[i:i + 256], dtype=np.float64) / 255.0
        s += c.sum((0, 1, 2))
        ss += (c ** 2).sum((0, 1, 2))
        n += c.shape[0] * c.shape[1] * c.shape[2]
    mean = s / n
    return mean.tolist(), np.sqrt(np.maximum(ss / n - mean ** 2, 0)).tolist()


def main():
    args = common_args(__doc__.splitlines()[0],
                       lambda p: p.add_argument("--force", action="store_true"))
    P = Paths(args)
    for split in ("train", "val", "test"):
        out = P.x_path(split)
        df = split_df(P, split)
        if out.exists() and not args.force:
            print(f"[skip] {out} exists")
            continue
        paths = [P.img_dir / r for r in df.relpath]
        arrs = Parallel(n_jobs=args.n_jobs, batch_size=16, verbose=1)(
            delayed(load_resized)(p, args.img_size) for p in paths)
        X = np.stack(arrs)
        np.save(out, X)
        print(f"{split}: {X.shape} -> {out} ({X.nbytes / 1e6:.0f} MB)")
        if split == "train":
            mean, std = channel_stats(X)
            P.stats_json.write_text(json.dumps({"mean": mean, "std": std, "n": len(X)}))
            print(f"train-only mean={np.round(mean, 4)} std={np.round(std, 4)}")


if __name__ == "__main__":
    main()
