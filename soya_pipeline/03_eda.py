#!/usr/bin/env python
"""Step 3 - Exploratory data analysis (run once, without --fold).

Tables first, because they decided the evaluation design on this dataset:
  * independent source groups per class (few groups = few truly independent samples);
  * class balance per split. Pixel statistics and sample grids use TRAIN only; the grid shows
    ONE image per source group.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import Paths, common_args, load_classes, load_manifest, load_xy


def per_image_stats(X):
    out = []
    for i in range(0, len(X), 256):
        c = np.asarray(X[i:i + 256], dtype=np.float32) / 255.0
        r, g, b = c[..., 0], c[..., 1], c[..., 2]
        out.append(np.stack([c.mean((1, 2, 3)), (2 * g - r - b).mean((1, 2))], 1))
    return np.concatenate(out)


def main():
    args = common_args(__doc__.splitlines()[0])
    P = Paths(args)
    classes, K = load_classes(P), len(load_classes(P))
    m = load_manifest(P)
    tr = m[m.split == "train"]
    summary = {"n_images": len(m)}

    ct = pd.crosstab(m.label, m.split)[["train", "val", "test"]]
    counts = tr.label.value_counts()
    summary["imbalance_max_over_min"] = float(counts.max() / counts.min())
    print("Images per class and split\n", ct.to_string(), f"\nImbalance (train): {summary['imbalance_max_over_min']:.2f}x")

    src = m.groupby("label").agg(images=("relpath", "size"), source_groups=("group", "nunique"),
                                 max_group_size=("group", lambda s: s.value_counts().max()))
    print("\nSource groups per class (without --group-regex every image is its own group)\n", src.to_string())

    ax = ct.plot(kind="bar", figsize=(8, 4), title="Images per class and split")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(P.eda / "class_distribution.png", dpi=130); plt.close()
    summary["resolutions"] = m[["width", "height"]].drop_duplicates().shape[0]

    X, y, trd = load_xy(P, "train")
    st = per_image_stats(X)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, v, t in zip(ax, st.T, ("Mean brightness", "Excess-green (2G-R-B)")):
        a.boxplot([v[y == k] for k in range(K)], showfliers=False)
        a.set_xticklabels(classes, rotation=30, ha="right"); a.set_title(t)
    plt.tight_layout(); plt.savefig(P.eda / "color_stats.png", dpi=130); plt.close()

    rng = np.random.default_rng(args.seed)
    n = 6
    fig, ax = plt.subplots(K, n, figsize=(n * 1.8, K * 1.8))
    for k in range(K):
        one_per_group = trd[trd.label_id == k].groupby("group").head(1)
        pick = one_per_group.index.to_numpy()
        pick = rng.choice(pick, min(n, len(pick)), replace=False)
        for j in range(n):
            ax[k, j].axis("off")
            if j < len(pick):
                ax[k, j].imshow(X[pick[j]])
            if j == 0:
                ax[k, j].set_title(classes[k], fontsize=7, loc="left")
    plt.tight_layout(); plt.savefig(P.eda / "samples_one_per_group.png", dpi=110); plt.close()

    (P.eda / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nEDA written to {P.eda}/")


if __name__ == "__main__":
    main()
