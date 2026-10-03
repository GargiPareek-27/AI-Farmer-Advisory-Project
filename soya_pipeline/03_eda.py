#!/usr/bin/env python
"""Step 3 - Exploratory data analysis.

Split sizes are shown for all splits; every content statistic (colour, brightness,
sample grids...) uses the TRAIN split only, so insights never come from val/test data.
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
        out.append(np.stack([c.mean((1, 2, 3)), (2 * g - r - b).mean((1, 2)),
                             r.mean((1, 2)), g.mean((1, 2)), b.mean((1, 2))], 1))
    return np.concatenate(out)


def boxplot(ax, values, y, classes, title):
    ax.boxplot([values[y == k] for k in range(len(classes))], showfliers=False)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_title(title)


def main():
    args = common_args(__doc__.splitlines()[0])
    P = Paths(args)
    classes, K = load_classes(P), len(load_classes(P))
    m = load_manifest(P)
    tr = m[m.split == "train"]
    summary = {"n_total": len(m), "n_per_split": m.split.value_counts().to_dict()}

    # 1) class balance per split -------------------------------------------------------
    ct = pd.crosstab(m.label, m.split)[["train", "val", "test"]]
    ax = ct.plot(kind="bar", figsize=(8, 4), title="Images per class and split")
    ax.set_ylabel("images"); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(P.eda / "class_distribution.png", dpi=130); plt.close()
    counts = tr.label.value_counts()
    summary["train_class_counts"] = counts.to_dict()
    summary["imbalance_ratio_max_over_min"] = float(counts.max() / counts.min())
    print(ct.to_string(), f"\nImbalance ratio (train): {summary['imbalance_ratio_max_over_min']:.2f}")

    # 2) raw image geometry / file size ---------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, c in enumerate(classes):
        s = tr[tr.label_id == k]
        ax[0].scatter(s.width, s.height, s=6, alpha=.5, label=c)
    ax[0].set_xlabel("width px"); ax[0].set_ylabel("height px"); ax[0].legend(fontsize=6)
    ax[0].set_title("Original resolutions (train)")
    ax[1].boxplot([tr[tr.label_id == k].bytes / 1e3 for k in range(K)], showfliers=False)
    ax[1].set_xticklabels(classes, rotation=30, ha="right"); ax[1].set_title("File size (KB)")
    plt.tight_layout(); plt.savefig(P.eda / "image_geometry.png", dpi=130); plt.close()
    summary["resolution_unique_count"] = int(tr[["width", "height"]].drop_duplicates().shape[0])
    summary["width_range"] = [int(tr.width.min()), int(tr.width.max())]
    summary["height_range"] = [int(tr.height.min()), int(tr.height.max())]

    # 3) near-duplicate structure ------------------------------------------------------------
    csz = m.cluster.value_counts()
    summary["clusters"] = int(len(csz)); summary["images_in_multi_image_clusters"] = int(csz[csz > 1].sum())
    plt.figure(figsize=(5, 3)); plt.hist(csz.values, bins=np.arange(1, csz.max() + 2) - .5)
    plt.yscale("log"); plt.title("Near-duplicate cluster sizes"); plt.tight_layout()
    plt.savefig(P.eda / "cluster_sizes.png", dpi=130); plt.close()

    # 4) pixel-level statistics (train only) ---------------------------------------------------
    X, y, _ = load_xy(P, "train")
    st = per_image_stats(X)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    boxplot(ax[0], st[:, 0], y, classes, "Mean brightness")
    boxplot(ax[1], st[:, 1], y, classes, "Excess-green index (2G-R-B)")
    plt.tight_layout(); plt.savefig(P.eda / "color_stats.png", dpi=130); plt.close()
    pd.DataFrame(st, columns=["brightness", "exg", "r", "g", "b"]).assign(label=tr.label.values) \
        .groupby("label").agg(["mean", "std"]).to_csv(P.eda / "color_stats_by_class.csv")

    # 5) sample grid + mean image per class -----------------------------------------------------
    rng = np.random.default_rng(args.seed)
    n = 6
    fig, ax = plt.subplots(K, n, figsize=(n * 1.6, K * 1.7))
    ax = np.atleast_2d(ax)
    for k in range(K):
        idx = np.where(y == k)[0]
        pick = rng.choice(idx, min(n, len(idx)), replace=False)
        for j in range(n):
            ax[k, j].axis("off")
            if j < len(pick):
                ax[k, j].imshow(X[pick[j]])
            if j == 0:
                ax[k, j].set_title(classes[k], fontsize=7, loc="left")
    plt.tight_layout(); plt.savefig(P.eda / "samples.png", dpi=110); plt.close()

    fig, ax = plt.subplots(1, K, figsize=(K * 2, 2.4))
    ax = np.atleast_1d(ax)
    for k in range(K):
        idx = np.where(y == k)[0][:300]
        ax[k].imshow(np.asarray(X[idx], dtype=np.float32).mean(0) / 255.0)
        ax[k].set_title(classes[k], fontsize=6); ax[k].axis("off")
    plt.tight_layout(); plt.savefig(P.eda / "mean_images.png", dpi=110); plt.close()

    (P.eda / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nEDA written to {P.eda}/ (png + eda_summary.json)")


if __name__ == "__main__":
    main()
