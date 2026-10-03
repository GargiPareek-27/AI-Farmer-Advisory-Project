#!/usr/bin/env python
"""Step 1 - Scan, validate, de-duplicate and split the dataset WITHOUT data leakage.

Leakage safeguards
  * Corrupt files are dropped.
  * Exact duplicates (same MD5) are dropped; if the same bytes appear under different
    labels, all copies are dropped (label noise).
  * Near-duplicates (perceptual hash, robust to flips / 90-degree rotations, i.e. the
    usual "augmented copies" and consecutive drone frames) are grouped into clusters.
  * The split is STRATIFIED + GROUP-AWARE: a cluster never spans train/val/test.
  * The split is decided once, here, on file metadata only - before any scaling,
    augmentation, feature selection or class-weight computation.
"""
import hashlib
import io

import imagehash
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from common import EXPECTED_CLASSES, IMG_EXT, Paths, common_args, set_seed
import json

T = Image.Transpose
DIHEDRAL = [None, T.FLIP_LEFT_RIGHT, T.FLIP_TOP_BOTTOM, T.ROTATE_90,
            T.ROTATE_180, T.ROTATE_270, T.TRANSPOSE, T.TRANSVERSE]
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def file_info(path, root, dihedral):
    rel = path.relative_to(root).as_posix()
    try:
        data = path.read_bytes()
        md5 = hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            im.draft("RGB", (128, 128))
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((64, 64))
            hashes = []
            for op in (DIHEDRAL if dihedral else DIHEDRAL[:1]):
                v = im if op is None else im.transpose(op)
                hashes.append(int(str(imagehash.phash(v)), 16))
        return dict(relpath=rel, label=path.parent.name, width=w, height=h,
                    bytes=len(data), md5=md5, hashes=hashes)
    except Exception as e:  # noqa: BLE001
        return dict(relpath=rel, error=repr(e))


def popcount64(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x)
    return _POP[x.view(np.uint8)].reshape(-1, 8).sum(1)


def cluster_near_duplicates(H: np.ndarray, thr: int) -> np.ndarray:
    """Union-find over pairs whose (best dihedral) pHash Hamming distance <= thr."""
    n = len(H)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    base = H[:, 0]
    for i in range(n - 1):
        x = np.bitwise_xor(H[i + 1:], base[i])               # (m, n_variants)
        d = popcount64(x.ravel()).reshape(len(x), -1).min(1)
        for j in np.nonzero(d <= thr)[0]:
            ra, rb = find(i), find(i + 1 + j)
            if ra != rb:
                parent[rb] = ra
    roots = np.array([find(i) for i in range(n)])
    return pd.factorize(roots)[0]


def group_split(df, test_size, val_size, seed):
    y, g = df.label_id.to_numpy(), df.cluster.to_numpy()
    k_test = round(1 / test_size)
    tv, te = next(StratifiedGroupKFold(k_test, shuffle=True, random_state=seed).split(df, y, g))
    k_val = round((1 - test_size) / val_size)
    rest = df.iloc[tv]
    tr, va = next(StratifiedGroupKFold(k_val, shuffle=True, random_state=seed)
                  .split(rest, rest.label_id, rest.cluster))
    split = np.empty(len(df), dtype=object)
    split[tv[tr]], split[tv[va]], split[te] = "train", "val", "test"
    return split


def main():
    def extra(p):
        p.add_argument("--test-size", type=float, default=0.20)
        p.add_argument("--val-size", type=float, default=0.10)
        p.add_argument("--hash-thr", type=int, default=6,
                       help="max pHash Hamming distance (of 64 bits) to call two images near-duplicates")
        p.add_argument("--no-dihedral", action="store_true",
                       help="skip flip/rotation-invariant hashing (faster, catches fewer augmented copies)")

    args = common_args(__doc__.splitlines()[0], extra)
    P = Paths(args)
    set_seed(args.seed)

    files = sorted(p for p in P.img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"No images found under {P.img_dir}. Check --data-root.")
    print(f"Found {len(files)} images in {P.img_dir}")

    info = Parallel(n_jobs=args.n_jobs, batch_size=16, verbose=2)(
        delayed(file_info)(p, P.img_dir, not args.no_dihedral) for p in files)
    raw = pd.DataFrame(info)
    dropped = []

    if "error" in raw:
        bad = raw[raw.error.notna()]
        dropped += [dict(relpath=r.relpath, reason=f"corrupt: {r.error}") for r in bad.itertuples()]
        raw = raw[raw.error.isna()].drop(columns="error")
    df = raw.reset_index(drop=True)

    # --- exact duplicates --------------------------------------------------------------
    conflict = df.groupby("md5").label.transform("nunique") > 1
    dropped += [dict(relpath=r, reason="same bytes under different labels") for r in df[conflict].relpath]
    df = df[~conflict]
    dup = df.duplicated("md5", keep="first")
    dropped += [dict(relpath=r, reason="exact duplicate") for r in df[dup].relpath]
    df = df[~dup].reset_index(drop=True)

    classes = sorted(df.label.unique())
    df["label_id"] = df.label.map({c: i for i, c in enumerate(classes)})
    print("\nClasses found (VERIFY these are your disease/pest/healthy folders):")
    print(df.label.value_counts().sort_index().to_string())
    if len(classes) != EXPECTED_CLASSES[args.dataset]:
        print(f"WARNING: expected {EXPECTED_CLASSES[args.dataset]} classes for '{args.dataset}', "
              f"found {len(classes)}. Class = name of the image's parent folder.")

    # --- near-duplicate clusters ---------------------------------------------------------
    H = np.array(df.hashes.tolist(), dtype=np.uint64)
    df["cluster"] = cluster_near_duplicates(H, args.hash_thr)
    df = df.drop(columns="hashes")
    sizes = df.cluster.value_counts()
    print(f"\n{len(sizes)} clusters for {len(df)} images; largest cluster = {sizes.iloc[0]} images; "
          f"{(sizes > 1).sum()} clusters contain near-duplicates.")
    if sizes.iloc[0] > 0.05 * len(df):
        print("WARNING: one cluster holds >5% of the data (uniform backgrounds can fool pHash). "
              "Lower --hash-thr if the split looks unbalanced.")

    # --- split ----------------------------------------------------------------------------
    df["split"] = group_split(df, args.test_size, args.val_size, args.seed)

    # --- hard leakage checks ----------------------------------------------------------------
    assert (df.groupby("cluster").split.nunique() == 1).all(), "cluster spans several splits!"
    assert (df.groupby("md5").split.nunique() == 1).all(), "identical file in several splits!"
    print("\nLeakage checks passed (no cluster / identical file crosses splits).")
    print(pd.crosstab(df.label, df.split, margins=True).to_string())

    df = df.sort_values(["split", "relpath"]).reset_index(drop=True)
    cols = ["relpath", "label", "label_id", "width", "height", "bytes", "md5", "cluster", "split"]
    df[cols].to_csv(P.manifest, index=False)
    P.classes_json.write_text(json.dumps(classes, indent=2))
    pd.DataFrame(dropped).to_csv(P.root / "dropped.csv", index=False)
    print(f"\nSaved {P.manifest}  (dropped {len(dropped)} files -> dropped.csv)")


if __name__ == "__main__":
    main()
