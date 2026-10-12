#!/usr/bin/env python
"""Step 1 - Scan, validate, de-duplicate and split the dataset WITHOUT data leakage.

Leakage safeguards
  * Corrupt files are dropped; exact duplicates (MD5) are dropped; identical bytes under
    different labels are dropped (label noise).
  * Near-duplicates (perceptual hash, robust to flips / 90-degree rotations) form clusters.
  * Optional `--group-regex` merges files that share a source id parsed from the file name
    (use it if the photos of one plant / plot / session are numbered together).
  * The split (70/10/20) AND `--folds` stratified-group CV folds are decided once, here,
    on metadata only. Later steps never re-split.
"""
import hashlib
import io
import json
import re

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from common import DEFAULT_FOLDS, EXPECTED_CLASSES, IMG_EXT, Paths, check_fold_coverage, common_args, set_seed

T = Image.Transpose
DIHEDRAL = [None, T.FLIP_LEFT_RIGHT, T.FLIP_TOP_BOTTOM, T.ROTATE_90,
            T.ROTATE_180, T.ROTATE_270, T.TRANSPOSE, T.TRANSVERSE]
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash(img: Image.Image) -> int:
    """64-bit DCT perceptual hash (same recipe as imagehash.phash, without the dependency)."""
    g = np.asarray(img.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = cv2.dct(g)[:8, :8]
    return int("".join("1" if b else "0" for b in (low > np.median(low)).ravel()), 2)


def file_info(path, root, dihedral):
    rel = path.relative_to(root).as_posix()
    try:
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            im.draft("RGB", (128, 128))
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((64, 64))
            hashes = [phash(im if op is None else im.transpose(op))
                      for op in (DIHEDRAL if dihedral else DIHEDRAL[:1])]
        return dict(relpath=rel, label=path.parent.name, width=w, height=h, bytes=len(data),
                    md5=hashlib.md5(data).hexdigest(), hashes=hashes)
    except Exception as e:  # noqa: BLE001
        return dict(relpath=rel, error=repr(e))


def source_info(rel: str, label: str, regex) -> str:
    """Source group id of a file (label-prefixed). Without --group-regex every file is its own group."""
    if regex is not None:
        m = regex.search(rel)
        return f"{label}/{m.group(1) if m else rel}"
    return f"{label}/{rel}"


def popcount64(x: np.ndarray) -> np.ndarray:
    return _POP[np.ascontiguousarray(x).view(np.uint8)].reshape(-1, 8).sum(1)


class DSU:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_clusters(H: np.ndarray, groups: np.ndarray, thr: int) -> np.ndarray:
    """Union images whose best-dihedral pHash distance <= thr, and images sharing a source group."""
    n = len(H)
    dsu = DSU(n)
    base = H[:, 0]
    for i in range(n - 1):
        d = popcount64(np.bitwise_xor(H[i + 1:], base[i]).ravel()).reshape(n - i - 1, -1).min(1)
        for j in np.flatnonzero(d <= thr):
            dsu.union(i, i + 1 + j)
    first = {}
    for i, g in enumerate(groups):
        dsu.union(i, first.setdefault(g, i))
    return pd.factorize(np.array([dsu.find(i) for i in range(n)]))[0]


def stratified_group_split(df, test_size, val_size, seed):
    y, g = df.label_id.to_numpy(), df.cluster.to_numpy()
    tv, te = next(StratifiedGroupKFold(round(1 / test_size), shuffle=True, random_state=seed).split(df, y, g))
    rest = df.iloc[tv]
    cv = StratifiedGroupKFold(round((1 - test_size) / val_size), shuffle=True, random_state=seed)
    tr, va = next(cv.split(rest, rest.label_id, rest.cluster))
    split = np.empty(len(df), dtype=object)
    split[tv[tr]], split[tv[va]], split[te] = "train", "val", "test"
    return split


def stratified_group_folds(df, k, seed):
    fold = np.full(len(df), -1)
    for i, (_, te) in enumerate(StratifiedGroupKFold(k, shuffle=True, random_state=seed)
                                .split(df, df.label_id, df.cluster)):
        fold[te] = i
    return fold


def main():
    def extra(p):
        p.add_argument("--test-size", type=float, default=0.20)
        p.add_argument("--val-size", type=float, default=0.10)
        p.add_argument("--folds", type=int, default=0, help="CV folds (0 = default of 5)")
        p.add_argument("--hash-thr", type=int, default=6, help="max pHash distance (of 64) for near-duplicates")
        p.add_argument("--no-dihedral", action="store_true", help="skip flip/rotation-invariant hashing")
        p.add_argument("--group-regex", default=None,
                       help="regex whose group 1 is a source id; files with the same id are never split (default: off)")

    args = common_args(__doc__.splitlines()[0], extra)
    P = Paths(args)
    set_seed(args.seed)
    folds = args.folds or DEFAULT_FOLDS[args.dataset]
    regex = re.compile(args.group_regex) if args.group_regex else None

    files = sorted(p for p in P.img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"No images found under {P.img_dir}. Check --data-root (and unzip the class zips).")
    print(f"Found {len(files)} images in {P.img_dir}")
    raw = pd.DataFrame(Parallel(n_jobs=args.n_jobs, batch_size=16, verbose=1)(
        delayed(file_info)(p, P.img_dir, not args.no_dihedral) for p in files))

    dropped = []
    if "error" in raw:
        bad = raw[raw.error.notna()]
        dropped += [dict(relpath=r.relpath, reason=f"corrupt: {r.error}") for r in bad.itertuples()]
        raw = raw[raw.error.isna()].drop(columns="error")
    df = raw.reset_index(drop=True)
    conflict = df.groupby("md5").label.transform("nunique") > 1
    dropped += [dict(relpath=r, reason="same bytes under different labels") for r in df[conflict].relpath]
    df = df[~conflict]
    dup = df.duplicated("md5", keep="first")
    dropped += [dict(relpath=r, reason="exact duplicate") for r in df[dup].relpath]
    df = df[~dup].reset_index(drop=True)

    classes = sorted(df.label.unique())
    df["label_id"] = df.label.map({c: i for i, c in enumerate(classes)})
    if len(classes) != EXPECTED_CLASSES[args.dataset]:
        print(f"WARNING: expected {EXPECTED_CLASSES[args.dataset]} classes, found {len(classes)}: {classes}")
    df["group"] = [source_info(r.relpath, r.label, regex) for r in df.itertuples()]

    H = np.array(df.hashes.tolist(), dtype=np.uint64)
    df["cluster"] = build_clusters(H, df.group.to_numpy(), args.hash_thr)
    df = df.drop(columns="hashes")
    df["split"] = stratified_group_split(df, args.test_size, args.val_size, args.seed)
    df["fold"] = stratified_group_folds(df, folds, args.seed)

    mixed = df.groupby("cluster").label.nunique()
    if (mixed > 1).any():
        bad = df[df.cluster.isin(mixed[mixed > 1].index)][["relpath", "label", "cluster"]]
        bad.to_csv(P.root / "mixed_label_clusters.csv", index=False)
        print(f"WARNING: {int((mixed > 1).sum())} cluster(s) contain images of several classes (near-identical images "
              "with different labels, or a too-coarse --group-regex); listed in mixed_label_clusters.csv.")

    check_fold_coverage(df, folds)                              # fail now, not after an hour of training

    # hard leakage checks ------------------------------------------------------------------
    for key in ("cluster", "group", "md5"):
        assert (df.groupby(key).split.nunique() == 1).all(), f"{key} spans several splits!"
        assert (df.groupby(key).fold.nunique() == 1).all(), f"{key} spans several folds!"

    summary = df.groupby("label").agg(images=("relpath", "size"), source_groups=("group", "nunique"),
                                      clusters=("cluster", "nunique"),
                                      largest_cluster=("cluster", lambda s: s.value_counts().max()))
    print("\nClasses (VERIFY these are your disease / pest / healthy folders):\n" + summary.to_string())
    print("\nsplit sizes:\n" + pd.crosstab(df.label, df.split, margins=True).to_string())
    if (pd.crosstab(df.label, df.split) == 0).any().any():
        print("WARNING: a class is missing from train, val or test in the single 70/10/20 split (too few independent "
              "sources). Use the CV folds instead: run steps 5-9 with --fold 0..K-1.")
    print("\nfold sizes:\n" + pd.crosstab(df.label, df.fold, margins=True).to_string())
    print("\nLeakage checks passed: no cluster / source group / identical file crosses splits or folds.")
    if (summary.largest_cluster > 0.4 * summary.images).any():
        print("WARNING: one cluster holds >40% of a class (uniform backgrounds can fool pHash, or a group "
              "pattern is too coarse); that class cannot be split. Lower --hash-thr / check --group-regex.")
    if (summary.source_groups < folds).any():
        print(f"WARNING: a class has fewer source groups than folds ({folds}); "
              "folds cannot be balanced - lower --folds.")
    if summary.source_groups.min() < 15:
        print("NOTE: few independent sources per class -> metrics are high-variance; "
              "prefer the CV summary (step 9) over the single split.")

    df = df.sort_values("relpath").reset_index(drop=True)      # stable row order = cache / feature order
    cols = ["relpath", "label", "label_id", "width", "height", "bytes", "md5", "group",
            "cluster", "split", "fold"]
    df[cols].to_csv(P.manifest, index=False)
    P.classes_json.write_text(json.dumps(classes, indent=2))
    pd.DataFrame(dropped).to_csv(P.root / "dropped.csv", index=False)
    stale = [*P.cache.glob("all_*"), *P.features.glob("all_*")]  # manifest changed -> row order changed
    for f in stale:
        f.unlink()
    print(f"\nSaved {P.manifest}  (dropped {len(dropped)} files -> dropped.csv; cleared {len(stale)} stale cache/feature files)")


if __name__ == "__main__":
    main()
