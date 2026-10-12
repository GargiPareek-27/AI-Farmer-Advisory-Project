#!/usr/bin/env python
"""Step 2 - Decode + resize every image ONCE into a single uint8 array (rows = manifest order).

All later steps (features, training, any fold / split) just index rows of this array, so no
script touches the original JPEGs again and nothing has to be re-cached per split.
Images are streamed straight into a memory-mapped file, so peak memory is one image, not two copies of the array.
"""
import os

import numpy as np
from joblib import Parallel, delayed

from common import Paths, common_args, load_manifest, load_resized, meta_state, write_meta


def load_named(path, size):
    try:
        return load_resized(path, size)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"cannot read {path}: {e!r}") from e


def main():
    args = common_args(__doc__.splitlines()[0], lambda p: p.add_argument("--force", action="store_true"))
    P = Paths(args)
    if P.x_path.exists() and not args.force and meta_state(P, P.x_path) == "ok":
        print(f"[skip] {P.x_path} is up to date (use --force to rebuild)")
        return
    paths = [P.img_dir / r for r in load_manifest(P).relpath]
    S, n = args.img_size, len(paths)
    tmp = P.x_path.with_name(f"all_x{S}.partial.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, S, S, 3))
    stream = Parallel(n_jobs=args.n_jobs, batch_size=16, return_as="generator")(
        delayed(load_named)(p, S) for p in paths)
    for i, arr in enumerate(stream):
        out[i] = arr
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"  cached {i + 1}/{n}", flush=True)
    out.flush()
    del out
    os.replace(tmp, P.x_path)                              # the final name only ever points at a complete file
    write_meta(P, P.x_path, size=S, n=n)
    print(f"({n}, {S}, {S}, 3) -> {P.x_path} ({n * S * S * 3 / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
