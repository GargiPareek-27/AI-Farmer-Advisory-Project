#!/usr/bin/env python
"""Step 4 - Features for ALL images, once: hand-crafted (colour / vegetation index / texture)
and frozen pretrained-CNN embeddings.

Nothing here is fitted on data (no scaler, no PCA, no selection), so computing it for every
row cannot leak; scaling etc. happens later inside sklearn Pipelines fitted on train only.
"""
import json

import numpy as np
from joblib import Parallel, delayed

from common import Paths, check_meta, common_args, write_meta
from features import feature_names, hand_vector


def main():
    def extra(p):
        p.add_argument("--batch-size", type=int, default=128)
        p.add_argument("--skip-hand", action="store_true")
        p.add_argument("--skip-deep", action="store_true")

    args = common_args(__doc__.splitlines()[0], extra)
    P = Paths(args)
    if args.backbone.startswith("dinov2") and args.img_size % 14:
        raise SystemExit(f"{args.backbone} needs --img-size divisible by 14 (use 224 or 308); "
                         f"run 02_cache_images.py --img-size 224 first")
    if not P.x_path.exists():
        raise SystemExit(f"{P.x_path} not found - run 02_cache_images.py --img-size {args.img_size} first")
    check_meta(P, P.x_path)
    X = np.load(P.x_path, mmap_mode="r")

    if not args.skip_hand:
        F = np.stack(Parallel(n_jobs=args.n_jobs, batch_size=32)(delayed(hand_vector)(X[i]) for i in range(len(X))))
        np.save(P.features / "all_hand.npy", F)
        write_meta(P, P.features / "all_hand.npy", n=len(F))
        (P.features / "hand_feature_names.json").write_text(json.dumps(feature_names(np.asarray(X[0]))))
        print(f"hand: {F.shape}")

    if not args.skip_deep:
        from models import IMAGENET_MEAN, IMAGENET_STD, build_embedder, forward_batches, get_device

        device = get_device()
        emb = build_embedder(args.backbone).to(device)
        E = forward_batches(emb, X, device, IMAGENET_MEAN, IMAGENET_STD, bs=args.batch_size)
        np.save(P.features / f"all_deep_{args.backbone}.npy", E)
        write_meta(P, P.features / f"all_deep_{args.backbone}.npy", n=len(E), backbone=args.backbone)
        print(f"deep: {E.shape} ({args.backbone}, device={device})")


if __name__ == "__main__":
    main()
