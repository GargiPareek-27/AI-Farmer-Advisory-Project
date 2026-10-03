#!/usr/bin/env python
"""Step 4 - Feature engineering: hand-crafted (colour / vegetation index / texture) and
frozen-CNN embeddings, for every split.

Nothing here is fitted on data (no scaler, no PCA, no selection), so there is no leakage;
scaling etc. happens later inside sklearn Pipelines fitted on train only.
Frozen-backbone embeddings need a single forward pass per image -> minutes even on CPU.
"""
import argparse
import json

import numpy as np
from joblib import Parallel, delayed

from common import Paths, common_args, load_xy
from features import feature_names, hand_vector


def main():
    def extra(p):
        p.add_argument("--backbone", default="mobilenet_v3_large",
                       choices=["mobilenet_v3_large", "efficientnet_b0", "resnet18", "resnet50"])
        p.add_argument("--batch-size", type=int, default=128)
        p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
        p.add_argument("--skip-hand", action="store_true")
        p.add_argument("--skip-deep", action="store_true")

    args = common_args(__doc__.splitlines()[0], extra)
    P = Paths(args)

    if not args.skip_hand:
        for split in ("train", "val", "test"):
            X, _, _ = load_xy(P, split)
            vecs = Parallel(n_jobs=args.n_jobs, batch_size=32)(
                delayed(hand_vector)(X[i]) for i in range(len(X)))
            F = np.stack(vecs)
            np.save(P.features / f"{split}_hand.npy", F)
            print(f"hand  {split}: {F.shape}")
            if split == "train":
                (P.features / "hand_feature_names.json").write_text(
                    json.dumps(feature_names(np.asarray(X[0]))))

    if not args.skip_deep:
        import torch
        from models import IMAGENET_MEAN, IMAGENET_STD, build_embedder, forward_batches, get_device

        device = get_device()
        emb = build_embedder(args.backbone, args.pretrained).to(device).to(memory_format=torch.channels_last)
        if args.pretrained:
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            st = json.loads(P.stats_json.read_text())
            mean, std = st["mean"], st["std"]
        for split in ("train", "val", "test"):
            X, _, _ = load_xy(P, split)
            E = forward_batches(emb, X, device, mean, std, bs=args.batch_size)
            np.save(P.features / f"{split}_deep_{args.backbone}.npy", E)
            print(f"deep  {split}: {E.shape} ({args.backbone}, device={device})")


if __name__ == "__main__":
    main()
