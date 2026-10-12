#!/usr/bin/env python
"""Step 5 - Audit the active split BEFORE training anything expensive.

(a) Source overlap  : share of val/test images whose source group is also in train (0 by construction; a sanity check).
(b) Near-copy check : cosine similarity of every val/test image to its nearest TRAIN image
                      (frozen deep features) and the macro-F1 of a plain 1-NN lookup.
                      A median similarity ~0.97+ and a 1-NN F1 ~1.0 mean the test set is
                      near-copies of train, i.e. the score measures memory, not generalisation.
(c) Colour probe    : macro-F1 from the 3 mean-RGB values alone (chance = 1/K).
Only features are used (no torch); features of every row are computed in step 4.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import Paths, check_meta, common_args, load_classes, load_manifest


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def main():
    args = common_args(__doc__.splitlines()[0])
    P = Paths(args)
    lines = []

    def say(msg):
        print(msg)
        lines.append(str(msg))

    m = load_manifest(P)
    y, K = m.label_id.to_numpy(), len(load_classes(P))
    mask = {s: (m.split == s).to_numpy() for s in ("train", "val", "test")}
    say(f"split = {P.tag}: " + ", ".join(f"{s} {int(v.sum())}" for s, v in mask.items()))

    seen = set(m.group[mask["train"]])
    for s in ("val", "test"):
        say(f"(a) {s}: {m.group[mask[s]].isin(seen).mean():.1%} of images share a source group with train")

    deep_f = P.features / f"all_deep_{args.backbone}.npy"
    deep = None
    if deep_f.exists():
        check_meta(P, deep_f)
        deep = np.load(deep_f)
        deep = deep / np.linalg.norm(deep, axis=1, keepdims=True)
        for s in ("val", "test"):
            S = deep[mask[s]] @ deep[mask["train"]].T
            p = np.percentile(S.max(1), [5, 25, 50, 75, 95]).round(3)
            f1 = f1_score(y[mask[s]], y[mask["train"]][S.argmax(1)], average="macro")
            flag = "   <-- near-copies of train: scores will be inflated" if np.median(S.max(1)) > 0.95 else ""
            say(f"(b) {s}: nearest-train similarity 5/25/50/75/95 = {p} | 1-NN macro-F1 {f1:.3f}{flag}")
    else:
        say(f"(b) skipped: {deep_f.name} missing (run step 4 without --skip-deep)")

    hand_f = P.features / "all_hand.npy"
    hand = np.load(hand_f) if hand_f.exists() else None
    if hand is not None:
        check_meta(P, hand_f)
        names = json.loads((P.features / "hand_feature_names.json").read_text())
        rgb = [i for i, n in enumerate(names) if n.startswith("rgb_") and n.endswith("_mean")]
        clf = logreg().fit(hand[mask["train"]][:, rgb], y[mask["train"]])
        f1 = f1_score(y[mask["test"]], clf.predict(hand[mask["test"]][:, rgb]), average="macro")
        say(f"(c) mean-RGB-only probe, test macro-F1 {f1:.3f} (chance {1 / K:.2f})")

    (P.reports / "audit.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
