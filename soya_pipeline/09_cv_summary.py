#!/usr/bin/env python
"""Step 9 - Summarise a cross-validation run (steps 5-8 repeated with --fold 0..K-1).

Every image is in the TEST fold exactly once and folds never share a source group, so the pooled
out-of-fold predictions are an honest estimate on ALL images (far tighter than one 20 % test set).
Reports per-fold macro-F1 (mean +- sd), the pooled macro-F1 with a cluster-bootstrap 95 % CI,
and per-class precision / recall.
"""
import json

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from common import Paths, common_args, load_classes
from metrics import cluster_bootstrap_f1


def main():
    args = common_args(__doc__.splitlines()[0], lambda p: p.add_argument(
        "--model-tag", default="cnn", help="cnn_efficientnet_b0_320_tta | cnn_ensemble3 | classical | classical-dinov2_vits14"))
    P = Paths(args)
    classes = load_classes(P)
    base = P.root / "reports"
    folds = sorted(d for d in base.glob("fold*") if d.is_dir())
    if not folds:
        raise SystemExit(f"No fold*/ report folders in {base} - run steps 6-8 with --fold 0..K-1 first.")
    parts = []
    for d in folds:
        stem = lambda f: f.name.removesuffix("_pred.csv")
        hits = sorted(f for f in d.glob("*_pred.csv") if stem(f) == args.model_tag or stem(f).startswith(args.model_tag + "_"))
        if len(hits) != 1:
            have = ", ".join(sorted(f.name.removesuffix("_pred.csv") for f in d.glob("*_pred.csv"))) or "nothing"
            raise SystemExit(f"{d.name}: need exactly one prediction file starting with '{args.model_tag}', found "
                             f"{len(hits)}. Available: {have}. Pass a longer --model-tag.")
        parts.append(pd.read_csv(hits[0]).assign(fold=d.name))
        print(f"{d.name}: {hits[0].name}")
    d = pd.concat(parts, ignore_index=True)
    # cluster ids are global (one manifest), so they stay valid across folds
    y, pred = d.label_id.to_numpy(), d.pred.to_numpy()
    per_fold = d.groupby("fold").apply(lambda g: f1_score(g.label_id, g.pred, average="macro"), include_groups=False)
    lo, hi = cluster_bootstrap_f1(y, pred, d.cluster.to_numpy(), 1000, args.seed)
    out = dict(model=args.model_tag, folds=len(folds), n=int(len(d)),
               per_fold_macro_f1=per_fold.round(4).to_dict(),
               mean_fold_macro_f1=float(per_fold.mean()), sd_fold_macro_f1=float(per_fold.std(ddof=1)) if len(per_fold) > 1 else 0.0,
               pooled_macro_f1=float(f1_score(y, pred, average="macro")), pooled_macro_f1_ci95=[lo, hi],
               pooled_accuracy=float(accuracy_score(y, pred)))
    print(json.dumps(out, indent=2))
    K = len(classes)
    print("\n" + classification_report(y, pred, labels=list(range(K)), target_names=classes, digits=3, zero_division=0))
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=list(range(K))), index=classes, columns=[c[:10] for c in classes])
    print("pooled confusion matrix (rows = true):\n" + cm.to_string())
    cv_dir = P.root / "reports" / "cv"
    cv_dir.mkdir(exist_ok=True)
    (cv_dir / f"{args.model_tag}_cv_summary.json").write_text(json.dumps(out, indent=2))
    cm.to_csv(cv_dir / f"{args.model_tag}_cv_confusion.csv")


if __name__ == "__main__":
    main()
