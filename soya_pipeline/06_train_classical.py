#!/usr/bin/env python
"""Step 6 - Fast classical models on engineered features (CPU, seconds to minutes).

Every model is an sklearn Pipeline (StandardScaler [+PCA] + classifier) fitted on TRAIN only;
model selection uses VAL. The TEST split is never loaded here.
"""
import time

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (DEFAULT_EMBED, Paths, classical_bundle, common_args, load_classes, load_features,
                    set_seed, split_df)


def make_clf(name, seed, n_jobs):
    if name == "logreg":
        return LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    if name == "svc":
        return SVC(C=10, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=n_jobs, class_weight="balanced_subsample", random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, early_stopping=True, class_weight="balanced", random_state=seed)
    raise ValueError(name)


def main():
    def extra(p):
        p.add_argument("--feature-sets", nargs="+", default=["hand", "deep", "both"], choices=["hand", "deep", "both"])
        p.add_argument("--models", nargs="+", default=["logreg", "svc", "hgb"], choices=["logreg", "svc", "rf", "hgb"])
        p.add_argument("--pca", type=int, default=0, help="PCA components (0 = off); fitted on train only")

    args = common_args(__doc__.splitlines()[0], extra)
    P = Paths(args)
    set_seed(args.seed)
    classes = load_classes(P)
    ytr, yva = split_df(P, "train").label_id.to_numpy(), split_df(P, "val").label_id.to_numpy()

    rows, best = [], (-1, None)
    for fs in args.feature_sets:
        try:
            Xtr, Xva = load_features(P, "train", fs, args.backbone), load_features(P, "val", fs, args.backbone)
        except FileNotFoundError as e:
            print(f"[skip '{fs}'] {e.filename} missing - run step 4")
            continue
        for mname in args.models:
            steps = [("scale", StandardScaler())]
            if args.pca:
                steps.append(("pca", PCA(args.pca, random_state=args.seed)))
            pipe = Pipeline(steps + [("clf", make_clf(mname, args.seed, args.n_jobs))])
            t0 = time.time()
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xva)
            r = dict(feature_set=fs, model=mname, n_features=Xtr.shape[1], fit_seconds=round(time.time() - t0, 1),
                     val_acc=accuracy_score(yva, pred), val_macro_f1=f1_score(yva, pred, average="macro"))
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
            if r["val_macro_f1"] > best[0]:
                best = (r["val_macro_f1"], dict(pipeline=pipe, feature_set=fs, backbone=args.backbone,
                                                model=mname, classes=classes))
    if best[1] is None:
        raise SystemExit("No model trained - run step 4 first.")
    res = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    suffix = "" if args.backbone == DEFAULT_EMBED else f"_{args.backbone}"
    res.to_csv(P.reports / f"classical_results{suffix}.csv", index=False)
    joblib.dump(best[1], classical_bundle(P, args.backbone))
    print("\n" + res.to_string(index=False))
    print(f"\nBest on VAL: {best[1]['feature_set']} + {best[1]['model']} (macro-F1={best[0]:.4f})")


if __name__ == "__main__":
    main()
