#!/usr/bin/env python
"""Step 5 - Fast classical models on engineered features (seconds to minutes, CPU only).

Leakage-proof by construction
  * Every model is an sklearn Pipeline (StandardScaler [+PCA] + classifier) fitted on TRAIN only.
  * Optional CV is StratifiedGroupKFold over near-duplicate clusters, inside TRAIN only.
  * Model selection uses the VAL split. The TEST split is never loaded here.
"""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import Paths, common_args, load_classes, load_features, set_seed, split_df


def make_clf(name, seed, n_jobs):
    if name == "logreg":
        return LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    if name == "svc":
        return SVC(C=10, kernel="rbf", gamma="scale", class_weight="balanced",
                   probability=True, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=n_jobs,
                                      class_weight="balanced_subsample", random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, early_stopping=True,
                                              class_weight="balanced", random_state=seed)
    raise ValueError(name)


def main():
    def extra(p):
        p.add_argument("--backbone", default="mobilenet_v3_large")
        p.add_argument("--feature-sets", nargs="+", default=["hand", "deep", "both"],
                       choices=["hand", "deep", "both"])
        p.add_argument("--models", nargs="+", default=["logreg", "svc", "rf", "hgb"],
                       choices=["logreg", "svc", "rf", "hgb"])
        p.add_argument("--cv", type=int, default=0, help="grouped CV folds on train (0 = skip, faster)")
        p.add_argument("--pca", type=int, default=0, help="PCA components (0 = off); fitted on train only")

    args = common_args(__doc__.splitlines()[0], extra)
    P = Paths(args)
    set_seed(args.seed)
    classes = load_classes(P)
    tr, va = split_df(P, "train"), split_df(P, "val")
    ytr, yva, groups = tr.label_id.to_numpy(), va.label_id.to_numpy(), tr.cluster.to_numpy()

    rows, best = [], (-1, None)
    for fs in args.feature_sets:
        try:
            Xtr = load_features(P, "train", fs, args.backbone)
            Xva = load_features(P, "val", fs, args.backbone)
        except FileNotFoundError as e:
            print(f"[skip feature set '{fs}'] {e.filename} missing - run 04_extract_features.py")
            continue
        for mname in args.models:
            steps = [("scale", StandardScaler())]
            if args.pca:
                steps.append(("pca", PCA(args.pca, random_state=args.seed)))
            steps.append(("clf", make_clf(mname, args.seed, args.n_jobs)))
            pipe = Pipeline(steps)

            cv_f1 = np.nan
            if args.cv:
                cv = StratifiedGroupKFold(args.cv, shuffle=True, random_state=args.seed)
                cv_f1 = cross_val_score(pipe, Xtr, ytr, groups=groups, cv=cv, scoring="f1_macro").mean()
            t0 = time.time()
            pipe.fit(Xtr, ytr)
            fit_s = time.time() - t0
            pred = pipe.predict(Xva)
            r = dict(feature_set=fs, model=mname, n_features=Xtr.shape[1], fit_seconds=round(fit_s, 1),
                     cv_macro_f1=cv_f1, val_acc=accuracy_score(yva, pred),
                     val_macro_f1=f1_score(yva, pred, average="macro"))
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
            if r["val_macro_f1"] > best[0]:
                best = (r["val_macro_f1"], dict(pipeline=pipe, feature_set=fs, backbone=args.backbone,
                                                model=mname, classes=classes))

    if best[1] is None:
        raise SystemExit("No model trained - extract features first.")
    res = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    res.to_csv(P.reports / "classical_results.csv", index=False)
    joblib.dump(best[1], P.models / "classical_best.joblib")
    print("\n" + res.to_string(index=False))
    print(f"\nBest on VAL: {best[1]['feature_set']} + {best[1]['model']} "
          f"(macro-F1={best[0]:.4f}) -> {P.models / 'classical_best.joblib'}")


if __name__ == "__main__":
    main()
