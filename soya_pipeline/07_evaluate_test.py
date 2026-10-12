#!/usr/bin/env python
"""Step 7 - Final, one-shot evaluation on the held-out TEST split (use --split val to debug).

Reports accuracy, balanced accuracy, macro/weighted F1, MCC, ROC-AUC (OvR), log-loss, ECE,
per-class report, confusion matrices, misclassified files and a cluster-bootstrap 95 % CI
(resampling near-duplicate clusters, not single images, so the CI is not over-optimistic).
Every TEST evaluation is logged: if you keep tuning against test you will see it in the log.
"""
import datetime as dt
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

from common import Paths, common_args, load_classes, load_features, load_xy, split_df


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece_score(y, proba, bins=15):
    conf, pred = proba.max(1), proba.argmax(1)
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def cluster_bootstrap_f1(y, pred, groups, n_boot, seed):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by_g = {g: np.where(groups == g)[0] for g in ug}
    scores = []
    for _ in range(n_boot):
        idx = np.concatenate([by_g[g] for g in rng.choice(ug, len(ug))])
        scores.append(f1_score(y[idx], pred[idx], average="macro"))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evaluate(tag, y, proba, df, classes, out, n_boot, seed, latency_ms):
    K = len(classes)
    pred = proba.argmax(1)
    m = dict(
        model=tag, n=int(len(y)), accuracy=accuracy_score(y, pred),
        balanced_accuracy=balanced_accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro"), weighted_f1=f1_score(y, pred, average="weighted"),
        macro_precision=precision_score(y, pred, average="macro", zero_division=0),
        macro_recall=recall_score(y, pred, average="macro", zero_division=0),
        mcc=matthews_corrcoef(y, pred), ece=ece_score(y, proba),
        log_loss=log_loss(y, np.clip(proba, 1e-9, 1), labels=list(range(K))),
        latency_ms_per_image=latency_ms)
    try:
        m["roc_auc_ovr_macro"] = roc_auc_score(y, proba / proba.sum(1, keepdims=True),
                                               multi_class="ovr", average="macro", labels=list(range(K)))
    except ValueError:
        m["roc_auc_ovr_macro"] = None
    m["macro_f1_ci95"] = cluster_bootstrap_f1(y, pred, df.cluster.to_numpy(), n_boot, seed)
    m = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}

    rep = classification_report(y, pred, labels=list(range(K)), target_names=classes, digits=4, zero_division=0)
    (out / f"{tag}_report.txt").write_text(rep)
    (out / f"{tag}_metrics.json").write_text(json.dumps(m, indent=2))

    cm = confusion_matrix(y, pred, labels=list(range(K)))
    fig, ax = plt.subplots(1, 2, figsize=(5 + 3.2 * K / 2, 3.6 + K * 0.35))
    for a, mat, title, fmt in ((ax[0], cm, "Counts", "d"),
                               (ax[1], cm / np.maximum(cm.sum(1, keepdims=True), 1), "Row-normalised", ".2f")):
        a.imshow(mat, cmap="Blues")
        a.set_xticks(range(K)); a.set_yticks(range(K))
        a.set_xticklabels(classes, rotation=45, ha="right", fontsize=7); a.set_yticklabels(classes, fontsize=7)
        a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(f"{tag} - {title}")
        for i in range(K):
            for j in range(K):
                a.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=7,
                       color="white" if mat[i, j] > mat.max() / 2 else "black")
    plt.tight_layout(); plt.savefig(out / f"{tag}_confusion.png", dpi=130); plt.close()

    wrong = df.assign(pred=[classes[i] for i in pred], confidence=proba.max(1))[pred != y]
    wrong[["relpath", "label", "pred", "confidence"]].sort_values("confidence", ascending=False) \
        .to_csv(out / f"{tag}_misclassified.csv", index=False)
    print(f"\n=== {tag} ===\n{rep}")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, indent=2))
    return m


def main():
    def extra(p):
        p.add_argument("--split", choices=["test", "val"], default="test")
        p.add_argument("--which", choices=["cnn", "classical", "both"], default="both")
        p.add_argument("--tta", action="store_true", help="flip test-time augmentation for the CNN")
        p.add_argument("--bootstrap", type=int, default=500)

    args = common_args(__doc__.splitlines()[0], extra)
    P = Paths(args)
    classes = load_classes(P)
    df = split_df(P, args.split)
    y = df.label_id.to_numpy()
    results = []

    cnn_path, cls_path = P.models / "cnn_best.pt", P.models / "classical_best.joblib"
    if args.which in ("cnn", "both") and cnn_path.exists():
        from models import forward_batches, get_device, load_checkpoint

        device = get_device()
        model, ck = load_checkpoint(cnn_path, device)
        X, _, _ = load_xy(P, args.split)
        t0 = time.time()
        logits = forward_batches(model, X, device, ck["mean"], ck["std"], tta=args.tta)
        lat = 1000 * (time.time() - t0) / len(X)
        results.append(evaluate(f"cnn_{ck['arch']}" + ("_tta" if args.tta else ""), y, softmax(logits),
                                df, classes, P.reports, args.bootstrap, args.seed, lat))
    elif args.which == "cnn":
        print(f"[skip] {cnn_path} not found")

    if args.which in ("classical", "both") and cls_path.exists():
        bundle = joblib.load(cls_path)
        Xf = load_features(P, args.split, bundle["feature_set"], bundle["backbone"])
        t0 = time.time()
        proba = bundle["pipeline"].predict_proba(Xf)
        lat = 1000 * (time.time() - t0) / len(Xf)   # classifier only, excludes feature extraction
        results.append(evaluate(f"classical_{bundle['feature_set']}_{bundle['model']}", y, proba,
                                df, classes, P.reports, args.bootstrap, args.seed, lat))
    elif args.which == "classical":
        print(f"[skip] {cls_path} not found")

    if not results:
        raise SystemExit("Nothing evaluated - train a model first (05 and/or 06).")

    if args.split == "test":
        log = P.reports / "test_eval_log.txt"
        prev = len(log.read_text().splitlines()) if log.exists() else 0
        with log.open("a") as fh:
            for r in results:
                fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')}\t{r['model']}\t"
                         f"macro_f1={r['macro_f1']:.4f}\n")
        if prev > 0:
            print(f"\nNOTE: this is TEST evaluation log entry #{prev + 1}. Choosing models / hyper-parameters "
                  f"based on repeated test scores leaks test information - tune on val only.")


if __name__ == "__main__":
    main()
