#!/usr/bin/env python
"""Step 8 - Evaluate on the held-out TEST split (use --split val while developing).

--cnn best   (default) the CNN checkpoint with the best VAL macro-F1 among those trained at --img-size
--cnn all    average the probabilities of every such checkpoint (ensemble)
--cnn <stem> one checkpoint, e.g. efficientnet_b0_320; a comma-separated list gives an explicit ensemble
--backbone   which frozen-embedding backbone's classical model to score (default mobilenet_v3_large)

Reports accuracy, balanced accuracy, macro/weighted F1, MCC, ROC-AUC, log-loss, ECE, per-class
report, confusion matrix, misclassified files, a cluster-bootstrap 95 % CI, and saves per-image
predictions (used by step 9). Every TEST evaluation is logged: tuning on test shows up in the log.
"""
import datetime as dt
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

from common import (Paths, classical_bundle, classical_tag, common_args, load_classes, load_features,
                    load_xy, split_df)
from metrics import cluster_bootstrap_f1, ece_score, softmax


def evaluate(tag, y, proba, df, classes, out, n_boot, seed, latency_ms):
    K, pred = len(classes), proba.argmax(1)
    m = dict(model=tag, n=int(len(y)), accuracy=accuracy_score(y, pred),
             balanced_accuracy=balanced_accuracy_score(y, pred),
             macro_f1=f1_score(y, pred, average="macro"), weighted_f1=f1_score(y, pred, average="weighted"),
             macro_precision=precision_score(y, pred, average="macro", zero_division=0),
             macro_recall=recall_score(y, pred, average="macro", zero_division=0),
             mcc=matthews_corrcoef(y, pred), ece=ece_score(y, proba),
             log_loss=log_loss(y, np.clip(proba, 1e-9, 1), labels=list(range(K))), latency_ms_per_image=latency_ms)
    try:
        m["roc_auc_ovr_macro"] = roc_auc_score(y, proba / proba.sum(1, keepdims=True), multi_class="ovr",
                                               average="macro", labels=list(range(K)))
    except ValueError:
        m["roc_auc_ovr_macro"] = None
    m["macro_f1_ci95"] = cluster_bootstrap_f1(y, pred, df.cluster.to_numpy(), n_boot, seed)
    m = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}

    rep = classification_report(y, pred, labels=list(range(K)), target_names=classes, digits=4, zero_division=0)
    (out / f"{tag}_report.txt").write_text(rep)
    (out / f"{tag}_metrics.json").write_text(json.dumps(m, indent=2))
    pd.DataFrame(proba, columns=[f"p_{c}" for c in classes]).assign(
        relpath=df.relpath.values, cluster=df.cluster.values, label_id=y, pred=pred).to_csv(out / f"{tag}_pred.csv", index=False)

    cm = confusion_matrix(y, pred, labels=list(range(K)))
    fig, ax = plt.subplots(1, 2, figsize=(5 + 3.2 * K / 2, 3.6 + K * 0.35))
    for a, mat, title, fmt in ((ax[0], cm, "Counts", "d"), (ax[1], cm / np.maximum(cm.sum(1, keepdims=True), 1), "Row-normalised", ".2f")):
        a.imshow(mat, cmap="Blues")
        a.set_xticks(range(K)); a.set_yticks(range(K))
        a.set_xticklabels(classes, rotation=45, ha="right", fontsize=7); a.set_yticklabels(classes, fontsize=7)
        a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(f"{tag} - {title}", fontsize=8)
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


def pick_checkpoints(P, which, size):
    cks = []
    for f in sorted(P.models.glob("cnn_*.json")):
        meta = json.loads(f.read_text())
        if meta["img_size"] == size and f.with_suffix(".pt").exists():
            cks.append((f.stem.removeprefix("cnn_"), meta["val_macro_f1"]))
    if not cks:
        return []
    print("CNN checkpoints at this size (val macro-F1): " + ", ".join(f"{s} {v:.4f}" for s, v in cks))
    if which == "all":
        return [s for s, _ in cks]
    if which == "best":
        return [max(cks, key=lambda t: t[1])[0]]
    wanted = [w.strip() for w in which.split(",") if w.strip()]
    names = [s for s, _ in cks]
    missing = [w for w in wanted if w not in names]
    if missing:
        raise SystemExit(f"checkpoint(s) {missing} not found at --img-size {size}; available: {names}")
    return wanted


def main():
    def extra(p):
        p.add_argument("--split", choices=["test", "val"], default="test")
        p.add_argument("--which", choices=["cnn", "classical", "both"], default="both")
        p.add_argument("--cnn", default="best", help="best | all | <stem> | <stem>,<stem>,... (explicit ensemble)")
        p.add_argument("--tta", action="store_true", help="flip test-time augmentation for the CNN")
        p.add_argument("--bootstrap", type=int, default=500)

    args = common_args(__doc__.splitlines()[0], extra)
    P = Paths(args)
    classes, df = load_classes(P), split_df(P, args.split)
    y = df.label_id.to_numpy()
    results = []

    stems = pick_checkpoints(P, args.cnn, args.img_size) if args.which in ("cnn", "both") else []
    if stems:
        from models import forward_batches, get_device, load_checkpoint

        device = get_device()
        X, _, _ = load_xy(P, args.split)
        probas, t0 = [], time.time()
        for s in stems:
            model, ck = load_checkpoint(P.models / f"cnn_{s}.pt", device)
            probas.append(softmax(forward_batches(model, X, device, ck["mean"], ck["std"], tta=args.tta)))
        lat = 1000 * (time.time() - t0) / len(X)
        name = f"cnn_{stems[0]}" if len(stems) == 1 else f"cnn_ensemble{len(stems)}"
        results.append(evaluate(name + ("_tta" if args.tta else ""), y, np.mean(probas, 0), df, classes,
                                P.reports, args.bootstrap, args.seed, lat))
    elif args.which == "cnn":
        print(f"[skip] no CNN checkpoint for --img-size {args.img_size} in {P.models}")

    cls_path = classical_bundle(P, args.backbone)
    if args.which in ("classical", "both") and cls_path.exists():
        bundle = joblib.load(cls_path)
        Xf = load_features(P, args.split, bundle["feature_set"], bundle["backbone"])
        t0 = time.time()
        proba = bundle["pipeline"].predict_proba(Xf)
        lat = 1000 * (time.time() - t0) / len(Xf)            # classifier only, excludes feature extraction
        results.append(evaluate(classical_tag(bundle), y, proba, df, classes,
                                P.reports, args.bootstrap, args.seed, lat))
    elif args.which == "classical":
        print(f"[skip] {cls_path} not found")

    if not results:
        raise SystemExit("Nothing evaluated - train a model first (steps 6 and/or 7).")
    if args.split == "test":
        log = P.reports / "test_eval_log.txt"
        prev = len(log.read_text().splitlines()) if log.exists() else 0
        with log.open("a") as fh:
            for r in results:
                fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')}\t{r['model']}\tmacro_f1={r['macro_f1']:.4f}\n")
        if prev > 0:
            print(f"\nNOTE: TEST evaluation log entry #{prev + 1} for {P.tag}. Choosing models / hyper-parameters "
                  "from repeated test scores leaks test information - tune on val only.")


if __name__ == "__main__":
    main()
