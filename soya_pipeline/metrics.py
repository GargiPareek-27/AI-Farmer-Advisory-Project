"""Metric helpers shared by the evaluation and CV-summary scripts."""
import numpy as np
from sklearn.metrics import f1_score


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


def cluster_bootstrap_f1(y, pred, groups, n_boot=500, seed=0):
    """95 % CI of macro-F1, resampling whole near-duplicate clusters (not single images)."""
    rng = np.random.default_rng(seed)
    by_g = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    keys = list(by_g)
    scores = []
    for _ in range(n_boot):
        idx = np.concatenate([by_g[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        scores.append(f1_score(y[idx], pred[idx], average="macro"))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
