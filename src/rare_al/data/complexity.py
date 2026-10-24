"""Data-complexity descriptors (Lorena et al., 2019) and minority-example types
(Napierala & Stefanowski, 2016), used by the meta-regression and guideline G7.

Descriptors can be computed on the *seed* labels (observable before deployment)
or on the full labeled pool (oracle descriptors, for explanation only).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def _sub(X, y, max_n, seed):
    if len(y) <= max_n:
        return X, y
    rng = np.random.RandomState(seed)
    rare = np.flatnonzero(y == 1)
    major = np.flatnonzero(y == 0)
    keep = np.r_[rare[: max_n // 2], rng.choice(major, max_n - min(len(rare), max_n // 2),
                                                 replace=False)]
    return X[keep], y[keep]


def f1_fisher(X, y):
    """F1 = 1/(1+max_j r_j), r_j = Fisher discriminant ratio of feature j (lower = easier)."""
    a, b = X[y == 1], X[y == 0]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    num = (a.mean(0) - b.mean(0)) ** 2
    den = a.var(0) + b.var(0) + 1e-12
    return float(1.0 / (1.0 + np.max(num / den)))


def n1_borderline(X, y):
    """Fraction of points incident to an MST edge joining different classes."""
    if len(np.unique(y)) < 2:
        return np.nan
    D = cdist(X, X)
    mst = minimum_spanning_tree(D).tocoo()
    border = set()
    for i, j in zip(mst.row, mst.col):
        if y[i] != y[j]:
            border.update((int(i), int(j)))
    return len(border) / len(y)


def n3_nn_error(X, y):
    """Leave-one-out 1-NN error rate."""
    if len(y) < 3:
        return np.nan
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    _, idx = nn.kneighbors(X)
    return float(np.mean(y[idx[:, 1]] != y))


def minority_types(X, y, k=5):
    """Shares of safe / borderline / rare / outlier minority examples (k=5)."""
    rare = np.flatnonzero(y == 1)
    if len(rare) == 0 or len(y) <= k:
        return {t: np.nan for t in ("safe", "borderline", "rare", "outlier")}
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X[rare])
    same = (y[idx[:, 1:]] == 1).sum(axis=1)
    return {"safe": float(np.mean(same >= 4)), "borderline": float(np.mean((same >= 2) & (same <= 3))),
            "rare": float(np.mean(same == 1)), "outlier": float(np.mean(same == 0))}


def describe(X, y, max_n=2000, seed=0, prefix=""):
    X, y = _sub(np.asarray(X, float), np.asarray(y, int), max_n, seed)
    n_rare = int((y == 1).sum())
    out = {"n": len(y), "n_rare": n_rare,
           "ir": float((len(y) - n_rare) / max(n_rare, 1)),
           "f1": f1_fisher(X, y), "n1": n1_borderline(X, y), "n3": n3_nn_error(X, y)}
    out.update({f"type_{k}": v for k, v in minority_types(X, y).items()})
    return {prefix + k: v for k, v in out.items()}
