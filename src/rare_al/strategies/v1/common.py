"""Verbatim port of the v1 (TMLR) acquisition function. Do not modify: v13 results depend on it."""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors

def positive_probability(clf, X):
    probs = clf.predict_proba(X)
    classes = list(clf.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return probs[:, classes.index(1)]

def margin_uncertainty(clf, X):
    p = positive_probability(clf, X)
    return 1.0 - np.abs(2.0 * p - 1.0)

def entropy(clf, X):
    p = np.clip(positive_probability(clf, X), 1e-12, 1 - 1e-12)
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def normalize(v):
    v = np.asarray(v, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    return (v - lo) / (hi - lo + 1e-12)

def nearest_distance(X, Y):
    if len(Y) == 0:
        return np.full(len(X), np.inf)
    nn = NearestNeighbors(n_neighbors=1).fit(Y)
    d, _ = nn.kneighbors(X)
    return d[:, 0]
