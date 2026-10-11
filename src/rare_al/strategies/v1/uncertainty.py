"""Verbatim port of the v1 (TMLR) acquisition function. Do not modify: v13 results depend on it."""
from __future__ import annotations
import numpy as np
from rare_al.strategies.v1.common import margin_uncertainty, entropy, positive_probability

def margin(clf, Xpool, batch, rng, **kwargs):
    return np.argsort(-margin_uncertainty(clf, Xpool), kind="stable")[:batch]

def entropy_select(clf, Xpool, batch, rng, **kwargs):
    return np.argsort(-entropy(clf, Xpool), kind="stable")[:batch]

def least_confident(clf, Xpool, batch, rng, **kwargs):
    p = positive_probability(clf, Xpool)
    score = 1.0 - np.maximum(p, 1-p)
    return np.argsort(-score, kind="stable")[:batch]

def random_select(clf, Xpool, batch, rng, **kwargs):
    return rng.choice(len(Xpool), min(batch, len(Xpool)), replace=False)

def robust_uncertainty(clf, Xpool, batch, rng, epsilon=0.10, **kwargs):
    """Epsilon-greedy uncertainty baseline.

    Within a batch, each query is the current most-uncertain remaining point
    with probability 1-epsilon and a uniformly random remaining point otherwise.
    """
    u = margin_uncertainty(clf, Xpool)
    remaining = list(range(len(Xpool)))
    selected = []
    for _ in range(min(batch, len(remaining))):
        if rng.rand() < epsilon:
            j = int(rng.randint(len(remaining)))
        else:
            vals = np.asarray([u[i] for i in remaining])
            j = int(np.argmax(vals))
        selected.append(remaining.pop(j))
    return np.asarray(selected, int)
