"""Verbatim port of the v1 (TMLR) acquisition function. Do not modify: v13 results depend on it."""
from __future__ import annotations
import numpy as np
from rare_al.strategies.v1.common import margin_uncertainty, normalize, nearest_distance

def rareguard(clf, Xpool, batch, rng, Xl=None, yl=None, candidate_mult=5, **kwargs):
    u = margin_uncertainty(clf, Xpool)
    if Xl is None or yl is None or np.sum(yl == 1) < 2:
        return np.argsort(-u)[:batch]
    m = min(len(Xpool), candidate_mult*batch)
    top = np.argsort(-u)[:m]
    dr = nearest_distance(Xpool[top], Xl[yl == 1])
    dm = nearest_distance(Xpool[top], Xl[yl == 0])
    affinity = dm / (dr + dm + 1e-12)
    score = 0.75*normalize(u[top]) + 0.25*affinity
    return top[np.argsort(-score)[:batch]]
