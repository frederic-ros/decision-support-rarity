"""Verbatim port of the v1 (TMLR) acquisition function. Do not modify: v13 results depend on it."""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors
from rare_al.strategies.v1.common import margin_uncertainty, normalize, nearest_distance

def constrained_cover(clf, Xpool, batch, rng, Xl=None, candidate_mult=5, **kwargs):
    u = margin_uncertainty(clf, Xpool)
    m = min(len(Xpool), candidate_mult*batch)
    top = np.argsort(-u)[:m]
    coverage = nearest_distance(Xpool[top], Xl)
    score = 0.70*normalize(u[top]) + 0.30*normalize(coverage)
    return top[np.argsort(-score)[:batch]]

def _delta(X, k=10, quantile=0.10):
    if len(X) <= 2:
        return 1e-12
    n = min(k+1, len(X))
    nn = NearestNeighbors(n_neighbors=n).fit(X)
    d, _ = nn.kneighbors(X)
    return max(float(np.quantile(d[:,-1], quantile)), 1e-12)

def probcover(clf, Xpool, batch, rng, Xl=None, delta=None, **kwargs):
    """Tabular adaptation of greedy delta-ball coverage.

    Standardized input features are used as the representation.
    """
    Xl = np.asarray(Xl)
    Xall = np.vstack([Xl, Xpool])
    n_l = len(Xl)
    if delta is None:
        delta = _delta(Xall)

    nn = NearestNeighbors(radius=delta).fit(Xall)
    neighbors = nn.radius_neighbors(Xall, radius=delta, return_distance=False)
    uncovered = np.ones(len(Xall), dtype=bool)
    for center in range(n_l):
        uncovered[np.asarray(neighbors[center], int)] = False

    candidates = np.arange(n_l, len(Xall))
    selected_positions = []
    for _ in range(min(batch, len(Xpool))):
        scores = np.array([
            uncovered[np.asarray(neighbors[c], int)].sum()
            for c in candidates
        ], float)
        scores[selected_positions] = -1
        best = scores.max()
        if best <= 0:
            break
        ties = np.flatnonzero(scores == best)
        pos = int(rng.choice(ties))
        selected_positions.append(pos)
        uncovered[np.asarray(neighbors[candidates[pos]], int)] = False

    result = [int(candidates[pos]-n_l) for pos in selected_positions]
    if len(result) < batch:
        used = set(result)
        for i in range(len(Xpool)):
            if i not in used:
                result.append(i)
                if len(result) >= batch:
                    break
    return np.asarray(result[:batch], int)
