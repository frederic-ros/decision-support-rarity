"""Verbatim port of the v1 (TMLR) acquisition function. Do not modify: v13 results depend on it."""
from __future__ import annotations
import numpy as np
from rare_al.learners import make_classifier
from rare_al.strategies.v1.common import positive_probability

def committee(clf, Xpool, batch, rng, Xl=None, yl=None, **kwargs):
    if Xl is None or yl is None:
        raise ValueError("Committee requires labeled data.")
    preds = []
    for q in range(5):
        idx = rng.choice(len(Xl), len(Xl), replace=True)
        member = make_classifier("logreg", int(rng.randint(2**31-1)))
        try:
            member.fit(Xl[idx], yl[idx])
            preds.append(positive_probability(member, Xpool))
        except Exception:
            preds.append(positive_probability(clf, Xpool))
    disagreement = np.var(np.vstack(preds), axis=0)
    return np.argsort(-disagreement)[:batch]
