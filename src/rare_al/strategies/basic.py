"""Small additions to the v1 uncertainty family."""
from __future__ import annotations

import numpy as np

from rare_al.learners import make_learner
from rare_al.strategies.base import Strategy


class MarginRandomTies(Strategy):
    """Margin with random tie-breaking (probability-resolution check, Section 4.4).

    Tree ensembles trained on few rare cases output many identical probabilities;
    v1 Margin breaks ties by pool order (stable sort).
    """
    name = "margin_rtb"

    def select(self, ctx):
        u = 1.0 - np.abs(2.0 * ctx.p_pool - 1.0)
        return np.lexsort((ctx.rng.rand(len(u)), -u))[: ctx.batch]


class CommitteeSameLearner(Strategy):
    """Query-by-committee whose members use the *target* learner.

    v1 ``committee`` always used logistic-regression members, including when the
    downstream learner was a Random Forest; this variant removes that mismatch.
    """
    name = "committee_learner"

    def select(self, ctx):
        n_members = int(self.params.get("n_members", 5))
        preds = []
        for _ in range(n_members):
            idx = ctx.rng.choice(len(ctx.Xl), len(ctx.Xl), replace=True)
            if len(np.unique(ctx.yl[idx])) < 2:
                preds.append(ctx.p_pool)
                continue
            m = make_learner(ctx.learner, int(ctx.rng.randint(2**31 - 1)))
            m.fit(ctx.Xl[idx], ctx.yl[idx])
            p = m.predict_proba(ctx.Xpool)
            preds.append(p[:, list(m.classes_).index(1)])
        return np.argsort(-np.var(np.vstack(preds), axis=0), kind="stable")[: ctx.batch]
