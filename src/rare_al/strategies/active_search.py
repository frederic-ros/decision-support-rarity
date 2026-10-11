"""Active search: find as many rare cases as possible under a budget.

* ``most_likely_positive`` -- greedy search on the learner's scores; what fraud
  investigators do when they review the highest-scored alerts
  (Attenberg & Provost, 2010).
* ``ens`` -- efficient nonmyopic search (Jiang et al., 2017) with the k-NN
  probability model of the original paper; batch selection by sequential
  simulation with most-likely fictional labels. It accounts for the delayed value
  of discovering new rare regions (remaining-budget look-ahead).
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from rare_al.strategies.base import Strategy, fill_unique


class MostLikelyPositive(Strategy):
    name = "most_likely_positive"

    def select(self, ctx):
        return np.argsort(-ctx.p_pool, kind="stable")[: ctx.batch]


class ENS(Strategy):
    name = "ens"

    def select(self, ctx):
        k = int(self.params.get("k", 50))
        gamma = float(self.params.get("gamma", 0.05))
        n_cand = int(self.params.get("n_candidates", 300))
        horizon_cap = int(self.params.get("horizon_cap", 500))
        Xall = np.vstack([ctx.Xl, ctx.Xpool])
        n_l, n_p = len(ctx.Xl), len(ctx.Xpool)
        kk = min(k, len(Xall) - 1)
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xall)
        _, nbr = nn.kneighbors(ctx.Xpool)
        nbr = nbr[:, 1:]                                   # neighbours of each pool point
        # sufficient statistics of the k-NN model over labeled neighbours
        lab_y = np.full(len(Xall), -1)
        lab_y[:n_l] = ctx.yl
        is_lab = lab_y >= 0
        num = np.array([(lab_y[r][is_lab[r]] == 1).sum() for r in nbr], float)
        den = np.array([is_lab[r].sum() for r in nbr], float)
        # reverse neighbours restricted to pool points: pool j is affected by labeling pool i
        rev = [[] for _ in range(n_p)]
        for j, r in enumerate(nbr):
            for a in r:
                if a >= n_l:
                    rev[a - n_l].append(j)
        rev = [np.asarray(v, int) for v in rev]
        available = np.ones(n_p, bool)
        chosen = []
        T = max(1, min(ctx.budget_remaining, horizon_cap))
        for step in range(min(ctx.batch, n_p)):
            p = (gamma + num) / (1.0 + den)
            p_av = np.where(available, p, -np.inf)
            horizon = max(0, T - step - 1)
            order = np.argsort(-p_av, kind="stable")
            cand = order[: min(n_cand, int(available.sum()))]
            if horizon == 0:
                best = int(cand[0])
            else:
                top_len = min(horizon + k + 1, int(available.sum()))
                top = order[:top_len]
                ptop = p[top]
                scores = np.empty(len(cand))
                for ci, i in enumerate(cand):
                    aff = rev[i][available[rev[i]]]
                    aff = aff[aff != i]
                    keep = ~np.isin(top, np.r_[aff, i])
                    base = ptop[keep]
                    exp = 0.0
                    for yv, w in ((1.0, p[i]), (0.0, 1.0 - p[i])):
                        merged = np.r_[base, (gamma + num[aff] + yv) / (2.0 + den[aff])]
                        h = min(horizon, len(merged))
                        exp += w * (np.partition(merged, len(merged) - h)[len(merged) - h:].sum()
                                    if h > 0 else 0.0)
                    scores[ci] = p[i] + exp
                best = int(cand[np.argmax(scores)])
            chosen.append(best)
            available[best] = False
            y_fict = 1.0 if p[best] >= 0.5 else 0.0          # most-likely fictional label
            aff = rev[best]
            num[aff] += y_fict
            den[aff] += 1.0
        return fill_unique(chosen, np.argsort(-ctx.p_pool), ctx.batch)
