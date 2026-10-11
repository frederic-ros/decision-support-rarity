"""Rare-class-targeted strategies, adapted to tabular data.

These are *controlled tabular adaptations*: standardized features replace learned
representations, and the downstream learner's rare-class score replaces network
outputs. See docs/STRATEGIES.md for the correspondence with the original papers.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.neighbors import NearestNeighbors

from rare_al.strategies.base import Strategy, fill_unique


def _sorted_all(ctx):
    """Scores of labeled + pool points, and the ascending order along the 1-D score line."""
    from rare_al.strategies.v1.common import positive_probability
    s_lab = positive_probability(ctx.clf, ctx.Xl)
    s = np.r_[s_lab, ctx.p_pool]
    lab = np.r_[ctx.yl, np.full(len(ctx.Xpool), -1)]
    # tiny deterministic jitter so that ties do not all collapse on one position
    order = np.lexsort((ctx.rng.rand(len(s)), s))
    return s, lab, order, len(ctx.Xl)


class SimilarFLQMI(Strategy):
    """SIMILAR with facility-location mutual information (Kothawade et al., 2021).

    Query set Q = labeled rare cases; RBF similarity on standardized features;
    greedy maximization of FLQMI(A;Q) = sum_q max_a s(q,a) + eta * sum_a max_q s(a,q).
    """
    name = "similar_flqmi"

    def select(self, ctx):
        Q = ctx.Xl[ctx.yl == 1]
        if len(Q) == 0:
            return ctx.margin_order()[: ctx.batch]
        eta = float(self.params.get("eta", 1.0))
        max_cand = int(self.params.get("max_candidates", 3000))
        cand = np.arange(len(ctx.Xpool))
        if len(cand) > max_cand:
            cand = np.sort(ctx.rng.choice(cand, max_cand, replace=False))
        D = np.sqrt(((ctx.Xpool[cand][:, None, :] - Q[None, :, :]) ** 2).sum(-1))  # |cand| x |Q|
        sigma = np.median(D) + 1e-12
        S = np.exp(-(D ** 2) / (2 * sigma ** 2))
        cover = np.zeros(len(Q))
        chosen = []
        avail = np.ones(len(cand), bool)
        for _ in range(min(ctx.batch, len(cand))):
            gain = np.maximum(S - cover[None, :], 0).sum(1) + eta * S.max(1)
            gain[~avail] = -np.inf
            j = int(np.argmax(gain))
            chosen.append(int(cand[j]))
            avail[j] = False
            cover = np.maximum(cover, S[j])
        return fill_unique(chosen, ctx.margin_order(), ctx.batch)


class Galaxy(Strategy):
    """GALAXY (Zhang et al., 2022), one-vs-rest linear graph on the score line.

    Points are ordered by rare-class score; labeled points of different classes that
    are adjacent along this line delimit gaps of unlabeled points. The midpoint of
    the shortest gap is queried (bisection); the batch is filled by repeatedly
    splitting the current shortest gap, then by Margin.
    """
    name = "galaxy"

    def select(self, ctx):
        _s, lab, order, n_l = _sorted_all(ctx)
        seq_lab = lab[order]
        pos_l = np.flatnonzero(seq_lab >= 0)
        gaps = []
        for a, b in zip(pos_l[:-1], pos_l[1:]):
            if seq_lab[a] != seq_lab[b] and b - a > 1:
                gaps.append((a + 1, b - 1))            # inclusive range of unlabeled positions
        chosen = []
        while gaps and len(chosen) < ctx.batch:
            gaps.sort(key=lambda g: g[1] - g[0])
            lo, hi = gaps.pop(0)
            mid = (lo + hi) // 2
            chosen.append(int(order[mid] - n_l))
            if mid - 1 >= lo:
                gaps.append((lo, mid - 1))
            if hi >= mid + 1:
                gaps.append((mid + 1, hi))
        return fill_unique(chosen, ctx.margin_order(), ctx.batch)


class Direct(Strategy):
    """DIRECT (Nuggehalli et al., 2025), one-vs-rest threshold search on the score line.

    The optimal separation threshold minimizes the class-balanced error of the
    labeled points along the score line; unlabeled points closest (in rank) to that
    threshold are queried, alternating above and below.
    """
    name = "direct"

    def select(self, ctx):
        _s, lab, order, n_l = _sorted_all(ctx)
        seq_lab = lab[order]
        pos_l = np.flatnonzero(seq_lab >= 0)
        y_l = seq_lab[pos_l]
        n_pos, n_neg = max((y_l == 1).sum(), 1), max((y_l == 0).sum(), 1)
        if (y_l == 1).sum() == 0:
            return ctx.margin_order()[: ctx.batch]
        # threshold between labeled positions c-1 and c: positives below c are errors,
        # negatives at or above c are errors.
        pos_below = np.r_[0, np.cumsum(y_l == 1)]
        neg_above = (y_l == 0).sum() - np.r_[0, np.cumsum(y_l == 0)]
        err = pos_below / n_pos + neg_above / n_neg
        c = int(np.argmin(err))
        cut = pos_l[c] if c < len(pos_l) else len(order)   # sequence position of threshold
        unl = np.flatnonzero(seq_lab < 0)
        below = unl[unl < cut][::-1]
        above = unl[unl >= cut]
        chosen = []
        i = j = 0
        while len(chosen) < ctx.batch and (i < len(above) or j < len(below)):
            if i < len(above):
                chosen.append(int(order[above[i]] - n_l)); i += 1
            if len(chosen) < ctx.batch and j < len(below):
                chosen.append(int(order[below[j]] - n_l)); j += 1
        return fill_unique(chosen, ctx.margin_order(), ctx.batch)


class AnchorAL(Strategy):
    """AnchorAL (Lesci & Vlachos, 2024): anchors -> sub-pool -> base strategy (Margin).

    Up to ``n_anchors`` labeled anchors per class (all rare anchors first); the
    sub-pool is the union of each anchor's nearest unlabeled neighbours, capped at
    ``subpool_size``; Margin ranks the sub-pool.
    """
    name = "anchoral"

    def select(self, ctx):
        n_anchors = int(self.params.get("n_anchors", 10))
        subpool_size = int(self.params.get("subpool_size", 500))
        anchors = []
        for cls in (1, 0):
            idx = np.flatnonzero(ctx.yl == cls)
            if len(idx) > n_anchors:
                idx = ctx.rng.choice(idx, n_anchors, replace=False)
            anchors.extend(idx.tolist())
        if not anchors:
            return ctx.margin_order()[: ctx.batch]
        per_anchor = max(1, math.ceil(subpool_size / len(anchors)))
        nn = NearestNeighbors(n_neighbors=min(per_anchor, len(ctx.Xpool))).fit(ctx.Xpool)
        _, nbr = nn.kneighbors(ctx.Xl[anchors])
        sub = np.unique(nbr.ravel())[:subpool_size]
        u = 1.0 - np.abs(2.0 * ctx.p_pool[sub] - 1.0)
        ranked = sub[np.argsort(-u, kind="stable")]
        return fill_unique(ranked, ctx.margin_order(), ctx.batch)


class ClassBalanced(Strategy):
    """Class-balanced acquisition in the spirit of Bengar et al. (2022).

    A rare-class quota moves the labeled class distribution towards balance; quota
    slots are filled with the most uncertain points among the highest-scored
    (predicted-rare) candidates, the other slots by Margin.
    """
    name = "class_balanced"

    def select(self, ctx):
        n_l, n_rare = len(ctx.yl), int((ctx.yl == 1).sum())
        deficit = max(0.0, (n_l + ctx.batch) / 2.0 - n_rare)
        quota = int(min(ctx.batch, math.ceil(min(deficit, ctx.batch * self.params.get("max_share", 0.5)))))
        by_p = np.argsort(-ctx.p_pool, kind="stable")[: max(quota * 5, quota)]
        u = 1.0 - np.abs(2.0 * ctx.p_pool[by_p] - 1.0)
        rare_slots = by_p[np.argsort(-u, kind="stable")][:quota]
        return fill_unique(rare_slots, ctx.margin_order(), ctx.batch)
