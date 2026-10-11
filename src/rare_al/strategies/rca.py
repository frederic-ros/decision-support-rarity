"""Risk-controlled acquisition (RCA): the framework instance of Section 3.7 (Option B).

Each batch = Margin anchor ((1-q_t) b queries) + discovery part (q_t b queries).
The discovery arm is chosen by Thompson sampling on the rate of *novel* rare
cases (rare cases farther from all previously labeled rare cases than the median
nearest-rare distance). q_t adapts multiplicatively within [q_min, q_max] and falls
back to q_min after ``patience`` rounds without any rare case. Hyperparameters are
to be fixed on the synthetic grid only.
"""
from __future__ import annotations

import numpy as np

from rare_al.strategies.base import Strategy, fill_unique
from rare_al.strategies.v1.common import nearest_distance


class RCA(Strategy):
    name = "rca"

    def __init__(self, **params):
        super().__init__(**params)
        from rare_al.strategies.registry import build
        self.arm_names = list(params.get("arms", ["most_likely_positive", "kmeans_diversity", "ens"]))
        self.arms = {a: build(a) for a in self.arm_names}
        self.q = float(params.get("q0", 0.25))
        self.q_min = float(params.get("q_min", 0.125))
        self.q_max = float(params.get("q_max", 0.5))
        self.step = float(params.get("step", 1.5))
        self.patience = int(params.get("patience", 3))
        self.alpha = {a: 1.0 for a in self.arm_names}
        self.beta = {a: 1.0 for a in self.arm_names}
        self._dry = 0
        self._arm = None
        self._disc = np.array([], int)

    def select(self, ctx):
        n_disc = max(1, int(round(self.q * ctx.batch)))
        draws = {a: ctx.rng.beta(self.alpha[a], self.beta[a]) for a in self.arm_names}
        self._arm = max(draws, key=draws.get)
        disc = list(self.arms[self._arm].select(ctx))[:n_disc]
        self._disc = np.asarray(disc, int)
        # novelty threshold computed before labels are revealed
        rare = ctx.Xl[ctx.yl == 1]
        if len(rare) >= 2:
            nearest = [np.partition(np.linalg.norm(rare - r, axis=1), 1)[1] for r in rare]
            self._thr = float(np.median(nearest))
        else:
            self._thr = -np.inf
        self._rare_before = rare
        return fill_unique(disc, ctx.margin_order(), ctx.batch)

    def observe(self, ctx, selected, labels):
        labels = np.asarray(labels, int)
        pos = {int(s): i for i, s in enumerate(selected)}
        disc_labels = np.array([labels[pos[int(i)]] for i in self._disc if int(i) in pos], int)
        disc_X = ctx.Xpool[[i for i in self._disc if int(i) in pos]]
        novel = 0
        if disc_labels.sum() > 0:
            Xr = disc_X[disc_labels == 1]
            d = nearest_distance(Xr, self._rare_before) if len(self._rare_before) else np.full(len(Xr), np.inf)
            novel = int(np.sum(d > self._thr))
        n = max(len(disc_labels), 1)
        self.alpha[self._arm] += novel / n
        self.beta[self._arm] += 1 - novel / n
        self.q = min(self.q_max, self.q * self.step) if novel > 0 else max(self.q_min, self.q / self.step)
        self._dry = 0 if labels.sum() > 0 else self._dry + 1
        if self._dry >= self.patience:
            self.q = self.q_min

    def diagnostics(self):
        return {"arm_mix": self._arm or "", "rca_q": self.q}
