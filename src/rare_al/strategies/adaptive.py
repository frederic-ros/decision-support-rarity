"""Adaptive strategy selection.

* ``tailor`` -- TAILOR (Zhang et al., 2023): Thompson sampling over candidate
  strategies, one draw per batch slot, with the class-balance reward of the
  original paper (inverse frequency of the revealed label in the labeled set).
* ``albl``   -- Active Learning by Learning (Hsu & Lin, 2015), batch version:
  EXP3 over strategies, one arm per batch, reward = importance-weighted,
  class-balanced accuracy of the updated learner on all queried points.
"""
from __future__ import annotations

import numpy as np

from rare_al.strategies.base import Strategy, fill_unique

DEFAULT_ARMS = ["margin", "kmeans_diversity", "most_likely_positive", "coreset", "random"]


def _make_arms(names):
    from rare_al.strategies.registry import build
    return {n: build(n) for n in names}


class Tailor(Strategy):
    name = "tailor"

    def __init__(self, **params):
        super().__init__(**params)
        self.arm_names = list(params.get("arms", DEFAULT_ARMS))
        self.arms = _make_arms(self.arm_names)
        self.alpha = {a: 1.0 for a in self.arm_names}
        self.beta = {a: 1.0 for a in self.arm_names}
        self._slot_arms: list[str] = []
        self._last_counts: dict = {}

    def select(self, ctx):
        lists = {a: list(self.arms[a].select(ctx)) for a in self.arm_names}
        ptr = {a: 0 for a in self.arm_names}
        chosen, slot_arms, seen = [], [], set()
        fallback = ctx.margin_order()
        while len(chosen) < ctx.batch:
            draws = {a: ctx.rng.beta(self.alpha[a], self.beta[a]) for a in self.arm_names}
            arm = max(draws, key=draws.get)
            lst = lists[arm]
            while ptr[arm] < len(lst) and int(lst[ptr[arm]]) in seen:
                ptr[arm] += 1
            if ptr[arm] < len(lst):
                idx = int(lst[ptr[arm]]); ptr[arm] += 1
            else:
                idx = next(int(i) for i in fallback if int(i) not in seen)
            chosen.append(idx); slot_arms.append(arm); seen.add(idx)
        self._slot_arms = slot_arms
        self._last_counts = {c: int((ctx.yl == c).sum()) for c in (0, 1)}
        return np.asarray(chosen, int)

    def observe(self, ctx, selected, labels):
        inv = {c: 1.0 / max(self._last_counts.get(c, 0), 1) for c in (0, 1)}
        top = max(inv.values())
        for arm, y in zip(self._slot_arms, labels):
            r = inv[int(y)] / top
            self.alpha[arm] += r
            self.beta[arm] += 1.0 - r

    def diagnostics(self):
        if not self._slot_arms:
            return {}
        vals, cnt = np.unique(self._slot_arms, return_counts=True)
        return {"arm_mix": ";".join(f"{v}:{c}" for v, c in zip(vals, cnt))}


class ALBL(Strategy):
    name = "albl"

    def __init__(self, **params):
        super().__init__(**params)
        self.arm_names = list(params.get("arms", DEFAULT_ARMS))
        self.arms = _make_arms(self.arm_names)
        self.gamma = float(params.get("gamma", 0.2))
        self.w = np.ones(len(self.arm_names))
        self._last = None               # (arm index, prob) awaiting reward
        self._queried_X, self._queried_y, self._queried_w = [], [], []

    def _probs(self):
        K = len(self.w)
        return (1 - self.gamma) * self.w / self.w.sum() + self.gamma / K

    def _reward(self, ctx):
        if not self._queried_X:
            return None
        X = np.vstack(self._queried_X)
        y = np.concatenate(self._queried_y)
        w = np.concatenate(self._queried_w)
        from rare_al.strategies.v1.common import positive_probability
        pred = (positive_probability(ctx.clf, X) >= 0.5).astype(int)
        acc = []
        for c in (0, 1):
            m = y == c
            if m.any():
                acc.append(np.sum(w[m] * (pred[m] == c)) / np.sum(w[m]))
        return float(np.mean(acc))

    def select(self, ctx):
        probs = self._probs()
        if self._last is not None:
            r = self._reward(ctx)
            if r is not None:
                a, pa = self._last
                self.w[a] *= np.exp(self.gamma * (r / pa) / len(self.w))
                self.w /= self.w.max()
        probs = self._probs()
        a = int(ctx.rng.choice(len(self.w), p=probs))
        self._last = (a, float(probs[a]))
        self._arm = self.arm_names[a]
        sel = self.arms[self._arm].select(ctx)
        return fill_unique(sel, ctx.margin_order(), ctx.batch)

    def observe(self, ctx, selected, labels):
        self._queried_X.append(ctx.Xpool[selected])
        self._queried_y.append(np.asarray(labels, int))
        self._queried_w.append(np.full(len(selected), 1.0 / self._last[1]))

    def diagnostics(self):
        return {"arm_mix": getattr(self, "_arm", "")}
