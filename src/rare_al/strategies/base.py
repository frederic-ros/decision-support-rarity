"""Strategy interface.

A strategy is instantiated once per trajectory (so that adaptive strategies can
keep state across rounds). At each round the engine calls ``select(ctx)`` and,
after the expert has labeled the batch, ``observe(ctx, selected, labels)``.
Indices are *local* to ``ctx.Xpool``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rare_al.strategies.v1.common import positive_probability


@dataclass
class Context:
    clf: object
    learner: str
    Xpool: np.ndarray
    Xl: np.ndarray
    yl: np.ndarray
    batch: int
    rng: np.random.RandomState
    round: int = 0
    budget_remaining: int = 0          # labels still to acquire, including this batch
    pool_ids: np.ndarray | None = None  # global training indices of the pool
    extra: dict = field(default_factory=dict)
    _p: np.ndarray | None = None

    @property
    def p_pool(self) -> np.ndarray:
        """Predicted rare-class probability on the pool (cached)."""
        if self._p is None:
            self._p = positive_probability(self.clf, self.Xpool)
        return self._p

    def margin_order(self) -> np.ndarray:
        u = 1.0 - np.abs(2.0 * self.p_pool - 1.0)
        return np.argsort(-u, kind="stable")


class Strategy:
    name = "base"

    def __init__(self, **params):
        self.params = params

    def select(self, ctx: Context) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def observe(self, ctx: Context, selected: np.ndarray, labels: np.ndarray) -> None:
        return None

    def diagnostics(self) -> dict:
        """Optional per-round diagnostics logged by the engine (e.g. chosen arm)."""
        return {}


class FunctionStrategy(Strategy):
    """Wraps a stateless v1-style function f(clf, Xpool, batch, rng, Xl, yl, **params)."""

    def __init__(self, fn, **params):
        super().__init__(**params)
        self.fn = fn

    def select(self, ctx):
        return np.asarray(self.fn(clf=ctx.clf, Xpool=ctx.Xpool, Xl=ctx.Xl, yl=ctx.yl,
                                  batch=ctx.batch, rng=ctx.rng, **self.params), int)


def fill_unique(primary, fallback, batch):
    """Concatenate index lists without duplicates up to ``batch`` items."""
    out, seen = [], set()
    for seq in (primary, fallback):
        for i in seq:
            i = int(i)
            if i not in seen:
                out.append(i)
                seen.add(i)
                if len(out) >= batch:
                    return np.asarray(out, int)
    return np.asarray(out, int)
