"""One strictly paired active-learning trajectory.

For a given *state* (dataset realization, split, initial labels, budget, seed)
every strategy starts from the identical labeled set and pool. ``state_seed``
drives split + initial labels + learner seeds and is shared by all strategies;
``method_seed`` drives strategy-internal randomness only. With a v1 strategy, a
v1 learner and the ``stratified`` split, the trajectory is identical to v1.
"""
from __future__ import annotations

import time

import numpy as np

from rare_al.data.splits import initial_mask, make_split
from rare_al.learners import make_learner
from rare_al.metrics.endpoints import round_metrics
from rare_al.strategies.base import Context
from rare_al.strategies.registry import build
from rare_al.strategies.v1.common import positive_probability


class _PriorModel:
    """Stand-in model while the labeled set has a single class (cold start)."""
    classes_ = np.array([0, 1])

    def __init__(self, prior=0.5):
        self.prior = prior

    def predict_proba(self, X):
        return np.tile([1 - self.prior, self.prior], (len(X), 1))


def fit_or_prior(learner, seed, X, y):
    if len(np.unique(y)) < 2:
        return _PriorModel(), True
    clf = make_learner(learner, seed)
    clf.fit(X, y)
    return clf, False


def run_trajectory(X, y, strategy, learner, state_seed, method_seed, n_init, batch, rounds,
                   test_fraction=0.3, start="warm", split="stratified", t=None, is_test=None,
                   strategy_params=None, capacities=(0.01, 0.05), cost_fn=1.0, cost_fp=1.0,
                   prepared=None):
    """Return a list of per-round dict rows.

    ``prepared`` may carry a precomputed (Xtr, Xte, ytr, yte) to avoid re-splitting
    for every strategy of the same state.
    """
    if prepared is None:
        prepared = make_split(X, y, state_seed, test_fraction, split, t, is_test)
    Xtr, Xte, ytr, yte = prepared
    labeled = initial_mask(ytr, n_init, state_seed, start)
    strat = build(strategy, **(strategy_params or {}))
    history = []
    for rnd in range(rounds + 1):
        t0 = time.perf_counter()
        clf, is_prior = fit_or_prior(learner, state_seed + rnd, Xtr[labeled], ytr[labeled])
        fit_s = time.perf_counter() - t0
        p = positive_probability(clf, Xte)
        row = {"round": rnd, "n_labeled": int(labeled.sum()),
               "n_rare_labeled": int(np.sum(ytr[labeled] == 1)), "prior_model": int(is_prior),
               "fit_s": fit_s}
        row.update(round_metrics(yte, p, capacities, cost_fn, cost_fp))
        if rnd == rounds:
            history.append(row)
            break
        pool = np.flatnonzero(~labeled)
        rng = np.random.RandomState(method_seed + 1009 * rnd)
        ctx = Context(clf=clf, learner=learner, Xpool=Xtr[pool], Xl=Xtr[labeled], yl=ytr[labeled],
                      batch=min(batch, len(pool)), rng=rng, round=rnd,
                      budget_remaining=(rounds - rnd) * batch, pool_ids=pool)
        t0 = time.perf_counter()
        local = np.asarray(strat.select(ctx), int)
        row["select_s"] = time.perf_counter() - t0
        if len(local) != ctx.batch or len(np.unique(local)) != len(local) or \
                (len(local) and (local.min() < 0 or local.max() >= len(pool))):
            raise RuntimeError(f"{strategy} returned an invalid batch: {local}")
        labels = ytr[pool[local]]
        row["batch_rare"] = int(labels.sum())
        strat.observe(ctx, local, labels)
        row.update(strat.diagnostics())
        labeled[pool[local]] = True
        history.append(row)
    return history
