"""Train/test splits, preprocessing, and initial labeled sets.

Split modes
-----------
* ``stratified`` -- v1 behaviour: stratified random split driven by the state seed.
* ``temporal``   -- test set = most recent ``test_fraction`` of cases (requires ``t``);
                    the state seed then only drives the initial labeled set.
* ``provided``   -- the dataset ships a train/test partition (``is_test``).

Preprocessing is fitted on the training pool only (transductive pool-based
setting): median imputation, then standardization.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _impute_scale(Xtr, Xte):
    Xtr = np.asarray(Xtr, float)
    Xte = np.asarray(Xte, float)
    if np.isnan(Xtr).any() or np.isnan(Xte).any():
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr)
        Xte = np.where(np.isnan(Xte), med, Xte)
    scaler = StandardScaler()
    return scaler.fit_transform(Xtr), scaler.transform(Xte)


def split_and_scale(X, y, split_seed, test_fraction):
    """v1-identical stratified split + scaling."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_fraction, stratify=y,
                                          random_state=split_seed)
    Xtr, Xte = _impute_scale(Xtr, Xte)
    return Xtr, Xte, ytr, yte


def make_split(X, y, state_seed, test_fraction=0.3, mode="stratified", t=None, is_test=None):
    if mode == "stratified":
        return split_and_scale(X, y, state_seed, test_fraction)
    if mode == "temporal":
        if t is None:
            raise ValueError("Temporal split requires a time vector `t`.")
        order = np.argsort(np.asarray(t), kind="stable")
        n_test = int(round(test_fraction * len(y)))
        te, tr = order[-n_test:], order[:-n_test]
    elif mode == "provided":
        if is_test is None:
            raise ValueError("Provided split requires `is_test`.")
        is_test = np.asarray(is_test, bool)
        te, tr = np.flatnonzero(is_test), np.flatnonzero(~is_test)
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    if y[te].sum() == 0 or y[tr].sum() == 0:
        raise ValueError("Split leaves one side without rare cases.")
    Xtr, Xte = _impute_scale(X[tr], X[te])
    return Xtr, Xte, y[tr], y[te]


def initial_mask(y, n_init, init_seed, start="warm"):
    """Initial labeled set.

    ``warm`` is v1-identical: one rare + one majority guaranteed, rest uniform.
    ``cold`` draws the whole initial set from the majority class (n_0^+ = 0).
    """
    rng = np.random.RandomState(init_seed)
    mask = np.zeros(len(y), dtype=bool)
    rare = np.flatnonzero(y == 1)
    major = np.flatnonzero(y == 0)
    if start == "cold":
        mask[rng.choice(major, min(n_init, len(major)), replace=False)] = True
        return mask
    if len(rare) == 0 or len(major) == 0:
        raise ValueError("Both classes must occur in the training set.")
    base = [rng.choice(rare), rng.choice(major)]
    remaining = np.setdiff1d(np.arange(len(y)), base)
    extra = rng.choice(remaining, min(max(0, n_init - 2), len(remaining)), replace=False)
    mask[np.r_[base, extra]] = True
    return mask
