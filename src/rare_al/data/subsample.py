"""Rarity-preserving subsampling.

v1 kept every rare case and subsampled the majority to reach ``n_max``, which
compressed the imbalance ratio (e.g. protein_homo 111 -> 5.2). Here subsampling is
stratified: the prevalence is preserved, and the target size is enlarged when
needed so that at least ``min_rare`` rare cases remain (hard ceiling ``n_hard_max``).
"""
from __future__ import annotations

import math

import numpy as np


def target_size(n, n_rare, n_max, min_rare, n_hard_max):
    prevalence = n_rare / n
    needed = math.ceil(min_rare / prevalence) if prevalence > 0 else n
    return int(min(n, max(n_max, min(needed, n_hard_max))))


def stratified_subsample(y, size, seed, strata=None):
    """Indices of a stratified subsample of ``size`` cases (sorted).

    Stratification is on the class label, or on ``strata`` if given (e.g. label x
    provided train/test flag). Every stratum keeps at least one case.
    """
    y = np.asarray(y)
    n = len(y)
    if size >= n:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    groups = np.asarray(y if strata is None else strata)
    keys, counts = np.unique(groups, return_counts=True)
    alloc = np.maximum(1, np.floor(size * counts / n).astype(int))
    alloc = np.minimum(alloc, counts)
    # distribute the remainder by largest fractional part
    rem = size - alloc.sum()
    frac = size * counts / n - np.floor(size * counts / n)
    for i in np.argsort(-frac):
        if rem <= 0:
            break
        if alloc[i] < counts[i]:
            alloc[i] += 1
            rem -= 1
    keep = [rng.choice(np.flatnonzero(groups == k), a, replace=False) for k, a in zip(keys, alloc)]
    return np.sort(np.concatenate(keep))


def legacy_cap(y, n_max, seed):
    """v1 behaviour (keeps all rare cases). Provided for the moderate stratum audit only."""
    y = np.asarray(y)
    if len(y) <= n_max:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    rare = np.flatnonzero(y == 1)
    major = np.flatnonzero(y == 0)
    n_major_keep = min(max(0, n_max - len(rare)), len(major))
    return np.sort(np.r_[rare, rng.choice(major, n_major_keep, replace=False)])
