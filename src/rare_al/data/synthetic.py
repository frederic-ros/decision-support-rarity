"""Controlled minority geometries (identical to v1 when ``min_rare=20``).

For extreme IR (200, 500) increase ``n`` so that ``n/(ir+1) >= min_rare``; otherwise
the realized IR is capped by the ``min_rare`` floor (see ``realized_ir``).
"""
from __future__ import annotations
import numpy as np

REGIMES = ("central","peripheral","overlap","multimodal","compact","diffuse")

def make_regime(regime, n, d, ir, sep, seed, min_rare=20):
    """Generate one binary rare-class geometry.

    The majority class is label 0 and the rare class is label 1.
    The imbalance ratio is n_majority / n_rare.
    """
    rng = np.random.RandomState(seed)
    n_rare = max(min_rare, int(round(n / (ir + 1))))
    n_major = n - n_rare

    if regime == "peripheral":
        X0 = rng.normal(0, 1, (n_major, d))
        mu = np.zeros(d); mu[0] = 2.0 * sep
        X1 = rng.normal(mu, 0.72, (n_rare, d))

    elif regime == "central":
        n_a = n_major // 2
        mu = np.zeros(d); mu[0] = 1.7 * sep
        X0 = np.vstack([
            rng.normal(mu, 0.82, (n_a, d)),
            rng.normal(-mu, 0.82, (n_major - n_a, d)),
        ])
        X1 = rng.normal(0, 0.45, (n_rare, d))

    elif regime == "overlap":
        X0 = rng.normal(0, 1, (n_major, d))
        mu = np.zeros(d); mu[0] = 0.55 * sep
        X1 = rng.normal(mu, 1.0, (n_rare, d))

    elif regime == "multimodal":
        X0 = rng.normal(0, 1, (n_major, d))
        n_a = n_rare // 2
        m1 = np.zeros(d); m1[0] = 1.7 * sep
        m2 = np.zeros(d); m2[min(1, d - 1)] = -1.7 * sep
        X1 = np.vstack([
            rng.normal(m1, 0.55, (n_a, d)),
            rng.normal(m2, 0.55, (n_rare - n_a, d)),
        ])

    elif regime == "compact":
        X0 = rng.normal(0, 1, (n_major, d))
        mu = np.zeros(d); mu[0] = 1.25 * sep
        X1 = rng.normal(mu, 0.30, (n_rare, d))

    elif regime == "diffuse":
        X0 = rng.normal(0, 0.8, (n_major, d))
        mu = np.zeros(d); mu[0] = 1.25 * sep
        X1 = rng.normal(mu, 1.6, (n_rare, d))

    else:
        raise ValueError(f"Unknown regime: {regime}")

    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n_major, int), np.ones(n_rare, int)]
    order = rng.permutation(n)
    return X[order], y[order]


def realized_ir(n, ir, min_rare=20):
    n_rare = max(min_rare, int(round(n / (ir + 1))))
    return (n - n_rare) / n_rare


def n_for_ir(ir, min_rare=20, n_min=2500):
    """Smallest pool size that realizes the nominal IR with at least ``min_rare`` rare cases."""
    return max(n_min, int(min_rare * (ir + 1)))
