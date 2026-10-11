"""Operational endpoints (Section 3.3 of the paper).

Round-level metrics are computed by the engine on the test set; trajectory-level
endpoints (labels-to-k, discovery rate, AUBC) are derived post hoc from the logged
per-round rows.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def auprc(y, p) -> float:
    return float(average_precision_score(y, p))


def recall_at_capacity(y, p, capacity: float) -> float:
    """Rare-class recall when the top ``capacity`` fraction of cases is flagged."""
    y = np.asarray(y)
    n_flag = max(1, int(math.ceil(capacity * len(y))))
    top = np.argsort(-np.asarray(p), kind="stable")[:n_flag]
    return float(y[top].sum() / max(y.sum(), 1))


def min_normalized_cost(y, p, cost_fn=1.0, cost_fp=1.0) -> float:
    """Minimum expected cost over thresholds, normalized by the cost of flagging nothing.

    0 = perfect, 1 = no better than flagging nothing. Ties in scores are cut together.
    """
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    # only cut where the score changes (plus the "flag nothing" option)
    cut = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]
    P = ys.sum()
    costs = np.r_[cost_fn * P, cost_fn * (P - tp[cut]) + cost_fp * fp[cut]]
    return float(costs.min() / max(cost_fn * P, 1e-12))


def round_metrics(y_test, p_test, capacities, cost_fn, cost_fp) -> dict:
    out = {"auprc": auprc(y_test, p_test),
           "recall": float(((p_test >= 0.5) & (y_test == 1)).sum() / max(y_test.sum(), 1)),
           "cost_norm": min_normalized_cost(y_test, p_test, cost_fn, cost_fp),
           "p_unique": int(len(np.unique(np.round(p_test, 12)))),
           "p_max": float(np.max(p_test))}
    for c in capacities:
        out[f"r_at_{c:g}"] = recall_at_capacity(y_test, p_test, c)
    return out


# ---------------------------------------------------------------- trajectory level
def labels_to_k(traj: pd.DataFrame, k: int) -> float:
    """Labels spent until at least k rare labels are in the labeled set (NaN if never)."""
    hit = traj.loc[traj.n_rare_labeled >= k, "n_labeled"]
    return float(hit.min()) if len(hit) else np.nan


def discovery_rate(traj: pd.DataFrame) -> float:
    first, last = traj.iloc[0], traj.iloc[-1]
    spent = last.n_labeled - first.n_labeled
    return float((last.n_rare_labeled - first.n_rare_labeled) / spent) if spent > 0 else np.nan


def aubc(traj: pd.DataFrame, metric="auprc") -> float:
    """Normalized area under the budget curve (trapezoid over labels spent)."""
    x = traj.n_labeled.to_numpy(float)
    y = traj[metric].to_numpy(float)
    if len(x) < 2 or x[-1] == x[0]:
        return float(y.mean())
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(y, x) / (x[-1] - x[0]))


def trajectory_endpoints(df: pd.DataFrame, keys, k_list=(1, 5, 10)) -> pd.DataFrame:
    """One row per trajectory: final-round columns + trajectory endpoints (vectorized).

    Censored labels-to-k (k rare labels never reached) are set to final budget + one
    batch and flagged in ``censored_k`` so that paired differences stay defined.
    """
    d = df.sort_values(list(keys) + ["round"]).reset_index(drop=True)
    g = d.groupby(list(keys), sort=False)
    last = d.loc[g["round"].idxmax()].set_index(list(keys))
    first = d.loc[g["round"].idxmin()].set_index(list(keys))
    out = last.copy()
    step = g["n_labeled"].diff()
    out["_step"] = step.groupby([d[k] for k in keys]).median().reindex(out.index)
    for k in k_list:
        hit = d[d.n_rare_labeled >= k].groupby(list(keys))["n_labeled"].min().reindex(out.index)
        out[f"censored_{k}"] = hit.isna().astype(int)
        out[f"labels_to_{k}"] = hit.fillna(out["n_labeled"] + out["_step"].fillna(0))
    spent = out["n_labeled"] - first["n_labeled"]
    out["discovery_rate"] = (out["n_rare_labeled"] - first["n_rare_labeled"]) / spent.where(spent > 0)
    y_prev = g["auprc"].shift()
    area = (step * (d["auprc"] + y_prev) / 2).groupby([d[k] for k in keys]).sum().reindex(out.index)
    out["aubc"] = (area / spent.where(spent > 0)).fillna(out["auprc"])
    out["final_round"] = out["round"]
    return out.drop(columns=["_step"]).reset_index()
