"""Paired differences to the reference strategy (within state, learner, budget, start)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rare_al.analysis.load import TRAJ_KEYS
from rare_al.metrics.endpoints import trajectory_endpoints

DEGENERATE = {"entropy", "least_confident"}   # binary-equivalent to Margin (controls)
PAIR_KEYS = ["state_id", "learner", "budget", "start"]
HIGHER_IS_BETTER = {"auprc": True, "recall": True, "cost_norm": False, "aubc": True,
                    "discovery_rate": True, "n_rare_labeled": True}


def final_rows(df: pd.DataFrame, endpoints: bool = True, k_list=(1, 5, 10)) -> pd.DataFrame:
    """One row per trajectory (final round), with trajectory endpoints if requested."""
    if endpoints:
        return trajectory_endpoints(df, TRAJ_KEYS, k_list)
    mx = df.groupby(TRAJ_KEYS)["round"].transform("max")
    return df[df["round"] == mx].copy()


def paired(final: pd.DataFrame, reference="margin", metrics=None) -> pd.DataFrame:
    """Add ``ref_<m>`` and ``delta_<m>`` columns; ``delta`` = AUPRC delta (v1 name)."""
    if metrics is None:
        metrics = [m for m in ["auprc", "recall", "cost_norm", "aubc", "discovery_rate",
                               "n_rare_labeled", "labels_to_1", "labels_to_5", "labels_to_10"]
                   if m in final] + [c for c in final.columns if c.startswith("r_at_")]
    ref = final[final.strategy == reference][PAIR_KEYS + metrics].rename(
        columns={m: f"ref_{m}" for m in metrics})
    out = final.merge(ref, on=PAIR_KEYS, how="inner")
    for m in metrics:
        out[f"delta_{m}"] = out[m] - out[f"ref_{m}"]
    out["delta"] = out["delta_auprc"]
    out["rel_delta"] = out["delta"] / out["ref_auprc"].clip(lower=1e-12)
    out["ref"] = out["ref_auprc"]
    return out


def dataset_unit(p: pd.DataFrame, value="delta", by=("learner", "budget", "start", "strategy")) -> pd.DataFrame:
    """Average seeds within dataset (inferential unit for real data)."""
    return p.groupby(list(by) + ["dataset", "domain"], dropna=False)[value].mean().reset_index()


def wide_by_unit(final: pd.DataFrame, metric="auprc", unit="dataset", learner=None, budget="main",
                 start="warm") -> pd.DataFrame:
    f = final
    if learner is not None:
        f = f[f.learner == learner]
    f = f[(f.budget == budget) & (f.start == start)]
    return f.groupby([unit, "strategy"])[metric].mean().unstack("strategy")
