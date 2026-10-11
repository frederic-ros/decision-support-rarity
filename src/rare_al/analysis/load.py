"""Loading results in the canonical v2 schema, including v1 (TMLR v13) archives.

Canonical trajectory schema (one row per round): identifiers
``experiment, dataset, domain, regime, ir, d, sep, seed, state_id, state_seed,
learner, budget, start, strategy`` + metrics (``auprc, recall, n_labeled,
n_rare_labeled`` and, for v2 runs, ``r_at_*, cost_norm, p_unique, select_s ...``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rare_al.data.registry import REGISTRY

TRAJ_KEYS = ["state_id", "learner", "budget", "start", "strategy"]


def load_trajectories(path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df


def from_v1(df: pd.DataFrame, experiment: str, budget: str = "main") -> pd.DataFrame:
    """Map a v1 raw CSV (synthetic_grid / rarity_stress / real_benchmark) to the v2 schema."""
    d = df.rename(columns={"clf": "learner", "method": "strategy"}).copy()
    d["strategy_name"] = d["strategy"]
    d["budget"] = budget
    d["start"] = "warm"
    if experiment == "synthetic":
        d["experiment"] = "synthetic"
        d["dataset"] = "syn_" + d["regime"].astype(str)
        d["domain"] = "synthetic"
        d["state_id"] = ("synthetic|" + d.regime.astype(str) + "|" + d.ir.astype(str) + "|" +
                         d.d.astype(str) + "|" + d.sep.astype(str) + "|" + d.seed.astype(str))
    elif experiment == "rarity":
        d["experiment"] = "rarity"
        d["dataset"] = "syn_" + d["regime"].astype(str)
        d["domain"] = "synthetic"
        d["state_id"] = "rarity|" + d.regime.astype(str) + "|" + d.ir.astype(str) + "|" + d.seed.astype(str)
    elif experiment == "real":
        d["experiment"] = "real"
        d["dataset"] = "imblearn_" + d["dataset"].astype(str)
        d["domain"] = d["dataset"].map(lambda x: REGISTRY[x].domain if x in REGISTRY else "moderate_stratum")
        d["origin"] = d["dataset"].map(lambda x: REGISTRY[x].origin if x in REGISTRY else None)
        d["state_id"] = "real|" + d.dataset + "|" + d.seed.astype(str)
    else:
        raise ValueError(experiment)
    return d


def load_v1_archive(root) -> dict:
    """Load the Zenodo v13 archive (``results/`` folder) into v2 frames."""
    root = Path(root)
    out = {}
    p = root / "paper1" / "synthetic_grid" / "synthetic_grid.csv"
    if p.exists():
        out["synthetic"] = from_v1(pd.read_csv(p), "synthetic", "main")
    p = root / "low_budget" / "synthetic_grid" / "synthetic_grid.csv"
    if p.exists():
        out["synthetic_low"] = from_v1(pd.read_csv(p), "synthetic", "low")
    p = root / "paper1" / "rarity_stress" / "rarity_stress.csv"
    if p.exists():
        out["rarity"] = from_v1(pd.read_csv(p), "rarity", "main")
    p = root / "paper1" / "real" / "real_benchmark.csv"
    if p.exists():
        out["real"] = from_v1(pd.read_csv(p), "real", "main")
    for tag, sub in (("oracle_lr", "paper1"), ("oracle_rf", "oracle_rf")):
        q = root / sub / "oracle"
        if (q / "oracle_points.csv").exists():
            out[tag] = q
    q = root / "paper1" / "observability"
    if q.exists():
        out["observability"] = q
    return out
