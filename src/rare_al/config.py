"""Configuration loading with defaults and validation.

Configurations are JSON files. A configuration may ``extends`` another one
(path relative to the file); keys are merged recursively.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULTS = {
    "master_seed": 20260822,
    "results_dir": "results/run",
    "n_jobs": 1,
    "reference_strategy": "margin",
    "learners": ["logreg", "rf"],
    "strategies": ["margin", "random"],
    "budgets": {"main": {"n_init": 30, "batch": 8, "rounds": 8}},
    "starts": ["warm"],
    "data": {"test_fraction": 0.3, "cache_dir": "data/cache"},
    "endpoints": {
        "capacities": [0.01, 0.05],       # fractions of the test set flagged (R@c)
        "k_list": [1, 5, 10],             # labels-to-k
        "cost_fn": 1.0, "cost_fp": 1.0,    # overridden per dataset when documented
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path) -> dict:
    path = Path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    return _merge(DEFAULTS, cfg) if not parent else cfg


def normalize_strategies(entries) -> list[dict]:
    """Strategies may be given as names or as {"name", "alias", "params"} dicts."""
    out = []
    for e in entries:
        if isinstance(e, str):
            out.append({"name": e, "alias": e, "params": {}})
        else:
            out.append({"name": e["name"], "alias": e.get("alias", e["name"]),
                        "params": e.get("params", {})})
    aliases = [s["alias"] for s in out]
    if len(set(aliases)) != len(aliases):
        raise ValueError(f"Duplicate strategy aliases: {aliases}")
    return out


def validate(cfg: dict) -> None:
    from rare_al.strategies.registry import REGISTRY
    from rare_al.learners import LEARNERS
    strategies = normalize_strategies(cfg["strategies"])
    names = {s["alias"] for s in strategies}
    if cfg["reference_strategy"] not in names:
        raise ValueError("The reference strategy must be part of `strategies`.")
    for s in strategies:
        if s["name"] not in REGISTRY:
            raise KeyError(f"Unknown strategy: {s['name']}")
    for lr in cfg["learners"]:
        if lr not in LEARNERS:
            raise KeyError(f"Unknown learner: {lr}")
    for name, b in cfg["budgets"].items():
        for key in ("n_init", "batch", "rounds"):
            if key not in b:
                raise KeyError(f"Budget regime {name} lacks {key}")
    for s in cfg["starts"]:
        if s not in ("warm", "cold"):
            raise ValueError(f"Unknown start regime: {s}")
