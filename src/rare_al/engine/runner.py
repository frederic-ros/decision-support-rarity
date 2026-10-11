"""Generic paired runner: states x budgets x starts x learners x strategies.

Results are appended to ``<out>/trajectories.csv`` (one row per round), which is
resumable: completed (state, learner, budget, start, strategy) keys are skipped.
Failures are logged to ``errors.csv`` without stopping the run.
"""
from __future__ import annotations

import traceback
from pathlib import Path

try:  # sklearn's Parallel/delayed propagate the sklearn thread config to workers,
    from sklearn.utils.parallel import Parallel, delayed   # which silences the UserWarning
except Exception:  # pragma: no cover  (very old scikit-learn)
    from joblib import Parallel, delayed

from rare_al.config import normalize_strategies
from rare_al.data.splits import make_split
from rare_al.engine.states import materialize
from rare_al.engine.trajectory import run_trajectory
from rare_al.io import append_rows, completed_keys
from rare_al.repro import stable_seed

KEYS = ["state_id", "learner", "budget", "start", "strategy"]
ID_FIELDS = ["experiment", "dataset", "domain", "origin", "regime", "ir", "ir_nominal", "d",
             "sep", "seed", "state_id", "state_seed", "split",
             "backbone", "positive_class", "ir_target"]


def _run_state(state, cfg, strategies, done):
    rows, errors = [], []
    try:
        X, y, t, is_test = materialize(state)
        prepared = make_split(X, y, state["state_seed"], cfg["data"]["test_fraction"],
                              state["split"], t, is_test)
    except Exception as exc:  # the whole state fails (e.g. degenerate split)
        return [], [{"state_id": state["state_id"], "error": repr(exc)}]
    ep = cfg["endpoints"]
    costs = state.get("costs") or {}
    capacities = sorted(set(ep["capacities"]) | ({state["capacity"]} if state.get("capacity") else set()))
    ident = {k: state.get(k) for k in ID_FIELDS}
    for bname, b in cfg["budgets"].items():
        for start in cfg["starts"]:
            for learner in cfg["learners"]:
                for s in strategies:
                    key = (state["state_id"], learner, bname, start, s["alias"])
                    if tuple(map(str, key)) in done:
                        continue
                    mseed = stable_seed(cfg["master_seed"], "method", s["alias"], state["state_seed"])
                    try:
                        hist = run_trajectory(
                            X, y, s["name"], learner, state["state_seed"], mseed,
                            b["n_init"], b["batch"], b["rounds"], cfg["data"]["test_fraction"],
                            start=start, split=state["split"], strategy_params=s["params"],
                            capacities=capacities, cost_fn=costs.get("cost_fn", ep["cost_fn"]),
                            cost_fp=costs.get("cost_fp", ep["cost_fp"]), prepared=prepared)
                    except Exception as exc:
                        errors.append({"state_id": state["state_id"], "learner": learner,
                                       "budget": bname, "start": start, "strategy": s["alias"],
                                       "error": repr(exc), "trace": traceback.format_exc()[-800:]})
                        continue
                    for h in hist:
                        rows.append({**ident, "learner": learner, "budget": bname, "start": start,
                                     "strategy": s["alias"], "strategy_name": s["name"],
                                     "method_seed": mseed, **h})
    return rows, errors


def run(cfg, states, out_dir, chunk=None, verbose=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target, err_path = out / "trajectories.csv", out / "errors.csv"
    strategies = normalize_strategies(cfg["strategies"])
    done = completed_keys(target, KEYS)
    n_expected = len(cfg["budgets"]) * len(cfg["starts"]) * len(cfg["learners"]) * len(strategies)
    todo = []
    for st in states:
        keys = [(st["state_id"], lr, b, s0, s["alias"]) for b in cfg["budgets"] for s0 in cfg["starts"]
                for lr in cfg["learners"] for s in strategies]
        if sum(tuple(map(str, k)) in done for k in keys) < n_expected:
            todo.append(st)
    if verbose:
        print(f"[runner] {len(todo)} states to run -> {target}", flush=True)
    n_jobs = int(cfg.get("n_jobs", 1))
    chunk = chunk or max(1, 4 * max(n_jobs, 1))
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        res = Parallel(n_jobs=n_jobs)(delayed(_run_state)(st, cfg, strategies, done) for st in part)
        for rows, errors in res:
            append_rows(rows, target)
            append_rows(errors, err_path)
        if verbose:
            print(f"[runner] {min(i + chunk, len(todo))}/{len(todo)} states", flush=True)
    return target
