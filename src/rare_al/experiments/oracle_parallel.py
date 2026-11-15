"""Sharded, parallel execution of the discovery oracle (revision, R2.6 / R3.4).

``rare_al.experiments.oracle.run`` is sequential. Extending the oracle to the random
forest and gradient-boosted learners (and to more base states) multiplies its cost,
so this wrapper splits the states across ``n_jobs`` worker processes, each writing to
its own shard directory (``<out>/shards/<i>/``; resumable through the oracle's own
completed-key logic), and then merges the shards into ``<out>/*.csv``.

Files already present in ``<out>`` (for instance the archived logistic-regression
oracle copied there before running) are kept and de-duplicated on merge, so only
missing (learner, state) pairs need to be computed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FILES = {"oracle_states.csv": ["learner", "state_id"],
         "oracle_regions.csv": ["learner", "state_id", "region_id"],
         "oracle_points.csv": ["learner", "state_id", "pool_loc"],
         "oracle_policies.csv": ["learner", "state_id", "policy"]}


def _shard_worker(cfg, states, out_dir):
    from rare_al.experiments import oracle as oracle_exp
    oracle_exp.run(cfg, states, out_dir)
    return str(out_dir)


def _done_pairs(out: Path) -> set:
    f = out / "oracle_states.csv"
    if not f.exists():
        return set()
    d = pd.read_csv(f, usecols=["learner", "state_id"])
    return set(zip(d.learner.astype(str), d.state_id.astype(str)))


def merge_shards(out_dir) -> dict:
    out = Path(out_dir)
    shards = sorted((out / "shards").glob("*")) if (out / "shards").exists() else []
    report = {}
    for name, keys in FILES.items():
        parts = [pd.read_csv(p) for p in [out / name] + [s / name for s in shards] if p.exists()]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        k = [c for c in keys if c in df.columns]
        df = df.drop_duplicates(subset=k, keep="first")
        df.to_csv(out / name, index=False)
        report[name] = len(df)
    return report


def run_parallel(cfg, states, out_dir, n_jobs=None, section="oracle"):
    """Run the oracle over ``states`` with ``n_jobs`` shards, then merge."""
    from joblib import Parallel, delayed
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_jobs = int(n_jobs or cfg[section].get("n_jobs", cfg.get("n_jobs", 1)))
    learners = cfg[section].get("learners", [cfg[section].get("classifier", "logreg")])
    # the oracle writes state ids as "<base state_id>|<round>"
    done = _done_pairs(out)
    rounds = range(1, cfg[section]["states"] + 1)

    def _complete(st):
        return all((lr, f"{st['state_id']}|{r}") in done for lr in learners for r in rounds)

    todo = [st for st in states if not _complete(st)]
    print(f"[oracle-parallel] {len(todo)}/{len(states)} base states to run "
          f"({len(learners)} learners) on {n_jobs} shards", flush=True)
    if todo:
        shards = [todo[i::n_jobs] for i in range(n_jobs)]
        Parallel(n_jobs=n_jobs)(
            delayed(_shard_worker)(cfg, sh, out / "shards" / f"{i:02d}")
            for i, sh in enumerate(shards) if sh)
    rep = merge_shards(out)
    print(f"[oracle-parallel] merged: {rep}", flush=True)
    return rep
