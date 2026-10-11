"""Context descriptors per state for the meta-regression (guideline G7).

``seed_*`` descriptors are computed on the initial labeled set (observable before
deployment); ``pool_*`` descriptors on the fully labeled pool (explanatory only).
"""
from __future__ import annotations

import pandas as pd

from rare_al.data.complexity import describe
from rare_al.data.splits import initial_mask, make_split
from rare_al.engine.states import materialize


def state_descriptors(states, cfg, budget="main", start="warm", max_n=2000):
    b = cfg["budgets"][budget]
    rows = []
    for st in states:
        X, y, t, is_test = materialize(st)
        Xtr, _Xte, ytr, _yte = make_split(X, y, st["state_seed"], cfg["data"]["test_fraction"],
                                          st["split"], t, is_test)
        lab = initial_mask(ytr, b["n_init"], st["state_seed"], start)
        row = {"state_id": st["state_id"], "dataset": st["dataset"]}
        row.update(describe(Xtr[lab], ytr[lab], max_n, st["state_seed"], prefix="seed_"))
        row.update(describe(Xtr, ytr, max_n, st["state_seed"], prefix="pool_"))
        rows.append(row)
    return pd.DataFrame(rows)
