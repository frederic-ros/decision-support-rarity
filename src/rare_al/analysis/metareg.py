"""Mixed-effects meta-regression of paired regret on context descriptors (Table `tab_metareg`).

Model per strategy family:  delta ~ C(ir_stratum) + tree_learner + long_budget
                                     + pool_f1 + pool_n1 + pool_type_rare_share ...
with a random intercept for dataset (states nested in datasets are averaged over seeds
first to keep the model tractable).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rare_al.learners import LEARNER_FAMILY
from rare_al.stats.core import mixed_effects
from rare_al.strategies.registry import family

IR_BINS = [0, 10, 50, 200, np.inf]
IR_LABELS = ["lt10", "10_50", "50_200", "ge200"]


def prepare_frame(p: pd.DataFrame, desc: pd.DataFrame | None = None, reference="margin"):
    d = p[p.strategy != reference].copy()
    d["family"] = d.strategy.map(family)
    d["tree_learner"] = d.learner.map(lambda x: int(LEARNER_FAMILY.get(x, "") != "linear"))
    d["long_budget"] = (d.budget == "long").astype(int)
    d["ir_stratum"] = pd.cut(d.ir.astype(float), IR_BINS, labels=IR_LABELS, right=False).astype(str)
    if desc is not None:
        d = d.merge(desc.drop(columns=["dataset"], errors="ignore"), on="state_id", how="left")
    return d


def fit_by_family(d: pd.DataFrame, descriptors=("pool_f1", "pool_n1", "pool_type_rare"),
                  response="delta", group="dataset", estimator="ols_cluster"):
    """Per-family meta-regression of paired regret on context descriptors.

    All descriptors are dataset-level, so the reported estimator is OLS with
    standard errors clustered by dataset (deterministic across platforms); pass
    ``estimator="auto"`` to try a dataset random intercept first (see
    :func:`rare_al.stats.core.mixed_effects`)."""
    terms = ["C(ir_stratum)", "tree_learner", "long_budget"] + [x for x in descriptors if x in d]
    terms = [t for t in terms if t.startswith("C(") or d[t].nunique() > 1]
    out = []
    for fam, g in d.groupby("family"):
        agg = g.groupby([group, "learner", "budget", "strategy"] + [t for t in terms if not t.startswith("C(")]
                        + ["ir_stratum"], dropna=False)[response].mean().reset_index()
        if agg[group].nunique() < 3:
            continue
        tab, _fit = mixed_effects(agg.dropna(), f"{response} ~ " + " + ".join(terms), group,
                                  estimator=estimator)
        tab.insert(0, "family", fam)
        out.append(tab)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
