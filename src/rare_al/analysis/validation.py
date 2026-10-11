"""Out-of-sample validation of the decision procedure (reviewer request).

Leave-one-domain-out: the strategy-selection rule is fit on all domains but one,
then applied to each held-out dataset using only its OBSERVABLE (initial-label)
descriptors, and the realized held-out gain of the choice is compared with two
references:

* ``margin``     -- always use the reference strategy (paired gain 0 by definition);
* ``mean_rank``  -- the single strategy with the best mean gain on the training
                    domains (context-free, risk-blind);

against two risk-aware rules fit on the training domains:

* ``risk_global``-- best mean training gain among strategies whose training tail
                    ``C_tau = P(delta < -tau)`` is within tolerance ``tau`` (alpha);
* ``risk_cond``  -- context-conditional: for each held-out dataset, the strategy
                    with the best mean gain over its ``k`` nearest training datasets
                    in observable-descriptor space, restricted to the tail-safe set.

The comparison with ``mean_rank`` isolates the contribution of the tail constraint
and of the descriptor conditioning; when the two rules select the same strategy
(the best-mean strategy is already tail-safe), the validation speaks only to the
value of conditioning on the learner, which is measured separately against a
learner-blind single-winner rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rare_al.analysis.paired import DEGENERATE


def _dataset_unit(paired_df, metric="delta"):
    """Average seeds -> one gain per (domain, dataset, learner, strategy)."""
    keep = paired_df[~paired_df.strategy.isin(DEGENERATE)].copy()
    return (keep.groupby(["domain", "dataset", "learner", "strategy"], dropna=False)[metric]
            .mean().reset_index())


def _nearest_datasets(test_ds, train_ds_feats, feats, k):
    """Indices of the k nearest training datasets by standardized descriptors."""
    cols = [c for c in feats if c in train_ds_feats.columns]
    if not cols or test_ds not in train_ds_feats.index:
        return list(train_ds_feats.index)            # fall back to all
    M = train_ds_feats[cols].to_numpy(float)
    mu, sd = np.nanmean(M, 0), np.nanstd(M, 0) + 1e-9
    Z = (M - mu) / sd
    z0 = (train_ds_feats.loc[[test_ds], cols].to_numpy(float) - mu) / sd
    d = np.sqrt(np.nansum((Z - z0) ** 2, axis=1))
    order = np.argsort(d)
    return list(train_ds_feats.index[order[:max(k, 1)]])


def leave_one_domain_out(paired_df, desc_df=None, reference="margin",
                         tau=0.10, k=5, metric="delta"):
    """Return a per-(held-out dataset, learner) table of realized gains for each rule.

    Selection and realized gain use the dataset unit (seeds averaged); the tail
    constraint ``C_tau`` is computed on the per-trajectory (per-seed) deltas, so
    that seed-averaging never hides lower-tail exposure.
    """
    raw = paired_df[~paired_df.strategy.isin(DEGENERATE) & (paired_df.strategy != reference)].copy()
    du = _dataset_unit(paired_df, metric)
    du = du[du.strategy != reference]
    domains = sorted(d for d in du.domain.dropna().unique())
    seed_cols = []
    ds_feat = None
    if desc_df is not None and len(desc_df):
        seed_cols = [c for c in desc_df.columns if c.startswith("seed_")]
        if seed_cols:
            ds_feat = desc_df.groupby("dataset")[seed_cols].mean()

    rows = []
    for dom in domains:
        tr, te = du[du.domain != dom], du[du.domain == dom]
        for learner in sorted(du.learner.unique()):
            trl, tel = tr[tr.learner == learner], te[te.learner == learner]
            if trl.empty or tel.empty:
                continue
            # tail from per-trajectory deltas (training domains, this learner)
            traw = raw[(raw.domain != dom) & (raw.learner == learner)]
            tail = traw.groupby("strategy")[metric].apply(lambda x: float((x < -tau).mean()))
            mean_tr = trl.groupby("strategy")[metric].mean()
            safe = list(tail[tail <= tau].index)
            mean_rank_pick = mean_tr.idxmax()
            safe_means = mean_tr[mean_tr.index.isin(safe)]
            risk_global_pick = safe_means.idxmax() if len(safe_means) else mean_rank_pick

            train_feats = None
            if ds_feat is not None:
                train_ds = [d for d in trl.dataset.unique() if d in ds_feat.index]
                train_feats = ds_feat.loc[train_ds] if train_ds else None

            for ds, g in tel.groupby("dataset"):
                gmap = g.set_index("strategy")[metric]
                # context-conditional pick
                cond_pick = risk_global_pick
                if train_feats is not None and len(train_feats):
                    nbrs = _nearest_datasets(ds, train_feats, seed_cols, k)
                    loc = trl[trl.dataset.isin(nbrs)].groupby("strategy")[metric].mean()
                    loc = loc[loc.index.isin(safe)] if len(safe) else loc
                    if len(loc):
                        cond_pick = loc.idxmax()
                rows.append({
                    "domain": dom, "learner": learner, "dataset": ds,
                    "margin": 0.0,
                    "mean_rank": float(gmap.get(mean_rank_pick, np.nan)),
                    "risk_global": float(gmap.get(risk_global_pick, np.nan)),
                    "risk_cond": float(gmap.get(cond_pick, np.nan)),
                    "oracle_best": float(gmap.max()),         # best in hindsight (upper ref)
                    "pick_mean_rank": mean_rank_pick,
                    "pick_risk_global": risk_global_pick,
                    "pick_risk_cond": cond_pick,
                })
    return pd.DataFrame(rows)


def summary(loo: pd.DataFrame, seed=11):
    """Per-rule held-out summary: mean gain, win rate vs Margin, worst 10%, and a
    Wilcoxon of risk_cond vs mean_rank across held-out datasets."""
    from rare_al.stats.core import boot_mean
    from scipy.stats import wilcoxon
    methods = ["margin", "mean_rank", "risk_global", "risk_cond", "oracle_best"]
    rows = []
    for m in methods:
        x = loo[m].dropna().to_numpy(float)
        mean, lo, hi = boot_mean(x, seed=seed)
        rows.append({"rule": m, "n": len(x), "mean_gain": mean, "ci_low": lo, "ci_high": hi,
                     "win_rate_vs_margin": float((x > 0).mean()),
                     "worst10": float(np.quantile(x, 0.10)) if len(x) else np.nan})
    out = pd.DataFrame(rows)
    # paired test: does the risk-aware context-conditional rule beat the mean-rank one?
    d = loo[["risk_cond", "mean_rank"]].dropna()
    diff = (d.risk_cond - d.mean_rank)
    try:
        stat, p = wilcoxon(diff) if (diff != 0).any() else (np.nan, 1.0)
    except ValueError:
        stat, p = np.nan, np.nan
    meta = {"risk_cond_vs_mean_rank_mean": float(diff.mean()) if len(diff) else np.nan,
            "wilcoxon_p": float(p), "n_pairs": int(len(diff))}
    return out, meta
