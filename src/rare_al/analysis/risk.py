"""RQ1/RQ2 analyses: acquisition-risk profiles, scale decomposition (C1), context
dependence (learner, budget) and real-data (dataset-unit) summaries.

Definitions and bootstrap seeds follow the v13 analysis so that archived numbers
are reproduced exactly from the v1 CSVs (see tests/test_v13_regression.py).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rare_al.analysis.paired import DEGENERATE
from rare_al.stats.core import boot_mean, bayesian_signed_rank, wilcoxon_vs_reference


def _eligible(p, reference="margin", exclude=DEGENERATE):
    return p[~p.strategy.isin(set(exclude) | {reference})]


def risk_profile(p: pd.DataFrame, by=("learner", "strategy"), tau=0.10, tau_rel=0.25,
                 reference="margin", seed=7, metric="delta") -> pd.DataFrame:
    """Risk profile R_tau = (mean [CI], SD, C_tau) + Q05, worst, relative tail, win rate."""
    rows = []
    for key, g in _eligible(p, reference).groupby(list(by)):
        x = g[metric]
        mean, lo, hi = boot_mean(x, seed=seed)
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row.update({"n": len(g), "mean": mean, "ci_low": lo, "ci_high": hi, "sd": float(x.std()),
                    "q05": float(x.quantile(.05)), "worst": float(x.min()),
                    "p_lt_tau": float((x < -tau).mean()), "win_rate": float((x > 0).mean())})
        if metric == "delta" and "ref" in g:
            row["p_lt_tau_rel"] = float((g.delta < -tau_rel * g.ref).mean())
        if "recall_delta" in g or "delta_recall" in g:
            row["mean_recall_delta"] = float(g.get("delta_recall", g.get("recall_delta")).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def recall_table(p, reference="margin", seed=21):
    rows = []
    for (lr, s), g in _eligible(p, reference).groupby(["learner", "strategy"]):
        m, lo, hi = boot_mean(g.delta_recall, seed=seed)
        rows.append({"learner": lr, "strategy": s, "mean": m, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def _pooled_sd_ratio_ci(g: pd.DataFrame, lo_ir, hi_ir, n=2000, seed=42, alpha=0.05):
    """95% seed-cluster bootstrap interval of  mean_s SD_hi(s) / mean_s SD_lo(s),
    where SD is pooled over all trajectories (seeds included) of strategy s at a
    given IR. Seeds are resampled with replacement and the pooled SD is rebuilt
    from per-(seed, strategy, ir) sufficient statistics (n, sum, sum of squares)."""
    g = g[g.ir.isin([lo_ir, hi_ir])]
    st = (g.assign(d2=g.delta ** 2)
            .groupby(["seed", "strategy", "ir"])
            .agg(n=("delta", "size"), s=("delta", "sum"), s2=("d2", "sum")))
    seeds = st.index.get_level_values("seed").unique().to_numpy()
    strategies = st.index.get_level_values("strategy").unique()
    # dense arrays [seed, strategy, ir]
    idx = pd.MultiIndex.from_product([seeds, strategies, [lo_ir, hi_ir]], names=["seed", "strategy", "ir"])
    st = st.reindex(idx).fillna(0.0)
    N = st.n.to_numpy().reshape(len(seeds), len(strategies), 2)
    S = st.s.to_numpy().reshape(len(seeds), len(strategies), 2)
    S2 = st.s2.to_numpy().reshape(len(seeds), len(strategies), 2)
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        pick = rng.randint(0, len(seeds), len(seeds))
        n_ = N[pick].sum(0); s_ = S[pick].sum(0); s2_ = S2[pick].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (s2_ - s_ ** 2 / n_) / (n_ - 1)           # [strategy, ir]
        sd = np.sqrt(np.clip(var, 0, None))
        ok = np.isfinite(sd).all(1) & (n_ > 1).all(1)
        if ok.sum() == 0 or sd[ok, 0].mean() <= 0:
            continue
        out.append(sd[ok, 1].mean() / sd[ok, 0].mean())
    if not out:
        return float("nan"), float("nan")
    return (float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))


def c1_decomposition(p: pd.DataFrame, reference="margin"):
    """Dual-scale Spearman trends, SD growth IR_max/IR_min with seed bootstrap,
    reference collapse factor, and mixed-effects slope of log SD on log IR (v13)."""
    import statsmodels.formula.api as smf
    o = _eligible(p, reference)
    rho_rows = []
    for (lr, reg, s), g in o.groupby(["learner", "regime", "strategy"]):
        by = g.groupby("ir").agg(sd_abs=("delta", "std"), sd_rel=("rel_delta", "std")).reset_index()
        rho_rows.append({"learner": lr, "geometry": reg, "strategy": s,
                         "rho_abs": spearmanr(by.ir, by.sd_abs).statistic,
                         "rho_rel": spearmanr(by.ir, by.sd_rel).statistic})
    rho = pd.DataFrame(rho_rows).groupby(["learner", "geometry"])[["rho_abs", "rho_rel"]].median().reset_index()
    rows = []
    for (lr, reg), g in o.groupby(["learner", "regime"]):
        by = g.groupby(["strategy", "ir"]).delta.std().reset_index(name="sd")
        lo_ir, hi_ir = by.ir.min(), by.ir.max()
        a = by[by.ir == lo_ir].set_index("strategy").sd
        b = by[by.ir == hi_ir].set_index("strategy").sd
        cm = a.index.intersection(b.index)
        factor = float(b[cm].mean() / a[cm].mean())
        ref = g.groupby("ir").ref.mean()
        # Seed-cluster bootstrap of the SAME pooled statistic (resample seeds with
        # replacement, recompute the pooled SD ratio), so the point estimate always
        # lies inside its interval. Sufficient statistics per (seed, strategy, ir)
        # make the 2000 resamples cheap.
        flo, fhi = _pooled_sd_ratio_ci(g[g.strategy.isin(cm)], lo_ir, hi_ir, n=2000, seed=42)
        z = by[by.sd > 0].copy()
        z["log_sd"], z["log_ir"] = np.log(z.sd), np.log(z.ir)
        slope, se = float("nan"), float("nan")
        if z.ir.nunique() >= 2 and z.strategy.nunique() >= 2 and len(z) >= 4:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = smf.mixedlm("log_sd ~ log_ir", z, groups=z["strategy"]).fit(
                        reml=False, method="lbfgs", disp=False)
                slope, se = float(fit.params["log_ir"]), float(fit.bse["log_ir"])
            except Exception:   # degenerate grid (e.g. a smoke run): leave NaN
                pass
        rows.append({"learner": lr, "geometry": reg, "sd_growth": factor, "sd_growth_ci_low": flo,
                     "sd_growth_ci_high": fhi, "margin_collapse": float(ref.loc[lo_ir] / ref.loc[hi_ir]),
                     "mixed_slope": slope, "slope_ci_low": slope - 1.96 * se, "slope_ci_high": slope + 1.96 * se})
    dec = pd.DataFrame(rows).merge(rho, on=["learner", "geometry"])
    dec["scale_degenerate"] = (dec.sd_growth_ci_high < 1.0) | ((dec.sd_growth_ci_low <= 1.0) & (dec.margin_collapse > 10))
    return dec


def learner_transfer(p: pd.DataFrame, a="logreg", b="rf", reference="margin", cell=("regime", "strategy")):
    means = _eligible(p, reference).groupby(list(cell) + ["learner"]).delta.mean().unstack("learner")
    means = means[[a, b]].dropna()
    cell_rho = spearmanr(means[a], means[b])
    glob = means.groupby("strategy").mean()
    g_rho = spearmanr(glob[a], glob[b])
    flip = np.sign(means[a]) * np.sign(means[b]) < 0
    return {"n_cells": int(len(means)), "cell_rho": float(cell_rho.statistic), "cell_p": float(cell_rho.pvalue),
            "global_rank_rho": float(g_rho.statistic), "global_rank_p": float(g_rho.pvalue),
            "raw_sign_flips": int(flip.sum()),
            "material_flips_005": int((flip & (means[a].abs() > .005) & (means[b].abs() > .005)).sum()),
            "material_flips_010": int((flip & (means[a].abs() > .01) & (means[b].abs() > .01)).sum())}, means


def budget_contrast(p_all: pd.DataFrame, low="low", high="main", reference="margin",
                    strategies=None, match=("regime", "ir", "d", "sep", "seed")):
    """Matched comparison of two budget regimes on their common state subset (v13 low-budget)."""
    a, b = p_all[p_all.budget == high], p_all[p_all.budget == low]
    for c in match:
        common = set(a[c]) & set(b[c])
        a, b = a[a[c].isin(common)], b[b[c].isin(common)]
    rows = []
    strategies = strategies or sorted(set(b.strategy) - DEGENERATE - {reference})
    for lr in sorted(b.learner.unique()):
        for s in strategies:
            gl, gh = b[(b.learner == lr) & (b.strategy == s)], a[(a.learner == lr) & (a.strategy == s)]
            ml, ll, hl = boot_mean(gl.delta, seed=11)
            mh, lh, hh = boot_mean(gh.delta, seed=12)
            rows.append({"learner": lr, "strategy": s, "low": ml, "low_ci_low": ll, "low_ci_high": hl,
                         "high": mh, "high_ci_low": lh, "high_ci_high": hh, "low_minus_high": ml - mh,
                         "n_low": len(gl), "n_high": len(gh)})
    return pd.DataFrame(rows)


def real_dataset_unit(p: pd.DataFrame, reference="margin", seed=31, rope=0.01, tau=0.10):
    """Dataset-unit summary: mean [CI over datasets], win rate, Wilcoxon-Holm, Bayesian
    signed-rank, and the share of datasets whose seed-level loss probability exceeds 10%."""
    rows = []
    elig = _eligible(p, reference)
    for (lr, bud, st), g in elig.groupby(["learner", "budget", "start"]):
        per_ds = g.groupby(["dataset", "strategy"]).agg(
            delta=("delta", "mean"), ptail=("delta", lambda x: float((x < -tau).mean()))).reset_index()
        wide = per_ds.pivot(index="dataset", columns="strategy", values="delta")
        wide[reference] = 0.0
        tests = wilcoxon_vs_reference(wide, reference).set_index("strategy")
        for s in wide.columns.drop(reference):
            x = wide[s].dropna()
            m, lo, hi = boot_mean(x, seed=seed)
            bs = bayesian_signed_rank(x.to_numpy(), rope=rope)
            pt = per_ds[per_ds.strategy == s].ptail
            rows.append({"learner": lr, "budget": bud, "start": st, "strategy": s, "n_datasets": len(x),
                         "mean": m, "ci_low": lo, "ci_high": hi, "win_rate": float((x > 0).mean()),
                         "p_holm": tests.loc[s, "p_holm"], **bs,
                         "share_ds_tail_gt10": float((pt > 0.10).mean())})
    return pd.DataFrame(rows)


def domain_risk(p: pd.DataFrame, reference="margin", tau=0.10, by=("domain", "learner", "strategy")):
    """Risk map input: mean delta and material-loss probability by domain."""
    rows = []
    for key, g in _eligible(p, reference).groupby(list(by)):
        ds = g.groupby("dataset").delta.mean()
        rows.append({**dict(zip(by, key)), "n_datasets": g.dataset.nunique(), "n": len(g),
                     "mean_ds": float(ds.mean()), "p_lt_tau": float((g.delta < -tau).mean()),
                     "q05": float(g.delta.quantile(.05))})
    return pd.DataFrame(rows)


def operational_summary(p: pd.DataFrame, reference="margin", seed=41):
    """Paired operational endpoints, dataset as unit (Table `tab_operational`)."""
    cols = [c for c in p.columns if c.startswith("delta_") and c != "delta_auprc"]
    rows = []
    for (lr, s), g in _eligible(p, reference).groupby(["learner", "strategy"]):
        row = {"learner": lr, "strategy": s}
        per = g.groupby("dataset")[cols].mean()
        for c in cols:
            m, lo, hi = boot_mean(per[c], seed=seed)
            row.update({c: m, f"{c}_ci_low": lo, f"{c}_ci_high": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def rare_label_mechanism(p: pd.DataFrame, reference="margin", bins=(-1, 2, 5, 10, 20, 10**6),
                         labels=("0-2", "3-5", "6-10", "11-20", ">20")):
    """Link between rare labels acquired and paired loss (RF tail diagnostic)."""
    o = _eligible(p, reference).copy()
    o["rare_bin"] = pd.cut(o.n_rare_labeled, list(bins), labels=list(labels))
    corr = o.groupby("learner").apply(lambda g: g.delta.corr(g.n_rare_labeled - g.ref_n_rare_labeled)
                                      if "ref_n_rare_labeled" in g else np.nan, include_groups=False)
    tab = o.groupby(["learner", "rare_bin"], observed=True).agg(
        mean_delta=("delta", "mean"), p_lt_010=("delta", lambda x: (x < -.1).mean()), n=("delta", "size")).reset_index()
    return tab, corr.rename("corr_delta_raregap").reset_index()


def rf_resolution_check(p: pd.DataFrame, reference="margin", tau=0.10):
    """Probability-resolution check (Section 4.4): do the RF tails survive learner variants
    and random tie-breaking? One row per (learner variant, strategy)."""
    rows = []
    for (lr, s), g in p[p.strategy != reference].groupby(["learner", "strategy"]):
        m, lo, hi = boot_mean(g.delta, seed=71)
        rows.append({"learner": lr, "strategy": s, "mean": m, "ci_low": lo, "ci_high": hi,
                     "p_lt_tau": float((g.delta < -tau).mean()), "worst": float(g.delta.min()),
                     "ref_prior_or_constant": float((g.get("ref_p_unique", pd.Series(2, index=g.index)) <= 1).mean())})
    return pd.DataFrame(rows)
