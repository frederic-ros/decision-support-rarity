"""Revision analyses for ESWA-D-26-42849 (reviewer requests R1-R3).

Every function takes the canonical paired frames of :mod:`rare_al.analysis.paired`
(one row per trajectory, ``delta`` = AUPRC difference to Margin in the same state) or
the oracle outputs, and returns tidy ``pandas`` frames. Nothing here re-runs an
experiment; the new runs are produced by the revision configs (``configs/eswa/rev_*``).

Map to the reviews
------------------
* :func:`noise_floor`, :func:`excess_tail`            -- R3.1 run-to-run noise baseline
* :func:`worst_case_anatomy`, :func:`lower_quantiles` -- R3.2 catastrophic worst cases
* :func:`test_composition`, :func:`fixed_test_paired` -- R3.3 rarity vs measurement noise
* :func:`rq3_inference`, :func:`query_vs_campaign`    -- R3.4, R2.6, R2.7 exploitability
* :func:`validation_alpha_sweep`                      -- R3.5, R2.2 does the tail rule bind?
* :func:`pilot_sample_size`, :func:`pilot_simulation` -- R3.6 test of the pilot step
* :func:`source_group`, :func:`strata_counts`         -- R3.7 datasets, sources, strata
* :func:`budget_at_rounds`                            -- R3 budget claim (matched settings)
* :func:`sensitivity_summary`                         -- R2.4, R1.3 learner settings
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as st

from rare_al.stats.core import holm_adjust

TAU = 0.10
REP_PREFIX = "margin_rep"


# ---------------------------------------------------------------------------- helpers
def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]
        prev = min(prev, p[idx] * n / (i + 1))
        adj[idx] = prev
    return np.minimum(adj, 1.0)


def cluster_boot(values, clusters, fn=np.mean, n=2000, seed=0, level=0.95):
    """Percentile cluster bootstrap of ``fn`` (applied to the concatenated values of the
    resampled clusters). Returns (estimate, lo, hi)."""
    v = np.asarray(values, float)
    c = np.asarray(clusters)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    uniq, inv = np.unique(c, return_inverse=True)
    groups = [v[inv == i] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    bs = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        bs[b] = fn(np.concatenate([groups[i] for i in pick]))
    a = (1 - level) / 2
    return float(fn(v)), float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


def cp_upper(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper bound for k events in n trials."""
    if n <= 0:
        return np.nan
    if k >= n:
        return 1.0
    return float(st.beta.ppf(conf, k + 1, n - k))


def _base_state(state_id: pd.Series) -> pd.Series:
    """Oracle state id without its trailing round index (independent unit)."""
    return state_id.astype(str).str.rsplit("|", n=1).str[0]


def _cell_id(df: pd.DataFrame) -> pd.Series:
    """Synthetic design cell (geometry, IR, d, separation) -- seeds are replicates."""
    return (df.regime.astype(str) + "|" + df.ir.astype(str) + "|" + df.d.astype(str) + "|"
            + df.sep.astype(str))


# ------------------------------------------------------------------ R3.1 noise floor
def noise_floor(p: pd.DataFrame, tau=TAU, by=("learner",), n_boot=2000):
    """Distribution of Margin-vs-Margin paired differences.

    Two noise components are reported separately:
    * ``tie_only``  -- ``margin_rtb``: random tie-breaking, identical learner seeds;
    * ``replicate`` -- ``margin_rep*``: random tie-breaking AND a shifted learner seed,
                       i.e. what two runs of the *same* policy in the *same* state differ by.
    Intervals are bootstraps over states (replicates of one state are one cluster)."""
    rows = []
    kinds = {"tie_only": p.strategy == "margin_rtb",
             "replicate": p.strategy.astype(str).str.startswith(REP_PREFIX)}
    for kind, mask in kinds.items():
        sub = p[mask]
        if sub.empty:
            continue
        for key, g in sub.groupby(list(by)):
            x = g.delta.to_numpy(float)
            cl = g.state_id.to_numpy()
            c, clo, chi = cluster_boot((x < -tau).astype(float), cl, n=n_boot, seed=101)
            q05, q05lo, q05hi = cluster_boot(x, cl, fn=lambda v: np.quantile(v, .05), n=n_boot, seed=102)
            key = key if isinstance(key, tuple) else (key,)
            rows.append({**dict(zip(by, key)), "noise": kind, "n_traj": len(x),
                         "n_states": g.state_id.nunique(), "n_replicates": g.strategy.nunique(),
                         "sd": float(x.std()), "q05": q05, "q05_ci_low": q05lo, "q05_ci_high": q05hi,
                         "q01": float(np.quantile(x, .01)), "worst": float(x.min()),
                         "c_tau": c, "c_tau_ci_low": clo, "c_tau_ci_high": chi,
                         "share_identical": float((np.abs(x) < 1e-12).mean())})
    return pd.DataFrame(rows)


def replicate_reference(p: pd.DataFrame) -> pd.DataFrame:
    """Per (state, learner): mean AUPRC of Margin over the original run and its replicates
    (the reference distribution's centre) and the number of runs averaged."""
    ref_runs = p[(p.strategy == "margin") | p.strategy.astype(str).str.startswith(REP_PREFIX)]
    keys = ["state_id", "learner", "budget", "start"]
    g = ref_runs.groupby(keys)
    return pd.DataFrame({"ref_mean": g.auprc.mean(), "ref_sd": g.auprc.std(),
                         "ref_min": g.auprc.min(), "n_ref_runs": g.size()}).reset_index()


def excess_tail(p: pd.DataFrame, tau=TAU, n_boot=2000, strategies=None):
    """Strategy tail net of run-to-run noise, per learner and strategy.

    * ``c_tau``          -- published reading: P(delta < -tau) against the single Margin run;
    * ``c_tau_noise``    -- the same quantity for the Margin replicates, on the same states;
    * ``excess``         -- c_tau - c_tau_noise, with a state-bootstrap interval of the
                            per-state paired difference (state = cluster);
    * ``c_tau_vs_mean``  -- P(P(a) - mean_r P(Margin_r) < -tau): regret measured against the
                            centre of the Margin replicate distribution instead of one run;
    * ``q05_vs_mean``    -- 5% quantile of that regret;
    * ``share_tail_in_unstable`` -- share of the strategy's material losses that occur in
                            states where at least one Margin replicate also loses > tau
                            against the original Margin run (chaotic states)."""
    reps = p[p.strategy.astype(str).str.startswith(REP_PREFIX)]
    if reps.empty:
        return pd.DataFrame()
    keys = ["state_id", "learner", "budget", "start"]
    noise_state = reps.assign(loss=(reps.delta < -tau).astype(float)).groupby(keys).loss.agg(
        noise_rate="mean", noise_any="max").reset_index()
    rr = replicate_reference(p)
    cand = p[~p.strategy.isin(["margin", "margin_rtb", "entropy", "least_confident"])
             & ~p.strategy.astype(str).str.startswith(REP_PREFIX)]
    if strategies is not None:
        cand = cand[cand.strategy.isin(strategies)]
    cand = cand.merge(noise_state, on=keys, how="inner").merge(rr, on=keys, how="inner")
    cand["loss"] = (cand.delta < -tau).astype(float)
    cand["delta_vs_mean"] = cand.auprc - cand.ref_mean
    rows = []
    for (lr, s), g in cand.groupby(["learner", "strategy"]):
        diff = g.loss - g.noise_rate
        e, elo, ehi = cluster_boot(diff.to_numpy(), g.state_id.to_numpy(), n=n_boot, seed=103)
        cm, cmlo, cmhi = cluster_boot((g.delta_vs_mean < -tau).astype(float).to_numpy(),
                                      g.state_id.to_numpy(), n=n_boot, seed=104)
        n_loss = g.loss.sum()
        rows.append({"learner": lr, "strategy": s, "n_states": len(g),
                     "mean": float(g.delta.mean()), "mean_vs_mean": float(g.delta_vs_mean.mean()),
                     "c_tau": float(g.loss.mean()), "c_tau_noise": float(g.noise_rate.mean()),
                     "excess": e, "excess_ci_low": elo, "excess_ci_high": ehi,
                     "excess_resolved": bool(elo > 0),
                     "c_tau_vs_mean": cm, "c_tau_vs_mean_ci_low": cmlo, "c_tau_vs_mean_ci_high": cmhi,
                     "q05": float(g.delta.quantile(.05)), "q05_vs_mean": float(g.delta_vs_mean.quantile(.05)),
                     "worst": float(g.delta.min()), "worst_vs_mean": float(g.delta_vs_mean.min()),
                     "share_tail_in_unstable": float((g.loss * g.noise_any).sum() / n_loss) if n_loss else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- R3.2 worst-case anatomy
def lower_quantiles(p: pd.DataFrame, qs=(0.01, 0.05), n_boot=2000, cluster="state_id"):
    """Lower quantiles of paired regret with bootstrap intervals, and the worst value with
    the number of distinct states among the k worst trajectories. Replaces the single
    'worst observed trajectory' by quantities whose uncertainty can be stated."""
    rows = []
    elig = p[~p.strategy.isin(["margin", "entropy", "least_confident"])
             & ~p.strategy.astype(str).str.startswith(REP_PREFIX)]
    for (lr, s), g in elig.groupby(["learner", "strategy"]):
        row = {"learner": lr, "strategy": s, "n": len(g), "worst": float(g.delta.min())}
        for q in qs:
            v, lo, hi = cluster_boot(g.delta.to_numpy(), g[cluster].to_numpy(),
                                     fn=lambda x, q=q: np.quantile(x, q), n=n_boot, seed=105)
            row.update({f"q{int(q*100):02d}": v, f"q{int(q*100):02d}_ci_low": lo,
                        f"q{int(q*100):02d}_ci_high": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def worst_case_anatomy(p: pd.DataFrame, threshold=-0.5, top=30):
    """Who produces the catastrophic losses, and why.

    Returns (pairs, by_state, summary):
    * ``pairs``    -- every (strategy, state, learner) with delta < threshold, with the
                      reference AUPRC, both final rare-label counts, the number of rare
                      test cases, test prevalence and two failure flags:
                      ``collapsed``   the strategy's final model ranks no better than ~2x
                                      prevalence (AUPRC <= 2 * test prevalence);
                      ``constant``    the strategy's final scores are (near) constant
                                      (``p_unique`` <= 2) -- the learner did not learn;
    * ``by_state`` -- per state: how many strategies lose > |threshold| there;
    * ``summary``  -- per learner: number of such trajectories, distinct states, share
                      explained by the top-5 states, share collapsed / constant, median
                      number of rare test cases in those states vs all states."""
    q = p[(p.delta < threshold) & ~p.strategy.astype(str).str.startswith(REP_PREFIX)].copy()
    if "n_test_pos" in q:
        q["test_prev"] = q.n_test_pos / q.n_test.where(q.n_test > 0)
    else:
        q["test_prev"] = np.nan
    q["collapsed"] = (q.auprc <= 2 * q.test_prev).astype(float).where(q.test_prev.notna())
    q["constant"] = (q.p_unique <= 2).astype(float) if "p_unique" in q else np.nan
    keep = [c for c in ["learner", "strategy", "state_id", "regime", "dataset", "ir", "d", "sep", "seed",
                        "ref_auprc", "auprc", "delta", "ref_n_rare_labeled", "n_rare_labeled",
                        "n_test_pos", "test_prev", "collapsed", "constant", "p_unique"] if c in q]
    pairs = q[keep].sort_values("delta")
    agg = {"n_strategies": ("strategy", "nunique"), "worst": ("delta", "min"),
           "ref_auprc": ("ref_auprc", "first")}
    if "n_test_pos" in q:
        agg["n_test_pos"] = ("n_test_pos", "first")
    by_state = (q.groupby(["learner", "state_id"]).agg(**agg).reset_index()
                .sort_values(["learner", "n_strategies"], ascending=[True, False]))
    summ = []
    for lr, g in q.groupby("learner"):
        bs = by_state[by_state.learner == lr]
        top5 = bs.nlargest(5, "n_strategies").n_strategies.sum()
        allp = (p[p.learner == lr].groupby("state_id").n_test_pos.first()
                if "n_test_pos" in p else None)
        summ.append({"learner": lr, "n_trajectories": len(g), "n_states": g.state_id.nunique(),
                     "n_strategies": g.strategy.nunique(), "share_top5_states": top5 / len(g),
                     "share_collapsed": float(g.collapsed.mean()) if g.collapsed.notna().any() else np.nan,
                     "share_constant": float(g.constant.mean()) if "p_unique" in g else np.nan,
                     "median_ref_auprc": float(g.ref_auprc.median()),
                     "median_test_pos_in_worst": float(g.n_test_pos.median()) if "n_test_pos" in g else np.nan,
                     "median_test_pos_all": float(allp.median()) if allp is not None else np.nan,
                     "share_ir_ge50": float((g.ir.astype(float) >= 50).mean()) if "ir" in g else np.nan})
    return pairs.head(top) if top else pairs, by_state, pd.DataFrame(summ)


# ---------------------------------------------- R3.3 test composition / fixed test set
def test_composition(states, cfg, max_states=None):
    """Number of rare test cases per state, recomputed from the state definitions
    (cheap for synthetic states; real/image states need the prepared caches)."""
    from rare_al.data.splits import make_split
    from rare_al.engine.states import materialize
    rows = []
    for i, stt in enumerate(states):
        if max_states and i >= max_states:
            break
        try:
            X, y, t, is_test = materialize(stt)
            _, _, ytr, yte = make_split(X, y, stt["state_seed"], cfg["data"]["test_fraction"],
                                        stt["split"], t, is_test)
        except Exception as exc:  # unavailable cache, degenerate split
            rows.append({"state_id": stt["state_id"], "error": repr(exc)})
            continue
        rows.append({"state_id": stt["state_id"], "kind": stt.get("kind"),
                     "dataset": stt.get("dataset"), "regime": stt.get("regime"), "ir": stt.get("ir"),
                     "d": stt.get("d"), "sep": stt.get("sep"), "seed": stt.get("seed"),
                     "backbone": stt.get("backbone"), "n_pool": len(ytr), "n_pool_pos": int(ytr.sum()),
                     "n_test": len(yte), "n_test_pos": int(yte.sum())})
    return pd.DataFrame(rows)


def test_composition_summary(tc: pd.DataFrame, by=("ir",)):
    g = tc.dropna(subset=["n_test_pos"]).groupby(list(by))
    return g.agg(n_states=("state_id", "size"), test_pos_min=("n_test_pos", "min"),
                 test_pos_median=("n_test_pos", "median"), test_pos_max=("n_test_pos", "max"),
                 pool_pos_median=("n_pool_pos", "median")).reset_index()


def fixed_test_paired(final: pd.DataFrame, reference="margin", prefix="fx"):
    """Paired frame in which AUPRC is measured on the fixed-positive evaluation set.

    The acquisition is identical to the published run; only the measurement changes.
    Columns ``delta``, ``rel_delta`` and ``ref`` are rebuilt from ``<prefix>_auprc`` so
    that :func:`rare_al.analysis.risk.c1_decomposition` can be applied unchanged."""
    col = f"{prefix}_auprc"
    if col not in final:
        raise KeyError(f"{col} not found: run configs/eswa/rev_synthetic.json first")
    keys = ["state_id", "learner", "budget", "start"]
    ref = final[final.strategy == reference][keys + [col]].rename(columns={col: "ref"})
    out = final.merge(ref, on=keys, how="inner")
    out["delta"] = out[col] - out["ref"]
    out["rel_delta"] = out["delta"] / out["ref"].clip(lower=1e-12)
    return out


def amplification_count(dec: pd.DataFrame) -> dict:
    """Number of learner x geometry cells with risk amplification (interval > 1, not
    scale-degenerate), as in the published '14 of 18'."""
    amp = (dec.sd_growth_ci_low > 1.0) & ~dec.scale_degenerate.astype(bool)
    return {"n_cells": int(len(dec)), "n_amplified": int(amp.sum()),
            "n_scale_degenerate": int(dec.scale_degenerate.astype(bool).sum())}


# ------------------------------------------------------------------ R3.4 / R2.6 RQ3
def _tost_p(x, delta):
    """Two one-sided t-tests of |mean(x)| < delta. Returns max of the two p-values."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3 or not np.isfinite(delta) or delta <= 0:
        return np.nan
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return 0.0 if abs(m) < delta else 1.0
    p_lo = 1 - st.t.cdf((m + delta) / se, n - 1)   # H0: mean <= -delta
    p_hi = st.t.cdf((m - delta) / se, n - 1)       # H0: mean >= +delta
    return float(max(p_lo, p_hi))


def rq3_inference(policies: pd.DataFrame, points: pd.DataFrame, k=5, tau=TAU, alpha=0.10,
                  eq_rel=0.5, eq_abs=None, n_boot=4000, exclude=("margin",)):
    """Exploitability with the inference the reviewers asked for (R3.4, R2.7).

    All tests use the *base state* (oracle state without its round index) as the
    independent unit: per-policy gains and opportunity levels are averaged within base
    state first, so the four rounds of one trajectory are never treated as independent.

    Per learner and policy:
    * gain test: two-sided one-sample t-test of the base-state mean gain, with Holm and
      Benjamini-Hochberg adjustment across the policies of that learner;
    * level difference G - O: t-interval, and a TOST equivalence test with margin
      ``delta_eq = eq_rel * O`` (pre-specified: half the opportunity level) or ``eq_abs``;
      Holm-adjusted across policies;
    * conditional level: O and G restricted to states whose candidate set contains at
      least one rare case (the zero-coded states are removed from both), with the ratio
      kappa_cond and a cluster-bootstrap interval;
    * C_tau with a cluster-bootstrap interval and the zero-event Clopper-Pearson bound."""
    from rare_al.analysis.oracle import label_only_states
    col = f"gain_vs_margin_k{k}"
    lo = label_only_states(points, k)
    lo = lo.set_index(["learner", "state_id"]) if "learner" in lo else lo.set_index("state_id")
    rows = []
    for lr, pl in policies.groupby("learner"):
        pl = pl[~pl.policy.isin(exclude)].copy()
        O = lo.loc[lr] if "learner" in lo.index.names else lo
        pl = pl.join(O[["delta", "nrare"]].rename(columns={"delta": "O"}), on="state_id")
        pl["G"] = pl[col]
        pl["base"] = _base_state(pl.state_id)
        for pol, g in pl.groupby("policy"):
            if (g.G.abs() < 1e-15).all():
                rows.append({"learner": lr, "policy": pol, "identical_to_margin": True})
                continue
            b = g.groupby("base").agg(G=("G", "mean"), O=("O", "mean"))
            b["D"] = b.G - b.O
            n = len(b)
            O_mean = float(b.O.mean())
            tg = st.ttest_1samp(b.G, 0.0)
            td = st.ttest_1samp(b.D, 0.0)
            se_g, se_d = b.G.std(ddof=1) / math.sqrt(n), b.D.std(ddof=1) / math.sqrt(n)
            tq = st.t.ppf(0.975, n - 1)
            tq90 = st.t.ppf(0.95, n - 1)
            delta_eq = eq_abs if eq_abs is not None else eq_rel * O_mean
            cond = g[g.nrare > 0]
            if len(cond):
                kc, kclo, kchi = _ratio_cluster(cond.G.to_numpy(), cond.O.to_numpy(),
                                                cond.base.to_numpy(), n_boot)
            else:
                kc = kclo = kchi = np.nan
            loss = (g.G < -tau).astype(float)
            c, clo, chi = cluster_boot(loss.to_numpy(), g.base.to_numpy(), n=n_boot, seed=106)
            rows.append({
                "learner": lr, "policy": pol, "identical_to_margin": False, "n_base": n,
                "n_states": len(g), "gain": float(b.G.mean()),
                "gain_ci_low": float(b.G.mean() - tq * se_g), "gain_ci_high": float(b.G.mean() + tq * se_g),
                "p_gain": float(tg.pvalue), "O": O_mean,
                "diff": float(b.D.mean()), "diff_ci_low": float(b.D.mean() - tq * se_d),
                "diff_ci_high": float(b.D.mean() + tq * se_d),
                "diff_ci90_low": float(b.D.mean() - tq90 * se_d), "diff_ci90_high": float(b.D.mean() + tq90 * se_d),
                "p_diff": float(td.pvalue), "delta_eq": float(delta_eq), "p_tost": _tost_p(b.D, delta_eq),
                "min_equivalence_margin": float(max(abs(b.D.mean() - tq90 * se_d), abs(b.D.mean() + tq90 * se_d))),
                "share_states_with_rare_candidate": float((g.nrare > 0).mean()),
                "O_cond": float(cond.O.mean()) if len(cond) else np.nan,
                "G_cond": float(cond.G.mean()) if len(cond) else np.nan,
                "kappa": float(b.G.mean() / O_mean) if O_mean else np.nan,
                "kappa_cond": kc, "kappa_cond_ci_low": kclo, "kappa_cond_ci_high": kchi,
                "c_tau": c, "c_tau_ci_low": clo, "c_tau_ci_high": chi,
                "c_tau_cp_upper": cp_upper(int(g.groupby("base").apply(
                    lambda h: (h.G < -tau).any(), include_groups=False).sum()), n),
                "rare_pick_rate": float(g.true_label.mean()) if "true_label" in g else np.nan})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    real = out[~out.identical_to_margin.astype(bool)].copy()
    for lr, idx in real.groupby("learner").groups.items():
        sub = real.loc[idx]
        out.loc[idx, "p_gain_holm"] = holm_adjust(sub.p_gain.fillna(1).to_numpy())
        out.loc[idx, "p_gain_bh"] = bh_adjust(sub.p_gain.fillna(1).to_numpy())
        out.loc[idx, "p_tost_holm"] = holm_adjust(sub.p_tost.fillna(1).to_numpy())
    out["gain_positive_holm"] = out.get("p_gain_holm", 1) < 0.05
    out["gain_positive_bh"] = out.get("p_gain_bh", 1) < 0.05
    out["equivalent_to_level"] = out.get("p_tost_holm", 1) < 0.05
    out["gain_positive_unadjusted"] = out.get("gain_ci_low", np.nan) > 0
    return out


def _ratio_cluster(num, den, clusters, n_boot=4000, seed=107):
    uniq, inv = np.unique(clusters, return_inverse=True)
    nb = len(uniq)
    sn = np.bincount(inv, weights=num, minlength=nb)
    sd = np.bincount(inv, weights=den, minlength=nb)
    rng = np.random.default_rng(seed)
    r = np.empty(n_boot)
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, nb, nb), minlength=nb)
        r[b] = (cnt @ sn) / (cnt @ sd) if (cnt @ sd) != 0 else np.nan
    r = r[np.isfinite(r)]
    est = num.sum() / den.sum() if den.sum() else np.nan
    return float(est), float(np.quantile(r, .025)), float(np.quantile(r, .975))


def query_vs_campaign(rq3: pd.DataFrame, campaign_p: pd.DataFrame, tau=TAU):
    """Query-level capture (RQ3) against full-campaign value *on the same states and under
    the same learner* (oracle-campaign run). Returns the merged per-policy table and, per
    learner, Spearman correlations across policies."""
    camp = (campaign_p.groupby(["learner", "strategy"])
            .agg(camp_mean=("delta", "mean"), camp_c_tau=("delta", lambda x: float((x < -tau).mean())),
                 camp_q05=("delta", lambda x: float(x.quantile(.05))))
            .reset_index().rename(columns={"strategy": "policy"}))
    m = rq3[~rq3.identical_to_margin.astype(bool)].merge(camp, on=["learner", "policy"], how="inner")
    corr = []
    for lr, g in m.groupby("learner"):
        if len(g) >= 4:
            r1 = st.spearmanr(g.gain, g.camp_mean)
            r2 = st.spearmanr(g.gain, g.camp_c_tau)
            corr.append({"learner": lr, "n_policies": len(g),
                         "rho_gain_vs_campaign_mean": float(r1.statistic), "p1": float(r1.pvalue),
                         "rho_gain_vs_campaign_tail": float(r2.statistic), "p2": float(r2.pvalue)})
    return m, pd.DataFrame(corr)


# ------------------------------------------------------- R3.7 datasets, sources, strata
_SOURCE_GROUPS = {
    "car_eval_34": "car_evaluation", "car_eval_4": "car_evaluation",
    "yeast_me2": "yeast", "yeast_ml8": "yeast",
    "abalone": "abalone", "abalone_19": "abalone",
    "thyroid_sick": "thyroid_garvan", "sick_euthyroid": "thyroid_garvan",
}


def source_group(dataset_id: str) -> str:
    """Independent source of a dataset. Extends the published grouping (Polish
    bankruptcy years, UNSW-NB15 attack categories) to the imbalanced-learn sets derived
    from one original collection (R3.7)."""
    d = str(dataset_id)
    if d.startswith("polish_bankruptcy"):
        return "polish_bankruptcy"
    if d.startswith("unsw_nb15"):
        return "unsw_nb15"
    name = d[len("imblearn_"):] if d.startswith("imblearn_") else d
    return _SOURCE_GROUPS.get(name, d)


IR_STRATA = [(0, 10, "IR<10"), (10, 50, "10<=IR<50"), (50, 200, "50<=IR<200"), (200, np.inf, "IR>=200")]


def ir_stratum(ir: float) -> str:
    for lo, hi, lab in IR_STRATA:
        if lo <= ir < hi:
            return lab
    return "nan"


def strata_counts(datasets: pd.DataFrame, ir_col="ir_eval"):
    """Datasets and independent sources per imbalance stratum and per domain.
    ``datasets`` needs columns ``dataset``, ``domain`` and ``ir_col`` (e.g. the manifest)."""
    d = datasets.copy()
    d["source"] = d.dataset.map(source_group)
    d["stratum"] = d[ir_col].astype(float).map(ir_stratum)
    by_stratum = d.groupby("stratum").agg(n_datasets=("dataset", "nunique"), n_sources=("source", "nunique"),
                                          datasets=("dataset", lambda x: ", ".join(sorted(x)))).reset_index()
    by_domain = d.groupby("domain").agg(n_datasets=("dataset", "nunique"),
                                        n_sources=("source", "nunique")).reset_index()
    return by_stratum, by_domain, d


# ------------------------------------------------- R3.5 / R2.2 does the tail rule bind?
def validation_alpha_sweep(p: pd.DataFrame, domain_col="domain", unit_col="dataset",
                           cluster_fn=source_group, alphas=(0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 1.0),
                           tau=TAU, reference="margin", candidates=None, n_boot=2000,
                           reference_proxy="margin_rtb"):
    """Leave-one-domain-out test of the tail constraint over a range of tolerances.

    For each held-out domain and learner, the training domains give per strategy a
    dataset-unit mean gain and a *trajectory-level* material-loss rate C_tau. Rules:
    ``mean_rank`` (best training mean), ``risk(alpha)`` (best training mean among
    strategies with C_tau <= alpha; Margin if none), ``learner_blind`` (best pooled mean
    over all learners, the same pick for every learner). Each pick is scored on the
    held-out units by its realized mean gain AND its realized trajectory-level loss rate.

    Because the reference has zero paired regret to itself by construction, a rule that
    falls back to Margin would otherwise be credited with a loss rate of exactly zero.
    When ``reference_proxy`` is available (Margin with random tie-breaking), a Margin
    pick is instead charged the proxy's held-out gain and loss rate, i.e. the run-to-run
    noise of deploying Margin itself.

    Returns (cells, summary): ``summary`` reports, per alpha, the share of cells where the
    risk rule picks differently from the mean-rank rule, and the paired differences
    risk - mean_rank in held-out gain and held-out loss rate, with cluster-bootstrap
    intervals over independent sources. If no alpha changes a pick, the tail constraint
    is untested by this benchmark, which is then reported as such."""
    elig = p[(p.strategy != reference) & ~p.strategy.isin(["entropy", "least_confident"])
             & ~p.strategy.astype(str).str.startswith(REP_PREFIX)]
    if candidates is not None:
        elig = elig[elig.strategy.isin(candidates)]
    du = elig.groupby([domain_col, unit_col, "learner", "strategy"]).agg(
        gain=("delta", "mean"), loss=("delta", lambda x: float((x < -tau).mean()))).reset_index()
    proxy = None
    if reference_proxy and (p.strategy == reference_proxy).any():
        proxy = (p[p.strategy == reference_proxy].groupby([unit_col, "learner"])
                 .agg(gain=("delta", "mean"), loss=("delta", lambda x: float((x < -tau).mean()))))
    rows = []
    for dom in sorted(du[domain_col].dropna().unique()):
        tr_all = elig[elig[domain_col] != dom]
        te = du[du[domain_col] == dom]
        pooled = tr_all.groupby("strategy").delta.mean()
        pooled[reference] = 0.0
        blind_pick = pooled.idxmax()
        for lr in sorted(du.learner.unique()):
            tr = tr_all[tr_all.learner == lr]
            if tr.empty:
                continue
            tr_unit = tr.groupby([unit_col, "strategy"]).delta.mean().reset_index()
            mean_tr = tr_unit.groupby("strategy").delta.mean()
            c_tr = tr.groupby("strategy").delta.apply(lambda x: float((x < -tau).mean()))
            # the anchor is always a candidate: gain 0, no material loss by construction
            mean_tr[reference], c_tr[reference] = 0.0, 0.0
            mr_pick = mean_tr.idxmax()
            picks = {"mean_rank": mr_pick, "learner_blind": blind_pick}
            for a in alphas:
                safe = mean_tr[c_tr.reindex(mean_tr.index) <= a]
                picks[f"risk_{a:g}"] = safe.idxmax()
            tel = te[te.learner == lr]
            for unit, g in tel.groupby(unit_col):
                gm = g.set_index("strategy")
                row = {"held_out": dom, "learner": lr, "unit": unit, "cluster": cluster_fn(unit)}
                for rule, s in picks.items():
                    row[f"pick_{rule}"] = s
                    if s == reference:
                        pr = proxy.loc[(unit, lr)] if proxy is not None and (unit, lr) in proxy.index else None
                        row[f"gain_{rule}"] = float(pr.gain) if pr is not None else 0.0
                        row[f"loss_{rule}"] = float(pr.loss) if pr is not None else 0.0
                    else:
                        row[f"gain_{rule}"] = float(gm.gain.get(s, np.nan))
                        row[f"loss_{rule}"] = float(gm.loss.get(s, np.nan))
                rows.append(row)
    cells = pd.DataFrame(rows)
    summ = []
    for rule in ["mean_rank", "learner_blind"] + [f"risk_{a:g}" for a in alphas]:
        g, glo, ghi = cluster_boot(cells[f"gain_{rule}"], cells.cluster, n=n_boot, seed=108)
        l, llo, lhi = cluster_boot(cells[f"loss_{rule}"], cells.cluster, n=n_boot, seed=109)
        row = {"rule": rule, "n_cells": len(cells), "gain": g, "gain_ci_low": glo, "gain_ci_high": ghi,
               "q10_gain": float(cells[f"gain_{rule}"].quantile(.10)),
               "loss_rate": l, "loss_ci_low": llo, "loss_ci_high": lhi}
        if rule.startswith("risk_"):
            changed = cells[f"pick_{rule}"] != cells["pick_mean_rank"]
            dg = cells[f"gain_{rule}"] - cells["gain_mean_rank"]
            dl = cells[f"loss_{rule}"] - cells["loss_mean_rank"]
            row.update({"share_changed": float(changed.mean())})
            for name, v in (("dgain", dg), ("dloss", dl)):
                m, lo_, hi_ = cluster_boot(v, cells.cluster, n=n_boot, seed=110)
                row.update({name: m, f"{name}_ci_low": lo_, f"{name}_ci_high": hi_})
        summ.append(row)
    return cells, pd.DataFrame(summ)


# --------------------------------------------------------------- R3.6 the pilot step
def pilot_sample_size(alphas=(0.05, 0.10, 0.20, 0.30), failures=(0, 1, 2), conf=0.95,
                      true_c=(0.05, 0.10, 0.30), n_list=(5, 10, 20, 30, 60)):
    """(i) Runs needed so that the one-sided Clopper-Pearson upper bound on C_tau is
    below alpha with f observed failures; (ii) probability that a pilot of n runs
    observes at least one material loss when the true loss probability is C."""
    need = []
    for a in alphas:
        for f in failures:
            n = f + 1
            while cp_upper(f, n, conf) > a and n < 10000:
                n += 1
            need.append({"alpha": a, "failures": f, "runs_needed": n})
    power = [{"true_c": c, "n_runs": n, "p_detect": 1 - (1 - c) ** n,
              "cp_upper_if_zero": cp_upper(0, n, conf)} for c in true_c for n in n_list]
    return pd.DataFrame(need), pd.DataFrame(power)


def pilot_simulation(p: pd.DataFrame, n_pilot=(2, 3, 5), tau=TAU, reference="margin",
                     candidates=None, guideline=None, n_rep=50, seed=0, n_boot=2000,
                     reference_proxy="margin_rtb", shortlist_size=3):
    """Empirical test of the paired pilot (Algorithm 2) on the real benchmark.

    For every dataset and learner, ``n_pilot`` of the dataset's seeds play the role of a
    labeled historical sample (the pilot); the remaining seeds are the deployment it must
    predict. Rules: ``pilot_mean`` (best pilot mean, Margin included at 0), ``pilot_screen``
    (same, after removing every strategy with at least one material loss in the pilot),
    ``guideline`` (no pilot: the learner-conditional default, e.g. DIRECT under LR and
    Margin under trees), and ``hindsight`` (best mean on the held-out seeds, upper ref.).
    ``pilot_shortlist`` follows Algorithm 2 more closely: the pilot only arbitrates between
    the anchor and the ``shortlist_size`` best strategies of the *other* datasets for this
    learner (the guideline-admissible candidates), instead of searching all candidates.
    As in :func:`validation_alpha_sweep`, a Margin pick is charged the held-out gain and
    loss of ``reference_proxy`` (Margin with random tie-breaking), i.e. its own noise.
    Repeated ``n_rep`` times over random pilot/held-out partitions."""
    elig = p[~p.strategy.isin(["entropy", "least_confident"])
             & ~p.strategy.astype(str).str.startswith(REP_PREFIX)]
    if candidates is not None:
        elig = elig[elig.strategy.isin(set(candidates) | {reference})]
    guideline = guideline or {"logreg": "direct", "rf": reference, "gbdt": reference}
    rng = np.random.default_rng(seed)
    rows = []
    unit_means = elig.groupby(["learner", "dataset", "strategy"]).delta.mean()
    proxy = (p[p.strategy == reference_proxy].pivot_table(index=["dataset", "learner"], columns="seed",
                                                          values="delta", aggfunc="mean")
             if reference_proxy and (p.strategy == reference_proxy).any() else None)
    for (ds, lr), g in elig.groupby(["dataset", "learner"]):
        w = g.pivot_table(index="seed", columns="strategy", values="delta", aggfunc="mean")
        if reference not in w:
            w[reference] = 0.0
        seeds = w.index.to_numpy()
        ref_h = (proxy.loc[(ds, lr)] if proxy is not None and (ds, lr) in proxy.index
                 else pd.Series(0.0, index=seeds))
        other = unit_means.loc[lr].drop(ds, level="dataset", errors="ignore").groupby("strategy").mean()
        other = other.drop(reference, errors="ignore")
        short = [x for x in other.sort_values(ascending=False).index[:shortlist_size] if x in w] + [reference]
        for npil in n_pilot:
            if npil >= len(seeds):
                continue
            for r in range(n_rep):
                pil = rng.choice(seeds, npil, replace=False)
                ho = np.setdiff1d(seeds, pil)
                wp, wh = w.loc[pil], w.loc[ho]
                pm = wp.mean()
                screen = pm[(wp < -tau).sum() == 0]
                picks = {"pilot_mean": pm.idxmax(), "pilot_screen": screen.idxmax() if len(screen) else reference,
                         "pilot_shortlist": pm[short].idxmax(),
                         "guideline": guideline.get(lr, reference), "hindsight": wh.mean().idxmax()}
                for rule, s in picks.items():
                    s = s if s in wh else reference
                    hv = ref_h.reindex(ho).fillna(0.0) if s == reference else wh[s]
                    rows.append({"dataset": ds, "cluster": source_group(ds), "learner": lr, "n_pilot": npil,
                                 "rep": r, "rule": rule, "pick": s, "gain": float(hv.mean()),
                                 "loss": float((hv < -tau).mean()),
                                 "agrees_hindsight": float(s == picks["hindsight"]),
                                 "picks_reference": float(s == reference)})
    sim = pd.DataFrame(rows)
    summ = []
    for (lr, npil, rule), g in sim.groupby(["learner", "n_pilot", "rule"]):
        per = g.groupby(["dataset", "cluster"]).agg(gain=("gain", "mean"), loss=("loss", "mean")).reset_index()
        m, lo, hi = cluster_boot(per.gain, per.cluster, n=n_boot, seed=111)
        l, llo, lhi = cluster_boot(per.loss, per.cluster, n=n_boot, seed=112)
        summ.append({"learner": lr, "n_pilot": npil, "rule": rule, "gain": m, "gain_ci_low": lo,
                     "gain_ci_high": hi, "loss_rate": l, "loss_ci_low": llo, "loss_ci_high": lhi,
                     "agrees_hindsight": float(g.agrees_hindsight.mean()),
                     "picks_reference": float(g.picks_reference.mean())})
    return sim, pd.DataFrame(summ)


# -------------------------------------------------------- R3 budget claim (matched)
def budget_at_rounds(traj: pd.DataFrame, rounds, reference="margin", strategies=None, n_boot=2000):
    """Paired AUPRC gain at intermediate rounds of the *same* trajectories, so that the
    initial set and the batch size are held fixed and only the number of labels varies."""
    from rare_al.stats.core import boot_mean
    keys = ["state_id", "learner", "budget", "start"]
    rows = []
    for r in rounds:
        d = traj[traj["round"] == r]
        ref = d[d.strategy == reference][keys + ["auprc"]].rename(columns={"auprc": "ref"})
        m = d.merge(ref, on=keys)
        m["delta"] = m.auprc - m.ref
        if strategies is not None:
            m = m[m.strategy.isin(strategies)]
        for (bud, lr, s), g in m[m.strategy != reference].groupby(["budget", "learner", "strategy"]):
            mean, lo, hi = boot_mean(g.delta, n=n_boot, seed=113)
            rows.append({"budget": bud, "learner": lr, "strategy": s, "round": r,
                         "n_labeled": int(g.n_labeled.median()), "mean": mean, "ci_low": lo,
                         "ci_high": hi, "c_tau": float((g.delta < -TAU).mean()), "n": len(g)})
    return pd.DataFrame(rows)


# ------------------------------------------------------ R2.4 / R1.3 learner settings
BASE_OF = {"rf_leaf1_500": "rf", "rf_leaf5": "rf", "rf_trees50": "rf", "rf_unweighted": "rf",
           "rf_nobootstrap": "rf", "gbdt_leaf5": "gbdt", "gbdt_unweighted": "gbdt",
           "logreg_unweighted": "logreg"}


def sensitivity_summary(p_var: pd.DataFrame, p_base: pd.DataFrame, unit="dataset", tau=TAU):
    """Compare each learner variant with its base learner on the common states and
    strategies: per-strategy unit-level mean gains, Spearman concordance of the strategy
    ordering, sign agreement, and the material-loss rate."""
    def unit_means(p):
        return p.groupby(["learner", "strategy", unit]).delta.mean().groupby(["learner", "strategy"]).mean()
    mv, mb = unit_means(p_var), unit_means(p_base)
    rows, conc = [], []
    for var in sorted(p_var.learner.unique()):
        base = BASE_OF.get(var)
        if base is None or base not in mb.index.get_level_values(0):
            continue
        a, b = mv.loc[var], mb.loc[base]
        common = a.index.intersection(b.index).drop("margin", errors="ignore")
        tail_v = p_var[p_var.learner == var].groupby("strategy").delta.apply(lambda x: float((x < -tau).mean()))
        tail_b = p_base[p_base.learner == base].groupby("strategy").delta.apply(lambda x: float((x < -tau).mean()))
        for s in common:
            rows.append({"variant": var, "base": base, "strategy": s, "gain_variant": a[s], "gain_base": b[s],
                         "c_tau_variant": tail_v.get(s, np.nan), "c_tau_base": tail_b.get(s, np.nan)})
        if len(common) >= 4:
            r = st.spearmanr(a[common], b[common])
            conc.append({"variant": var, "base": base, "n_strategies": len(common),
                         "spearman": float(r.statistic), "p": float(r.pvalue),
                         "sign_agreement": float((np.sign(a[common]) == np.sign(b[common])).mean()),
                         "n_beating_margin_variant": int((a[common] > 0.005).sum()),
                         "n_beating_margin_base": int((b[common] > 0.005).sum())})
    return pd.DataFrame(rows), pd.DataFrame(conc)
