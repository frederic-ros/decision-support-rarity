"""RQ3 analyses: label-only discovery opportunity (one- and k-step) and exploitability."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rare_al.stats.core import boot_mean, boot_ratio


def _ensure_k1(points: pd.DataFrame) -> pd.DataFrame:
    if "gain_vs_margin_k1" not in points:
        mg = points[points.is_margin_top == 1].set_index(_skeys(points))["gain"]
        points = points.join(mg.rename("_mg"), on=_skeys(points))
        points["gain_vs_margin_k1"] = points.gain - points._mg
        points = points.drop(columns="_mg")
    return points


def _skeys(df):
    return ["learner", "state_id"] if "learner" in df else ["state_id"]


def label_only_states(points: pd.DataFrame, k=1, sources=None) -> pd.DataFrame:
    """Per state: mean k-step gain of rare candidates relative to the Margin-top query
    (0 if no rare candidate -- v13 convention)."""
    points = _ensure_k1(points)
    col = f"gain_vs_margin_k{k}"
    if sources is not None and "cand_source" in points:
        points = points[points.cand_source.isin(sources) | (points.is_margin_top == 1)]
    keys = _skeys(points)
    g = points.groupby(keys)
    rare_mean = points[points.true_label == 1].groupby(keys)[col].mean()
    out = g.agg(geometry=("regime", "first"), dataset=("dataset", "first") if "dataset" in points else ("regime", "first"))
    out["delta"] = rare_mean.reindex(out.index).fillna(0.0)
    out["nrare"] = points[points.true_label == 1].groupby(keys).size().reindex(out.index).fillna(0).astype(int)
    mt = points[points.is_margin_top == 1].set_index(keys)
    out["margin_gain"] = mt["gain"].reindex(out.index)
    return out.reset_index()


def label_only_summary(points, k=1, sources=None, learner=None):
    """Headline label-only opportunity with v13 bootstrap seeds (overall 14, by geometry 13)."""
    s = label_only_states(points, k, sources)
    if learner is not None and "learner" in s:
        s = s[s.learner == learner]
    by = []
    for reg, g in s.groupby("geometry"):
        m, lo, hi = boot_mean(g.delta, seed=13)
        by.append({"geometry": reg, "mean": m, "ci_low": lo, "ci_high": hi})
    m, lo, hi = boot_mean(s.delta, seed=14)
    mg = float(s.margin_gain.mean())
    return {"k": k, "mean": m, "ci_low": lo, "ci_high": hi, "margin_gain_mean": mg,
            "headroom_to_margin_gain_ratio": m / mg if mg else np.nan, "n_states": len(s)}, pd.DataFrame(by)


def _tail_ci(x, thr, n=3000, seed=57):
    """P(x < -thr) with a percentile bootstrap interval."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(seed)
    p = float((x < -thr).mean())
    bs = [(rng.choice(x, len(x)) < -thr).mean() for _ in range(n)]
    return p, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _base_ids(state_ids) -> np.ndarray:
    """Base-state id = state_id without its trailing round index (the independent unit)."""
    return pd.Index(state_ids).str.rsplit("|", n=1).str[0].to_numpy()


def _cluster_boot(stats: dict, clusters: np.ndarray, n=4000, seed=71):
    """Cluster bootstrap (resampling base states) of means and ratios of means.

    ``stats`` maps a name to an array aligned with ``clusters``. Returns a function
    ``q(expr)`` where ``expr`` combines bootstrap mean vectors, e.g. ``q(lambda m: m["G"]-m["O"])``
    gives (lo, hi) of the resampled mean difference."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    nb = len(uniq)
    rng = np.random.default_rng(seed)
    cnt = np.zeros((n, nb))
    for b in range(n):
        cnt[b] = np.bincount(rng.integers(0, nb, nb), minlength=nb)
    size = np.bincount(inv, minlength=nb)
    tot = cnt @ size
    means = {k: (cnt @ np.bincount(inv, weights=np.asarray(v, float), minlength=nb)) / tot
             for k, v in stats.items()}

    def q(expr):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = expr(means)
        v = v[np.isfinite(v)]
        return (float(np.quantile(v, .025)), float(np.quantile(v, .975))) if len(v) else (np.nan, np.nan)
    return q


def exploitability(policies: pd.DataFrame, points: pd.DataFrame, k_list=(1,), sources=None,
                   tau=0.10, alpha=0.10, tau_sens=0.01):
    """kappa(g) = mean gain of policy g's first query / mean label-only reference
    opportunity, on the same states, with separate CIs for numerator and denominator
    and a paired bootstrap CI for the ratio.

    The pre-specified reading (Sec. 3.5) classifies a policy as exploitable when
    kappa >= 0.5, the mean-gain interval excludes zero, AND the material-loss
    probability C_tau = P(gain < -tau) is within the tolerance alpha at the upper
    bound of its bootstrap interval. P(gain < -tau_sens) is reported as a
    sensitivity threshold only.

    Complementary (post hoc, descriptive) readings added with the same states:
    * ``level_diff``: the paired within-state difference G - O between the policy's
      gain and the label-only opportunity level, with a cluster-bootstrap interval
      over independent base states. ``at_level`` = gain interval excludes zero and the
      difference interval includes zero; ``below_level`` = difference interval < 0.
    * ``kappa_ceiling``: the gain relative to the hindsight ceiling (mean over states
      of the best single candidate query), a maximized and therefore optimistic
      reference, so this ratio is a lower bound on the captured share and lies in [0, 1].
    * mechanism: ``rare_pick_rate`` (share of states in which the policy's query is a
      rare case), ``gain_when_rare`` / ``gain_when_not_rare`` (conditional mean gains),
      ``p_gain_pos`` / ``p_gain_neg`` (share of states with a positive / negative gain)."""
    rows = []
    for k in k_list:
        lo_states = label_only_states(points, k, sources).set_index(_skeys(points))
        ceil = lo_states["delta"]
        col = f"gain_vs_margin_k{k}"
        pts_k = _ensure_k1(points)
        hind = pts_k.groupby(_skeys(pts_k))[col].max()          # hindsight best single query
        for (lr, pol), g in policies.groupby(["learner", "policy"]):
            g = g.set_index(["learner", "state_id"])
            den = ceil.reindex(g.index)
            num = g[col]
            nm, nlo, nhi = boot_mean(num, seed=51)
            dm, dlo, dhi = boot_mean(den, seed=52)
            r, rlo, rhi = boot_ratio(num.to_numpy(), den.to_numpy(), seed=53)
            c, clo, chi = _tail_ci(num, tau)
            cs, _, _ = _tail_ci(num, tau_sens)
            # --- complementary paired / bounded / mechanistic readings (cluster bootstrap) ---
            hd = hind.reindex(g.index)
            lab = g.true_label.to_numpy().astype(float)
            ok = np.isfinite(num.to_numpy()) & np.isfinite(den.to_numpy()) & np.isfinite(hd.to_numpy())
            G, O, Hc = num.to_numpy()[ok], den.to_numpy()[ok], hd.to_numpy()[ok]
            clusters = _base_ids(g.index.get_level_values("state_id"))[ok]
            qb = _cluster_boot({"G": G, "O": O, "H": Hc}, clusters)
            d_lo, d_hi = qb(lambda m: m["G"] - m["O"])
            kc_lo, kc_hi = qb(lambda m: m["G"] / m["H"])
            kcl_lo, kcl_hi = qb(lambda m: m["G"] / m["O"])
            extra = {"level_diff": float((G - O).mean()), "level_diff_ci_low": d_lo, "level_diff_ci_high": d_hi,
                     "kappa_cluster_ci_low": kcl_lo, "kappa_cluster_ci_high": kcl_hi,
                     "ceiling": float(Hc.mean()), "kappa_ceiling": float(G.mean() / Hc.mean()) if Hc.mean() else np.nan,
                     "kappa_ceiling_ci_low": kc_lo, "kappa_ceiling_ci_high": kc_hi,
                     "gain_when_rare": float(G[lab[ok] == 1].mean()) if (lab[ok] == 1).any() else np.nan,
                     "gain_when_not_rare": float(G[lab[ok] == 0].mean()) if (lab[ok] == 0).any() else np.nan,
                     "p_gain_pos": float((G > 0).mean()), "p_gain_neg": float((G < 0).mean())}
            # one-sided 95% Clopper-Pearson upper bound on C_tau using the number of
            # independent base states (state_id without its trailing round index), so
            # that an observed zero does not read as a proven zero.
            from scipy.stats import beta as _beta
            n_base = int(g.index.get_level_values("state_id").str.rsplit("|", n=1).str[0].nunique())
            # zero-event bound: with no loss observed in n_base independent base states,
            # the one-sided 95% upper bound is beta.ppf(0.95, 1, n_base) (~ 3/n_base).
            cp_up = float(_beta.ppf(0.95, 1, n_base))
            rows.append({"k": k, "learner": lr, "policy": pol, "n_states": len(g), "n_base_states": n_base,
                         "gain": nm, "gain_ci_low": nlo, "gain_ci_high": nhi,
                         "reference": dm, "reference_ci_low": dlo, "reference_ci_high": dhi,
                         "kappa": r, "kappa_ci_low": rlo, "kappa_ci_high": rhi,
                         "numerator_excludes_zero": bool(nlo > 0 or nhi < 0),
                         "rare_pick_rate": float(g.true_label.mean()),
                         "c_tau": c, "c_tau_ci_low": clo, "c_tau_ci_high": chi,
                         "c_tau_cp_upper": cp_up, "c_tau_sens": cs, **extra})
    out = pd.DataFrame(rows)
    # tolerance judged at the upper bound of the bootstrap interval (pre-specified); when
    # no loss was observed the bootstrap interval degenerates to [0, 0], so the zero-event
    # bound on the number of independent base states is used instead.
    out["c_tau_upper"] = np.where(out.c_tau > 0, out.c_tau_ci_high, out.c_tau_cp_upper)
    out["within_alpha"] = out.c_tau_upper <= alpha
    out["gain_positive"] = out.gain_ci_low > 0
    out["boundary"] = (out.gain_ci_low > 0) & (out.gain_ci_low < 1e-4)   # indistinguishable from zero
    out["exploitable"] = (out.kappa >= 0.5) & out.gain_positive & out.within_alpha
    out["at_level"] = out.gain_positive & (out.level_diff_ci_low <= 0) & (out.level_diff_ci_high >= 0)
    out["below_level"] = out.level_diff_ci_high < 0
    return out


def regional_probe_kappa(obs_dir):
    """v13 regional probe: best net gain over the confidence sweep / regional ceiling."""
    from pathlib import Path
    obs_dir = Path(obs_dir)
    sweep = pd.read_csv(obs_dir / "compass_sweep.csv")
    actions = pd.read_csv(obs_dir / "actions.csv")
    best = sweep.loc[sweep.net_mean_delta.idxmax()]
    use = actions.confidence >= best.eta
    num = np.where(use, actions.compass_delta_vs_margin, 0.0)
    den = actions.regional_ceiling_vs_margin.to_numpy()
    nm, nlo, nhi = boot_mean(num, seed=61)
    r, rlo, rhi = boot_ratio(num, den, seed=62)
    return {"eta": float(best.eta), "coverage": float(best.coverage), "gain": nm, "gain_ci_low": nlo,
            "gain_ci_high": nhi, "ceiling": float(den.mean()), "kappa": r, "kappa_ci_low": rlo,
            "kappa_ci_high": rhi}


def paired_horizon_differences(points: pd.DataFrame, k_list=(1, 3, 5), sources=None, n=5000, seed=0):
    """Within-state differences O_kb - O_ka on the same states (no-rare-candidate states count as 0),
    with percentile bootstrap intervals. Returns {(ka, kb): (mean, lo, hi)}."""
    S = {k: label_only_states(points, k, sources).set_index(_skeys(points))["delta"] for k in k_list}
    rng = np.random.RandomState(seed)
    out = {}
    ks = sorted(k_list)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            d = (S[b] - S[a]).to_numpy(float)
            bs = [rng.choice(d, len(d)).mean() for _ in range(n)]
            out[(a, b)] = (float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return out
