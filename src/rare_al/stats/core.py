"""Statistical toolbox: bootstrap, paired tests, critical differences, Bayesian
signed-rank test, and mixed-effects meta-regression."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as st


def boot_mean(x, n=2000, seed=0):
    """Mean and percentile bootstrap 95% CI (v1 definition, same RNG)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(x.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def boot_ratio(num, den, n=2000, seed=0):
    """Ratio of means with a paired bootstrap CI (resampling the same units)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (n, len(num)))
    r = num[idx].mean(1) / den[idx].mean(1)
    return float(num.mean() / den.mean()), float(np.quantile(r, .025)), float(np.quantile(r, .975))


def holm_adjust(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def wilcoxon_vs_reference(wide: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Rows = units (e.g. datasets), columns = strategies. Holm-adjusted Wilcoxon."""
    rows = []
    for m in [c for c in wide.columns if c != reference]:
        d = (wide[m] - wide[reference]).dropna()
        try:
            stat, p = st.wilcoxon(d) if (d != 0).any() else (np.nan, 1.0)
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append({"strategy": m, "n_units": len(d), "mean_delta": d.mean(),
                     "win_rate": float((d > 0).mean()), "wilcoxon_stat": stat, "p_raw": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_adjust(out.p_raw.fillna(1.0))
    return out


def friedman_nemenyi(wide: pd.DataFrame, alpha=0.05) -> dict:
    """Average ranks (1 = best, higher metric is better), Friedman test and Nemenyi CD."""
    w = wide.dropna()
    k, n = w.shape[1], w.shape[0]
    ranks = w.rank(axis=1, ascending=False).mean()
    stat, p = st.friedmanchisquare(*[w[c].to_numpy() for c in w.columns]) if k >= 3 and n >= 2 else (np.nan, np.nan)
    q = st.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"avg_ranks": ranks.sort_values(), "friedman_stat": stat, "friedman_p": p, "cd": cd,
            "n_units": n, "k": k}


def bayesian_signed_rank(x, rope=0.01, prior_strength=0.5, n_samples=20000, seed=0):
    """Bayesian signed-rank test (Benavoli et al., 2017), Monte-Carlo implementation.

    ``x``: paired differences (strategy - reference), one per unit. Returns the
    posterior probabilities that the difference is below -rope, within the ROPE,
    or above +rope.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    z = np.r_[0.0, x]                          # prior pseudo-observation at 0
    n = len(z)
    rng = np.random.default_rng(seed)
    alpha = np.r_[prior_strength, np.ones(n - 1)]
    W = rng.dirichlet(alpha, n_samples)        # n_samples x n
    s = z[:, None] + z[None, :]                # Walsh averages
    left = (s < -2 * rope).astype(float)
    right = (s > 2 * rope).astype(float)
    pl = np.einsum("si,ij,sj->s", W, left, W)
    pr = np.einsum("si,ij,sj->s", W, right, W)
    pc = 1.0 - pl - pr
    win = np.argmax(np.c_[pl, pc, pr], axis=1)
    return {"p_left": float(np.mean(win == 0)), "p_rope": float(np.mean(win == 1)),
            "p_right": float(np.mean(win == 2))}


def _tidy(params, ci_low, ci_high, pvals, method):
    return pd.DataFrame({"term": list(params.index), "coef": params.values,
                         "ci_low": np.asarray(ci_low, float), "ci_high": np.asarray(ci_high, float),
                         "p": pvals.reindex(params.index).values, "method": method})


def _degenerate(fit) -> bool:
    """A dataset-level categorical (e.g. ir_stratum) competes with the per-group random
    intercept, leaving it unidentified: the group variance collapses to ~0 and the
    affected fixed-effect CIs blow up. Detect that so we can fall back to cluster-robust OLS."""
    try:
        gv = float(np.asarray(fit.cov_re).ravel()[0])
    except Exception:
        gv = np.nan
    if np.isfinite(gv) and gv < 1e-8:
        return True
    ci = fit.conf_int()
    width = (ci[1] - ci[0]).to_numpy(float)
    if not np.all(np.isfinite(width)) or np.nanmax(np.abs(width)) > 1e3:
        return True
    return bool(getattr(fit, "converged", True) is False)


def cluster_ols(df: pd.DataFrame, formula: str, group: str):
    """OLS with standard errors clustered by ``group`` (the inferential unit).

    This is the estimator used for the paper's meta-regression: every descriptor in
    that model (imbalance stratum, pool complexity, minority-type profile) is a
    dataset-level covariate, so a per-dataset random intercept is confounded with
    them and whether its variance collapses to zero depends on the optimizer and
    platform. Cluster-robust OLS identifies the between-dataset terms deterministically."""
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": df[group]})
    ci = ols.conf_int()
    return _tidy(ols.params, ci[0].values, ci[1].values, ols.pvalues, "ols_cluster"), ols


def mixed_effects(df: pd.DataFrame, formula: str, group: str, vc: dict | None = None,
                  estimator: str = "auto"):
    """Linear mixed model with a per-``group`` random intercept (statsmodels).

    ``estimator="ols_cluster"`` skips the mixed model and returns :func:`cluster_ols`.
    With ``"auto"`` the mixed model is fitted and, if the random effect is degenerate
    (a group-level covariate nested in ``group`` is confounded with the random
    intercept), the model is refit as cluster-robust OLS. The ``method`` column
    records which estimator produced each row. Note that the degeneracy test is
    optimizer-dependent, so ``"auto"`` is not guaranteed to be reproducible across
    platforms; use ``"ols_cluster"`` for reported tables."""
    if estimator == "ols_cluster":
        return cluster_ols(df, formula, group)
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm(formula, df, groups=df[group], vc_formula=vc).fit(
            reml=False, method="lbfgs", disp=False)
        if _degenerate(fit):
            return cluster_ols(df, formula, group)
    ci = fit.conf_int()
    return _tidy(fit.params, ci[0].values, ci[1].values, fit.pvalues, "mixedlm"), fit
