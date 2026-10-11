"""LaTeX writers for the paper tables. File names and labels match the ESWA manuscript."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rare_al.strategies.registry import display

LEARNER_NAME = {"logreg": "LR", "rf": "RF", "gbdt": "GBDT", "rf_nobootstrap": "RF (no bootstrap)",
                "rf_leaf1_500": "RF (500 trees, leaf 1)", "rf_unweighted": "RF (unweighted)",
                "logreg_unweighted": "LR (unweighted)", "gbdt_unweighted": "GBDT (unweighted)",
                "lightgbm": "LightGBM", "xgboost": "XGBoost", "tabpfn": "TabPFN"}


def _f(x, d=4, sign=True):
    if pd.isna(x):
        return "--"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _ci(m, lo, hi, d=4):
    return f"{_f(m, d)} [{_f(lo, d)},{_f(hi, d)}]"


def _wrap(body_lines, caption, label, colspec, header, size="scriptsize", resize=False, sideways=False):
    """``sideways=True`` emits a landscape ``sidewaystable`` (package ``rotating``) so that a
    wide table keeps a readable font instead of being shrunk by ``resizebox``."""
    env = "sidewaystable" if sideways else "table"
    lines = [f"\\begin{{{env}}}[t]" if not sideways else f"\\begin{{{env}}}", r"\centering", f"\\{size}",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    tab = [f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r"\\", r"\midrule", *body_lines,
           r"\bottomrule", r"\end{tabular}"]
    if resize:
        width = r"\textheight" if sideways else r"\textwidth"
        tab = [f"\\resizebox{{{width}}}{{!}}{{" + tab[0]] + tab[1:-1] + [tab[-1] + "}"]
    return "\n".join(lines + tab + [f"\\end{{{env}}}"]) + "\n"


def _star(lo, hi):
    """Dagger when the 95% interval excludes zero."""
    return r"$^{\dagger}$" if (lo > 0 or hi < 0) else ""


def tab_risk(risk: pd.DataFrame, learners, order=None, label="tab:risk", tau=0.10):
    """Compact body table: the SAME three columns for every learner (mean, 5% quantile,
    material-loss probability). Intervals, SD and worst case are in the full table."""
    order = order or list(risk[risk.learner == learners[0]].sort_values("mean", ascending=False).strategy)
    body = []
    for s in order:
        cells = [display(s)]
        for l in learners:
            r = risk[(risk.learner == l) & (risk.strategy == s)]
            if r.empty:
                cells += ["--", "--", "--"]
                continue
            r = r.iloc[0]
            cells += [_f(r["mean"], 3) + _star(r.ci_low, r.ci_high), _f(r.q05, 2), f"{100 * r.p_lt_tau:.0f}"]
        body.append(" & ".join(cells) + r"\\")
    group = " & ".join(f"\\multicolumn{{3}}{{c}}{{{LEARNER_NAME.get(l, l)}}}" for l in learners)
    rules = " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(learners)))
    head = f"Strategy & {group}\\\\ {rules}\n & " + " & ".join(
        f"mean & $q_{{05}}$ & $\\mathcal C_{{{tau:.2f}}}$\\%" for _ in learners)
    cap = (f"Acquisition-risk profile on the synthetic grid at the final budget, paired difference to \\Margin{{}} "
           f"(AUPRC). For each learner: mean paired gain ($\\dagger$: bootstrap 95\\% interval excludes zero), "
           f"5\\% quantile, and material-loss probability $\\mathcal C_{{{tau:.2f}}}=P(\\Delta<-{tau:.2f})$ in \\%. "
           f"Intervals, dispersion and worst cases are in Table~\\ref{{tab:riskfull}}.")
    return _wrap(body, cap, label, "l" + "rrr" * len(learners), head, size="small")


def tab_risk_full(risk: pd.DataFrame, learners, order=None, label="tab:riskfull", tau=0.10):
    """Appendix: one panel per learner with mean [CI], SD, q05, worst and C_tau."""
    order = order or list(risk[risk.learner == learners[0]].sort_values("mean", ascending=False).strategy)
    body = []
    for l in learners:
        body.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{LEARNER_NAME.get(l, l)}}}}}\\\\")
        for s in order:
            r = risk[(risk.learner == l) & (risk.strategy == s)]
            if r.empty:
                continue
            r = r.iloc[0]
            body.append(f"{display(s)} & {_ci(r['mean'], r.ci_low, r.ci_high, 3)} & {_f(r.sd, 3, sign=False)} & "
                        f"{_f(r.q05, 3)} & {_f(r.worst, 2)} & {100 * r.p_lt_tau:.1f}\\\\")
        body.append(r"\midrule")
    body = body[:-1]
    head = f"Strategy & mean [95\\% CI] & SD & $q_{{05}}$ & worst & $\\mathcal C_{{{tau:.2f}}}$\\%"
    cap = ("Complete acquisition-risk profile on the synthetic grid (paired difference to \\Margin{}, AUPRC), "
           "one panel per learner: bootstrap 95\\% interval, standard deviation, 5\\% quantile, worst paired "
           f"trajectory and material-loss probability $\\mathcal C_{{{tau:.2f}}}$.")
    return _wrap(body, cap, label, "lrrrrr", head, size="scriptsize")


def tab_c1(dec: pd.DataFrame, label="tab:c1v52"):
    body = [f"{LEARNER_NAME.get(r.learner, r.learner)} & {r.geometry} & {_f(r.rho_abs, 3)} & {_f(r.rho_rel, 3)} & "
            f"{r.sd_growth:.2f} [{r.sd_growth_ci_low:.2f},{r.sd_growth_ci_high:.2f}] & {r.margin_collapse:.1f} & "
            f"{_ci(r.mixed_slope, r.slope_ci_low, r.slope_ci_high, 3)}\\\\" for r in dec.itertuples()]
    cap = ("Rarity-risk amplification with scale decomposition and mixed-effects inference. $\\rho_{abs}$, "
           "$\\rho_{rel}$: median per-strategy Spearman correlations of regret SD with IR. SD growth compares the "
           "highest to the lowest IR; reference drop is the corresponding \\Margin{} AUPRC ratio.")
    head = r"Learner & Geometry & $\rho_{abs}$ & $\rho_{rel}$ & SD growth [95\% CI] & Ref.\ drop & $\beta_{\log IR}$ [95\% CI]"
    return _wrap(body, cap, label, "llrrrrr", head)


def tab_budget(bc: pd.DataFrame, low_label="26 labels", high_label="matched 94 labels", label="tab:lowbudget"):
    body = [f"{LEARNER_NAME.get(r.learner, r.learner)} & {display(r.strategy)} & {_ci(r.low, r.low_ci_low, r.low_ci_high)} & "
            f"{_ci(r.high, r.high_ci_low, r.high_ci_high)} & {_f(r.low_minus_high)}\\\\" for r in bc.itertuples()]
    head = f"Learner & Strategy & {low_label} [95\\% CI] & {high_label} [95\\% CI] & difference"
    cap = "Budget sensitivity on the matched common subset of states (paired difference to \\Margin{})."
    return _wrap(body, cap, label, "llrrr", head)


def tab_recall(rt: pd.DataFrame, learners, strategies, label="tab:recall"):
    body = []
    for s in strategies:
        cells = [display(s)]
        for l in learners:
            r = rt[(rt.learner == l) & (rt.strategy == s)]
            cells.append(_ci(r.iloc[0]["mean"], r.iloc[0].ci_low, r.iloc[0].ci_high) if len(r) else "--")
        body.append(" & ".join(cells) + r"\\")
    head = "Strategy & " + " & ".join(f"{LEARNER_NAME.get(l, l)} $\\Delta$ recall [95\\% CI]" for l in learners)
    cap = ("Rare-class recall difference to \\Margin{} at the final budget. Positive recall can coexist with "
           "negative AUPRC, separating discovery from exploitation.")
    return _wrap(body, cap, label, "l" + "r" * len(learners), head, size="small")


def tab_real(rd: pd.DataFrame, learners, strategies=None, label="tab:real", bayes=False):
    """Compact body: mean [95% CI] and win rate per learner; Holm and posteriors in the full table."""
    strategies = strategies or list(rd[rd.learner == learners[0]].sort_values("mean", ascending=False).strategy)
    body = []
    for s in strategies:
        cells = [display(s)]
        for l in learners:
            r = rd[(rd.learner == l) & (rd.strategy == s)]
            if r.empty:
                cells += ["--", "--"]
                continue
            r = r.iloc[0]
            cells += [_ci(r["mean"], r.ci_low, r.ci_high, 3), f"{100 * r.win_rate:.0f}"]
        body.append(" & ".join(cells) + r"\\")
    group = " & ".join(f"\\multicolumn{{2}}{{c}}{{{LEARNER_NAME.get(l, l)}}}" for l in learners)
    rules = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(learners)))
    head = f"Strategy & {group}\\\\ {rules}\n & " + " & ".join("mean [95\\% CI] & win\\%" for _ in learners)
    cap = ("Real-data benchmark (42 datasets, dataset as unit, seeds averaged within dataset): mean paired AUPRC "
           "gain vs.\\ \\Margin{} with bootstrap 95\\% interval over datasets, and share of datasets improving on "
           "\\Margin{}. Holm-adjusted tests and Bayesian posteriors are in Table~\\ref{tab:realfull}.")
    return _wrap(body, cap, label, "l" + "rr" * len(learners), head, size="small", resize=True)


def tab_real_full(rd: pd.DataFrame, learners, strategies=None, label="tab:realfull"):
    """Appendix: per learner, mean [CI], win, Holm p, posterior worse/equivalent/better."""
    strategies = strategies or list(rd[rd.learner == learners[0]].sort_values("mean", ascending=False).strategy)
    body = []
    for l in learners:
        body.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{LEARNER_NAME.get(l, l)}}}}}\\\\")
        for s in strategies:
            r = rd[(rd.learner == l) & (rd.strategy == s)]
            if r.empty:
                continue
            r = r.iloc[0]
            ph = "$<0.001$" if r.p_holm < 0.001 else f"{r.p_holm:.3f}"
            body.append(f"{display(s)} & {_ci(r['mean'], r.ci_low, r.ci_high, 3)} & {100 * r.win_rate:.0f} & {ph} & "
                        f"{r.p_left:.2f} / {r.p_rope:.2f} / {r.p_right:.2f} & {100 * r.share_ds_tail_gt10:.0f}\\\\")
        body.append(r"\midrule")
    body = body[:-1]
    head = ("Strategy & mean [95\\% CI] & win\\% & Holm $p$ & $P(<\\,/\\,\\approx\\,/\\,>)$ & "
            "tail\\%")
    cap = ("Complete real-data results, one panel per learner. Holm-adjusted Wilcoxon $p$ against \\Margin{}; "
           "Bayesian signed-rank posterior of worse / practically equivalent (ROPE $\\pm0.01$) / better; "
           "tail\\% = share of datasets whose seed-level material-loss probability exceeds 10\\%.")
    return _wrap(body, cap, label, "lrrrrr", head, size="scriptsize")


def tab_oracle(by_lr, by_rf, s_lr, s_rf, label="tab:oraclev52"):
    geos = ["central", "overlap", "compact", "multimodal", "diffuse", "peripheral"]
    L, R = by_lr.set_index("geometry")["mean"], by_rf.set_index("geometry")["mean"]
    body = [f"{g.capitalize()} & {_f(L.get(g))} & {_f(R.get(g))}\\\\" for g in geos if g in L or g in R]
    body += [r"\midrule", f"Overall headroom & {_f(s_lr['mean'])} & {_f(s_rf['mean'])}\\\\",
             f"Margin one-step gain & {_f(s_lr['margin_gain_mean'])} & {_f(s_rf['margin_gain_mean'])}\\\\",
             f"Headroom / Margin gain & {s_lr['headroom_to_margin_gain_ratio']:.2f}$\\times$ & "
             f"{s_rf['headroom_to_margin_gain_ratio']:.2f}$\\times$\\\\"]
    cap = "Non-maximized label-only rare-discovery headroom by geometry (one-step)."
    return _wrap(body, cap, label, "lrr", "Geometry / quantity & LR & RF", size="small")


def tab_c5(c5: pd.DataFrame, label="tab:c5v52"):
    body = []
    for m, g in c5.groupby("model", sort=False):
        for r in g.itertuples():
            body.append(f"{'ExtraTrees' if m == 'extratrees' else 'LogReg'} & {r.features.replace('_', ' ')} & "
                        f"{r.groupcv:.3f} & {r.loro:.3f} & {_f(r.gap, 3)}\\\\")
        body.append(r"\addlinespace")
    cap = "Predictability of the best region from observable descriptors (GroupCV vs leave-one-geometry-out)."
    return _wrap(body[:-1], cap, label, "llrrr", "Model & Descriptors & GroupCV & LORO & Gap", size="small")


def tab_exploit(ex: pd.DataFrame, probe: dict | None = None, k=1, label="tab:exploit", tau=0.10, alpha=0.10):
    """Exploitability table. Columns: gain [CI]; paired within-state difference to the
    label-only opportunity level G-O [cluster CI]; kappa (ratio to that level) and
    kappa_ceiling (ratio to the hindsight best single query, in [0,1]); material-loss
    probability C_tau [CI]; the pre-specified verdict. Policies that never leave the
    Margin query (gain identically zero) are grouped in one line."""
    d = ex[ex.k == k].copy()
    zero = d[(d.gain.abs() < 1e-9) & (d.gain_ci_high.abs() < 1e-9)]
    nz = d[~d.index.isin(zero.index)].sort_values("gain", ascending=False)
    body = []
    if probe:
        body.append(f"Regional probe$^{{a}}$ & {_ci(probe['gain'], probe['gain_ci_low'], probe['gain_ci_high'])} & -- & "
                    f"{probe['kappa']:.2f} [{probe['kappa_ci_low']:.2f},{probe['kappa_ci_high']:.2f}] & -- & -- & --\\\\")
        body.append(r"\midrule")
    for r in nz.itertuples():
        if r.exploitable:
            verdict = "criterion met (boundary)" if r.boundary else "criterion met"
        elif r.gain_ci_low > 0 and r.within_alpha:
            verdict = "partial"
        else:
            verdict = "no"
        level = "at level" if r.at_level else ("below" if r.below_level else "between")
        ctau = (f"0.0 ($<${100 * r.c_tau_cp_upper:.1f}$^{{b}}$)" if r.c_tau == 0
                else f"{100 * r.c_tau:.1f} [{100 * r.c_tau_ci_low:.1f},{100 * r.c_tau_ci_high:.1f}]")
        body.append(f"{display(r.policy)} & {_ci(r.gain, r.gain_ci_low, r.gain_ci_high)} & "
                    f"{_ci(r.level_diff, r.level_diff_ci_low, r.level_diff_ci_high)} ({level}) & "
                    f"{r.kappa:.2f} [{r.kappa_ci_low:.2f},{r.kappa_ci_high:.2f}] & "
                    f"{r.kappa_ceiling:.2f} [{r.kappa_ceiling_ci_low:.2f},{r.kappa_ceiling_ci_high:.2f}] & "
                    f"{ctau} & {verdict}\\\\")
    if len(zero):
        z = zero.iloc[0]
        names = ", ".join(display(x) for x in zero.policy)
        body.append(r"\midrule")
        body.append(f"\\multicolumn{{7}}{{p{{0.95\\textheight}}}}{{\\emph{{Identical to \\Margin{{}} at the tested query "
                    f"(gain $\\equiv0$, $\\kappa=0$, $\\mathcal C_{{{tau:.2f}}}=0$; paired difference to the opportunity level "
                    f"{_ci(z.level_diff, z.level_diff_ci_low, z.level_diff_ci_high)}, below): {names}.}}}}\\\\")
    cap = (f"Exploitability of deployable policies on the synthetic linear-learner states ($k={k}$ continuation). "
           f"$\\overline{{\\mathcal G}}$: mean paired gain of the policy's query vs.\\ \\Margin{{}} [95\\% CI]. "
           f"$\\overline{{\\mathcal G}}-\\overline{{\\mathcal O}}$: paired within-state difference to the label-only "
           f"opportunity level [cluster-bootstrap CI over independent base states]; \\emph{{at level}}: gain interval excludes zero "
           f"and the difference interval includes zero; \\emph{{below}}: difference interval entirely negative. "
           f"$\\kappa$: ratio to the label-only level [paired bootstrap CI]; $\\kappa_{{\\max}}$: ratio to the hindsight best single "
           f"candidate query (a maximized, optimistic reference, so a lower bound on the captured share) [cluster CI]. "
           f"$\\mathcal C_{{{tau:.2f}}}$: material-loss probability in \\% [CI]. Verdict: pre-specified criterion "
           f"($\\kappa\\ge0.5$, gain interval excluding zero, $\\mathcal C_{{{tau:.2f}}}$ within $\\alpha={alpha:.2f}$ at its upper bound; "
           f"partial: positive gain within tolerance but $\\kappa<0.5$; boundary: gain lower bound positive but below $10^{{-4}}$). "
           f"$^{{b}}$ No material loss observed; one-sided 95\\% Clopper--Pearson upper bound on the number of independent base states."
           + (" $^{a}$ Relative to its regional reference level." if probe else ""))
    return _wrap(body, cap, label, "lrrrrrl",
                 r"Policy & $\overline{\mathcal G}$ [CI] & $\overline{\mathcal G}-\overline{\mathcal O}$ [CI] & $\kappa$ [CI] & $\kappa_{\max}$ [CI] & $\mathcal C_{0.10}$ \% [CI] & verdict",
                 size="footnotesize", resize=True, sideways=True)


def tab_mechanism(ex: pd.DataFrame, k=1, label="tab:mechanism", tau_sens=0.01, min_gain=1e-9):
    """How the query-level gain arises: share of states in which the policy's query is a
    rare case, conditional mean gains given a rare / non-rare query, share of states with a
    positive and a negative gain, and the share of small losses P(loss > tau_sens)."""
    d = ex[(ex.k == k) & (ex.gain.abs() >= min_gain)].sort_values("gain", ascending=False)
    body = []
    for r in d.itertuples():
        gr = "--" if not np.isfinite(r.gain_when_rare) else _f(r.gain_when_rare, 4)
        body.append(f"{display(r.policy)} & {100 * r.rare_pick_rate:.0f} & {gr} & {_f(r.gain_when_not_rare, 4)} & "
                    f"{100 * r.p_gain_pos:.0f} / {100 * r.p_gain_neg:.0f} & {100 * r.c_tau_sens:.0f}\\\\")
    mg = ex[(ex.k == k) & (ex.policy == "margin")]
    note = (f" \\Margin{{}} itself queries a rare case in {100 * float(mg.rare_pick_rate.iloc[0]):.0f}\\% of states."
            if len(mg) else "")
    cap = (f"Mechanism of the query-level gain ($k={k}$ continuation, same states as Table~\\ref{{tab:exploit}}): "
           f"share of states in which the policy's query is a rare case; mean paired gain when it is and when it is not; "
           f"share of states with a positive / negative gain; share of states losing more than ${tau_sens}$ AUPRC."
           + note + " Policies identical to \\Margin{} at the tested query are omitted.")
    return _wrap(body, cap, label, "lrrrrr",
                 r"Policy & rare query \% & $\overline{\mathcal G}\mid$rare & $\overline{\mathcal G}\mid$not rare & win / loss \% & $P(\text{loss}>0.01)$ \%",
                 size="scriptsize", resize=True)


REPRESENTATIVE = ["direct", "most_likely_positive", "class_balanced", "tailor", "uherding", "dcf",
                  "kmeans_diversity", "coreset", "typiclust", "probcover", "random"]


def tab_operational(op: pd.DataFrame, capacity_col=None, label="tab:operational", strategies=None, learners=None):
    """Body: representative strategies only; the complete table is tab_operational_full."""
    strategies = strategies or [s for s in REPRESENTATIVE if s in set(op.strategy)]
    learners = learners or [l for l in ("logreg", "gbdt") if l in set(op.learner)]
    sub = op[op.strategy.isin(strategies) & op.learner.isin(learners)]
    return _tab_operational(sub, capacity_col, label,
                            "Operational endpoints for representative strategies (paired differences to \\Margin{}, "
                            "dataset as unit; all strategies and learners in Table~\\ref{tab:operationalfull})")


def tab_operational_full(op: pd.DataFrame, capacity_col=None, label="tab:operationalfull"):
    return _tab_operational(op, capacity_col, label, "Operational endpoints for all strategies and learners "
                            "(paired differences to \\Margin{}, dataset as unit)")


def _tab_operational(op, capacity_col, label, lead):
    cap_col = capacity_col or next((c for c in op.columns if c.startswith("delta_r_at_") and not c.endswith(("low", "high"))), None)
    cols = [("delta_labels_to_5", "$\\Delta L_5$", 1), ("delta_discovery_rate", "$\\Delta$DR", 3),
            (cap_col, "$\\Delta$R@$c$", 3), ("delta_cost_norm", "$\\Delta$Cost", 3), ("delta_aubc", "$\\Delta$AUBC", 3)]
    cols = [c for c in cols if c[0] in op]
    body = []
    for l in [x for x in ("logreg", "rf", "gbdt") if x in set(op.learner)]:
        body.append(f"\\multicolumn{{{1 + len(cols)}}}{{l}}{{\\emph{{{LEARNER_NAME.get(l, l)}}}}}\\\\")
        for _, r in op[op.learner == l].iterrows():
            body.append(f"{display(r['strategy'])} & " + " & ".join(_f(r[c], d) for c, _n, d in cols) + r"\\")
        body.append(r"\midrule")
    body = body[:-1]
    head = "Strategy & " + " & ".join(n for _c, n, _d in cols)
    cap = (lead + ". $L_5$: labels needed to acquire 5 rare cases (negative = faster); DR: rare cases per label; "
           "R@$c$: recall at the investigation capacity; Cost: normalized expected cost (negative = cheaper); "
           "AUBC: area under the budget curve.")
    return _wrap(body, cap, label, "l" + "r" * len(cols), head, size="small")


FAMILY_NAME = {"active_search": "Active search", "adaptive": "Adaptive selection", "disagreement": "Disagreement",
               "framework_instance": "Framework instance (RCA)", "hybrid": "Hybrid", "passive": "Passive (random)",
               "probe": "Internal probe", "rare_targeted": "Rare-targeted", "reference": "Reference (Margin)",
               "representativeness": "Representativeness", "uncertainty": "Uncertainty"}
TERM_NAME = {"tree_learner": "Tree learner (vs linear)", "long_budget": "Long budget",
             "C(ir_stratum)[T.lt10]": "$\\IR<10$ (vs 10--50)", "C(ir_stratum)[T.50_200]": "$50\\le\\IR<200$ (vs 10--50)",
             "C(ir_stratum)[T.ge200]": "$\\IR\\ge200$ (vs 10--50)",
             "pool_f1": "F1 overlap (pool)", "pool_n1": "N1 border (pool)", "pool_n3": "N3 1-NN error (pool)",
             "pool_type_rare": "Rare-type share (pool)", "seed_f1": "F1 overlap (seed)", "seed_n1": "N1 border (seed)",
             "seed_type_rare": "Rare-type share (seed)"}
KEY_TERMS = ["tree_learner", "C(ir_stratum)[T.ge200]", "pool_type_rare", "pool_f1"]   # 4 columns fit the printed width at footnotesize


def _pstars(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))


def tab_metareg(mr: pd.DataFrame, terms=None, label="tab:metareg"):
    """Body: families as rows, a few readable key terms as columns, coefficient with significance stars."""
    terms = [t for t in (terms or KEY_TERMS) if t in set(mr.term)]
    fams = [f for f in FAMILY_NAME if f in set(mr.family) and f != "probe"]   # internal probes: full table only
    body = []
    for f in fams:
        cells = [FAMILY_NAME[f]]
        for t in terms:
            r = mr[(mr.family == f) & (mr.term == t)]
            cells.append(f"{_f(r.iloc[0].coef, 3)}{_pstars(r.iloc[0].p)}" if len(r) else "--")
        body.append(" & ".join(cells) + r"\\")
    head = "Family & " + " & ".join(TERM_NAME.get(t, t) for t in terms)
    cap = ("Meta-regression of seed-averaged paired regret (AUPRC) per strategy family: key coefficients "
           "(the remaining strata, the internal probes and all intervals are in Table~\\ref{tab:metaregfull}). "
           "Learner and imbalance stratum are observable before deployment; descriptors marked (pool) are computed "
           "on the fully labeled pool and are explanatory only. Reference stratum $10\\le\\IR<50$. "
           "Stars: $p<0.05$ (*), $0.01$ (**), $0.001$ (***).")
    return _wrap(body, cap, label, "@{}l" + "r" * len(terms) + "@{}", head, size="footnotesize")


def tab_metareg_full(mr: pd.DataFrame, label="tab:metaregfull"):
    """Appendix: every term with 95% interval, one panel per family."""
    fams = [f for f in FAMILY_NAME if f in set(mr.family)]
    terms = [t for t in TERM_NAME if t in set(mr.term)]
    body = []
    for f in fams:
        body.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{{FAMILY_NAME[f]}}}}}\\\\")
        for t in terms:
            r = mr[(mr.family == f) & (mr.term == t)]
            if len(r):
                r = r.iloc[0]
                body.append(f"{TERM_NAME[t]} & {_ci(r.coef, r.ci_low, r.ci_high, 3)} & {r.p:.3f}\\\\")
        body.append(r"\midrule")
    body = body[:-1]
    cap = ("Complete meta-regression coefficients per family with 95\\% intervals (OLS with standard errors "
           "clustered by dataset). Reference stratum $10\\le\\IR<50$.")
    return _wrap(body, cap, label, "lrr", "Term & coefficient [95\\% CI] & $p$", size="scriptsize")


def tab_cost(times: pd.DataFrame, seconds: pd.DataFrame | None = None, label="tab:cost", pool_note=None):
    """Ratios to Margin per learner, plus absolute seconds per round under the linear learner when given."""
    learners = list(times.columns)
    body = []
    for s in times.index:
        cells = [display(s)] + [f"{times.loc[s, l]:.1f}" for l in learners]
        if seconds is not None:
            v = seconds.loc[s, "logreg"] if ("logreg" in seconds.columns and s in seconds.index) else float("nan")
            cells.append(f"{v:.2f}" if np.isfinite(v) else "--")
        body.append(" & ".join(cells) + r"\\")
    head = "Strategy & " + " & ".join(f"{LEARNER_NAME.get(l, l)} $\\times$" for l in learners) \
        + (" & LR s/round" if seconds is not None else "")
    cap = ("Mean selection time per round relative to \\Margin{} ($\\times$) for each learner, and absolute seconds "
           "per round under the linear learner (median pool size " + (pool_note or "\\TBD{pool size}") +
           "; hardware: \\TBD{CPU model, cores}). Ratios above about $10\\times$ under LR correspond to several "
           "seconds per batch and are incompatible with interactive annotation.")
    return _wrap(body, cap, label, "l" + "r" * len(learners) + ("r" if seconds is not None else ""), head, size="small")


def tab_domains(manifest: dict, label="tab:domains"):
    rows = sorted(manifest.values(), key=lambda m: (m["domain"], m["dataset"]))
    body = [f"{m['domain'].replace('_', ' ')} & {m['dataset'].replace('_', chr(92) + '_')} & {m['origin'].replace('_', ' ')} & "
            f"{m['ir_nominal']:.1f} & {m['ir_eval']:.1f} & {m['n_eval']} & {m['n_rare_eval']} & {m['split']}\\\\" for m in rows]
    head = "Domain & Dataset & Origin & IR nom. & IR eval. & $n$ & rare & split"
    cap = "Real-data benchmark after rarity-preserving subsampling (nominal vs evaluated imbalance)."
    return _wrap(body, cap, label, "lllrrrrl", head, resize=True)
