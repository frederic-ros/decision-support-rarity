"""``rare-al revision-report``: every analysis requested in the ESWA-D-26-42849 reviews.

Inputs (each section runs only if its inputs exist; a failing section is logged and
skipped, as in ``rare-al analyze``):

* ``<campaign>/{synthetic,real,image,oracle}``  -- the reported campaign (Zenodo archive);
* ``<revision>/synthetic``        -- configs/eswa/rev_synthetic.json  (noise + fixed test);
* ``<revision>/oracle``           -- configs/eswa/rev_oracle.json     (LR/RF/GBDT oracle);
* ``<revision>/oracle_campaign``  -- configs/eswa/rev_oracle.json     (campaigns, same states);
* ``<revision>/budget``           -- configs/eswa/rev_budget.json     (matched budget);
* ``<revision>/sensitivity``      -- configs/eswa/rev_sensitivity.json (learner settings).

Outputs: ``<out>/stats/rev_*.csv`` and ``<out>/tables/rev_*.tex`` plus
``<out>/revision_report.md`` listing what was computed from what.
"""
from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from rare_al.analysis import revision as R
from rare_al.analysis.load import load_trajectories
from rare_al.analysis.paired import final_rows, paired
from rare_al.strategies.registry import display

# "logreg" is the liblinear LR in results/campaign and the lbfgs LR in a swap-learner tree
LEARNER_NAME = {"logreg": "LR", "logreg_lbfgs": "LR (lbfgs)", "rf": "RF", "gbdt": "GBDT"}
EXTRA = ["fx_auprc", "auprc", "n_rare_labeled", "recall"]


def _fmt(x, nd=3, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _ci(m, lo, hi, nd=3):
    return f"{_fmt(m, nd)} [{_fmt(lo, nd)},{_fmt(hi, nd)}]"


def _pct(x, nd=1):
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}"


def _tabular(header, rows, align):
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(map(str, r)) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


class RevisionBuilder:
    def __init__(self, cfg, campaign, revision, out, cache_dir=None, n_boot=2000):
        self.cfg = cfg
        self.campaign = Path(campaign) if campaign else None
        self.revision = Path(revision) if revision else None
        self.out = Path(out)
        for sub in ("stats", "tables"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        self.n_boot = n_boot
        self.report = []
        self.ref = cfg.get("reference_strategy", "margin")

    # ------------------------------------------------------------------ io helpers
    def _find(self, root, sub, file="trajectories.csv"):
        if root is None:
            return None
        p = root / sub / file
        return p if p.exists() else None

    def _stat(self, name, df):
        if df is not None and len(df):
            df.to_csv(self.out / "stats" / name, index=False)

    def _tex(self, name, text):
        (self.out / "tables" / name).write_text(text, encoding="utf-8")

    def _log(self, section, source, note=""):
        self.report.append({"section": section, "source": str(source), "note": note})
        print(f"[revision] {section:28s} <- {source} {note}", flush=True)

    def _paired(self, path, budget="main"):
        df = load_trajectories(path)
        if "budget" in df and budget is not None:
            df = df[df.budget == budget]
        fin = final_rows(df)
        metrics = [m for m in ["auprc", "recall", "cost_norm", "aubc", "discovery_rate", "n_rare_labeled",
                               "fx_auprc"] if m in fin]
        return df, fin, paired(fin, self.ref, metrics=metrics)

    # ------------------------------------------------------- R3.1 / R3.2 / R3.3
    def synthetic_noise(self):
        path = self._find(self.revision, "synthetic")
        if not path:
            self._log("noise floor + fixed test", "missing", "run configs/eswa/rev_synthetic.json")
            return
        df, fin, p = self._paired(path)
        nf = R.noise_floor(p, n_boot=self.n_boot)
        nf_ir = R.noise_floor(p, by=("learner", "ir"), n_boot=self.n_boot)
        ex = R.excess_tail(p, n_boot=self.n_boot)
        self._stat("rev_noise_floor.csv", nf)
        self._stat("rev_noise_floor_by_ir.csv", nf_ir)
        self._stat("rev_excess_tail.csv", ex)
        rows = []
        for _, r in nf.sort_values(["noise", "learner"]).iterrows():
            rows.append([LEARNER_NAME.get(r.learner, r.learner), r.noise.replace("_", " "),
                         int(r.n_states), _fmt(r.sd, 3, False), _fmt(r.q05), _fmt(r.worst, 2),
                         f"{_pct(r.c_tau)} [{_pct(r.c_tau_ci_low)},{_pct(r.c_tau_ci_high)}]"])
        self._tex("rev_noise_floor.tex", _tabular(
            ["Learner", "Noise", "States", "SD", r"$q_{0.05}$", "Worst", r"$\mathcal C_{0.10}$ (\%)"],
            rows, "llrrrrr"))
        rows = []
        for _, r in ex.sort_values(["learner", "excess"], ascending=[True, False]).iterrows():
            rows.append([display(r.strategy), LEARNER_NAME.get(r.learner, r.learner), _fmt(r["mean"]),
                         _pct(r.c_tau), _pct(r.c_tau_noise),
                         f"{_pct(r.excess)} [{_pct(r.excess_ci_low)},{_pct(r.excess_ci_high)}]",
                         _pct(r.c_tau_vs_mean), _fmt(r.q05_vs_mean, 2), _pct(r.share_tail_in_unstable, 0)])
        self._tex("rev_excess_tail.tex", _tabular(
            ["Strategy", "Learner", "Mean", r"$\mathcal C$", r"$\mathcal C_{\rm noise}$", "Excess [95\\% CI]",
             r"$\mathcal C$ vs.\ mean ref.", r"$q_{0.05}$ vs.\ mean ref.", r"in unstable states (\%)"],
            rows, "llrrrrrrr"))
        self._log("noise floor + excess tail", path, f"{p.state_id.nunique()} states")
        # R3.2 on the revision run (has n_test_pos and p_unique)
        self._worst_cases(p, "rev_synthetic", path)
        # R3.3 composition + fixed-test rarity analysis
        tc = fin.groupby(["state_id", "regime", "ir"]).agg(n_test_pos=("n_test_pos", "first"),
                                                           n_test=("n_test", "first")).reset_index()
        tc["n_pool_pos"] = np.nan
        self._stat("rev_test_positives_synthetic.csv",
                   tc.groupby("ir").n_test_pos.describe().reset_index())
        if "fx_auprc" in fin:
            from rare_al.analysis.risk import c1_decomposition, risk_profile
            # Margin replicates are noise controls, not strategies: excluded from the dispersion analysis
            p = p[~p.strategy.astype(str).str.startswith(R.REP_PREFIX)]
            fin = fin[~fin.strategy.astype(str).str.startswith(R.REP_PREFIX)]
            dec_orig = c1_decomposition(p)
            pfx = R.fixed_test_paired(fin, self.ref)
            dec_fx = c1_decomposition(pfx)
            self._stat("rev_c1_original_test.csv", dec_orig)
            self._stat("rev_c1_fixed_test.csv", dec_fx)
            a, b = R.amplification_count(dec_orig), R.amplification_count(dec_fx)
            m = dec_orig.merge(dec_fx, on=["learner", "geometry"], suffixes=("_orig", "_fx"))
            rows = [[LEARNER_NAME.get(r.learner, r.learner), r.geometry,
                     f"{r.sd_growth_orig:.2f} [{r.sd_growth_ci_low_orig:.2f},{r.sd_growth_ci_high_orig:.2f}]",
                     f"{r.sd_growth_fx:.2f} [{r.sd_growth_ci_low_fx:.2f},{r.sd_growth_ci_high_fx:.2f}]",
                     f"{r.margin_collapse_orig:.1f}", f"{r.margin_collapse_fx:.1f}",
                     "yes" if (r.sd_growth_ci_low_fx > 1 and not r.scale_degenerate_fx) else "no"]
                    for _, r in m.iterrows()]
            self._tex("rev_c1_fixed_test.tex", _tabular(
                ["Learner", "Geometry", "SD growth, original test", "SD growth, 100 test positives",
                 "Ref.\\ drop (orig.)", "Ref.\\ drop (fixed)", "Amplified"], rows, "llrrrrl"))
            rk = risk_profile(pfx.assign(ref=pfx.ref), by=("learner", "strategy"))
            self._stat("rev_risk_fixed_test.csv", rk)
            self._log("fixed-test rarity analysis", path,
                      f"amplified cells: original {a['n_amplified']}/{a['n_cells']}, "
                      f"fixed test {b['n_amplified']}/{b['n_cells']}")

    def _worst_cases(self, p, tag, source):
        pairs, by_state, summ = R.worst_case_anatomy(p, threshold=-0.5, top=None)
        lq = R.lower_quantiles(p, n_boot=self.n_boot)
        self._stat(f"rev_worst_pairs_{tag}.csv", pairs)
        self._stat(f"rev_worst_by_state_{tag}.csv", by_state)
        self._stat(f"rev_worst_summary_{tag}.csv", summ)
        self._stat(f"rev_lower_quantiles_{tag}.csv", lq)
        rows = [[LEARNER_NAME.get(r.learner, r.learner), int(r.n_trajectories), int(r.n_states),
                 int(r.n_strategies), _pct(r.share_top5_states, 0), _pct(r.share_collapsed, 0),
                 _pct(r.share_constant, 0), _fmt(r.median_ref_auprc, 2, False),
                 _fmt(r.median_test_pos_in_worst, 0, False), _fmt(r.median_test_pos_all, 0, False),
                 _pct(r.share_ir_ge50, 0)] for _, r in summ.iterrows()]
        self._tex(f"rev_worst_summary_{tag}.tex", _tabular(
            ["Learner", "Traj.", "States", "Strategies", "Top-5 states (\\%)", "Collapsed (\\%)",
             "Constant (\\%)", "Ref.\\ AUPRC", "Test pos.\\ (worst)", "Test pos.\\ (all)", "IR$\\ge$50 (\\%)"],
            rows, "lrrrrrrrrrr"))
        self._log(f"worst-case anatomy ({tag})", source, f"{len(pairs)} trajectories with delta < -0.5")

    def campaign_synthetic(self):
        path = self._find(self.campaign, "synthetic")
        if not path:
            self._log("published synthetic (worst cases)", "missing", "Zenodo results/campaign/synthetic")
            return
        df, fin, p = self._paired(path)
        if "n_test_pos" not in p:   # published runs predate the n_test_pos column: recompute
            from rare_al.engine.states import synthetic_states
            tc = R.test_composition(list(synthetic_states(self.cfg)), self.cfg)
            self._stat("rev_test_positives_by_state.csv", tc)
            self._stat("rev_test_positives_summary.csv", R.test_composition_summary(tc, ("ir",)))
            p = p.merge(tc[["state_id", "n_test_pos", "n_test"]], on="state_id", how="left")
        self._worst_cases(p, "published_synthetic", path)
        from rare_al.reporting.revision_figures import fig_mean_tail_zoom
        (self.out / "figures").mkdir(parents=True, exist_ok=True)
        fig_mean_tail_zoom(p, self.out / "figures" / "fig_mean_tail_rf_v52.pdf")
        self._log("fig_mean_tail_rf_v52.pdf (zoom, R1.9)", path)
        # leave-one-geometry-out alpha sweep on the synthetic grid (tails bind under GBDT)
        q = p.assign(cell=R._cell_id(p))
        from rare_al.strategies.registry import role
        cand = [s for s in q.strategy.unique() if role(s) in ("published", "instance", "reference")]
        cells, summ = R.validation_alpha_sweep(q, domain_col="regime", unit_col="cell", candidates=cand,
                                               cluster_fn=lambda c: c, n_boot=self.n_boot,
                                               alphas=(0.0, 0.01, 0.02, 0.03, 0.05, 0.10))
        self._stat("rev_validation_synthetic_cells.csv", cells)
        self._stat("rev_validation_synthetic_sweep.csv", summ)
        self._tex("rev_validation_synthetic_sweep.tex", self._sweep_tex(summ))
        self._log("alpha sweep (leave-one-geometry-out)", path)

    # ---------------------------------------------------------------- R3.4 / R2.6
    def oracle(self):
        root = None
        for base in (self.revision, self.campaign):
            if self._find(base, "oracle", "oracle_policies.csv"):
                root = base / "oracle"
                break
        if root is None:
            self._log("RQ3 inference", "missing", "oracle_points/policies.csv")
            return
        pts, pol = pd.read_csv(root / "oracle_points.csv"), pd.read_csv(root / "oracle_policies.csv")
        ks = sorted(int(c.split("k")[-1]) for c in pol.columns if c.startswith("gain_vs_margin_k"))
        allk = []
        for k in ks:
            t = R.rq3_inference(pol, pts, k=k, n_boot=self.n_boot)
            t.insert(0, "k", k)
            allk.append(t)
        rq3 = pd.concat(allk, ignore_index=True)
        self._stat("rev_rq3_inference.csv", rq3)
        kk = max(ks)
        t = rq3[(rq3.k == kk) & ~rq3.identical_to_margin.astype(bool)].sort_values(["learner", "gain"],
                                                                                    ascending=[True, False])
        rows = [[LEARNER_NAME.get(r.learner, r.learner), display(r.policy),
                 _ci(r.gain, r.gain_ci_low, r.gain_ci_high, 4), f"{r.p_gain_holm:.2f}",
                 _ci(r["diff"], r.diff_ci_low, r.diff_ci_high, 4), f"{r.p_tost_holm:.2f}",
                 _fmt(r.min_equivalence_margin, 4, False), f"{r.kappa_cond:.2f}",
                 f"{_pct(r.c_tau)} [{_pct(r.c_tau_ci_low)},{_pct(r.c_tau_ci_high)}]"] for _, r in t.iterrows()]
        self._tex("rev_rq3_inference.tex", _tabular(
            ["Learner", "Policy", r"$\overline{\mathcal G}$ [95\% CI]", "Holm $p$",
             r"$\overline{\mathcal G}-\overline{\mathcal O}$ [95\% CI]", "TOST $p$ (Holm)",
             "Min.\\ equiv.\\ margin", r"$\kappa_{\rm cond}$", r"$\mathcal C_{0.10}$ (\%)"], rows, "llrrrrrrr"))
        self._log("RQ3 inference (Holm/BH/TOST, conditional level)", root,
                  f"learners={sorted(rq3.learner.unique())}, k={ks}")
        camp = self._find(self.revision, "oracle_campaign")
        if camp:
            _, _, pc = self._paired(camp)
            m, corr = R.query_vs_campaign(rq3[rq3.k == kk], pc)
            self._stat("rev_query_vs_campaign.csv", m)
            self._stat("rev_query_vs_campaign_corr.csv", corr)
            self._log("query vs campaign, same states", camp, corr.to_string(index=False) if len(corr) else "")
        else:
            self._log("query vs campaign, same states", "missing", "run --only oracle_campaign")

    # ------------------------------------------------------------- R3.5 / R3.6 / R3.7
    def _sweep_tex(self, summ):
        rows = []
        for _, r in summ.iterrows():
            rule = r.rule.replace("risk_", r"risk-aware, $\alpha=") + ("$" if r.rule.startswith("risk_") else "")
            rule = {"mean_rank": "Mean-rank pick (per learner)", "learner_blind": "Learner-blind winner"}.get(r.rule, rule)
            rows.append([rule, _ci(r.gain, r.gain_ci_low, r.gain_ci_high), _fmt(r.q10_gain),
                         f"{_pct(r.loss_rate)} [{_pct(r.loss_ci_low)},{_pct(r.loss_ci_high)}]",
                         _pct(r.get("share_changed", np.nan), 0),
                         _ci(r.get("dgain", np.nan), r.get("dgain_ci_low", np.nan), r.get("dgain_ci_high", np.nan))
                         if r.rule.startswith("risk_") else "--",
                         f"{_pct(r.get('dloss', np.nan))} [{_pct(r.get('dloss_ci_low', np.nan))},"
                         f"{_pct(r.get('dloss_ci_high', np.nan))}]" if r.rule.startswith("risk_") else "--"])
        return _tabular(["Rule", r"Held-out $\Delta$AUPRC [95\% CI]", "10th pct.", r"Held-out $\mathcal C_{0.10}$ (\%)",
                         "Pick changed (\\%)", r"$\Delta$gain vs mean-rank", r"$\Delta\mathcal C$ vs mean-rank (pts)"],
                        rows, "lrrrrrr")

    def real(self):
        path = self._find(self.campaign, "real")
        if not path:
            self._log("validation sweep / pilot / strata", "missing", "Zenodo results/campaign/real")
            return
        _, fin, p = self._paired(path)
        cells, summ = R.validation_alpha_sweep(p, n_boot=self.n_boot)
        self._stat("rev_validation_real_cells.csv", cells)
        self._stat("rev_validation_real_sweep.csv", summ)
        self._tex("rev_validation_real_sweep.tex", self._sweep_tex(summ))
        self._log("alpha sweep (leave-one-domain-out)", path,
                  f"max share changed = {summ.share_changed.max():.2f}" if "share_changed" in summ else "")
        from rare_al.strategies.registry import role
        cand = [s for s in p.strategy.unique() if role(s) in ("published", "instance", "reference")]
        cells, summ = R.validation_alpha_sweep(p, candidates=cand, n_boot=self.n_boot,
                                               alphas=(0.0, 0.01, 0.02, 0.03, 0.05, 0.10))
        self._stat("rev_validation_real_cells_published.csv", cells)
        self._stat("rev_validation_real_sweep_published.csv", summ)
        self._tex("rev_validation_real_sweep_published.tex", self._sweep_tex(summ))
        sim, ps = R.pilot_simulation(p, candidates=cand, n_boot=self.n_boot)
        self._stat("rev_pilot_simulation.csv", ps)
        rows = [[LEARNER_NAME.get(r.learner, r.learner), int(r.n_pilot), r.rule.replace("_", " "),
                 _ci(r.gain, r.gain_ci_low, r.gain_ci_high),
                 f"{_pct(r.loss_rate)} [{_pct(r.loss_ci_low)},{_pct(r.loss_ci_high)}]",
                 _pct(r.agrees_hindsight, 0), _pct(r.picks_reference, 0)] for _, r in ps.iterrows()]
        self._tex("rev_pilot_simulation.tex", _tabular(
            ["Learner", "Pilot runs", "Rule", r"Held-out $\Delta$AUPRC [95\% CI]", r"Held-out $\mathcal C_{0.10}$ (\%)",
             "Agrees w.\\ hindsight (\\%)", "Keeps Margin (\\%)"], rows, "lrlrrrr"))
        self._log("pilot simulation", path)
        # strata and sources
        from rare_al.data.prepare import read_manifest
        man = read_manifest(self.cache_dir) if self.cache_dir else {}
        if man:
            ds = pd.DataFrame([{"dataset": k, "domain": v.get("domain"), "ir_eval": v.get("ir_eval"),
                                "n_rare_eval": v.get("n_rare_eval"), "origin": v.get("origin")}
                               for k, v in man.items()])
        else:
            ds = fin.groupby("dataset").agg(domain=("domain", "first"), ir_eval=("ir", "first")).reset_index()
        ds = ds[ds.dataset.isin(set(fin.dataset))]
        bs, bd, detail = R.strata_counts(ds)
        self._stat("rev_strata_sources.csv", bs)
        self._stat("rev_domain_sources.csv", bd)
        self._stat("rev_dataset_sources.csv", detail)
        self._tex("rev_strata_sources.tex", _tabular(
            ["Stratum", "Datasets", "Independent sources", "Datasets"],
            [[r.stratum, int(r.n_datasets), int(r.n_sources), r.datasets.replace("_", r"\_")]
             for _, r in bs.iterrows()], "lrrp{8cm}"))
        self._log("strata and sources", "manifest" if man else path,
                  f"{detail.dataset.nunique()} datasets, {detail.source.nunique()} sources")
        # test positives of the real datasets (from the manifest / caches)
        try:
            from rare_al.engine.states import real_states
            tc = R.test_composition(list(real_states(self.cfg, man)), self.cfg) if man else None
            if tc is not None and len(tc):
                self._stat("rev_test_positives_real.csv",
                           tc.groupby("dataset").agg(n_test_pos=("n_test_pos", "median"),
                                                     n_pool_pos=("n_pool_pos", "median")).reset_index())
                self._log("test positives (real)", "caches")
        except Exception as exc:
            self._log("test positives (real)", "skipped", repr(exc))

    # ------------------------------------------------------------ budget / sensitivity
    def budget(self):
        rows = []
        path = self._find(self.revision, "budget/synthetic")
        if path:
            df = load_trajectories(path)
            b = R.budget_at_rounds(df[df.budget == "low_matched"], rounds=[4, 21], n_boot=self.n_boot)
            rows.append(b)
            self._log("matched budget (n_init=10, b=4)", path)
        cpath = self._find(self.campaign, "synthetic")
        if cpath:
            df = load_trajectories(cpath)
            b = R.budget_at_rounds(df[df.budget == "main"], rounds=[2, 8, 20], n_boot=self.n_boot,
                                   strategies=["typiclust", "probcover", "coreset", "kmeans_diversity",
                                               "direct", "most_likely_positive", "class_balanced"])
            rows.append(b)
            self._log("budget at intermediate rounds (n_init=30, b=8)", cpath)
        if rows:
            self._stat("rev_budget_matched.csv", pd.concat(rows, ignore_index=True))
        else:
            self._log("matched budget", "missing", "run configs/eswa/rev_budget.json")

    # ------------------------------------------------ revision: LR intercept regularization
    INTERCEPT_ROWS = ["direct", "most_likely_positive", "class_balanced", "galaxy", "tailor", "rca",
                      "kmeans_diversity", "coreset", "random"]

    def intercept(self):
        """liblinear (penalized intercept, published) vs. lbfgs (unpenalized intercept) on the
        same states: per-strategy gains vs. Margin, Margin's noise floor under lbfgs, and the
        threshold diagnostics (share of pool >= 0.5, rare-case lift of the queries)."""
        found, eff_all = False, []
        for sub, unit in (("real", "dataset"), ("image", "dataset"), ("synthetic", "state_id"),
                          ("rarity", "state_id")):
            path, base = self._find(self.revision, f"lbfgs/{sub}"), self._find(self.campaign, sub)
            if not (path and base):
                continue
            found = True
            dv, _, pv = self._paired(path)
            _, _, pb = self._paired(base)
            pv = pv[pv.learner == "logreg_lbfgs"]
            pb = pb[(pb.learner == "logreg") & pb.state_id.isin(set(pv.state_id))]
            reps = pv.strategy.astype(str).str.startswith("margin_rep")
            eff = R.intercept_effect(pb, pv[~reps], unit=unit,
                                     cluster_fn=R.source_group if sub == "real" else None,
                                     n_boot=self.n_boot)
            eff.insert(0, "benchmark", sub)
            eff_all.append(eff)
            self._stat(f"rev_intercept_{sub}.csv", eff)
            if reps.any():
                self._stat(f"rev_noise_floor_lbfgs_{sub}.csv", R.noise_floor(pv, n_boot=self.n_boot))
                self._stat(f"rev_excess_tail_lbfgs_{sub}.csv", R.excess_tail(pv, n_boot=self.n_boot))
            self._stat(f"rev_threshold_lbfgs_{sub}.csv", R.threshold_diagnostics(dv))
            d = eff[eff.strategy == "direct"].set_index("solver")
            note = ", ".join(f"{k}: DIRECT-Margin {_fmt(r.gain)} [{_fmt(r.gain_ci_low)}, {_fmt(r.gain_ci_high)}]"
                             for k, r in d.iterrows())
            self._log(f"LR intercept ({sub})", path, note)
        thr = []
        for sub in ("real", "image", "synthetic"):
            path = self._find(self.revision, f"threshold/{sub}")
            if path:
                t = R.threshold_diagnostics(load_trajectories(path))
                t.insert(0, "benchmark", sub)
                thr.append(t)
        if thr:
            self._stat("rev_threshold_diagnostics.csv", pd.concat(thr, ignore_index=True))
            self._log("threshold diagnostics", self.revision / "threshold", "")
        if not found:
            self._log("LR intercept", "missing", "run configs/eswa/rev_lbfgs.json")
            return
        eff = pd.concat(eff_all, ignore_index=True)
        benches = [b for b in ("real", "image", "synthetic") if b in set(eff.benchmark)]
        rows = []
        for s in [x for x in self.INTERCEPT_ROWS if x in set(eff.strategy)]:
            row = [display(s)]
            for b in benches:
                for solver in ("liblinear", "lbfgs"):
                    r = eff[(eff.benchmark == b) & (eff.strategy == s) & (eff.solver == solver)]
                    row.append(_ci(r.gain.iloc[0], r.gain_ci_low.iloc[0], r.gain_ci_high.iloc[0])
                               if len(r) else "--")
            rows.append(row)
        row = ["Margin final AUPRC"]
        for b in benches:
            for solver in ("liblinear", "lbfgs"):
                r = eff[(eff.benchmark == b) & (eff.solver == solver)]
                row.append(_fmt(r.margin_auprc_all.iloc[0], 3, False) if len(r) else "--")
        rows.append(row)
        header = ["Strategy"] + [f"{b} ({sv})" for b in benches for sv in ("liblinear", "lbfgs")]
        self._tex("tab_intercept.tex", _tabular(header, rows, "l" + "c" * (len(header) - 1)))

    def sensitivity(self):
        found = False
        for sub in ("real", "synthetic"):
            path = self._find(self.revision, f"sensitivity/{sub}")
            base = self._find(self.campaign, sub)
            if not (path and base):
                continue
            found = True
            _, _, pv = self._paired(path)
            _, _, pb = self._paired(base)
            lb = self._find(self.revision, f"lbfgs/{sub}")       # base of the LR-intercept variants
            if lb is not None:
                _, _, pl = self._paired(lb)
                pl = pl[(pl.learner == "logreg_lbfgs") & ~pl.strategy.astype(str).str.startswith("margin_rep")]
                pb = pd.concat([pb, pl], ignore_index=True)
            unit = "dataset" if sub == "real" else "state_id"
            pb = pb[pb.state_id.isin(set(pv.state_id))]
            rows, conc = R.sensitivity_summary(pv, pb, unit=unit)
            self._stat(f"rev_sensitivity_{sub}.csv", rows)
            self._stat(f"rev_sensitivity_{sub}_concordance.csv", conc)
            self._log(f"learner sensitivity ({sub})", path, conc.to_string(index=False) if len(conc) else "")
        if not found:
            self._log("learner sensitivity", "missing", "run configs/eswa/rev_sensitivity.json")

    def image(self):
        path = self._find(self.campaign, "image")
        if not path:
            self._log("image (Fig. 8, task counts)", "missing", "Zenodo results/campaign/image")
            return
        df = load_trajectories(path)
        df = df[df.budget == "main"] if "budget" in df else df
        fin = final_rows(df, endpoints=False)
        p = paired(fin, self.ref, metrics=["auprc", "n_rare_labeled"])
        from rare_al.reporting.revision_figures import fig_image_two_panels
        (self.out / "figures").mkdir(parents=True, exist_ok=True)
        t = fig_image_two_panels(p, self.out / "figures" / "fig_image_backbones.pdf", n_boot=self.n_boot)
        self._stat("rev_image_task_clustered.csv", t)
        tasks = fin.groupby(["dataset", "positive_class", "ir_target"]).state_id.nunique().reset_index()
        self._stat("rev_image_tasks.csv", tasks)
        self._log("image: Fig. 8 two panels + task counts", path,
                  f"{fin.dataset.nunique()} datasets, {len(tasks)} tasks, {fin.state_id.nunique()} states")

    def sample_size(self):
        need, power = R.pilot_sample_size()
        self._stat("rev_pilot_sample_size.csv", need)
        self._stat("rev_pilot_power.csv", power)
        self._tex("rev_pilot_sample_size.tex", _tabular(
            [r"Tolerance $\alpha$", "0 losses", "1 loss", "2 losses"],
            [[f"{a:.2f}"] + [int(need[(need.alpha == a) & (need.failures == f)].runs_needed.iloc[0])
                              for f in (0, 1, 2)] for a in sorted(need.alpha.unique())], "lrrr"))
        self._log("pilot sample size (analytic)", "closed form")

    def finish(self):
        rep = pd.DataFrame(self.report)
        lines = ["# Revision analyses report", "", "| section | source | note |", "|---|---|---|"]
        lines += [f"| {r.section} | {r.source} | {str(r.note).splitlines()[0] if r.note else ''} |"
                  for r in rep.itertuples()]
        (self.out / "revision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return rep


def build_revision(cfg, campaign, revision, out, cache_dir=None, n_boot=2000):
    b = RevisionBuilder(cfg, campaign, revision, out, cache_dir, n_boot)
    for name, fn in [("sample_size", b.sample_size), ("synthetic_noise", b.synthetic_noise),
                     ("campaign_synthetic", b.campaign_synthetic), ("image", b.image), ("oracle", b.oracle),
                     ("real", b.real),
                     ("budget", b.budget), ("sensitivity", b.sensitivity), ("intercept", b.intercept)]:
        try:
            fn()
        except Exception as exc:  # fault tolerant, like rare-al analyze
            b._log(name, "FAILED", f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
    return b.finish()
