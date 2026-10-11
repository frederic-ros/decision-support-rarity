"""Build every table and figure of the ESWA manuscript from available results.

Source priority for each asset: new v2 runs (``--runs``) > archived v13 results
(``--v13``) > missing. ``assets_report.md`` lists the provenance of every asset,
which maps directly onto the [v13] / [re-run] / [TBD] marks of the manuscript.
All intermediate statistics are written as CSV under ``<out>/stats``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rare_al.analysis import oracle as oan
from rare_al.analysis import risk
from rare_al.analysis.load import load_trajectories, load_v1_archive
from rare_al.analysis.paired import final_rows, paired
from rare_al.reporting import figures as F
from rare_al.reporting import tables as T
from rare_al.strategies.registry import display

MAIN_TABLE_ORDER = ["kmeans_diversity", "dcf", "coreset", "uherding", "adaptive_repulsion", "constrained_cover",
                    "typiclust", "probcover", "committee", "random"]


class Builder:
    def __init__(self, out, v13=None, runs=None, reference="margin"):
        self.out = Path(out)
        (self.out / "tables").mkdir(parents=True, exist_ok=True)
        (self.out / "figures").mkdir(parents=True, exist_ok=True)
        (self.out / "stats").mkdir(parents=True, exist_ok=True)
        self.v13 = load_v1_archive(v13) if v13 else {}
        self.runs = Path(runs) if runs else None
        self.ref = reference
        self.report = []

    # ------------------------------------------------------------------ helpers
    def _log(self, asset, source, note=""):
        self.report.append({"asset": asset, "source": source, "note": note})
        print(f"[assets] {asset:32s} <- {source} {note}", flush=True)

    def _write(self, name, text):
        (self.out / "tables" / name).write_text(text, encoding="utf-8")

    def _stat(self, name, df):
        df.to_csv(self.out / "stats" / name, index=False)

    def _new(self, sub, file="trajectories.csv"):
        if self.runs is None:
            return None
        p = self.runs / sub / file
        return p if p.exists() else None

    def _paired(self, df):
        return paired(final_rows(df), self.ref)

    # ------------------------------------------------------------------ RQ1/RQ2 synthetic
    def synthetic(self):
        p_new = self._new("synthetic")
        if p_new:
            df, src = load_trajectories(p_new), "new"
        elif "synthetic" in self.v13:
            df, src = self.v13["synthetic"], "v13"
        else:
            for a in ("tab_risk_v52.tex", "tab_c1_v52.tex", "tab_context_v52.tex", "tab_recall_v52.tex"):
                self._log(a, "missing")
            return
        main = df[df.budget == "main"] if "budget" in df else df
        p = self._paired(main)
        learners = [l for l in ["logreg", "rf", "gbdt"] if l in set(p.learner)]
        rk = risk.risk_profile(p)
        self._stat("synthetic_risk_profile.csv", rk)
        order = [s for s in MAIN_TABLE_ORDER if s in set(rk.strategy)] + \
                sorted(set(rk.strategy) - set(MAIN_TABLE_ORDER) - {"rareguard"})
        self._write("tab_risk_v52.tex", T.tab_risk(rk, learners, order))
        self._write("tab_risk_full.tex", T.tab_risk_full(rk, learners, order))
        self._log("tab_risk_v52.tex", src)
        dec = risk.c1_decomposition(p)
        self._stat("c1_decomposition.csv", dec)
        self._write("tab_c1_v52.tex", T.tab_c1(dec))
        F.fig_c1(dec, self.out / "figures" / "fig_c1_v52.pdf")
        self._log("tab_c1_v52.tex + fig_c1_v52.pdf", src)
        rt = risk.recall_table(p)
        self._stat("recall.csv", rt)
        self._write("tab_recall_v52.tex", T.tab_recall(rt, learners, [s for s in ["typiclust", "probcover", "random",
                    "committee", "kmeans_diversity", "coreset", "rareguard"] if s in set(rt.strategy)]))
        self._log("tab_recall_v52.tex", src)
        if {"logreg", "rf"} <= set(p.learner):
            tr, cells = risk.learner_transfer(p)
            (self.out / "stats" / "learner_transfer.json").write_text(json.dumps(tr, indent=2))
            self._write("tab_context_v52.tex", self._tab_context(rk))
            self._log("tab_context_v52.tex", src, f"cell rho={tr['cell_rho']:.3f}")
        tail_learner = "rf" if "rf" in learners else learners[-1]
        F.fig_mean_tail(rk, tail_learner, self.out / "figures" / "fig_mean_tail_rf_v52.pdf")
        F.fig_tail_vs_ir(p, tail_learner, ["kmeans_diversity", "coreset", "typiclust", "probcover", "random", "dcf"],
                         self.out / "figures" / "fig_tail_vs_ir_rf.pdf")
        self._log("fig_mean_tail_rf_v52.pdf + fig_tail_vs_ir_rf.pdf", src)
        tab, corr = risk.rare_label_mechanism(p)
        self._stat("rare_label_mechanism.csv", tab)
        self._stat("rare_label_mechanism_corr.csv", corr)
        F.fig_rare_mechanism(tab, self.out / "figures" / "fig_rare_mechanism.pdf")
        self._log("fig_rare_mechanism.pdf", src, "new diagnostic (RF tail vs rare labels acquired)")
        # budget contrast
        low = df[df.budget == "low"] if "budget" in df and (df.budget == "low").any() else self.v13.get("synthetic_low")
        if low is not None:
            pl = self._paired(low)
            bc = risk.budget_contrast(pd.concat([p, pl]), strategies=[s for s in ["coreset", "kmeans_diversity",
                                      "probcover", "typiclust"] if s in set(pl.strategy)])
            self._stat("budget_low_vs_main.csv", bc)
            self._write("tab_low_budget_v52.tex", T.tab_budget(bc))
            F.fig_budget(bc, self.out / "figures" / "fig_low_budget_v52.pdf")
            self._log("tab_low_budget_v52.tex + fig_low_budget_v52.pdf", "new" if p_new and (df.budget == "low").any() else "v13")

    def _tab_context(self, rk):
        rows = []
        for s in ["coreset", "committee", "typiclust", "kmeans_diversity", "dcf", "uherding"]:
            a = rk[(rk.learner == "logreg") & (rk.strategy == s)]
            b = rk[(rk.learner == "rf") & (rk.strategy == s)]
            if len(a) and len(b):
                a, b = a.iloc[0]["mean"], b.iloc[0]["mean"]
                rows.append(f"{display(s)} & {T._f(a)} & {T._f(b)} & {T._f(b - a)}\\\\")
        return T._wrap(rows, "Context dependence of acquisition value: final-budget mean paired gain by learner.",
                       "tab:context", "lrrr", "Strategy & LR & RF & RF$-$LR", size="small")

    # ------------------------------------------------------------------ real data
    def real(self):
        p_new = self._new("real")
        if p_new:
            df, src = load_trajectories(p_new), "new"
        elif "real" in self.v13:
            df, src = self.v13["real"], "v13 (moderate stratum, IR-compressed)"
        else:
            self._log("tab_real_v52.tex", "missing")
            return
        p = self._paired(df[df.budget == "main"])
        learners = [l for l in ["logreg", "rf", "gbdt"] if l in set(p.learner)]
        rd = risk.real_dataset_unit(p)
        self._stat("real_dataset_unit.csv", rd)
        self._write("tab_real_v52.tex", T.tab_real(rd, learners))
        self._write("tab_real_full.tex", T.tab_real_full(rd, learners))
        F.fig_forest(rd, learners, self.out / "figures" / "fig_forest_real.pdf")
        self._log("tab_real_full.tex + fig_forest_real.pdf", "new")
        self._log("tab_real_v52.tex", src)
        from rare_al.stats.core import friedman_nemenyi
        for lr in learners:
            wide = p[p.learner == lr].groupby(["dataset", "strategy"]).auprc.mean().unstack()
            wide = wide.drop(columns=[c for c in ["entropy", "least_confident"] if c in wide])
            fn = friedman_nemenyi(wide)
            F.fig_cd(fn["avg_ranks"], fn["cd"], self.out / "figures" / f"fig_cd_{lr}.pdf",
                     f"{T.LEARNER_NAME.get(lr, lr)}: Friedman p={fn['friedman_p']:.2g}, {fn['n_units']} datasets")
            self._stat(f"friedman_{lr}.csv", fn["avg_ranks"].rename("avg_rank").reset_index())
        self._log("fig_cd_<learner>.pdf", src)
        dr = risk.domain_risk(p)
        self._stat("domain_risk.csv", dr)
        if dr.domain.nunique() > 1:
            F.fig_riskmap(dr, self.out / "figures" / "fig_riskmap.pdf")
            mat = dr.pivot_table(index="strategy", columns=["domain", "learner"], values="mean_ds")
            tail = dr.pivot_table(index="strategy", columns=["domain", "learner"], values="p_lt_tau")
            mat.columns = [f"{a}|{b}" for a, b in mat.columns]     # 'domain|learner' -> one panel per learner
            tail.columns = mat.columns
            F.fig_heatmap(mat, tail, self.out / "figures" / "fig_heatmap.pdf")
            self._log("fig_riskmap.pdf + fig_heatmap.pdf", src)
        else:
            self._log("fig_riskmap.pdf + fig_heatmap.pdf", "missing", "needs >1 domain (new real runs)")
        if "cost_norm" in p:
            op = risk.operational_summary(p)
            self._stat("operational.csv", op)
            self._write("tab_operational.tex", T.tab_operational(op))
            self._write("tab_operational_full.tex", T.tab_operational_full(op))
            self._log("tab_operational.tex", src)
            if "select_s" in df:
                t = df.groupby(["strategy", "learner"]).select_s.mean().unstack()
                self._write("tab_cost.tex", T.tab_cost(t.div(t.loc[self.ref]), seconds=t))
                self._log("tab_cost.tex", src)
        else:
            self._log("tab_operational.tex + tab_cost.tex", "missing", "requires v2 runs")

    def image(self):
        """Two-embedding image benchmark (reviewer requirement): paired risk by
        backbone, so DINOv2 (self-supervised) vs ResNet-18 (ImageNet-supervised)
        are compared under the identical protocol."""
        p_new = self._new("image")
        if not p_new:
            self._log("tab_image_v52.tex + fig_image_backbones.pdf", "missing",
                      "needs image runs (rare-al extract-images + run --only image)")
            return
        df = load_trajectories(p_new)
        if "backbone" not in df.columns or df.empty:
            self._log("tab_image_v52.tex", "missing", "no backbone column in image runs")
            return
        p = self._paired(df[df.budget == "main"] if "budget" in df else df)
        rk = risk.risk_profile(p, by=("backbone", "learner", "strategy"))
        self._stat("image_risk_profile.csv", rk)
        # LaTeX table: mean AUPRC delta vs reference [95% CI], win rate, by backbone x strategy (logreg)
        lr = "logreg" if "logreg" in set(rk.learner) else sorted(set(rk.learner))[0]
        sub = rk[rk.learner == lr].copy()
        backbones = sorted(sub.backbone.unique())
        strategies = sorted(sub.strategy.unique())
        lines = [r"\begin{tabular}{l" + "cc" * len(backbones) + "}", r"\toprule",
                 "Strategy & " + " & ".join(
                     [rf"\multicolumn{{2}}{{c}}{{{T.BACKBONE_NAME.get(b, b) if hasattr(T,'BACKBONE_NAME') else b}}}"
                      for b in backbones]) + r" \\",
                 " & " + " & ".join([r"$\Delta$AUPRC & win"] * len(backbones)) + r" \\", r"\midrule"]
        for s in strategies:
            cells = []
            for b in backbones:
                r = sub[(sub.backbone == b) & (sub.strategy == s)]
                if len(r):
                    cells.append(f"{r['mean'].iloc[0]:+.3f}")
                    cells.append(f"{100 * r['win_rate'].iloc[0]:.0f}\\%")
                else:
                    cells += ["--", "--"]
            lines.append(f"{s} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        self._write("tab_image_v52.tex", "\n".join(lines))
        self._log("tab_image_v52.tex", "new", f"{len(backbones)} backbones, learner={lr}")
        # Figure: discovery (mean rare labelled at final round) by strategy, grouped by backbone
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            fin = final_rows(df[df.budget == "main"] if "budget" in df else df)
            disc = fin[fin.learner == lr].groupby(["backbone", "strategy"]).n_rare_labeled.mean()
            fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(strategies)), 3.2))
            w = 0.8 / max(len(backbones), 1)
            x = np.arange(len(strategies))
            for i, b in enumerate(backbones):
                vals = [disc.get((b, s), np.nan) for s in strategies]
                ax.bar(x + i * w, vals, w, label=b)
            ax.set_xticks(x + w * (len(backbones) - 1) / 2)
            ax.set_xticklabels(strategies, rotation=60, ha="right", fontsize=7)
            ax.set_ylabel("mean rare found (final round)")
            ax.set_title(f"Image discovery by embedding ({lr})")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(self.out / "figures" / "fig_image_backbones.pdf")
            plt.close(fig)
            self._log("fig_image_backbones.pdf", "new")
        except Exception as exc:  # pragma: no cover
            self._log("fig_image_backbones.pdf", "missing", f"plot failed: {exc}")

    def domains(self, cache_dir):
        from rare_al.data.prepare import read_manifest
        man = read_manifest(cache_dir) if cache_dir else {}
        if man:
            self._write("tab_domains_generated.tex", T.tab_domains(man))
            self._log("tab_domains_generated.tex", "manifest")
        else:
            self._log("tab_domains_generated.tex", "missing", "run prepare-data")

    # ------------------------------------------------------------------ RQ3
    def oracle(self):
        by, summ, src = {}, {}, None
        new = self._new("oracle", "oracle_points.csv")
        if new:
            pts = pd.read_csv(new)
            src = "new"
            ks = sorted(int(c.split("k")[-1]) for c in pts.columns if c.startswith("gain_vs_margin_k"))
            multi = []
            for lr in sorted(pts.learner.unique()):
                for k in ks:
                    s, b = oan.label_only_summary(pts[pts.learner == lr], k)
                    multi.append({"learner": lr, **s})
                    if k == 1:
                        by[lr], summ[lr] = b, s
            multi = pd.DataFrame(multi)
            self._stat("label_only_by_k.csv", multi)
            if len(ks) > 1:
                paired = oan.paired_horizon_differences(pts, ks) if hasattr(oan, "paired_horizon_differences") else None
                F.fig_multistep(multi, self.out / "figures" / "fig_multistep.pdf", paired=paired)
                self._log("fig_multistep.pdf", src)
            pol = self._new("oracle", "oracle_policies.csv")
            if pol:
                ex = oan.exploitability(pd.read_csv(pol), pts, ks)
                self._stat("exploitability.csv", ex)
                probe = oan.regional_probe_kappa(self.v13["observability"]) if "observability" in self.v13 else None
                self._write("tab_exploit.tex", T.tab_exploit(ex, probe, k=max(ks)))
                self._write("tab_mechanism.tex", T.tab_mechanism(ex, k=max(ks)))
                self._log("tab_exploit.tex + tab_mechanism.tex", src)
        else:
            for tag, lr in (("oracle_lr", "logreg"), ("oracle_rf", "rf")):
                if tag in self.v13:
                    pts = pd.read_csv(self.v13[tag] / "oracle_points.csv")
                    summ[lr], by[lr] = oan.label_only_summary(pts)
                    src = "v13"
            if "observability" in self.v13:
                probe = oan.regional_probe_kappa(self.v13["observability"])
                (self.out / "stats" / "regional_probe.json").write_text(json.dumps(probe, indent=2))
                self._log("stats/regional_probe.json", "v13",
                          f"gain CI [{probe['gain_ci_low']:+.5f},{probe['gain_ci_high']:+.5f}]")
            self._log("tab_exploit.tex + fig_multistep.pdf", "missing", "requires the v2 oracle run")
        if {"logreg", "rf"} <= set(summ):
            self._write("tab_oracle_v52.tex", T.tab_oracle(by["logreg"], by["rf"], summ["logreg"], summ["rf"]))
            (self.out / "stats" / "label_only.json").write_text(json.dumps(summ, indent=2, default=float))
            F.fig_label_oracle(by, self.out / "figures" / "fig_label_oracle_v52.pdf")
            self._log("tab_oracle_v52.tex + fig_label_oracle_v52.pdf", src)

    def c5(self, fast=False):
        regions = None
        new = self._new("oracle", "oracle_regions.csv")
        if new:
            regions, src = pd.read_csv(new), "new"
            regions = regions[regions.learner == "logreg"] if "learner" in regions else regions
        elif "oracle_lr" in self.v13:
            regions, src = pd.read_csv(self.v13["oracle_lr"] / "oracle_regions.csv"), "v13"
        if regions is None or fast:
            self._log("tab_c5_v52.tex", "skipped" if fast else "missing")
            return
        from rare_al.analysis.c5 import c5_table
        c5 = c5_table(regions)
        self._stat("c5.csv", c5)
        self._write("tab_c5_v52.tex", T.tab_c5(c5))
        self._log("tab_c5_v52.tex", src)

    def metareg(self):
        p_new = self._new("real")
        desc = self._new("real", "descriptors.csv")
        if not (p_new and desc):
            self._log("tab_metareg.tex", "missing", "requires v2 real runs + `rare-al descriptors`")
            return
        from rare_al.analysis.metareg import fit_by_family, prepare_frame
        d = prepare_frame(self._paired(load_trajectories(p_new)), pd.read_csv(desc))
        mr = fit_by_family(d)
        self._stat("metareg.csv", mr)
        self._write("tab_metareg.tex", T.tab_metareg(mr))
        self._write("tab_metareg_full.tex", T.tab_metareg_full(mr))
        self._log("tab_metareg.tex", "new")

    def rf_check(self):
        p_new = self._new("rf_check")
        if not p_new:
            self._log("tab_rf_check.tex", "missing", "run configs/eswa/10_rf_check.json (priority P0)")
            return
        p = self._paired(load_trajectories(p_new))
        rc = risk.rf_resolution_check(p)
        self._stat("rf_check.csv", rc)
        piv_m = rc.pivot(index="strategy", columns="learner", values="mean")
        piv_t = rc.pivot(index="strategy", columns="learner", values="p_lt_tau")
        learners = list(piv_m.columns)
        body = [display(s) + " & " + " & ".join(f"{T._f(piv_m.loc[s, l])} ({100 * piv_t.loc[s, l]:.1f}\\%)"
                                                 for l in learners) + r"\\" for s in piv_m.index]
        head = "Strategy & " + " & ".join(T.LEARNER_NAME.get(l, l) for l in learners)
        self._write("tab_rf_check.tex", T._wrap(body, "Probability-resolution check: mean paired gain "
                    "(share of states losing more than 0.10 AUPRC) under Random-Forest variants.",
                    "tab:rfcheck", "l" + "r" * len(learners), head, resize=True))
        self._log("tab_rf_check.tex", "new")

    def regimes(self):
        for sub, label in (("budget_long", "long budget"), ("cold_start", "cold start")):
            p_new = self._new(sub)
            if not p_new:
                self._log(f"tab_{sub}.tex", "missing", f"run the {label} configuration")
                continue
            p = self._paired(load_trajectories(p_new))
            rd = risk.real_dataset_unit(p) if (p.experiment == "real").any() else risk.risk_profile(p)
            self._stat(f"{sub}.csv", rd)
            op = risk.operational_summary(p) if "dataset" in p else None
            if op is not None:
                self._stat(f"{sub}_operational.csv", op)
                self._write(f"tab_{sub}.tex", T.tab_operational(op, label=f"tab:{sub.replace('_', '')}"))
            self._log(f"tab_{sub}.tex", "new")

    def finish(self):
        rep = pd.DataFrame(self.report)
        lines = ["# Paper assets report", "", "| asset | source | note |", "|---|---|---|"]
        lines += [f"| {r.asset} | {r.source} | {r.note} |" for r in rep.itertuples()]
        (self.out / "assets_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return rep


def build_all(out, v13=None, runs=None, cache_dir=None, fast=False):
    b = Builder(out, v13, runs)
    stages = [("synthetic", b.synthetic), ("real", b.real), ("image", b.image),
              ("domains", lambda: b.domains(cache_dir)), ("oracle", b.oracle),
              ("c5", lambda: b.c5(fast=fast)), ("metareg", b.metareg),
              ("rf_check", b.rf_check), ("regimes", b.regimes)]
    for name, fn in stages:
        try:
            fn()
        except Exception as exc:   # one degenerate section must not sink the report
            import traceback
            b._log(f"[{name}]", "failed", f"{type(exc).__name__}: {exc}")
            print(traceback.format_exc()[-600:], flush=True)
    return b.finish()
