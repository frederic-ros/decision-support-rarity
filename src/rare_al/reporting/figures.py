"""Figures of the paper (vector PDF, matplotlib only)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rare_al.reporting.tables import LEARNER_NAME  # noqa: E402
from rare_al.strategies.registry import display, family  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
FAMILY_COLOR = {"reference": "#444444", "uncertainty": "#777777", "passive": "#999999",
                "disagreement": "#8c564b", "representativeness": "#1f77b4", "hybrid": "#17becf",
                "rare_targeted": "#d62728", "active_search": "#ff7f0e", "adaptive": "#2ca02c",
                "framework_instance": "#9467bd", "probe": "#bcbd22"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_c1(dec, path):
    """SD growth per learner x geometry, log scale, grouped by learner, degenerate cells hollow."""
    d = dec.copy()
    d["lr"] = d.learner.map(lambda x: LEARNER_NAME.get(x, x))
    d = d.sort_values(["learner", "geometry"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(d))
    err = np.clip(np.vstack([d.sd_growth - d.sd_growth_ci_low, d.sd_growth_ci_high - d.sd_growth]), 0.0, None)
    lcol = {"logreg": "#1f77b4", "rf": "#d62728", "gbdt": "#2ca02c"}
    for i in range(len(d)):
        deg = bool(d.scale_degenerate[i])
        ax.errorbar(d.sd_growth[i], y[i], xerr=err[:, i:i + 1], fmt="o", ms=5, color=lcol.get(d.learner[i], "k"),
                    mfc="white" if deg else lcol.get(d.learner[i], "k"), capsize=2, lw=1)
    # separators between learner groups
    for lr, g in d.groupby("learner", sort=False):
        ax.axhspan(g.index.min() - 0.5, g.index.max() + 0.5, color=lcol.get(lr, "k"), alpha=0.05, lw=0)
        ax.text(0.09, float(np.mean(g.index.to_numpy())), LEARNER_NAME.get(lr, lr), va="center", ha="left", fontsize=8, fontweight="bold")
    ax.axvline(1, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(y, list(d.geometry))
    ax.set_xlabel("Growth of absolute regret SD, IR=100 vs IR=1 (log scale; seed-cluster bootstrap 95% CI)")
    ax.set_xlim(left=0.08)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color="k", ls="", label="amplified"),
                       Line2D([], [], marker="o", color="k", mfc="white", ls="", label="scale-degenerate (reference collapses)")],
              frameon=False, fontsize=7, loc="lower right")
    _save(fig, path)


def fig_mean_tail(risk, learner, path, tau=0.10, n_label=8):
    """Mean gain vs loss probability. Only the most informative strategies are labelled (largest |mean| and
    largest tails); the rest are numbered and listed in a side legend to avoid overlapping text."""
    r = risk[risk.learner == learner].reset_index(drop=True)
    r["score"] = r["mean"].abs().rank() + r.p_lt_tau.rank()
    lab = set(r.nlargest(n_label, "score").strategy)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    handles = []
    # labelled points: cluster near-coincident points, then stack their labels in a column to the
    # right of the cluster at evenly spaced absolute heights, with leader lines (no overlap possible)
    labelled = r[r.strategy.isin(lab)].reset_index(drop=True)
    xs, ys = labelled["mean"].to_numpy(), 100 * labelled.p_lt_tau.to_numpy()
    xr = (r["mean"].max() - r["mean"].min()) or 1.0
    yr = (100 * (r.p_lt_tau.max() - r.p_lt_tau.min())) or 1.0
    clusters = []
    for j in range(len(labelled)):
        for cl in clusters:
            if any(abs(xs[j] - xs[k]) < 0.18 * xr and abs(ys[j] - ys[k]) < 0.18 * yr for k in cl):
                cl.append(j); break
        else:
            clusters.append([j])
    pos = {}
    xmin = r["mean"].min()
    side = {}
    for cl in clusters:
        cl = sorted(cl, key=lambda k: -ys[k])
        right = max(xs[k] for k in cl) < xmin + 0.55 * xr        # room on the right?
        x_text = (max(xs[k] for k in cl) + 0.07 * xr) if right else (min(xs[k] for k in cl) - 0.07 * xr)
        top = max(ys[k] for k in cl) + 0.02 * yr
        step = 0.075 * yr
        for n, k in enumerate(cl):
            pos[k] = (x_text, top - n * step)
            side[k] = "left" if right else "right"
    for i, row in r.iterrows():
        c = FAMILY_COLOR.get(family(row.strategy), "k")
        ax.scatter(row["mean"], 100 * row.p_lt_tau, color=c, s=28, zorder=3, edgecolor="k", lw=0.4)
    for k, srow in labelled.iterrows():
        ax.annotate(display(srow.strategy), (xs[k], ys[k]), xytext=pos[k], textcoords="data", fontsize=8,
                    va="center", ha=side[k],
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555", shrinkA=0, shrinkB=3))
    for i, row in r.iterrows():
        if row.strategy not in lab:
            ax.annotate(str(i + 1), (row["mean"], 100 * row.p_lt_tau), fontsize=6, xytext=(3, 2),
                        textcoords="offset points", color="#444444")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Mean paired AUPRC gain vs Margin")
    ax.set_ylabel(f"P(loss > {tau:.2f}) [%]")
    ax.set_title(LEARNER_NAME.get(learner, learner), loc="left", fontsize=9)
    others = [f"{i + 1}: {display(s)}" for i, s in enumerate(r.strategy) if s not in lab]
    ax.text(1.02, 0.98, "\n".join(others), transform=ax.transAxes, fontsize=7, va="top", ha="left")
    from matplotlib.lines import Line2D
    fams = sorted({family(s) for s in r.strategy})
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=FAMILY_COLOR.get(f, "k"), label=f.replace("_", " "))
                       for f in fams], fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, title="strategy family", title_fontsize=7.5)
    fig.subplots_adjust(right=0.72)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_tail_vs_ir(p, learner, strategies, path, tau=0.10, precomputed=None):
    """Loss probability vs IR with distinct line styles/markers and seed-bootstrap bands; legend outside."""
    styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), "-"]
    marks = "osD^v<>P"
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for i, s in enumerate(strategies):
        if precomputed is not None:
            g = precomputed[(precomputed.learner == learner) & (precomputed.strategy == s)].sort_values("ir")
            if g.empty:
                continue
            x, yv, lo, hi = g.ir, g.p, g.lo, g.hi
        else:
            gs = p[(p.learner == learner) & (p.strategy == s)]
            t = gs.groupby("ir").delta.apply(lambda v: 100 * (v < -tau).mean())
            x, yv, lo, hi = t.index, t.values, None, None
        c = FAMILY_COLOR.get(family(s), "k")
        ax.plot(x, yv, ls=styles[i % len(styles)], marker=marks[i % len(marks)], ms=5, lw=1.4, color=c, label=display(s))
        if lo is not None:
            ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Imbalance ratio (log scale)")
    ax.set_ylabel(f"P(loss > {tau:.2f}) [%]")
    ax.set_title(LEARNER_NAME.get(learner, learner), loc="left", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_budget(bc, path):
    fig, axes = plt.subplots(1, bc.learner.nunique(), figsize=(6.2, 2.8), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (lr, g) in zip(axes, bc.groupby("learner")):
        x = np.arange(len(g))
        ax.bar(x - 0.2, g.high, 0.4, label="main budget", color="#999999")
        ax.bar(x + 0.2, g.low, 0.4, label="low budget", color="#1f77b4")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, [display(s) for s in g.strategy], rotation=30, ha="right")
        ax.set_title(LEARNER_NAME.get(lr, lr))
    axes[0].set_ylabel("Mean paired AUPRC gain")
    axes[0].legend(fontsize=7, frameon=False)
    _save(fig, path)


def fig_label_oracle(by: dict, path):
    """``by``: {learner: DataFrame(geometry, mean, ci_low, ci_high)}."""
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    geos = sorted(set().union(*[set(b.geometry) for b in by.values()]))
    w = 0.8 / len(by)
    for i, (lr, b) in enumerate(by.items()):
        b = b.set_index("geometry").reindex(geos)
        x = np.arange(len(geos)) + (i - (len(by) - 1) / 2) * w
        ax.bar(x, b["mean"], w, yerr=[b["mean"] - b.ci_low, b.ci_high - b["mean"]], capsize=2,
               label=LEARNER_NAME.get(lr, lr))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(geos)), geos)
    ax.set_ylabel("Label-only headroom (AUPRC)")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, path)


def fig_multistep(summ: pd.DataFrame, path, paired: dict | None = None):
    """``summ``: learner, k, mean, ci_low, ci_high. ``paired``: {(k_a, k_b): (diff, lo, hi)} annotated in-plot."""
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for lr, g in summ.groupby("learner"):
        g = g.sort_values("k")
        ax.errorbar(g.k, g["mean"], yerr=[g["mean"] - g.ci_low, g.ci_high - g["mean"]], marker="o",
                    capsize=3, label=LEARNER_NAME.get(lr, lr))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Horizon k (rounds)")
    ax.set_ylabel("Label-only reference opportunity (AUPRC)")
    ax.set_xticks(sorted(summ.k.unique()))
    if paired:
        txt = "\n".join(f"O$_{b}$ $-$ O$_{a}$ = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]" for (a, b), (d, lo, hi) in paired.items())
        ax.text(0.02, 0.97, "paired within-state differences:\n" + txt, transform=ax.transAxes, fontsize=6.5, va="top")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _save(fig, path)


def fig_forest(rd: pd.DataFrame, learners, path, strategies=None):
    """Forest plot of mean paired gain vs Margin with 95% CI, one column per learner, dataset as unit."""
    strategies = strategies or list(rd[rd.learner == learners[0]].sort_values("mean").strategy)
    # Rendered at the printed text width (5.4 in, elsarticle 12pt) so that fonts are not scaled down.
    fig, axes = plt.subplots(1, len(learners), figsize=(5.4, 0.21 * len(strategies) + 1.0),
                             sharey=True, squeeze=False)
    y = np.arange(len(strategies))
    for ax, lr in zip(axes[0], learners):
        g = rd[rd.learner == lr].set_index("strategy").reindex(strategies)
        for i, s in enumerate(strategies):
            r = g.loc[s]
            c = FAMILY_COLOR.get(family(s), "k")
            sig = (r.ci_low > 0) or (r.ci_high < 0)
            ax.errorbar(r["mean"], y[i], xerr=[[r["mean"] - r.ci_low], [r.ci_high - r["mean"]]], fmt="o", ms=4,
                        color=c, mfc=c if sig else "white", capsize=2, lw=1)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_title(LEARNER_NAME.get(lr, lr), fontsize=9)
        ax.set_xlabel("Mean gain [95% CI]", fontsize=8)
        ax.tick_params(axis="x", labelsize=7.5)
    axes[0][0].set_yticks(y, [display(s) for s in strategies], fontsize=8)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="k", label="CI excludes 0"),
                        Line2D([], [], marker="o", ls="", color="k", mfc="white", label="CI includes 0")],
               fontsize=7.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_riskmap(dr: pd.DataFrame, path, tau=0.10, ncols=2):
    """One panel per domain on a grid (legend in the spare cell), shared axes, families = colour,
    learners = marker. Sized for a full text-width figure so that fonts stay legible."""
    doms = sorted(dr.domain.unique())
    nrows = int(np.ceil((len(doms) + 1) / ncols))
    # 5.4 in = printed text width; with ncols=2 each panel is ~2.6 in wide and fonts print at their nominal size.
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4, 2.3 * nrows), sharey=True, sharex=True, squeeze=False)
    learners = [l for l in ("logreg", "rf", "gbdt") if l in set(dr.learner)]
    markers = dict(zip(learners, "os^"))
    flat = axes.ravel()
    for ax, dom in zip(flat, doms):
        g = dr[dr.domain == dom]
        for r in g.itertuples():
            ax.scatter(r.mean_ds, 100 * r.p_lt_tau, marker=markers.get(r.learner, "o"), s=28,
                       color=FAMILY_COLOR.get(family(r.strategy), "k"), edgecolor="k", lw=0.3, alpha=0.9)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_title(dom.replace("_", " "), fontsize=10)
        ax.tick_params(labelsize=9)
    for ax in flat[len(doms):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Mean gain vs Margin", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"datasets with loss > {tau:.2f} [%]", fontsize=9)
    from matplotlib.lines import Line2D
    fams = sorted({family(s) for s in dr.strategy.unique()})
    h = [Line2D([], [], marker="o", ls="", color=FAMILY_COLOR.get(f, "k"), label=f.replace("_", " ")) for f in fams]
    h += [Line2D([], [], marker=markers[l], ls="", color="k", mfc="white", label=LEARNER_NAME.get(l, l)) for l in learners]
    flat[len(doms)].legend(handles=h, fontsize=8.5, frameon=False, loc="center", ncol=2, title="family / learner", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_cd(ranks: pd.Series, cd: float, path, title=""):
    """Critical-difference diagram (Demsar, 2006)."""
    ranks = ranks.sort_values()
    k = len(ranks)
    fig, ax = plt.subplots(figsize=(6.4, 0.35 * k / 2 + 1.4))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.85, lo, hi, color="k")
    for i in range(lo, hi + 1):
        ax.vlines(i, 0.83, 0.87, color="k")
        ax.text(i, 0.9, str(i), ha="center", fontsize=7)
    ax.hlines(0.97, lo, lo + cd, color="r", lw=2)
    ax.text(lo + cd / 2, 0.99, f"CD={cd:.2f}", ha="center", fontsize=7, color="r")
    half = (k + 1) // 2
    for j, (s, r) in enumerate(ranks.items()):
        left = j < half
        yy = 0.75 - 0.6 * ((j if left else k - 1 - j) + 1) / (half + 1)
        xx = lo - 0.4 if left else hi + 0.4
        ax.plot([r, r, xx], [0.85, yy, yy], color="k", lw=0.6)
        ax.text(xx, yy, f"{display(s)} ({r:.2f})", ha="right" if left else "left", va="center", fontsize=7)
    # cliques: groups within CD
    vals = ranks.to_numpy()
    y0 = 0.8
    for i in range(k):
        j = np.searchsorted(vals, vals[i] + cd, side="right") - 1
        if j > i and (i == 0 or np.searchsorted(vals, vals[i - 1] + cd, side="right") - 1 < j):
            ax.hlines(y0, vals[i], vals[j], color="k", lw=2.5)
            y0 -= 0.03
    ax.set_title(title, fontsize=8)
    _save(fig, path)


def fig_heatmap(mat: pd.DataFrame, tail: pd.DataFrame | None, path, alpha=0.10):
    """Three panels (one per learner), same strategy order, shared colour scale; a cross marks cells whose
    share of datasets with a material loss exceeds ``alpha``. ``mat`` columns are 'domain|learner'."""
    cols = list(mat.columns)
    learners = [l for l in ("logreg", "rf", "gbdt") if any(str(c).endswith("|" + l) for c in cols)]
    if not learners:                       # fall back to a single panel
        learners, split = [None], False
    else:
        split = True
    # Rows sorted by the mean gain under the first learner (descending), so the panels share one order.
    first = [c for c in cols if (not split) or str(c).endswith("|" + learners[0])]
    order = mat[first].mean(axis=1).sort_values(ascending=False).index
    mat = mat.loc[order]
    tail = tail.loc[order] if tail is not None else None
    v = np.nanmax(np.abs(mat.to_numpy()))
    # 5.4 in = printed text width (colour bar included); rows sized so fonts print at nominal size.
    fig, axes = plt.subplots(1, len(learners), figsize=(5.4, 0.2 * mat.shape[0] + 1.2),
                             sharey=True, squeeze=False)
    for ax, lr in zip(axes[0], learners):
        sub = mat[[c for c in cols if (not split) or str(c).endswith("|" + lr)]]
        im = ax.imshow(sub.to_numpy(), cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(sub.shape[1]), [str(c).split("|")[0].replace("_", " ") for c in sub.columns], rotation=45, ha="right", fontsize=8.5)
        if tail is not None:
            for i, s in enumerate(sub.index):
                for j, c in enumerate(sub.columns):
                    if tail.loc[s, c] > alpha:
                        ax.text(j, i, "x", ha="center", va="center", fontsize=8)
        ax.set_title(LEARNER_NAME.get(lr, lr) if lr else "", fontsize=9)
    axes[0][0].set_yticks(range(mat.shape[0]), [display(s) for s in mat.index], fontsize=8.5)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="mean paired AUPRC gain vs Margin", shrink=0.6, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("mean paired AUPRC gain vs Margin", fontsize=8.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_rare_mechanism(tab: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for lr, g in tab.groupby("learner"):
        ax.plot(range(len(g)), 100 * g.p_lt_010, marker="o", label=LEARNER_NAME.get(lr, lr))
        ax.set_xticks(range(len(g)), [str(b) for b in g.rare_bin])
    ax.set_xlabel("Rare labels acquired at the final budget")
    ax.set_ylabel("P(loss > 0.10) [%]")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, path)
