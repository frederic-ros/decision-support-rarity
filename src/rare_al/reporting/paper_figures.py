"""Figures of the ESWA revision, drawn at the printed text width (5.4 in, elsarticle 12 pt), so that
fonts print at their nominal size (labels 7-8 pt).

Two steps, both run by ``rare-al paper-figures``:

* ``prepare`` reads the result trees and writes small CSV files in ``<out>/figure_data``;
* ``plot`` draws every PDF from those CSV files only, so the figures can be redrawn without the
  trajectories (``--plot-only``). The CSV files are released with the paper.

Inputs (defaults follow REVISION_ESWA.md):
    --campaign   results/campaign              published campaign (LR = liblinear, penalized intercept)
    --lbfgs      results/campaign_lbfgs        swapped tree (LR = lbfgs) with ``rare-al analyze`` assets
    --revision   results/revision_lbfgs/assets ``rare-al revision-report`` outputs (stats/rev_*.csv)
    --oracle     results/revision/oracle       oracle points (unpenalized LR; RF and GBDT when present)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator  # noqa: E402

from rare_al.strategies.registry import display, family  # noqa: E402

W = 5.4                                    # printed text width (in)
INK, MUTED, GRID, BAND = "#0b0b0b", "#6f6e69", "#e4e3df", "#f1efe8"
LCOL = {"logreg": "#2a78d6", "rf": "#eb6834", "gbdt": "#1baf7a", "pooled": "#0b0b0b", "lr_pen": "#8a8984"}
LMARK = {"logreg": "o", "rf": "s", "gbdt": "^", "pooled": "D", "lr_pen": "D"}
LNAME = {"logreg": "LR", "rf": "RF", "gbdt": "GBDT", "pooled": "All learners",
         "lr_pen": "LR, penalized intercept (liblinear)"}
FAMILY_COLOR = {"reference": "#444444", "uncertainty": "#777777", "passive": "#999999",
                "disagreement": "#8c564b", "representativeness": "#2a78d6", "hybrid": "#17becf",
                "rare_targeted": "#d62728", "active_search": "#ff7f0e", "adaptive": "#2ca02c",
                "framework_instance": "#9467bd", "probe": "#bcbd22", "control": "#bbbbbb"}
CONTROLS = {"committee", "robust_uncertainty", "margin_rtb", "rca", "constrained_cover", "rareguard",
            "adaptive_repulsion"}
SHORT = {"most_likely_positive": "Most-likely-pos.", "kmeans_diversity": "K-means div.",
         "committee": "Committee (LR)", "committee_learner": "Committee (same)", "class_balanced": "Class-balanced",
         "similar_flqmi": "SIMILAR", "constrained_cover": "Constrained cover", "robust_uncertainty": "Robust uncert.",
         "adaptive_repulsion": "Adaptive repulsion", "margin_rtb": "Margin (ties)"}
TAU = 0.10


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def short(s):
    return SHORT.get(s, display(s))


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def _cboot(values, clusters, n=4000, seed=7):
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    u = np.unique(clusters)
    idx = {c: np.flatnonzero(clusters == c) for c in u}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for b in range(n):
        ii = np.concatenate([idx[c] for c in rng.choice(u, len(u))])
        out[b] = values[ii].mean()
    return float(values.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _repel(y, min_gap, lo=None, hi=None):
    """1-D label positions: keep order, enforce a minimum gap (used for stacked labels)."""
    order = np.argsort(y)
    pos = np.array(y, float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    if hi is not None and len(pos) and pos[-1] > hi:
        pos -= pos[-1] - hi
        for i in range(len(pos) - 2, -1, -1):
            pos[i] = min(pos[i], pos[i + 1] - min_gap)
    if lo is not None and len(pos) and pos[0] < lo:
        pos += lo - pos[0]
    out = np.empty_like(pos)
    out[order] = pos
    return out


def _place_labels(ax, xs, ys, texts, fontsize=6.2, marker_px=4.0, cands=None, color=INK):
    """Greedy label placement in display space: for each point, the first candidate offset (in points)
    whose text box overlaps neither an earlier label nor any marker is used; a leader line is drawn when
    the label is not adjacent to its point."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tr = ax.transData
    pts = tr.transform(np.column_stack([xs, ys]))
    dpi = fig.dpi / 72.0
    cands = cands or [(4, 0), (-4, 0), (4, 6), (4, -6), (-4, 6), (-4, -6), (10, 10), (10, -10), (-10, 10),
                      (-10, -10), (0, 9), (0, -9), (16, 0), (-16, 0), (14, 14), (14, -14), (-14, 14), (-14, -14),
                      (22, 4), (-22, 4), (22, -8), (-22, -8)]
    m = marker_px * dpi
    boxes = [(px - m, py - m, px + m, py + m) for px, py in pts]
    placed = []
    for (px, py), (x, y), t in zip(pts, zip(xs, ys), texts):
        probe = ax.text(0, 0, t, fontsize=fontsize)
        bb = probe.get_window_extent(renderer=rend)
        probe.remove()
        w, h = bb.width, bb.height
        best = None
        for dx, dy in cands:
            ox, oy = dx * dpi, dy * dpi
            x0 = px + ox if dx >= 0 else px + ox - w
            if dx == 0:
                x0 = px - w / 2
            y0 = py + oy - h / 2
            box = (x0, y0, x0 + w, y0 + h)
            hit = any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in boxes + placed)
            if not hit:
                best = (dx, dy, box)
                break
        if best is None:
            dx, dy = cands[0]
            best = (dx, dy, (px, py, px + w, py + h))
        dx, dy, box = best
        placed.append(box)
        far = abs(dx) > 6 or abs(dy) > 7
        ax.annotate(t, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize, color=color,
                    ha="left" if dx > 0 else ("right" if dx < 0 else "center"), va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.35, color=MUTED, shrinkA=0, shrinkB=1.5) if far else None)


# ============================================================================ prepare
KEYS = ["state_id", "learner", "budget", "start"]
COLS = KEYS + ["strategy", "round", "auprc", "n_rare_labeled", "dataset", "ir", "backbone", "positive_class",
               "ir_target"]


def _final_paired(path, budget="main"):
    """Final round of every trajectory, paired with Margin in the same state (chunked, low memory).
    Returns (all rows of the budget with the columns needed for yields, paired final rows)."""
    head = pd.read_csv(path, nrows=0).columns
    use = [c for c in COLS if c in head]
    parts, last = [], {}
    for ch in pd.read_csv(path, usecols=use, chunksize=400_000, low_memory=False):
        if "budget" in ch:
            ch = ch[ch.budget.astype(str) == budget]
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    for c in ("learner", "strategy", "budget", "start", "state_id"):
        if c in df:
            df[c] = df[c].astype(str)
    key = [k for k in KEYS if k in df] + ["strategy"]
    df = df.drop_duplicates(key + ["round"])
    fin = df[df["round"] == df.groupby(key)["round"].transform("max")]
    k2 = [k for k in KEYS if k in fin]
    ref = fin[fin.strategy == "margin"][k2 + ["auprc", "n_rare_labeled"]].rename(
        columns={"auprc": "ref_auprc", "n_rare_labeled": "ref_n_rare_labeled"})
    p = fin.merge(ref, on=k2, how="inner")
    p["delta"] = p.auprc - p.ref_auprc
    return df, p


def _per_strategy(p, unit, cluster_fn=None):
    rows = []
    u = p.groupby(["learner", "strategy", unit]).agg(delta=("delta", "mean"), rare=("n_rare_labeled", "mean"),
                                                    ref_rare=("ref_n_rare_labeled", "mean"),
                                                    ref=("ref_auprc", "mean")).reset_index()
    for (lr, s), g in u.groupby(["learner", "strategy"]):
        if s == "margin":
            continue
        cl = g[unit].map(cluster_fn) if cluster_fn else g[unit]
        m, lo, hi = _cboot(g.delta.to_numpy(), cl.to_numpy())
        rows.append(dict(learner=str(lr), strategy=str(s), mean=m, lo=lo, hi=hi, rare=g.rare.mean(),
                         margin_rare=g.ref_rare.mean(), margin_auprc=g.ref.mean()))
    return pd.DataFrame(rows)


def _yield(df):
    g = df.groupby(["learner", "state_id", "strategy"]).n_rare_labeled
    acq = (g.max() - g.min()).unstack()
    return {str(lr): float(a["margin"].sum() / a["random"].sum()) for lr, a in acq.groupby(level=0)}


def prepare(campaign, lbfgs, revision, oracle, out, log=print):
    from rare_al.analysis.oracle import label_only_states
    from rare_al.analysis.revision import source_group
    campaign, lbfgs, revision, oracle = map(Path, (campaign, lbfgs, revision, oracle))
    fd = Path(out) / "figure_data"
    fd.mkdir(parents=True, exist_ok=True)
    stats = lbfgs / "assets" / "stats"
    for f in ("synthetic_risk_profile", "c1_decomposition", "real_dataset_unit", "domain_risk", "c5"):
        if (stats / f"{f}.csv").exists():
            shutil.copy2(stats / f"{f}.csv", fd / f"{f}.csv")
            log(f"[figdata] {f}.csv <- {stats}")
    rs = revision / "stats" if (revision / "stats").exists() else revision
    for f in ("rev_query_vs_campaign", "rev_validation_real_cells_published", "rev_validation_synthetic_cells",
              "rev_excess_tail"):
        src = rs / f"{f}.csv"
        if src.exists():
            shutil.copy2(src, fd / f"{f}.csv")
            log(f"[figdata] {f}.csv <- {rs}")
    # tail probability by imbalance ratio (synthetic grid)
    syn = lbfgs / "synthetic" / "trajectories.csv"
    if syn.exists():
        _, p = _final_paired(syn)
        rows = []
        for (lr, s, ir), g in p.groupby(["learner", "strategy", "ir"]):
            v = (g.delta.to_numpy() < -TAU).astype(float)
            m, lo, hi = _cboot(v, np.arange(len(v)), n=1000, seed=3)
            rows.append(dict(learner=str(lr), strategy=str(s), ir=float(ir), p=100 * m, lo=100 * lo, hi=100 * hi))
        pd.DataFrame(rows).to_csv(fd / "tail_by_ir.csv", index=False)
        log("[figdata] tail_by_ir.csv")
        gap, pairs = mechanism_data(p)
        gap.to_csv(fd / "mechanism_gap.csv", index=False)
        pairs.to_csv(fd / "mechanism_pairs.csv", index=False)
        log("[figdata] mechanism_gap.csv + mechanism_pairs.csv")
    # configuration effect: LR penalized (campaign) vs LR unpenalized, RF, GBDT (swapped tree)
    rows, yl = [], []
    for bench, unit, cl in (("real", "dataset", source_group), ("image", "dataset", None)):
        pen, new = campaign / bench / "trajectories.csv", lbfgs / bench / "trajectories.csv"
        if not (pen.exists() and new.exists()):
            continue
        dpen, ppen = _final_paired(pen)
        ppen = ppen[ppen.learner == "logreg"]
        dnew, pnew = _final_paired(new)
        a = _per_strategy(ppen, unit, cl).assign(config="lr_pen")
        b = _per_strategy(pnew, unit, cl).rename(columns={"learner": "config"})
        rows.append(pd.concat([a.drop(columns="learner"), b]).assign(bench=bench))
        ypen = _yield(dpen[dpen.learner == "logreg"])
        for k, v in ypen.items():
            yl.append(dict(bench=bench, config="lr_pen", yield_=v))
        for k, v in _yield(dnew).items():
            yl.append(dict(bench=bench, config=k, yield_=v))
        if bench == "image":
            pi = pnew[pnew.learner == "logreg"].copy()
            pi["task"] = pi.dataset.astype(str) + "|" + pi.positive_class.astype(str) + "|" + pi.ir_target.astype(str)
            ib = []
            for bb, g in pi.groupby("backbone"):
                t = _per_strategy(g, "task")
                ib.append(t.assign(backbone=bb))
            pd.concat(ib).to_csv(fd / "image_backbones.csv", index=False)
            log("[figdata] image_backbones.csv")
    if rows:
        pd.concat(rows).to_csv(fd / "config_effect.csv", index=False)
        pd.DataFrame(yl).rename(columns={"yield_": "yield"}).to_csv(fd / "config_yield.csv", index=False)
        log("[figdata] config_effect.csv + config_yield.csv")
    # discovery opportunity by horizon (unpenalized LR), base state as unit
    pts = oracle / "oracle_points.csv"
    if pts.exists():
        P = pd.read_csv(pts)
        orows, drows = [], []
        for lr in sorted(P.learner.unique()):
            st = {}
            for k in (1, 3, 5):
                s = label_only_states(P[P.learner == lr].drop(columns="learner"), k)
                s["base"] = s.state_id.astype(str).str.rsplit("|", n=1).str[0]
                st[k] = s.groupby("base").delta.mean()
                m, lo, hi = _cboot(st[k].to_numpy(), st[k].index.to_numpy(), seed=5)
                orows.append(dict(learner=lr, k=k, mean=m, lo=lo, hi=hi))
            for a, b in ((1, 3), (1, 5), (3, 5)):
                d = (st[b] - st[a]).dropna()
                m, lo, hi = _cboot(d.to_numpy(), d.index.to_numpy(), seed=6)
                drows.append(dict(learner=lr, k_from=a, k_to=b, diff=m, lo=lo, hi=hi))
        pd.DataFrame(orows).to_csv(fd / "opportunity_by_k.csv", index=False)
        pd.DataFrame(drows).to_csv(fd / "opportunity_diffs.csv", index=False)
        log("[figdata] opportunity_by_k.csv + opportunity_diffs.csv")
    # validation by learner and tolerance
    vb = []
    for bench, f in (("real", "rev_validation_real_cells_published"), ("synthetic", "rev_validation_synthetic_cells")):
        if (fd / f"{f}.csv").exists():
            vb.append(validation_by_learner(pd.read_csv(fd / f"{f}.csv")).assign(bench=bench))
    if vb:
        pd.concat(vb).to_csv(fd / "validation_by_learner.csv", index=False)
        log("[figdata] validation_by_learner.csv")
    return fd


GAP_EDGES = (-np.inf, -10.5, -4.5, -0.5, 0.5, 4.5, np.inf)
GAP_LABELS = ("\u2264\u221211", "\u221210 to \u22125", "\u22124 to \u22121", "0", "+1 to +4", "\u2265+5")


def mechanism_data(p, reference="margin", rerun="margin_rtb", exclude=("entropy", "least_confident"),
                   total=-0.9, min_n=30):
    """Discovery mechanism of the lower tail (synthetic grid, final round).

    ``gap``: P(delta < -TAU) by rare-label gap (rare labels acquired by the policy minus those acquired by the
    anchor in the same state), for the other policies pooled and for the anchor's tie-breaking rerun, with a
    state-cluster bootstrap interval; bins with fewer than ``min_n`` trajectories are dropped.
    ``pairs``: rare labels of policy and anchor for every tie-breaking rerun and every near-total loss
    (delta < ``total``)."""
    q = p[(p.strategy != reference) & ~p.strategy.isin(set(exclude))].copy()
    q["group"] = np.where(q.strategy == rerun, "rerun", "policies")
    q["gap"] = q.n_rare_labeled - q.ref_n_rare_labeled
    q["gap_bin"] = pd.cut(q.gap, list(GAP_EDGES), labels=list(GAP_LABELS))
    rows = []
    for (lr, g, b), h in q.groupby(["learner", "group", "gap_bin"], observed=True):
        if len(h) < min_n:
            continue
        v = (h.delta.to_numpy() < -TAU).astype(float)
        m, lo, hi = _cboot(v, h.state_id.to_numpy(), n=1000, seed=5)
        rows.append(dict(learner=str(lr), group=g, gap_bin=str(b), order=GAP_LABELS.index(str(b)),
                         p=100 * m, lo=100 * lo, hi=100 * hi, n=len(h)))
    gap = pd.DataFrame(rows)
    sel = (q.group == "rerun") | (q.delta < total)
    pairs = q.loc[sel, ["learner", "strategy", "state_id", "n_rare_labeled", "ref_n_rare_labeled", "delta"]].copy()
    pairs["kind"] = np.where(pairs.strategy == rerun, "rerun", "near_total")
    return gap, pairs.rename(columns={"n_rare_labeled": "rare", "ref_n_rare_labeled": "ref_rare"})


def validation_by_learner(c, alphas=("0", "0.01", "0.02", "0.05", "0.1")):
    rows = []
    for a in alphas:
        if f"loss_risk_{a}" not in c:
            continue
        c = c.assign(dl=100 * (c[f"loss_risk_{a}"] - c.loss_mean_rank), dg=c[f"gain_risk_{a}"] - c.gain_mean_rank,
                     ch=(c[f"pick_risk_{a}"] != c.pick_mean_rank).astype(float))
        for lr, g in list(c.groupby("learner")) + [("pooled", c)]:
            m, lo, hi = _cboot(g.dl.to_numpy(), g.cluster.to_numpy(), seed=11)
            mg, lg, hg = _cboot(g.dg.to_numpy(), g.cluster.to_numpy(), seed=11)
            rows.append(dict(alpha=float(a), learner=lr, dloss=m, dloss_lo=lo, dloss_hi=hi, dgain=mg,
                             dgain_lo=lg, dgain_hi=hg, changed=g.ch.mean(), loss_mean_rank=100 * g.loss_mean_rank.mean(),
                             loss_risk=100 * g[f"loss_risk_{a}"].mean(), n=len(g)))
    return pd.DataFrame(rows)


# ============================================================================ plot
def fig_headline(rp, path, xmin=-0.05, ymin=0.06, ymax=45):
    rp = rp.copy()
    rp["C"] = 100 * rp.p_lt_tau
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.85), sharey=True)
    lab = {"logreg": ["committee", "direct", "dcf", "typiclust"],
           "rf": ["dcf", "uherding", "kmeans_diversity", "class_balanced", "typiclust"],
           "gbdt": ["tailor", "kmeans_diversity", "class_balanced", "typiclust"]}
    for ax, lr in zip(axs, ("logreg", "rf", "gbdt")):
        d = rp[(rp.learner == lr) & ~rp.strategy.isin(["margin", "entropy", "least_confident", "margin_rtb"])].copy()
        fl = rp[(rp.learner == lr) & (rp.strategy == "margin_rtb")]
        floor = float(fl.C.iloc[0]) if len(fl) else 0.0
        ax.axvspan(-0.0055, 0.0055, color=BAND, zorder=0, lw=0)
        if floor > 0:
            ax.axhspan(ymin, floor, color="#dcdad3", zorder=0.5, lw=0)
            ax.axhline(floor, color=MUTED, lw=0.8, ls="--", zorder=1)
            inside = floor / ymin > 3
            ax.text(xmin + 0.001, ymin * 1.25 if inside else floor * 1.15,
                    f"anchor noise floor {floor:.1f}%" if floor >= 1 else f"anchor noise floor {floor:.2f}%",
                    color=INK if inside else MUTED, fontsize=6.5, va="bottom")
        else:
            ax.text(xmin + 0.001, ymin * 1.2, "anchor noise floor 0\n(deterministic learner)", color=MUTED,
                    fontsize=6.5, va="bottom")
        d["off"] = d["mean"] < xmin
        d["xp"] = d["mean"].clip(lower=xmin + 0.0015)
        d["yp"] = d.C.clip(lower=ymin * 1.3)
        near = d["mean"].abs() < 0.0055
        col = LCOL[lr]
        for sub, face, z in ((d[~near], "#bdbbb4", 2), (d[near], col, 3)):
            pub = ~sub.strategy.isin(CONTROLS)
            for m in ("o", "<"):
                sel = (sub.off if m == "<" else ~sub.off)
                ax.scatter(sub.xp[pub & sel], sub.yp[pub & sel], s=20, marker=m, c=face, edgecolors="white",
                           linewidths=0.5, zorder=z)
                ax.scatter(sub.xp[~pub & sel], sub.yp[~pub & sel], s=18, marker=m, facecolors="white",
                           edgecolors=face, linewidths=1.0, zorder=z)
        # labels: off-scale and far-left points to the right of the point; near-neutral ones stacked on the left
        far = d[d.strategy.isin(lab[lr]) & (d["mean"] < -0.015)]
        for r in far.itertuples():
            txt = f"{short(r.strategy)} ({r.mean:+.3f})".replace("-", "\u2212") if r.off else short(r.strategy)
            ax.annotate(txt, (r.xp, r.yp), (r.xp + 0.004, r.yp * (0.55 if r.yp > 20 else 1.9)), fontsize=6.5,
                        color=INK, va="center",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.4, shrinkA=1, shrinkB=2))
        nn = d[d.strategy.isin(lab[lr]) & (d["mean"] >= -0.015)].sort_values("yp", ascending=False)
        if len(nn):
            # near-neutral labels go to the free area at the left, below every point lying left of the band,
            # stacked downwards in the order of their points (no label over a data point or a leader line)
            left = d[(d.xp < -0.0055) & (d.xp > -0.03)]
            top = float(left.yp.min()) / 1.35 if len(left) else ymax / 2
            ys, prev = [], None
            for r in nn.itertuples():
                y = min(r.yp, top) if prev is None else min(r.yp, prev / 1.55)
                ys.append(max(y, ymin * 2.2))
                prev = ys[-1]
            for (r, y) in zip(nn.itertuples(), ys):
                ax.annotate(short(r.strategy), (r.xp, r.yp), (-0.0125, y), fontsize=6.5, color=INK, ha="right",
                            va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.4, shrinkA=1, shrinkB=2))
        nd = d[d["mean"].abs() < 0.0055]
        ttl = {"logreg": "LR", "rf": "Random forest", "gbdt": "Gradient-boosted trees"}[lr]
        ax.set_title(f"{ttl}\nnear-neutral: {nd.C.min():.1f}\u2013{nd.C.max():.1f}%", loc="left",
                     color=INK, fontsize=7.5, linespacing=1.3)
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, 0.012)
        ax.yaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10, 30]))
        ax.yaxis.set_major_formatter(FixedFormatter(["0.1", "0.3", "1", "3", "10", "30"]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xticks([-0.04, -0.02, 0.0])
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    axs[1].set_xlabel("Mean paired AUPRC gain vs. Margin")
    axs[0].set_ylabel(r"$P(\Delta<-0.10)$ (%), log scale")
    h = [Line2D([], [], marker="o", ls="", mfc=MUTED, mec="white", ms=5, label="published method"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, mew=1.0, ms=5, label="control, probe or RCA"),
         Line2D([], [], marker="<", ls="", mfc=MUTED, mec="white", ms=5, label="mean below $-0.05$"),
         Line2D([], [], marker="s", ls="", mfc=BAND, mec=BAND, ms=7, label=r"near-neutral, $|$mean$|\leq0.005$"),
         Line2D([], [], color=MUTED, ls="--", lw=0.8, label="anchor noise floor (tie-breaking rerun)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.4)
    _save(fig, path)


def fig_c1(dec, path):
    geos = ["central", "compact", "diffuse", "multimodal", "overlap", "peripheral"]
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.15), sharey=True, sharex=True)
    for ax, lr in zip(axs, ("logreg", "rf", "gbdt")):
        g = dec[dec.learner == lr].set_index("geometry").reindex(geos)
        y = np.arange(len(geos))[::-1]
        for yi, (geo, r) in zip(y, g.iterrows()):
            deg = bool(r.scale_degenerate)
            ax.plot([max(r.sd_growth_ci_low, 0.05), r.sd_growth_ci_high], [yi, yi], color=LCOL[lr], lw=1.0)
            ax.scatter([r.sd_growth], [yi], s=22, marker=LMARK[lr], facecolors="white" if deg else LCOL[lr],
                       edgecolors=LCOL[lr], linewidths=1.0, zorder=3)
        ax.axvline(1, color=INK, lw=0.6, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(0.07, 80)
        ax.set_xticks([0.1, 1, 10])
        ax.set_xticklabels(["0.1", "1", "10"])
        ax.set_title({"logreg": "LR", "rf": "RF", "gbdt": "GBDT"}[lr], loc="left", color=INK)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_yticks(np.arange(len(geos))[::-1])
        ax.set_yticklabels(geos)
        ax.tick_params(axis="y", length=0, labelcolor=INK)
    axs[1].set_xlabel("Growth of the SD of absolute regret, IR = 100 vs. IR = 1 (log scale, 95% CI)")
    h = [Line2D([], [], marker="o", ls="", mfc=MUTED, mec=MUTED, ms=5, label="amplification"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, mew=1.0, ms=5,
                label="scale-degenerate (the reference AUPRC collapses)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.6)
    _save(fig, path)


def fig_config(ce, cy, path, strategies=("direct", "most_likely_positive", "class_balanced", "tailor", "galaxy", "coreset")):
    cfg = [("lr_pen", "white"), ("logreg", None), ("rf", None), ("gbdt", None)]
    fig = plt.figure(figsize=(W, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.62], wspace=0.12)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    off = np.linspace(-0.27, 0.27, 4)
    for k, (ax, bench) in enumerate(zip(axs[:2], ("real", "image"))):
        for i, s in enumerate(strategies):
            y0 = len(strategies) - 1 - i
            if i % 2 == 0:
                ax.axhspan(y0 - 0.5, y0 + 0.5, color="#f6f5f1", lw=0, zorder=0)
            for j, (c, face) in enumerate(cfg):
                r = ce[(ce.bench == bench) & (ce.config == c) & (ce.strategy == s)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = y0 - off[j]
                ax.plot([r.lo, r.hi], [y, y], color=LCOL[c], lw=1.0, zorder=2, solid_capstyle="round")
                ax.scatter([r["mean"]], [y], marker=LMARK[c], s=18, facecolors=face or LCOL[c], edgecolors=LCOL[c],
                           linewidths=1.0, zorder=3)
        ax.axvline(0, color=INK, lw=0.6, zorder=1)
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels([short(s) for s in strategies][::-1] if k == 0 else [])
        ax.tick_params(axis="y", length=0, labelcolor=INK)
        ax.set_ylim(-0.6, len(strategies) - 0.4)
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.set_xlabel("Gain vs. Margin [95% CI]")
        ax.set_title({"real": "A. Tabular (42 datasets)", "image": "B. Images (15 datasets)"}[bench], loc="left", color=INK)
        for sp in ("left",):
            ax.spines[sp].set_visible(False)
    axs[0].set_xlim(-0.08, 0.10)
    axs[0].set_xticks([-0.05, 0, 0.05, 0.1])
    axs[1].set_xlim(-0.26, 0.30)
    axs[1].set_xticks([-0.2, 0, 0.2])
    ax = axs[2]
    for i, bench in enumerate(("real", "image")):
        y0 = 1 - i
        for j, (c, face) in enumerate(cfg):
            r = cy[(cy.bench == bench) & (cy.config == c)]
            if r.empty:
                continue
            v = float(r["yield"].iloc[0])
            y = y0 - off[j] * 0.9
            ax.plot([1, v], [y, y], color=LCOL[c], lw=0.8, zorder=2)
            ax.scatter([v], [y], marker=LMARK[c], s=18, facecolors=face or LCOL[c], edgecolors=LCOL[c], linewidths=1.0, zorder=3)
            ax.text(v * 1.3, y, f"{v:.1f}" if v < 10 else f"{v:.0f}", va="center", fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 70)
    ax.set_xticks([1, 3, 10, 30])
    ax.set_xticklabels(["1", "3", "10", "30"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([])
    ax.text(0.85, 1.42, "Tabular", fontsize=7, color=INK, fontweight="bold", va="center")
    ax.text(0.85, 0.42, "Images", fontsize=7, color=INK, fontweight="bold", va="center")
    ax.set_ylim(-0.6, 1.55)
    ax.axvline(1, color=INK, lw=0.6)
    ax.set_xlabel("Margin's rare yield\n(× random sampling)")
    ax.set_title("C. Anchor yield", loc="left", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.spines["left"].set_visible(False)
    h = [Line2D([], [], marker=LMARK[c], ls="-", lw=1.0, color=LCOL[c], mfc=face or LCOL[c], mec=LCOL[c], ms=4.5,
                label={"lr_pen": "LR, penalized intercept", "logreg": "LR, unpenalized intercept",
                       "rf": "Random forest", "gbdt": "Gradient-boosted trees"}[c]) for c, face in cfg]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.0, handletextpad=0.4)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.91, bottom=0.27)
    _save(fig, path)


def fig_forest(rd, path):
    learners = ["logreg", "rf", "gbdt"]
    strategies = list(rd[rd.learner == "logreg"].sort_values("mean").strategy)
    strategies = [s for s in strategies if s not in ("margin", "entropy", "least_confident")]
    fig, axes = plt.subplots(1, 3, figsize=(W, 0.2 * len(strategies) + 1.3), sharey=True)
    y = np.arange(len(strategies))
    for ax, lr in zip(axes, learners):
        g = rd[rd.learner == lr].set_index("strategy").reindex(strategies)
        for i, s in enumerate(strategies):
            r = g.loc[s]
            c = FAMILY_COLOR.get(family(s), "k")
            sig = (r.ci_low > 0) or (r.ci_high < 0)
            ax.plot([r.ci_low, r.ci_high], [y[i], y[i]], color=c, lw=1.0)
            ax.scatter([r["mean"]], [y[i]], s=16, color=c if sig else "white", edgecolors=c, linewidths=1.0, zorder=3)
        ax.axvline(0, color=INK, lw=0.6, ls="--")
        ax.set_title(LNAME[lr], color=INK)
        ax.set_xlabel("Paired gain [95% CI]")
        ax.grid(axis="x", color=GRID, lw=0.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([display(s) for s in strategies])
    axes[0].tick_params(axis="y", length=0, labelcolor=INK)
    fams = []
    for s in strategies[::-1]:
        f = family(s)
        if f not in fams:
            fams.append(f)
    h = [Line2D([], [], marker="o", ls="", color=FAMILY_COLOR.get(f, "k"), ms=4.5, label=f.replace("_", " ")) for f in fams]
    h += [Line2D([], [], marker="o", ls="", color=INK, ms=4.5, label="interval excludes 0"),
          Line2D([], [], marker="o", ls="", color=INK, mfc="white", ms=4.5, label="interval includes 0")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, -0.005),
               columnspacing=0.9, handletextpad=0.3)
    nrow = int(np.ceil(len(h) / 4))
    hgt = fig.get_size_inches()[1]
    fig.tight_layout(rect=(0, (0.135 * nrow + 0.03) / hgt, 1, 1), w_pad=0.5)
    _save(fig, path)


def _lr_name(df, prefer=("logreg_lbfgs", "logreg")):
    have = set(df.learner) if "learner" in df else set()
    return next((n for n in prefer if n in have), None)


def fig_rq3(opp, diffs, qvc, path):
    from scipy import stats
    o = opp[opp.learner == _lr_name(opp)].sort_values("k")
    q = qvc[qvc.learner == _lr_name(qvc)] if "learner" in qvc else qvc
    q = q[(q.k == 5) & (~q.identical_to_margin.astype(bool))].copy().sort_values("gain")
    fig = plt.figure(figsize=(W, 2.9))
    ax = fig.add_axes([0.115, 0.16, 0.15, 0.72])
    ax.axhline(0, color=INK, lw=0.6)
    ax.vlines(o.k, o.lo, o.hi, color=LCOL["logreg"], lw=1.2)
    ax.plot(o.k, o["mean"], "-o", color=LCOL["logreg"], ms=4.5, mec="white", lw=1.0)
    ax.text(0.97, 0.97, "\n".join(f"$k$={int(r.k)}: {r.mean:+.4f}" for r in o.itertuples()), transform=ax.transAxes,
            fontsize=6.2, va="top", ha="right", color=INK)
    ax.set_xticks([1, 3, 5])
    ax.set_xlim(0.4, 5.6)
    ax.set_xlabel("Horizon $k$ (rounds)")
    ax.set_ylabel("Opportunity (AUPRC)")
    ax.set_title("A. Opportunity", loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax = fig.add_axes([0.445, 0.16, 0.21, 0.72])
    y = np.arange(len(q))
    ax.axvline(0, color=INK, lw=0.6)
    O5 = float(q.O.iloc[0])
    ax.axvline(O5, color=LCOL["logreg"], lw=0.8, ls="--")
    ax.text(O5, len(q) - 0.3, " level ($k$=5)", color=LCOL["logreg"], fontsize=6.3, va="bottom")
    ax.hlines(y, q.gain_ci_low, q.gain_ci_high, color=MUTED, lw=1.0)
    ax.scatter(q.gain, y, s=12, color=INK, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([short(p) for p in q.policy], fontsize=6.3, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(q) + 0.9)
    ax.set_xticks([-0.002, 0, 0.002, 0.004])
    ax.set_xticklabels(["\u2212.002", "0", ".002", ".004"])
    ax.set_xlabel("Query-level gain [95% CI]")
    ax.set_title(r"B. Holm $p\geq0.45$ for all", loc="left", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.spines["left"].set_visible(False)
    ax = fig.add_axes([0.75, 0.16, 0.19, 0.72])
    rho, pv = stats.spearmanr(q.gain, q.camp_mean)
    ax.axhline(0, color=INK, lw=0.6)
    ax.axvline(0, color=INK, lw=0.6)
    ax.scatter(q.gain, q.camp_mean, s=14, color=LCOL["logreg"], edgecolors="white", lw=0.5, zorder=3)
    offs = {"rca": (-2, 6, "right"), "galaxy": (0, -11, "center"),
            "similar_flqmi": (-4, 4, "right"), "anchoral": (4, -1, "left"), "typiclust": (5, -2, "left")}
    names = {"rca": "RCA, Committee (LR)"}
    for r in q.itertuples():
        if r.policy in offs:
            dx, dy, ha = offs[r.policy]
            ax.annotate(names.get(r.policy, short(r.policy)), (r.gain, r.camp_mean), (dx, dy),
                        textcoords="offset points", fontsize=6.3, color=INK, ha=ha)
    ax.set_xlabel("Query-level gain ($k$=5)")
    ax.set_ylabel("Campaign gain (same states)")
    ax.set_title("C. Query vs. campaign", loc="left", color=INK)
    ax.set_xticks([0, 0.001, 0.002])
    ax.set_xticklabels(["0", ".001", ".002"])
    ax.text(0.97, 0.12, f"Spearman ρ = {rho:.2f}\n(p = {pv:.2f})", transform=ax.transAxes, fontsize=6.5,
            color=INK, ha="right")
    ax.grid(color=GRID, lw=0.5)
    _save(fig, path)


def fig_validation(vb, path, alphas=(0.0, 0.01, 0.02, 0.05, 0.1)):
    ser = ["logreg", "rf", "gbdt", "pooled"]
    vb = vb[vb.alpha.isin(alphas)]
    fig, axs = plt.subplots(2, 2, figsize=(W, 3.6), sharex=True)
    off = np.linspace(-0.27, 0.27, 4)
    for j, bench in enumerate(("real", "synthetic")):
        for i, (m, lo, hi, lab) in enumerate((("dloss", "dloss_lo", "dloss_hi", "Δ loss rate (pts)"),
                                              ("dgain", "dgain_lo", "dgain_hi", "Δ mean gain"))):
            ax = axs[i, j]
            ax.axhline(0, color=INK, lw=0.6, zorder=1)
            for k, lr in enumerate(ser):
                d = vb[(vb.bench == bench) & (vb.learner == lr)].sort_values("alpha")
                if d.empty:
                    continue
                x = np.arange(len(d)) + off[k]
                ax.vlines(x, d[lo], d[hi], color=LCOL[lr], lw=1.0, zorder=2)
                ax.scatter(x, d[m], marker=LMARK[lr], s=14, color=LCOL[lr], edgecolors="white", linewidths=0.4, zorder=3)
            if i == 0:
                d = vb[(vb.bench == bench) & (vb.learner == "pooled")].sort_values("alpha")
                top = vb[vb.bench == bench][hi].max()
                for x, v in zip(range(len(d)), d.changed):
                    ax.text(x, top + 0.55, f"{100 * v:.0f}%", ha="center", fontsize=6.3, color=MUTED)
                ax.text(-0.5, top + 1.35, "picks changed:", fontsize=6.3, color=MUTED)
                ax.set_ylim(top=top + 2.1)
                ax.set_title({"real": "Real data, held-out domains", "synthetic": "Synthetic grid, held-out geometries"}[bench],
                             loc="left", color=INK)
            if j == 0:
                ax.set_ylabel(lab)
            ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
            if i == 1:
                ax.set_xticks(range(len(alphas)))
                ax.set_xticklabels([f"{a:g}" for a in alphas])
                ax.set_xlabel(r"Tolerance $\alpha$ (training-fold loss rate)")
    h = [Line2D([], [], marker=LMARK[lr], color=LCOL[lr], ls="", ms=4.5, label=LNAME[lr]) for lr in ser]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.5, w_pad=1.0)
    _save(fig, path)


def fig_mean_tail(rp, path, learner="rf"):
    """Supplementary: all policies under one learner, with a zoom on the near-neutral ones (numbered)."""
    r = rp[(rp.learner == learner) & ~rp.strategy.isin(["margin", "entropy", "least_confident"])].copy()
    r["C"] = 100 * r.p_lt_tau
    fig = plt.figure(figsize=(W, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 0.5], hspace=0.5, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    for t in r.itertuples():
        c = FAMILY_COLOR.get(family(t.strategy), "k")
        ctl = t.strategy in CONTROLS
        ax.scatter(t.mean, t.C, s=18, facecolors="white" if ctl else c, edgecolors=c, linewidths=1.0, zorder=3)
    big = r[(r.C > 9) | (r["mean"] < -0.03)].sort_values("C", ascending=False)
    ax.set_ylim(-1.5, r.C.max() + 4)
    ax.set_xlim(r["mean"].min() - 0.008, 0.012)
    _place_labels(ax, big["mean"].to_numpy(), big.C.to_numpy(), [short(x) for x in big.strategy], fontsize=6.3,
                  cands=[(6, 0), (6, 8), (6, -8), (14, 12), (14, -12), (24, 0), (24, 16), (24, -16), (34, 6), (34, -10)])
    ax.set_xticks([-0.12, -0.08, -0.04, 0])
    ax.set_xticklabels(["\u22120.12", "\u22120.08", "\u22120.04", "0"])
    zx = (-0.022, 0.009)
    ax.add_patch(plt.Rectangle((zx[0], -0.5), zx[1] - zx[0], 9, fill=False, ls=":", lw=0.8, ec=MUTED))
    ax.axvline(0, color=INK, lw=0.6, ls="--")
    ax.set_xlabel("Mean paired gain vs. Margin")
    ax.set_ylabel(r"$P(\Delta<-0.10)$ (% of states)")
    ax.set_title("A. All policies", loc="left", color=INK)
    ax.grid(color=GRID, lw=0.5)
    ax = fig.add_subplot(gs[0, 1])
    z = r[(r["mean"] >= zx[0]) & (r["mean"] <= zx[1]) & (r.C <= 8.5)].sort_values("C", ascending=False).reset_index(drop=True)
    for t in z.itertuples():
        c = FAMILY_COLOR.get(family(t.strategy), "k")
        ctl = t.strategy in CONTROLS
        ax.scatter(t.mean, t.C, s=18, facecolors="white" if ctl else c, edgecolors=c, linewidths=1.0, zorder=3)
    ax.axvline(0, color=INK, lw=0.6, ls="--")
    ax.set_xlim(zx[0] - 0.002, zx[1] + 0.002)
    ax.set_ylim(-0.4, 8.6)
    _place_labels(ax, z["mean"].to_numpy(), z.C.to_numpy(), [str(i + 1) for i in range(len(z))])
    ax.set_xlabel("Mean paired gain vs. Margin")
    ax.set_title("B. Zoom on the near-neutral policies", loc="left", color=INK)
    ax.grid(color=GRID, lw=0.5)
    axl = fig.add_subplot(gs[1, :])
    axl.axis("off")
    items = [f"{i + 1}. {display(s)}" + ("$^{c}$" if s in CONTROLS else "") for i, s in enumerate(z.strategy)]
    ncol = 3
    per = int(np.ceil(len(items) / ncol))
    for c in range(ncol):
        axl.text(c / ncol, 1.0, "\n".join(items[c * per:(c + 1) * per]), fontsize=6.3, va="top", color=INK,
                 transform=axl.transAxes)
    fams = []
    for s in r.strategy:
        f = family(s)
        if f not in fams:
            fams.append(f)
    h = [Line2D([], [], marker="o", ls="", color=FAMILY_COLOR.get(f, "k"), ms=4.5, label=f.replace("_", " ")) for f in sorted(fams)]
    h.append(Line2D([], [], marker="o", ls="", color=MUTED, mfc="white", ms=4.5, label="open, $^{c}$: control / probe"))
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=6.3, bbox_to_anchor=(0.5, -0.01),
               columnspacing=0.8, handletextpad=0.3)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.11)
    _save(fig, path)


def fig_tail_vs_ir(tb, path, learner="rf",
                   strategies=("typiclust", "probcover", "random", "coreset", "dcf", "kmeans_diversity")):
    cols = ["#2a78d6", "#eb6834", "#8a8984", "#6250d6", "#1baf7a", "#e34948"]
    marks = "os^Dv<"
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    for s, c, m in zip(strategies, cols, marks):
        g = tb[(tb.learner == learner) & (tb.strategy == s)].sort_values("ir")
        if g.empty:
            continue
        ax.fill_between(g.ir, g.lo, g.hi, color=c, alpha=0.12, lw=0)
        ax.plot(g.ir, g.p, marker=m, ms=4, lw=1.2, color=c, label=display(s))
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "5", "10", "20", "50", "100"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Imbalance ratio (log scale)")
    ax.set_ylabel(r"$P(\Delta<-0.10)$ (%)")
    ax.set_title({"rf": "Random forest", "gbdt": "Gradient-boosted trees", "logreg": "LR"}[learner], loc="left", color=INK)
    ax.grid(color=GRID, lw=0.5)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    _save(fig, path)


def fig_mechanism(gap, pairs, path, learners=("rf", "gbdt", "logreg")):
    """A: material-loss probability by rare-label gap, policies pooled (solid) and the anchor's tie-breaking
    rerun (dashed). B: rare labels acquired by the anchor and by the policy, for the tie-breaking reruns (grey)
    and the near-total losses (coloured), tree ensembles."""
    fig = plt.figure(figsize=(W, 2.65))
    ax = fig.add_axes([0.085, 0.30, 0.40, 0.58])
    bx = fig.add_axes([0.60, 0.30, 0.38, 0.58])
    xs = np.arange(len(GAP_LABELS))
    off = {"rf": -0.12, "gbdt": 0.0, "logreg": 0.12}
    for lr in learners:
        g = gap[(gap.learner == lr) & (gap.group == "policies")].sort_values("order")
        if g.empty:
            continue
        x = g.order.to_numpy() + off[lr]
        ax.errorbar(x, g.p, yerr=[g.p - g.lo, g.hi - g.p], color=LCOL[lr], marker=LMARK[lr], ms=3.8, lw=1.2,
                    elinewidth=0.7, capsize=0, label=f"{LNAME[lr]}, other policies")
    for lr in learners:
        g = gap[(gap.learner == lr) & (gap.group == "rerun")].sort_values("order")
        if len(g) < 2:
            continue
        x = g.order.to_numpy() + off[lr]
        ax.errorbar(x, g.p, yerr=[g.p - g.lo, g.hi - g.p], color=LCOL[lr], marker=LMARK[lr], ms=3.8, lw=1.0,
                    ls=(0, (3, 2)), mfc="white", elinewidth=0.7, capsize=0,
                    label=f"{LNAME[lr]}, Margin tie-breaking rerun")
    ax.set_xticks(xs, GAP_LABELS, rotation=30, ha="right")
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Rare-label gap to Margin at the final budget")
    ax.set_ylabel(r"$P(\Delta<-0.10)$ (%)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_title("A  Losses follow the discovery gap", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper right", fontsize=6.2, handlelength=2.2, borderaxespad=0.2)
    rng = np.random.default_rng(11)

    def jit(v):
        return np.asarray(v, float) * np.exp(rng.uniform(-0.07, 0.07, len(v)))
    rr = pairs[(pairs.kind == "rerun") & pairs.learner.isin(["rf", "gbdt"])]
    lost = rr.delta < -TAU
    h1 = bx.scatter(jit(rr.ref_rare[~lost]), jit(rr.rare[~lost]), s=4, color="#c9c7c1", lw=0, zorder=1)
    h2 = bx.scatter(jit(rr.ref_rare[lost]), jit(rr.rare[lost]), s=9, facecolor="white", edgecolor=MUTED, lw=0.6,
                    zorder=2)
    hs = []
    for lr in ("rf", "gbdt"):
        nt = pairs[(pairs.kind == "near_total") & (pairs.learner == lr)]
        if nt.empty:
            continue
        hs.append((bx.scatter(jit(nt.ref_rare), jit(nt.rare), s=11, marker=LMARK[lr], color=LCOL[lr], lw=0,
                              alpha=0.85, zorder=3), LNAME[lr]))
    kw = dict(frameon=False, fontsize=6.2, title_fontsize=6.2, handletextpad=0.3, borderaxespad=0.2,
              markerscale=1.3, alignment="left")
    lg = bx.legend([h1, h2], ["loss \u2264 0.10", "loss > 0.10"], title="Margin, tie-breaking rerun",
                   loc="upper left", **kw)
    bx.add_artist(lg)
    if hs:
        bx.legend([h for h, _ in hs], [t for _, t in hs], title="Policy, loss > 0.9", loc="upper left",
                  bbox_to_anchor=(0.0, 0.70), **kw)
    lim = (0.8, max(200, float(pairs.ref_rare.max()) * 1.3))
    bx.plot(lim, lim, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=0)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlim(lim)
    bx.set_ylim(lim)
    ticks = [1, 2, 5, 10, 20, 50, 100]
    for a in (bx.xaxis, bx.yaxis):
        a.set_major_locator(FixedLocator(ticks))
        a.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        a.set_minor_locator(NullLocator())
    bx.set_xlabel("Rare labels acquired by Margin")
    bx.set_ylabel("Rare labels acquired by the policy")
    bx.grid(color=GRID, lw=0.5)
    bx.set_title("B  Near-total losses: no discovery", loc="left", color=INK)
    _save(fig, path)


def fig_heatmap(dr, path, alpha=0.10):
    dom_short = {"cybersecurity": "Cyber", "finance": "Finance", "industry": "Industry", "medicine": "Medicine",
                 "moderate_stratum": "Moderate", "moderate stratum": "Moderate"}
    doms = sorted(dr.domain.unique())
    learners = ["logreg", "rf", "gbdt"]
    mat = dr.pivot_table(index="strategy", columns=["learner", "domain"], values="mean_ds")
    tail = dr.pivot_table(index="strategy", columns=["learner", "domain"], values="p_lt_tau")
    mat = mat.drop(index=[s for s in ("margin", "entropy", "least_confident") if s in mat.index])
    order = mat["logreg"].mean(axis=1).sort_values(ascending=False).index
    mat, tail = mat.loc[order], tail.loc[order]
    v = float(np.nanmax(np.abs(mat.to_numpy())))
    fig, axes = plt.subplots(1, 3, figsize=(W, 0.19 * len(order) + 1.25), sharey=True)
    for ax, lr in zip(axes, learners):
        sub = mat[lr][doms]
        im = ax.imshow(sub.to_numpy(), cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(len(doms)))
        ax.set_xticklabels([dom_short.get(d, d) for d in doms], rotation=55, ha="right", fontsize=6.6)
        for i, s in enumerate(sub.index):
            for j, d in enumerate(doms):
                if tail[lr].loc[s, d] > alpha:
                    ax.text(j, i, "×", ha="center", va="center", fontsize=6.5, color=INK)
        ax.set_title(LNAME[lr], color=INK)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([display(s) for s in order], fontsize=6.6, color=INK)
    fig.subplots_adjust(left=0.27, right=0.86, top=0.95, bottom=0.12, wspace=0.08)
    cax = fig.add_axes([0.885, 0.35, 0.018, 0.4])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label("Mean paired gain vs. Margin", fontsize=7)
    _save(fig, path)


def fig_multistep(opp, diffs, path):
    lr = _lr_name(opp)
    o = opp[opp.learner == lr].sort_values("k")
    d = diffs[diffs.learner == lr]
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.axhline(0, color=INK, lw=0.6)
    ax.vlines(o.k, o.lo, o.hi, color=LCOL["logreg"], lw=1.2)
    ax.plot(o.k, o["mean"], "-o", color=LCOL["logreg"], ms=4.5, mec="white", lw=1.0)
    ax.set_xticks([1, 3, 5])
    ax.set_xlim(0.5, 5.5)
    ax.set_xlabel("Horizon $k$ (rounds)")
    ax.set_ylabel("Opportunity level (AUPRC)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    txt = "\n".join(f"$O_{{{int(r.k_to)}}}-O_{{{int(r.k_from)}}}$ = {r.diff:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]"
                    for r in d.itertuples())
    ax.text(0.98, 0.97, "paired, within base state:\n" + txt, transform=ax.transAxes, fontsize=6.2, va="top",
            ha="right", color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.6))
    fig.tight_layout()
    _save(fig, path)


def fig_c5(c5, path):
    names = {"all": "all", "geometry": "geometry", "uncertainty": "uncertainty", "ranking": "ranking",
             "state_progress": "state progress"}
    feats = [f for f in names if f in set(c5.features)]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.4), sharey=True)
    for ax, (model, title) in zip(axs, (("extratrees", "Extra trees"), ("logreg", "Logistic regression"))):
        g = c5[c5.model == model].set_index("features").reindex(feats)
        y = np.arange(len(feats))[::-1]
        ax.barh(y + 0.18, g.groupcv, 0.34, color=LCOL["logreg"], label="grouped CV")
        ax.barh(y - 0.18, g.loro, 0.34, color=LCOL["rf"], label="leave one geometry out")
        ax.axvline(0.5, color=INK, lw=0.6, ls="--")
        ax.set_xlim(0.45, 0.70)
        ax.set_yticks(y)
        ax.set_yticklabels([names[f] for f in feats], color=INK)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("AUROC")
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="x", color=GRID, lw=0.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.0)
    _save(fig, path)


def fig_image_backbones(ib, path):
    bb = [("dinov2_vits14", "DINOv2 ViT-S/14", LCOL["logreg"]), ("resnet18", "ResNet-18", LCOL["rf"])]
    order = (ib[ib.backbone == "dinov2_vits14"].sort_values("mean").strategy.tolist())
    order = [s for s in order if s not in ("entropy", "least_confident")]
    fig, axs = plt.subplots(1, 2, figsize=(W, 0.19 * len(order) + 1.2), sharey=True,
                            gridspec_kw=dict(width_ratios=[1.2, 1.0]))
    y = np.arange(len(order))
    for k, (b, name, c) in enumerate(bb):
        g = ib[ib.backbone == b].set_index("strategy").reindex(order)
        yy = y + (0.17 if k == 0 else -0.17)
        axs[0].hlines(yy, g.lo, g.hi, color=c, lw=1.0)
        axs[0].scatter(g["mean"], yy, s=12, color=c, zorder=3, label=name)
        axs[1].barh(yy, g.rare, 0.32, color=c, label=name)
        axs[1].axvline(float(g.margin_rare.mean()), color=c, lw=0.8, ls=":")
    axs[0].axvline(0, color=INK, lw=0.6, ls="--")
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([display(s) for s in order], color=INK)
    axs[0].tick_params(axis="y", length=0)
    axs[0].set_xlabel("Gain vs. Margin [95% CI]")
    axs[0].set_title("A. Gain under the linear head", loc="left", color=INK)
    axs[1].set_xlabel("Rare cases labeled")
    axs[1].set_title("B. Rare cases found", loc="left", color=INK)
    for ax in axs:
        ax.grid(axis="x", color=GRID, lw=0.5)
    axs[1].legend(handles=axs[1].get_legend_handles_labels()[0] + [Line2D([], [], color=MUTED, ls=":", lw=0.8)],
                  labels=axs[1].get_legend_handles_labels()[1] + ["Margin"], frameon=False, loc="lower right")
    fig.tight_layout(w_pad=0.8)
    _save(fig, path)


def plot(out, log=print):
    _style()
    out = Path(out)
    fd = out / "figure_data"

    def has(*names):
        return all((fd / f"{n}.csv").exists() for n in names)

    def rd(n):
        return pd.read_csv(fd / f"{n}.csv")
    jobs = [
        ("fig_headline_tail.pdf", ("synthetic_risk_profile",), lambda p: fig_headline(rd("synthetic_risk_profile"), p)),
        ("fig_c1_v52.pdf", ("c1_decomposition",), lambda p: fig_c1(rd("c1_decomposition"), p)),
        ("fig_config_effect.pdf", ("config_effect", "config_yield"), lambda p: fig_config(rd("config_effect"), rd("config_yield"), p)),
        ("fig_forest_real.pdf", ("real_dataset_unit",), lambda p: fig_forest(rd("real_dataset_unit"), p)),
        ("fig_rq3.pdf", ("opportunity_by_k", "opportunity_diffs", "rev_query_vs_campaign"),
         lambda p: fig_rq3(rd("opportunity_by_k"), rd("opportunity_diffs"), rd("rev_query_vs_campaign"), p)),
        ("fig_validation.pdf", ("validation_by_learner",), lambda p: fig_validation(rd("validation_by_learner"), p)),
        ("fig_mean_tail_rf_v52.pdf", ("synthetic_risk_profile",), lambda p: fig_mean_tail(rd("synthetic_risk_profile"), p)),
        ("fig_tail_vs_ir_rf.pdf", ("tail_by_ir",), lambda p: fig_tail_vs_ir(rd("tail_by_ir"), p)),
        ("fig_mechanism.pdf", ("mechanism_gap", "mechanism_pairs"),
         lambda p: fig_mechanism(rd("mechanism_gap"), rd("mechanism_pairs"), p)),
        ("fig_heatmap.pdf", ("domain_risk",), lambda p: fig_heatmap(rd("domain_risk"), p)),
        ("fig_multistep.pdf", ("opportunity_by_k", "opportunity_diffs"),
         lambda p: fig_multistep(rd("opportunity_by_k"), rd("opportunity_diffs"), p)),
        ("fig_c5_v52.pdf", ("c5",), lambda p: fig_c5(rd("c5"), p)),
        ("fig_image_backbones.pdf", ("image_backbones",), lambda p: fig_image_backbones(rd("image_backbones"), p)),
    ]
    for name, need, fn in jobs:
        if has(*need):
            fn(out / name)
            log(f"[figure] {name}")
        else:
            log(f"[figure] {name}: missing {[n for n in need if not (fd / f'{n}.csv').exists()]}")


def build(campaign, lbfgs, revision, oracle, out, plot_only=False, log=print):
    Path(out).mkdir(parents=True, exist_ok=True)
    if not plot_only:
        prepare(campaign, lbfgs, revision, oracle, out, log=log)
    plot(out, log=log)
