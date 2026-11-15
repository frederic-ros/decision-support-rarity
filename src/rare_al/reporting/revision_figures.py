"""Revision figures (R1.9, R1.10, R2.12).

* :func:`fig_mean_tail_zoom` -- Fig. 3: mean paired gain vs. material-loss probability
  with a zoom panel on the near-neutral strategies, contiguous numbering aligned with
  the list, controls and internal probes drawn as open markers.
* :func:`fig_image_two_panels` -- Fig. 8: both panels announced by the caption
  (A: paired gain under the linear head with task-resampled intervals; B: rare cases
  found), full text width, readable strategy names.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rare_al.analysis.revision import cluster_boot  # noqa: E402
from rare_al.reporting.figures import FAMILY_COLOR  # noqa: E402
from rare_al.strategies.registry import display, family, role  # noqa: E402

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42})
FAMNAME = {"active_search": "active search", "adaptive": "adaptive", "disagreement": "disagreement",
           "framework_instance": "framework instance (RCA)", "hybrid": "hybrid", "passive": "passive (random)",
           "rare_targeted": "rare-targeted", "representativeness": "representativeness",
           "uncertainty": "uncertainty", "reference": "reference variant", "probe": "internal probe",
           "control": "control"}
EXCLUDE = ["margin", "entropy", "least_confident"]       # binary-identical to the reference


def fig_mean_tail_zoom(p: pd.DataFrame, path, learner="rf", tau=0.10,
                       offsets=None, zoom_mean=-0.025, zoom_c=9.0):
    """``p``: paired frame (one row per trajectory) of the synthetic grid."""
    g = p[(p.learner == learner) & ~p.strategy.isin(EXCLUDE)]
    r = g.groupby("strategy").delta.agg(mean="mean", C=lambda x: 100 * (x < -tau).mean()).reset_index()
    r["fam"] = r.strategy.map(family)
    r["role"] = r.strategy.map(role)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"width_ratios": [1, 1.15]})
    zoom = (r["mean"] > zoom_mean) & (r.C < zoom_c)
    for ax, sub, title in ((a1, r, "A. All strategies"), (a2, r[zoom], "B. Zoom on near-neutral means")):
        for _, row in sub.iterrows():
            c = FAMILY_COLOR.get(row.fam, "k")
            ax.scatter(row["mean"], row.C, s=30, zorder=3, edgecolor=c, lw=1.1,
                       facecolor="white" if row.role in ("probe", "control") else c)
        ax.axvline(0, color="k", ls="--", lw=0.7)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Mean paired AUPRC gain vs. Margin" if ax is a2 else "Mean paired gain")
    a1.set_ylim(-1.5, max(38.5, r.C.max() + 4))
    a1.set_ylabel(r"$\mathcal{C}_{0.10}=P(\Delta<-0.10)$ (% of states)")
    off = offsets or {"committee": (-2, -11), "random": (7, 0), "typiclust": (-2, 5), "probcover": (5, -1)}
    for _, row in r[~zoom].iterrows():
        a1.annotate(display(row.strategy), (row["mean"], row.C), xytext=off.get(row.strategy, (4, -3)),
                    textcoords="offset points", fontsize=6.8)
    zs = r[zoom].sort_values(["C", "mean"], ascending=[False, True]).reset_index(drop=True)
    names = []
    for i, row in zs.iterrows():
        a2.annotate(str(i + 1), (row["mean"], row.C), xytext=(3, 2), textcoords="offset points", fontsize=6.8)
        tag = " (probe)" if row.role == "probe" else " (control)" if row.role == "control" else ""
        names.append(f"{i + 1}. {display(row.strategy)}{tag}")
    a2.text(1.03, 1.0, "\n".join(names), transform=a2.transAxes, va="top", fontsize=6.6, linespacing=1.25)
    a1.add_patch(plt.Rectangle((zoom_mean, -0.5), 0.035, zoom_c + 0.5, fill=False, ls=":", lw=0.8, ec="grey"))
    fams = sorted(set(r.fam) - {"control"})
    h = [plt.Line2D([], [], marker="o", ls="", color=FAMILY_COLOR.get(f, "k"), label=FAMNAME.get(f, f))
         for f in fams]
    h.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="open: control / internal probe"))
    fig.tight_layout(rect=(0, 0.16, 0.83, 1))
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.45, 0.0))
    fig.savefig(path)
    plt.close(fig)
    return r


def fig_image_two_panels(p_img: pd.DataFrame, path, learner="logreg", n_boot=2000):
    """``p_img``: paired frame of the image benchmark (dataset, positive_class, ir_target)."""
    d = p_img.copy()
    d["task"] = d.dataset.astype(str) + "|" + d.positive_class.astype(str) + "|" + d.ir_target.astype(str)
    d = d[(d.learner == learner)]
    rows = []
    for (bb, s), g in d[~d.strategy.isin(EXCLUDE)].groupby(["backbone", "strategy"], observed=True):
        m, lo, hi = cluster_boot(g.delta, g.task, n=n_boot, seed=3)
        rows.append({"backbone": bb, "strategy": s, "mean": m, "lo_task": lo, "hi_task": hi})
    t = pd.DataFrame(rows)
    order = t[t.backbone == "dinov2_vits14"].sort_values("mean").strategy.tolist()
    rare = d.groupby(["backbone", "strategy"], observed=True).n_rare_labeled.mean()
    ref_rare = d[d.strategy == "margin"].groupby("backbone", observed=True).n_rare_labeled.mean()
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.2, 5.2), sharey=True)
    y = np.arange(len(order))
    for bb, col, lab, off in (("dinov2_vits14", "#1f77b4", "DINOv2 ViT-S/14", -0.18),
                              ("resnet18", "#ff7f0e", "ResNet-18", 0.18)):
        s = t[t.backbone == bb].set_index("strategy").reindex(order)
        b1.errorbar(s["mean"], y + off, xerr=[s["mean"] - s.lo_task, s.hi_task - s["mean"]], fmt="o", ms=3.5,
                    color=col, ecolor=col, elinewidth=0.9, capsize=1.5, label=lab)
        b2.barh(y + off, [rare.get((bb, x), np.nan) for x in order], height=0.36, color=col, label=lab)
        if bb in ref_rare:
            b2.axvline(ref_rare[bb], color=col, ls=":", lw=1)
    b1.axvline(0, color="k", ls="--", lw=0.7)
    b1.set_yticks(y)
    b1.set_yticklabels([display(x) for x in order])
    b1.set_xlabel("Paired AUPRC gain vs. Margin (95% CI, tasks resampled)")
    b2.set_xlabel("Rare cases labeled at the final round\n(dotted: Margin)")
    b1.set_title("A. Gain under the linear head", loc="left")
    b2.set_title("B. Rare cases found", loc="left")
    b2.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return t
