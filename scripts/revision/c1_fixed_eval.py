r"""Rarity effect on a fixed evaluation set (Table S11): SD growth from IR=1 to IR=100 on the original test sets
and on 100 rare test cases, per learner and geometry. LR from the lbfgs tree (unpenalized intercept, which logs the
fixed set), RF and GBDT from rev_synthetic (trajectories identical to the campaign). Margin replicates are excluded.

Usage (repository root):
    python scripts/revision/c1_fixed_eval.py results/campaign_lbfgs results/revision out.csv
"""
import sys

import pandas as pd

from rare_al.analysis.load import load_trajectories
from rare_al.analysis.paired import final_rows, paired
from rare_al.analysis.revision import fixed_test_paired
from rare_al.analysis.risk import c1_decomposition


def pair(path, learners):
    df = load_trajectories(path)
    df = df[df.budget == "main"] if "budget" in df else df
    df = df[df.learner.isin(learners) & ~df.strategy.astype(str).str.startswith("margin_rep")]
    fin = final_rows(df)
    return fin, paired(fin, "margin", metrics=[m for m in ["auprc", "fx_auprc"] if m in fin])


lbfgs, revision, out = sys.argv[1:4]
parts = []
for path, lrs in ((f"{revision}/synthetic/trajectories.csv", ["rf", "gbdt"]),
                  (f"{lbfgs}/synthetic/trajectories.csv", ["logreg"])):
    fin, p = pair(path, lrs)
    parts.append(c1_decomposition(p).merge(c1_decomposition(fixed_test_paired(fin, "margin")),
                                           on=["learner", "geometry"], suffixes=("_orig", "_fx")))
m = pd.concat(parts, ignore_index=True)
for s in ("orig", "fx"):
    m[f"amplified_{s}"] = (m[f"sd_growth_ci_low_{s}"] > 1) & ~m[f"scale_degenerate_{s}"].astype(bool)
m.to_csv(out, index=False)
print(f"amplified cells: original {int(m.amplified_orig.sum())}/18, fixed test {int(m.amplified_fx.sum())}/18 -> {out}")
