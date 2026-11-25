r"""Rare cases acquired by Margin and DIRECT relative to random sampling (Tables 3 and S20).

Usage (repository root):
    python scripts/revision/rare_yield.py results/campaign results/campaign_lbfgs out.csv
(results/campaign: LR with a penalized intercept; results/campaign_lbfgs: unpenalized LR, RF, GBDT)
"""
import sys

import pandas as pd

from rare_al.reporting.paper_figures import _final_paired

pub, lbfgs, out = sys.argv[1:4]
rows = []
for tree, tag in ((pub, "pub"), (lbfgs, "lbfgs")):
    for bench in ("real", "image"):
        df, _ = _final_paired(f"{tree}/{bench}/trajectories.csv")
        g = df.groupby(["learner", "state_id", "strategy"]).n_rare_labeled
        acq = (g.max() - g.min()).unstack()
        for lr, a in acq.groupby(level=0):
            if tag == "pub" and lr != "logreg":
                continue
            name = {"logreg": "LR penalized" if tag == "pub" else "LR unpenalized", "rf": "RF", "gbdt": "GBDT"}[lr]
            rows.append(dict(bench=bench, learner=name, margin=a["margin"].sum() / a["random"].sum(),
                             direct=a["direct"].sum() / a["random"].sum(), margin_acquired=a["margin"].mean(),
                             random_acquired=a["random"].mean()))
pd.DataFrame(rows).drop_duplicates(["bench", "learner"]).to_csv(out, index=False)
print(pd.DataFrame(rows).round(2).to_string(index=False))
