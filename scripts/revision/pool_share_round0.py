"""Share of the pool scored at or above 0.5 at the start of a campaign (round 0), real and image states.

For every state the preparation of the campaigns is reproduced exactly (materialize, split, 30 warm-start
labels); LR is fitted on the seed labels with a penalized intercept (`logreg`, liblinear), an unpenalized
intercept (`logreg_lbfgs`) and the mechanism control (`logreg_iscale100`), and the pool is scored.
No campaign is run (a few minutes). Output: results/revision/pool_share_round0.csv.

Usage (from the repository root): python scripts/revision/pool_share_round0.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rare_al.config import load_config
from rare_al.data.prepare import read_manifest
from rare_al.data.splits import initial_mask, make_split
from rare_al.engine import states as S
from rare_al.engine.trajectory import fit_or_prior
from rare_al.strategies.v1.common import positive_probability

cfg = load_config("configs/eswa/campaign.json")
manifest = read_manifest(cfg["data"]["cache_dir"])
n_init = cfg["budgets"]["main"]["n_init"]
LEARNERS = ("logreg", "logreg_lbfgs", "logreg_iscale100")

rows = []
for bench, gen in (("real", lambda: S.real_states(cfg, manifest)), ("image", lambda: S.image_states(cfg))):
    for st in gen():
        try:
            X, y, t, is_test = S.materialize(st)
            Xtr, Xte, ytr, yte = make_split(X, y, st["state_seed"], cfg["data"]["test_fraction"],
                                            st["split"], t, is_test)
            lab = initial_mask(ytr, n_init, st["state_seed"], "warm")
        except Exception as exc:
            print("skip", st["state_id"], repr(exc)[:80], flush=True)
            continue
        r = {"bench": bench, "state_id": st["state_id"], "dataset": st["dataset"],
             "prev": float(np.mean(ytr[~lab] == 1))}
        for lr in LEARNERS:
            clf, _ = fit_or_prior(lr, st["state_seed"], Xtr[lab], ytr[lab])
            r[lr] = float(np.mean(positive_probability(clf, Xtr[~lab]) >= 0.5))
        rows.append(r)
    print(bench, "done", flush=True)

d = pd.DataFrame(rows)
out = Path(cfg["results_dir"]).parent / "revision" / "pool_share_round0.csv"
out.parent.mkdir(parents=True, exist_ok=True)
d.to_csv(out, index=False)
print(d.groupby("bench")[["prev", *LEARNERS]].median().round(4).to_string())
print(d.groupby("bench")[["prev", *LEARNERS]].mean().round(4).to_string())
print("written:", out)
