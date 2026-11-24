"""Smoke tests for the paper figures (rare-al paper-figures, plot step)."""
import numpy as np
import pandas as pd

from rare_al.reporting import paper_figures as P


def _risk():
    rows = []
    for lr in ("logreg", "rf", "gbdt"):
        for i, s in enumerate(["typiclust", "probcover", "dcf", "uherding", "kmeans_diversity", "class_balanced",
                               "committee", "direct", "tailor", "coreset", "margin_rtb"]):
            rows.append(dict(learner=lr, strategy=s, mean=-0.12 + 0.012 * i if i < 2 else 0.001 * (i - 5),
                             p_lt_tau=0.3 - 0.025 * i if s != "margin_rtb" else 0.001))
    return pd.DataFrame(rows)


def test_validation_by_learner_and_plot(tmp_path):
    rng = np.random.default_rng(0)
    n = 30
    cells = pd.DataFrame({"learner": np.repeat(["logreg", "rf", "gbdt"], n), "cluster": np.tile(np.arange(10), 9),
                          "pick_mean_rank": "a", "gain_mean_rank": rng.normal(0, 0.01, 3 * n),
                          "loss_mean_rank": rng.uniform(0, 0.1, 3 * n)})
    for a in ("0", "0.01", "0.02", "0.05", "0.1"):
        cells[f"pick_risk_{a}"] = np.where(rng.uniform(size=3 * n) < 0.5, "a", "b")
        cells[f"gain_risk_{a}"] = cells.gain_mean_rank + rng.normal(0, 0.002, 3 * n)
        cells[f"loss_risk_{a}"] = (cells.loss_mean_rank - 0.01).clip(lower=0)
    vb = P.validation_by_learner(cells)
    assert set(vb.learner) == {"logreg", "rf", "gbdt", "pooled"}
    assert (vb[vb.alpha == 0.02].dloss <= 0).all()
    P._style()
    P.fig_validation(pd.concat([vb.assign(bench="real"), vb.assign(bench="synthetic")]), tmp_path / "v.pdf")
    assert (tmp_path / "v.pdf").stat().st_size > 1000


def test_headline_and_mean_tail(tmp_path):
    P._style()
    P.fig_headline(_risk(), tmp_path / "h.pdf")
    P.fig_mean_tail(_risk(), tmp_path / "m.pdf")
    assert (tmp_path / "h.pdf").exists() and (tmp_path / "m.pdf").exists()


def test_plot_skips_missing_inputs(tmp_path):
    (tmp_path / "figure_data").mkdir()
    _risk().to_csv(tmp_path / "figure_data" / "synthetic_risk_profile.csv", index=False)
    msgs = []
    P.plot(tmp_path, log=msgs.append)
    assert (tmp_path / "fig_headline_tail.pdf").exists()
    assert any("missing" in m for m in msgs)


def test_mechanism(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for lr in ("logreg", "rf", "gbdt"):
        for st in range(60):
            ref = int(rng.integers(5, 40))
            rows.append(dict(learner=lr, state_id=str(st), strategy="margin", n_rare_labeled=ref,
                             ref_n_rare_labeled=ref, delta=0.0))
            for s in ("typiclust", "coreset", "margin_rtb", "entropy"):
                r = max(1, ref + int(rng.integers(-15, 6)) if s != "margin_rtb" else ref)
                d = -0.95 if r == 1 else (-0.2 if r < ref - 10 else rng.normal(0, 0.02))
                rows.append(dict(learner=lr, state_id=str(st), strategy=s, n_rare_labeled=r,
                                 ref_n_rare_labeled=ref, delta=d))
    gap, pairs = P.mechanism_data(pd.DataFrame(rows), min_n=5)
    assert set(gap.group) <= {"policies", "rerun"} and "entropy" not in set(pairs.strategy)
    pol = gap[(gap.group == "policies") & (gap.learner == "rf")].set_index("order").p
    assert pol.loc[0] > pol.loc[3]
    assert {"rerun"} <= set(pairs.kind)
    P._style()
    P.fig_mechanism(gap, pairs, tmp_path / "mech.pdf")
    assert (tmp_path / "mech.pdf").stat().st_size > 1000
