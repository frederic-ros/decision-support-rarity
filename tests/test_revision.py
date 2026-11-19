"""Fast tests of the revision additions (no data, no network)."""
import numpy as np
import pandas as pd

from rare_al.analysis import revision as R
from rare_al.config import _merge, normalize_strategies
from rare_al.data.splits import make_split
from rare_al.data.synthetic import make_regime
from rare_al.engine.trajectory import run_trajectory


def test_defaults_reproduce_published_trajectory():
    X, y = make_regime("compact", 600, 4, 20, 2.0, 11)
    prep = make_split(X, y, 11, 0.3)
    a = run_trajectory(X, y, "margin", "logreg", 11, 5, 20, 4, 3, prepared=prep)
    b = run_trajectory(X, y, "margin", "logreg", 11, 5, 20, 4, 3, prepared=prep, learner_seed_offset=0,
                       eval_sets=None)
    assert [r["auprc"] for r in a] == [r["auprc"] for r in b]
    assert a[-1]["n_test_pos"] == int(prep[3].sum())


def test_eval_set_is_logged_and_does_not_change_acquisition():
    X, y = make_regime("peripheral", 600, 4, 20, 2.0, 3)
    prep = make_split(X, y, 3, 0.3)
    Xe, ye = make_regime("peripheral", 400, 4, 20, 2.0, 99)
    a = run_trajectory(X, y, "margin", "rf", 3, 5, 20, 4, 2, prepared=prep)
    b = run_trajectory(X, y, "margin", "rf", 3, 5, 20, 4, 2, prepared=prep, eval_sets={"fx": (Xe, ye)})
    assert [r["n_rare_labeled"] for r in a] == [r["n_rare_labeled"] for r in b]
    assert "fx_auprc" in b[-1] and b[-1]["fx_n_pos"] == int(ye.sum())


def test_config_replace_and_offsets():
    out = _merge({"budgets": {"main": 1}}, {"budgets": {"__replace__": True, "low": 2}})
    assert out["budgets"] == {"low": 2}
    s = normalize_strategies(["margin", {"name": "margin_rtb", "alias": "margin_rep1", "learner_seed_offset": 7}])
    assert s[0]["learner_seed_offset"] == 0 and s[1]["learner_seed_offset"] == 7


def test_multiplicity_and_bounds():
    p = np.array([0.01, 0.02, 0.5])
    assert np.all(R.bh_adjust(p) >= p)
    need, _ = R.pilot_sample_size(alphas=(0.05,), failures=(0,))
    assert int(need.runs_needed.iloc[0]) == 59
    assert R.source_group("imblearn_car_eval_4") == R.source_group("imblearn_car_eval_34")
    assert R.source_group("polish_bankruptcy_y3") == "polish_bankruptcy"


def test_validation_sweep_tail_rule_binds():
    rng = np.random.default_rng(0)
    rows = []
    for dom in ("a", "b", "c"):
        for ds in range(3):
            for seed in range(10):
                rows.append({"domain": dom, "dataset": f"{dom}{ds}", "learner": "rf", "seed": seed,
                             "strategy": "margin", "delta": 0.0})
                rows.append({"domain": dom, "dataset": f"{dom}{ds}", "learner": "rf", "seed": seed,
                             "strategy": "heavy", "delta": 0.10 if seed < 9 else -0.5})
                rows.append({"domain": dom, "dataset": f"{dom}{ds}", "learner": "rf", "seed": seed,
                             "strategy": "safe", "delta": 0.01 + 0.001 * rng.standard_normal()})
    p = pd.DataFrame(rows)
    cells, summ = R.validation_alpha_sweep(p, cluster_fn=lambda d: d, alphas=(0.05, 1.0), n_boot=50,
                                           reference_proxy=None)
    assert (cells["pick_mean_rank"] == "heavy").all()
    assert (cells["pick_risk_0.05"] == "safe").all()
    row = summ.set_index("rule").loc["risk_0.05"]
    assert row.share_changed == 1.0 and row.dloss < 0


# ------------------------------------------------------------ LR intercept regularization
def test_lbfgs_intercept_is_not_shrunk():
    from rare_al.learners import make_learner
    X, y = make_regime("peripheral", 3000, 6, 100, 2.0, 5)
    rng = np.random.RandomState(0)
    idx = np.r_[rng.choice(np.flatnonzero(y == 1), 5, replace=False),
                rng.choice(np.flatnonzero(y == 0), 45, replace=False)]
    fit = {n: make_learner(n, 0).fit(X[idx], y[idx]) for n in ("logreg", "logreg_lbfgs", "logreg_iscale100")}
    share = {n: float((m.predict_proba(X)[:, 1] >= 0.5).mean()) for n, m in fit.items()}
    # the penalized intercept puts more of the (99% majority) pool above 0.5
    assert share["logreg"] > share["logreg_lbfgs"]
    # with the intercept penalty divided by 1e4, liblinear matches lbfgs
    assert abs(share["logreg_iscale100"] - share["logreg_lbfgs"]) < 0.01
    assert abs(fit["logreg_iscale100"].intercept_[0] - fit["logreg_lbfgs"].intercept_[0]) < 0.05


def test_threshold_logging_does_not_change_acquisition():
    X, y = make_regime("compact", 600, 4, 20, 2.0, 11)
    prep = make_split(X, y, 11, 0.3)
    for s in ("margin", "random"):
        a = run_trajectory(X, y, s, "logreg", 11, 5, 20, 4, 3, prepared=prep)
        b = run_trajectory(X, y, s, "logreg", 11, 5, 20, 4, 3, prepared=prep, log_threshold=True)
        assert [r["auprc"] for r in a] == [r["auprc"] for r in b]
        assert [r["n_rare_labeled"] for r in a] == [r["n_rare_labeled"] for r in b]
        assert 0.0 <= b[0]["pool_share_p05"] <= 1.0 and 0.0 < b[0]["pool_prev"] < 0.5
        assert "pool_share_p05" not in a[0]


def test_threshold_diagnostics_lift():
    t = pd.DataFrame({"state_id": ["s"] * 3, "learner": ["logreg"] * 3, "strategy": ["margin"] * 3,
                      "pool_share_p05": [0.4, 0.4, np.nan], "pool_prev": [0.01, 0.01, np.nan],
                      "batch_rare": [0, 1, np.nan]})
    d = R.threshold_diagnostics(t, batch=8)
    assert np.isclose(d.lift_median.iloc[0], 1 / 0.16) and np.isclose(d.share_over_prev.iloc[0], 40)


def test_swap_learner(tmp_path):
    from rare_al.tools.swap_learner import swap_learner
    base, add = tmp_path / "base" / "real", tmp_path / "add" / "real"
    base.mkdir(parents=True), add.mkdir(parents=True)
    pd.DataFrame({"learner": ["logreg", "rf"], "state_id": ["a", "a"], "strategy": ["margin"] * 2,
                  "auprc": [0.1, 0.2]}).to_csv(base / "trajectories.csv", index=False)
    pd.DataFrame({"learner": ["logreg_lbfgs"] * 2, "state_id": ["a"] * 2, "strategy": ["margin", "margin_rep1"],
                  "auprc": [0.3, 0.31]}).to_csv(add / "trajectories.csv", index=False)
    swap_learner(tmp_path / "base", tmp_path / "add", tmp_path / "out", sections=["real"])
    o = pd.read_csv(tmp_path / "out" / "real" / "trajectories.csv")
    assert sorted(o.learner) == ["logreg", "rf"] and set(o.strategy) == {"margin"}
    assert o.set_index("learner").loc["logreg", "auprc"] == 0.3
    assert o.set_index("learner").loc["logreg", "learner_impl"] == "logreg_lbfgs"
    pd.DataFrame({"learner": ["logreg"], "state_id": ["b"], "strategy": ["margin"], "auprc": [0.1]}).to_csv(
        base / "trajectories.csv", mode="a", header=False, index=False)
    try:
        swap_learner(tmp_path / "base", tmp_path / "add", tmp_path / "out2", sections=["real"])
        raise AssertionError("incomplete lbfgs run must be refused")
    except RuntimeError:
        pass


def test_swap_learner_same_tree_and_mapping(tmp_path):
    from rare_al.tools.swap_learner import swap_learner
    o = tmp_path / "rev" / "oracle"
    o.mkdir(parents=True)
    pd.DataFrame({"learner": ["logreg", "logreg_lbfgs", "rf"], "state_id": ["a"] * 3,
                  "gain": [1, 2, 3]}).to_csv(o / "oracle_states.csv", index=False)
    (tmp_path / "rev" / "noise_real" / "real").mkdir(parents=True)
    (tmp_path / "lb" / "real").mkdir(parents=True)
    pd.DataFrame({"learner": ["logreg", "rf"], "state_id": ["s"] * 2, "strategy": ["margin_rep1"] * 2,
                  "auprc": [0.1, 0.2]}).to_csv(tmp_path / "rev" / "noise_real" / "real" / "trajectories.csv", index=False)
    pd.DataFrame({"learner": ["logreg_lbfgs"], "state_id": ["s"], "strategy": ["margin_rep1"],
                  "auprc": [0.5]}).to_csv(tmp_path / "lb" / "real" / "trajectories.csv", index=False)
    swap_learner(tmp_path / "rev", tmp_path / "rev", tmp_path / "out", sections=["oracle"])
    s = pd.read_csv(tmp_path / "out" / "oracle" / "oracle_states.csv")
    assert sorted(s.learner) == ["logreg", "rf"] and s.set_index("learner").gain["logreg"] == 2
    swap_learner(tmp_path / "rev", tmp_path / "lb", tmp_path / "out2", sections=["noise_real/real:real"],
                 keep_replicates=True)
    t = pd.read_csv(tmp_path / "out2" / "noise_real" / "real" / "trajectories.csv")
    assert t.set_index("learner").auprc["logreg"] == 0.5


def test_swap_learner_drops_extra_states(tmp_path):
    from rare_al.tools.swap_learner import swap_learner
    base, add = tmp_path / "base" / "image", tmp_path / "add" / "image"
    base.mkdir(parents=True), add.mkdir(parents=True)
    pd.DataFrame({"learner": ["logreg", "rf"], "state_id": ["a", "a"], "strategy": ["margin"] * 2,
                  "auprc": [0.1, 0.2]}).to_csv(base / "trajectories.csv", index=False)
    pd.DataFrame({"learner": ["logreg_lbfgs"] * 2, "state_id": ["a", "new"], "strategy": ["margin"] * 2,
                  "auprc": [0.3, 0.4]}).to_csv(add / "trajectories.csv", index=False)
    log = swap_learner(tmp_path / "base", tmp_path / "add", tmp_path / "out", sections=["image"])
    o = pd.read_csv(tmp_path / "out" / "image" / "trajectories.csv")
    assert set(o.state_id) == {"a"} and log[0]["extra_states_dropped"] == 1
