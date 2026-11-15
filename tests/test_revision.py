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
