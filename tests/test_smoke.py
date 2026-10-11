"""Offline smoke tests: the 27 strategies run, and image tasks flow through the
engine with two embeddings. No network, no GPU, a few seconds."""
from __future__ import annotations

import numpy as np
import pytest

from rare_al.strategies.registry import REGISTRY, build
from rare_al.engine.trajectory import run_trajectory
from rare_al.data.synthetic import make_regime


def test_registry_has_27():
    assert len(REGISTRY) == 27


@pytest.mark.parametrize("method", REGISTRY)
def test_each_strategy_runs(method):
    X, y = make_regime("peripheral", n=800, d=10, ir=20, sep=1.5, seed=0)
    hist = run_trajectory(X, y, method, "logreg", state_seed=0, method_seed=1,
                          n_init=30, batch=8, rounds=3, test_fraction=0.3)
    assert len(hist) == 4
    assert hist[-1]["n_rare_labeled"] >= hist[0]["n_rare_labeled"]


def test_image_task_two_embeddings(tmp_path):
    from rare_al.data.images import build_rare_task, ImageTaskSpec
    rng = np.random.default_rng(0)
    for dim in (384, 512):                       # the two backbone dims
        ymc = rng.integers(0, 10, 4000)
        centers = rng.normal(0, 2.5, (10, dim))
        X = centers[ymc] + rng.normal(0, 1, (4000, dim))
        spec = ImageTaskSpec(dataset="toy", backbone="dinov2_vits14" if dim == 384 else "resnet18",
                             rarity="one_vs_rest", positive_class=0, target_ir=49,
                             min_rare=20, max_pool=3000)
        Xt, yt, meta = build_rare_task(X, ymc, spec, rng=rng)
        assert Xt.shape[1] == dim
        assert meta["n_rare"] >= 20
        assert meta["min_rare_satisfied"]
        hist = run_trajectory(Xt, yt, "most_likely_positive", "logreg", 0, 1, 30, 8, 3)
        assert hist[-1]["n_rare_labeled"] >= hist[0]["n_rare_labeled"]


def test_ir_preserved_not_compressed():
    """Natural-rarity subsampling keeps the nominal IR (the v13 bug is gone)."""
    from rare_al.data.subsample import stratified_subsample, target_size
    rng = np.random.RandomState(0)
    n, n_rare = 50000, 450                       # IR ~ 110
    y = np.r_[np.zeros(n - n_rare, int), np.ones(n_rare, int)]
    size = target_size(n, n_rare, n_max=10000, min_rare=50, n_hard_max=60000)
    keep = stratified_subsample(y, size, seed=0)
    ir_before = (n - n_rare) / n_rare
    ys = y[keep]
    ir_after = (ys == 0).sum() / max((ys == 1).sum(), 1)
    assert abs(ir_after - ir_before) / ir_before < 0.15   # preserved within 15%


def test_rare27_xlsx_loader(tmp_path):
    """The offline Rare27 loader reads feature_* + target_binary."""
    import pandas as pd
    from rare_al.data.loaders import load_rare27_xlsx
    df = pd.DataFrame({"feature_000": [0.1, 0.2, 0.3, 0.4],
                       "feature_001": [1.0, 2.0, 3.0, 4.0],
                       "target_original": [-1, -1, -1, 1],
                       "target_binary": [0, 0, 0, 1]})
    df.to_excel(tmp_path / "toy.xlsx", index=False)
    d = load_rare27_xlsx(folder=tmp_path, file="toy.xlsx")
    assert d["X"].shape == (4, 2)
    assert d["y"].tolist() == [0, 0, 0, 1]


def test_fetch_domains_placement(tmp_path):
    """fetch_domains resolves and renames files out of a (nested) archive."""
    import sys, zipfile
    sys.path.insert(0, "scripts")
    import fetch_domains as fd
    z = tmp_path / "secom.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("secom/secom.data", "1 2 3\n")
        zf.writestr("secom/secom_labels.data", "-1 a b\n")
    out = tmp_path / "raw/secom"; out.mkdir(parents=True)
    placed, missing = fd._place_from_archive(z, out, fd.DATASETS["secom"]["pick"])
    ok, _ = fd.verify(fd.DATASETS["secom"], out)
    assert ok and not missing


def test_leave_one_domain_out_prefers_tail_safe():
    """The risk-aware rule avoids a high-mean strategy with an unsafe lower tail."""
    import numpy as np, pandas as pd
    from rare_al.analysis import validation as V
    rng = np.random.default_rng(0)
    doms = {"a": ["a1", "a2", "a3"], "b": ["b1", "b2", "b3"],
            "c": ["c1", "c2", "c3"], "d": ["d1", "d2", "d3"]}
    rows = []
    for dom, dss in doms.items():
        for ds in dss:
            for _ in range(12):
                rows.append({"domain": dom, "dataset": ds, "learner": "logreg", "strategy": "margin", "delta": 0.0})
                rows.append({"domain": dom, "dataset": ds, "learner": "logreg", "strategy": "gambler",
                             "delta": rng.normal(0.25, 0.03) if rng.random() > 0.25 else rng.normal(-0.25, 0.04)})
                rows.append({"domain": dom, "dataset": ds, "learner": "logreg", "strategy": "steady",
                             "delta": rng.normal(0.05, 0.02)})
    loo = V.leave_one_domain_out(pd.DataFrame(rows), None, reference="margin", tau=0.10)
    summ, meta = V.summary(loo)
    assert set(loo.pick_mean_rank.unique()) == {"gambler"}   # chases the high mean
    assert set(loo.pick_risk_global.unique()) == {"steady"}  # avoids the unsafe tail
    assert float(summ.set_index("rule").loc["margin", "mean_gain"]) == 0.0
