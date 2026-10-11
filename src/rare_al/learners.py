"""Downstream learners.

``logreg`` and ``rf`` are identical to v1 (both class-weighted). The other entries
support the ESWA design:

* ``gbdt``            -- gradient-boosted trees (sklearn HistGradientBoosting,
                         class-weighted); ``lightgbm`` / ``xgboost`` if installed.
* ``*_unweighted``    -- learner-side ablation without class weighting. NOTE: in v1
                         the reference learners were ALREADY class-weighted, so the
                         paper's "Margin + class-weighted learner" arm is the v1
                         default; the informative ablation is the unweighted learner.
* ``rf_nobootstrap``  -- probability-resolution check (Section 4.4): without bootstrap
                         no tree is grown on a resample containing zero rare cases.
* ``rf_leaf1_500``    -- probability-resolution check: finer probability grid.
* ``tabpfn``          -- optional tabular foundation model (if installed).
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _logreg(seed, weighted=True):
    return LogisticRegression(max_iter=1500, class_weight="balanced" if weighted else None,
                              solver="liblinear", random_state=seed)


def _rf(seed, weighted=True, bootstrap=True, n_estimators=200, min_samples_leaf=2):
    cw = ("balanced_subsample" if bootstrap else "balanced") if weighted else None
    return RandomForestClassifier(n_estimators=n_estimators, class_weight=cw,
                                  min_samples_leaf=min_samples_leaf, bootstrap=bootstrap,
                                  random_state=seed, n_jobs=1)


def _gbdt(seed, weighted=True):
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=2,
        l2_regularization=1.0, class_weight="balanced" if weighted else None,
        early_stopping=False, random_state=seed)


def _lightgbm(seed):
    import lightgbm as lgb  # optional dependency
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=15, min_child_samples=2,
                              class_weight="balanced", random_state=seed, verbose=-1)


def _xgboost(seed):
    import xgboost as xgb  # optional dependency
    return xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             random_state=seed, eval_metric="logloss", n_jobs=1)


def _tabpfn(seed):
    from tabpfn import TabPFNClassifier  # optional dependency
    return TabPFNClassifier(random_state=seed)


LEARNERS = {
    "logreg": lambda s: _logreg(s),
    "logreg_unweighted": lambda s: _logreg(s, weighted=False),
    "rf": lambda s: _rf(s),
    "rf_unweighted": lambda s: _rf(s, weighted=False),
    "rf_nobootstrap": lambda s: _rf(s, bootstrap=False),
    "rf_leaf1_500": lambda s: _rf(s, n_estimators=500, min_samples_leaf=1),
    "gbdt": lambda s: _gbdt(s),
    "gbdt_unweighted": lambda s: _gbdt(s, weighted=False),
    "lightgbm": _lightgbm,
    "xgboost": _xgboost,
    "tabpfn": _tabpfn,
}

LEARNER_FAMILY = {
    "logreg": "linear", "logreg_unweighted": "linear",
    "rf": "bagged_trees", "rf_unweighted": "bagged_trees", "rf_nobootstrap": "bagged_trees",
    "rf_leaf1_500": "bagged_trees",
    "gbdt": "boosted_trees", "gbdt_unweighted": "boosted_trees", "lightgbm": "boosted_trees",
    "xgboost": "boosted_trees", "tabpfn": "foundation",
}


def make_learner(name: str, seed: int):
    if name not in LEARNERS:
        raise ValueError(f"Unknown learner: {name}")
    return LEARNERS[name](int(seed))


make_classifier = make_learner  # v1 name
