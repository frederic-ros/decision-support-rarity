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
* ``rf_leaf5`` / ``gbdt_leaf5`` -- revision sensitivity (R2.4, R1.3): coarser leaves.
* ``rf_trees50``      -- revision sensitivity: fewer trees (coarser probability grid).
* ``tabpfn``          -- optional tabular foundation model (if installed).
* ``logreg_lbfgs``    -- revision: the published ``logreg`` (liblinear) L2-penalizes the
                         intercept like any other weight (intercept_scaling=1); with a
                         few dozen labels the intercept stays near zero and the 0.5
                         threshold falls inside the majority mass. ``logreg_lbfgs`` is the
                         same model (L2, C=1, class-balanced) with an unpenalized intercept.
* ``logreg_iscale100`` -- mechanism control: liblinear with intercept_scaling=100, i.e.
                         the intercept penalty divided by 10^4 with the solver unchanged.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _logreg(seed, weighted=True):
    # Published configuration. NOTE: with the liblinear solver the intercept is L2-penalized
    # like any other weight (intercept_scaling=1); with few labels it stays near zero.
    return LogisticRegression(max_iter=1500, class_weight="balanced" if weighted else None,
                              solver="liblinear", random_state=seed)


def _logreg_lbfgs(seed, weighted=True):
    """Same model (L2, C=1, class-balanced) with an unpenalized intercept (lbfgs solver).
    Revision: robustness of every logistic-regression result to intercept regularization."""
    return LogisticRegression(max_iter=1500, class_weight="balanced" if weighted else None,
                              solver="lbfgs", C=1.0, random_state=seed)


def _logreg_iscale(seed, scale=100.0, weighted=True):
    """liblinear with a (nearly) unpenalized intercept: isolates the intercept penalty
    from the choice of solver."""
    return LogisticRegression(max_iter=1500, class_weight="balanced" if weighted else None,
                              solver="liblinear", intercept_scaling=scale, random_state=seed)


def _rf(seed, weighted=True, bootstrap=True, n_estimators=200, min_samples_leaf=2):
    cw = ("balanced_subsample" if bootstrap else "balanced") if weighted else None
    return RandomForestClassifier(n_estimators=n_estimators, class_weight=cw,
                                  min_samples_leaf=min_samples_leaf, bootstrap=bootstrap,
                                  random_state=seed, n_jobs=1)


def _gbdt(seed, weighted=True, min_samples_leaf=2):
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=min_samples_leaf,
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
    "logreg_lbfgs": lambda s: _logreg_lbfgs(s),
    "logreg_lbfgs_unweighted": lambda s: _logreg_lbfgs(s, weighted=False),
    "logreg_iscale100": lambda s: _logreg_iscale(s),
    "rf": lambda s: _rf(s),
    "rf_unweighted": lambda s: _rf(s, weighted=False),
    "rf_nobootstrap": lambda s: _rf(s, bootstrap=False),
    "rf_leaf1_500": lambda s: _rf(s, n_estimators=500, min_samples_leaf=1),
    "rf_leaf5": lambda s: _rf(s, min_samples_leaf=5),
    "rf_trees50": lambda s: _rf(s, n_estimators=50),
    "gbdt": lambda s: _gbdt(s),
    "gbdt_leaf5": lambda s: _gbdt(s, min_samples_leaf=5),
    "gbdt_unweighted": lambda s: _gbdt(s, weighted=False),
    "lightgbm": _lightgbm,
    "xgboost": _xgboost,
    "tabpfn": _tabpfn,
}

LEARNER_FAMILY = {
    "logreg": "linear", "logreg_unweighted": "linear", "logreg_lbfgs": "linear",
    "logreg_lbfgs_unweighted": "linear", "logreg_iscale100": "linear",
    "rf": "bagged_trees", "rf_unweighted": "bagged_trees", "rf_nobootstrap": "bagged_trees",
    "rf_leaf1_500": "bagged_trees", "rf_leaf5": "bagged_trees", "rf_trees50": "bagged_trees",
    "gbdt": "boosted_trees", "gbdt_leaf5": "boosted_trees", "gbdt_unweighted": "boosted_trees", "lightgbm": "boosted_trees",
    "xgboost": "boosted_trees", "tabpfn": "foundation",
}


def make_learner(name: str, seed: int):
    if name not in LEARNERS:
        raise ValueError(f"Unknown learner: {name}")
    return LEARNERS[name](int(seed))


make_classifier = make_learner  # v1 name
