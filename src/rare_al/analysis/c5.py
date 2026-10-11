"""Matched observability analysis (v13 C5): WHERE prediction with ExtraTrees and
regularized logistic regression, GroupCV vs leave-one-geometry-out, descriptor ablations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_GROUPS = {
    "uncertainty": ["uncertainty_mean", "uncertainty_max", "uncertainty_std", "prare_mean", "prare_max"],
    "geometry": ["region_size_fraction", "dist_labeled_mean", "dist_labeled_min", "dist_rare_mean",
                 "dist_majority_mean", "within_dispersion"],
    "ranking": ["margin_rank_mean", "margin_rank_min", "margin_rank_max"],
    "state_progress": ["labeled_rare_fraction", "budget_fraction"],
}
FEATURE_GROUPS["all"] = sum(FEATURE_GROUPS.values(), [])


def _model(name, seed):
    if name == "extratrees":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("model", ExtraTreesClassifier(n_estimators=100, min_samples_leaf=3, max_features=0.8,
                                                        class_weight="balanced", random_state=seed, n_jobs=-1))])
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear",
                                                  max_iter=3000, random_state=seed))])


def _splits(df, mode):
    if mode == "groupcv":
        n_groups = df["state_id"].nunique()
        n_splits = max(2, min(5, n_groups))
        yield from ((f"fold{k}", tr, te) for k, (tr, te) in
                    enumerate(GroupKFold(n_splits=n_splits).split(df, groups=df["state_id"])))
    else:
        arr = df["regime"].to_numpy()
        for r in sorted(df.regime.unique()):
            yield r, np.flatnonzero(arr != r), np.flatnonzero(arr == r)


def where_auc(regions: pd.DataFrame, model="extratrees", features="all", mode="groupcv"):
    df = regions.reset_index(drop=True)
    y = df.contains_point_oracle.to_numpy()
    pred = np.full(len(df), np.nan)
    for k, (_lab, tr, te) in enumerate(_splits(df, mode)):
        m = _model(model, k)
        m.fit(df.iloc[tr][FEATURE_GROUPS[features]], y[tr])
        pred[te] = m.predict_proba(df.iloc[te][FEATURE_GROUPS[features]])[:, 1]
    return float(roc_auc_score(y, pred)), pred


def c5_table(regions: pd.DataFrame, n_boot=1000):
    rows = []
    for model in ("extratrees", "logreg"):
        for feats in ("all", "geometry", "uncertainty", "ranking", "state_progress"):
            g, pred = where_auc(regions, model, feats, "groupcv")
            l, _ = where_auc(regions, model, feats, "loro")
            rows.append({"model": model, "features": feats, "groupcv": g, "loro": l, "gap": g - l})
    return pd.DataFrame(rows)
