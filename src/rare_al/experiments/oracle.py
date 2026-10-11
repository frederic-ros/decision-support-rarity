"""Discovery oracle: one-step and k-step label-only opportunity + policy capture.

For each oracle state (a point on a pure-Margin trajectory), candidate queries are
evaluated retrospectively: the candidate is labeled, the learner refitted, then the
trajectory continues with Margin for ``k-1`` further batches; test AUPRC is recorded
after each step. The reference is the same continuation started from the
Margin-top candidate, so that ``gain_vs_margin_k1`` reproduces the v1 one-step
quantity (``gain - margin_gain``).

Outputs (``<out>/``):
* ``oracle_points.csv``   -- one row per candidate (v1 columns + ``ap_k*``,
                             ``gain_vs_margin_k*``, ``cand_source``);
* ``oracle_states.csv``   -- one row per state (v1 columns);
* ``oracle_regions.csv``  -- region descriptors for the observability diagnostic (v1);
* ``oracle_policies.csv`` -- first query of each deployable policy at the same state,
                             with its k-step gains (numerator of the exploitability kappa).

With ``k_list=[1]``, ``candidates="margin_top"`` and no policies, the run is
identical to v1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score

from rare_al.data.splits import initial_mask, make_split
from rare_al.engine.states import materialize
from rare_al.engine.trajectory import fit_or_prior
from rare_al.io import append_rows, completed_keys
from rare_al.repro import stable_seed
from rare_al.strategies.base import Context
from rare_al.strategies.registry import build
from rare_al.strategies.v1.common import margin_uncertainty, nearest_distance, positive_probability

REGION_FEATURES = [
    "region_size_fraction", "uncertainty_mean", "uncertainty_max", "uncertainty_std",
    "prare_mean", "prare_max", "dist_labeled_mean", "dist_labeled_min",
    "dist_rare_mean", "dist_majority_mean", "within_dispersion",
    "margin_rank_mean", "margin_rank_min", "margin_rank_max",
    "labeled_rare_fraction", "budget_fraction",
]


def oracle_synthetic_states(cfg, section="oracle"):
    """v1-identical oracle states (tag 'oracle-state')."""
    p = cfg[section]
    for regime in p["regimes"]:
        for ir in p["irs"]:
            for d in p["dims"]:
                for sep in p["seps"]:
                    for s in range(p["seeds"]):
                        seed = stable_seed(cfg["master_seed"], "oracle-state", regime, ir, d, sep, s)
                        yield {"experiment": "oracle", "kind": "synthetic", "dataset": f"syn_{regime}",
                               "domain": "synthetic", "regime": regime, "ir": ir, "d": d, "sep": sep,
                               "seed": s, "n_samples": p["n_samples"], "min_rare": p.get("min_rare", 20),
                               "state_seed": seed, "split": "stratified",
                               "state_id": f"{regime}|{ir}|{d}|{sep}|{s}"}


def _continuation_aps(Xtr, ytr, labeled, first, Xte, yte, learner, seed0, kmax, batch):
    """AUPRC after adding ``first`` then kmax-1 Margin batches (one value per step)."""
    lab = labeled.copy()
    lab[int(first)] = True
    aps = []
    for j in range(kmax):
        clf, _ = fit_or_prior(learner, seed0 + j, Xtr[lab], ytr[lab])
        aps.append(float(average_precision_score(yte, positive_probability(clf, Xte))))
        if j == kmax - 1:
            break
        pool = np.flatnonzero(~lab)
        u = margin_uncertainty(clf, Xtr[pool])
        lab[pool[np.argsort(-u, kind="stable")[:batch]]] = True
    return aps


def _region_features(Xp, top, labels, rid, u, pr, order, Xl, yl):
    idx = np.asarray(top[labels == rid], int)
    if len(idx) == 0:
        return None
    Xr = Xp[idx]
    rare, major = Xl[yl == 1], Xl[yl == 0]
    dL = nearest_distance(Xr, Xl)
    dR = nearest_distance(Xr, rare) if len(rare) else np.full(len(Xr), 20.)
    dM = nearest_distance(Xr, major)
    center = Xr.mean(axis=0)
    disp = float(np.mean(np.sqrt(((Xr - center) ** 2).sum(axis=1)))) if len(Xr) > 1 else 0.
    rankmap = {int(i): r for r, i in enumerate(order)}
    ranks = np.asarray([rankmap[int(i)] for i in idx], float)
    vals = np.asarray([
        len(idx) / max(len(top), 1), u[idx].mean(), u[idx].max(), u[idx].std(),
        pr[idx].mean(), pr[idx].max(), dL.mean(), dL.min(), dR.mean(), dM.mean(), disp,
        ranks.mean() / max(len(order) - 1, 1), ranks.min() / max(len(order) - 1, 1),
        ranks.max() / max(len(order) - 1, 1), float(np.mean(yl == 1)), len(yl) / (len(yl) + len(Xp)),
    ], float)
    vals[~np.isfinite(vals)] = 20.
    return dict(zip(REGION_FEATURES, vals))


def run(cfg, states, out_dir, section="oracle"):
    p = cfg[section]
    b0 = p.get("budget", cfg["budgets"][next(iter(cfg["budgets"]))])
    n_init, batch = b0["n_init"], b0["batch"]
    rounds_total = b0.get("rounds", 8)
    k_list = sorted(p.get("k_list", [1]))
    kmax = max(k_list)
    learners = p.get("learners", [p.get("classifier", "logreg")])
    policies = p.get("policies", [])
    mode = p.get("candidates", "margin_top")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f_states, f_regions = out / "oracle_states.csv", out / "oracle_regions.csv"
    f_points, f_pol = out / "oracle_points.csv", out / "oracle_policies.csv"
    done = completed_keys(f_states, ["learner", "state_id"])

    for st in states:
        X, y, t, is_test = materialize(st)
        Xtr, Xte, ytr, yte = make_split(X, y, st["state_seed"], cfg["data"]["test_fraction"],
                                        st["split"], t, is_test)
        ident = {k: st.get(k) for k in ("dataset", "domain", "regime", "ir", "d", "sep", "seed")}
        for learner in learners:
            labeled = initial_mask(ytr, n_init, st["state_seed"], "warm")
            for rnd in range(1, p["states"] + 1):
                sid = f"{st['state_id']}|{rnd}"
                seed_r = st["state_seed"] + rnd
                clf, _ = fit_or_prior(learner, seed_r, Xtr[labeled], ytr[labeled])
                pool = np.flatnonzero(~labeled)
                Xp = Xtr[pool]
                u = margin_uncertainty(clf, Xp)
                pr = positive_probability(clf, Xp)
                order = np.argsort(-u, kind="stable")
                b = min(batch, len(order))
                if (learner, sid) in done:
                    labeled[pool[order[:b]]] = True
                    continue
                m = min(len(order), p["candidate_mult"] * b)
                top = np.asarray(order[:m], int)
                cand = {int(c): "margin_top" for c in top}
                if mode == "extended":
                    rng = np.random.RandomState(stable_seed(cfg["master_seed"], "oracle-cand", sid, learner))
                    for c in np.argsort(-pr, kind="stable")[: p.get("n_top_prob", b)]:
                        cand.setdefault(int(c), "top_prob")
                    rest = np.setdiff1d(np.arange(len(pool)), list(cand))
                    for c in rng.choice(rest, min(p.get("n_random", 2 * b), len(rest)), replace=False):
                        cand.setdefault(int(c), "random")
                base = float(average_precision_score(yte, positive_probability(clf, Xte)))
                seed_c = seed_r if kmax == 1 else seed_r  # same seeds for candidate and reference
                aps = {c: _continuation_aps(Xtr, ytr, labeled, pool[c], Xte, yte, learner, seed_c,
                                            kmax, b) for c in cand}
                margin_loc = int(order[0])
                gains = {c: aps[c][0] - base for c in cand}          # v1 one-step gain
                point_loc = max((c for c in cand if cand[c] == "margin_top"), key=gains.get)
                # ---- regions on the Margin-top candidates (v1 observability diagnostic)
                k_reg = min(p.get("n_regions", 4), len(top), len(np.unique(Xp[top], axis=0)))
                labels = (KMeans(n_clusters=k_reg, n_init=10, random_state=seed_r).fit_predict(Xp[top])
                          if k_reg > 1 else np.zeros(len(top), int))
                loc2reg = {int(top[i]): int(labels[i]) for i in range(len(top))}
                rankmap = {int(x): r for r, x in enumerate(order)}
                region_best, region_margin = {}, {}
                for rid in np.unique(labels):
                    members = [int(top[i]) for i in range(len(top)) if int(labels[i]) == int(rid)]
                    region_best[int(rid)] = max(gains[x] for x in members)
                    region_margin[int(rid)] = min(members, key=lambda x: rankmap[x])
                oracle_region = max(region_best, key=region_best.get)
                regional_loc = region_margin[oracle_region]
                srow = {"state_id": sid, "learner": learner, **ident, "round": rnd, "base_auprc": base,
                        "margin_gain": gains[margin_loc], "regional_oracle_gain": gains[regional_loc],
                        "point_oracle_gain": gains[point_loc],
                        "regional_ceiling_vs_margin": gains[regional_loc] - gains[margin_loc],
                        "point_ceiling_vs_margin": gains[point_loc] - gains[margin_loc],
                        "oracle_region": oracle_region, "n_regions": len(np.unique(labels)),
                        "n_rare_labeled": int(ytr[labeled].sum()), "n_candidates": len(cand)}
                for k in k_list:
                    srow[f"margin_ap_k{k}"] = aps[margin_loc][k - 1]
                rrows = []
                for rid in np.unique(labels):
                    feats = _region_features(Xp, top, labels, int(rid), u, pr, order, Xtr[labeled], ytr[labeled])
                    feats.update({"state_id": sid, "learner": learner, **ident, "round": rnd,
                                  "region_id": int(rid),
                                  "contains_point_oracle": int(int(rid) == loc2reg[point_loc]),
                                  "is_best_region": int(int(rid) == oracle_region),
                                  "region_best_point_gain": region_best[int(rid)],
                                  "region_margin_point_gain": gains[region_margin[int(rid)]],
                                  "base_auprc": base})
                    rrows.append(feats)
                prows = []
                for c, src in cand.items():
                    row = {"state_id": sid, "learner": learner, **ident, "round": rnd, "pool_loc": c,
                           "cand_source": src, "margin_rank": rankmap[c], "uncertainty": float(u[c]),
                           "prare": float(pr[c]), "true_label": int(ytr[pool[c]]), "gain": gains[c],
                           "is_point_oracle": int(c == point_loc), "is_margin_top": int(c == margin_loc),
                           "region_id": loc2reg.get(c, -1)}
                    for k in k_list:
                        row[f"ap_k{k}"] = aps[c][k - 1]
                        row[f"gain_vs_margin_k{k}"] = aps[c][k - 1] - aps[margin_loc][k - 1]
                    prows.append(row)
                # ---- deployable policies: their first query at this state
                pol_rows = []
                for pol in policies:
                    name = pol if isinstance(pol, str) else pol["name"]
                    alias = pol if isinstance(pol, str) else pol.get("alias", name)
                    params = {} if isinstance(pol, str) else pol.get("params", {})
                    ctx = Context(clf=clf, learner=learner, Xpool=Xp, Xl=Xtr[labeled], yl=ytr[labeled],
                                  batch=1, rng=np.random.RandomState(
                                      stable_seed(cfg["master_seed"], "oracle-policy", alias, sid, learner)),
                                  round=rnd, budget_remaining=(rounds_total - rnd) * batch, pool_ids=pool)
                    pick = int(build(name, **params).select(ctx)[0])
                    ap = aps[pick] if pick in aps else _continuation_aps(
                        Xtr, ytr, labeled, pool[pick], Xte, yte, learner, seed_c, kmax, b)
                    prow = {"state_id": sid, "learner": learner, **ident, "round": rnd, "policy": alias,
                            "pool_loc": pick, "true_label": int(ytr[pool[pick]]),
                            "is_margin_top": int(pick == margin_loc)}
                    for k in k_list:
                        prow[f"gain_vs_margin_k{k}"] = ap[k - 1] - aps[margin_loc][k - 1]
                    pol_rows.append(prow)
                append_rows([srow], f_states)
                append_rows(rrows, f_regions)
                append_rows(prows, f_points)
                append_rows(pol_rows, f_pol)
                labeled[pool[order[:b]]] = True       # advance with a pure Margin batch
                print("ORACLE", learner, sid, flush=True)
