"""Verbatim port of the v1 (TMLR) acquisition function. Do not modify: v13 results depend on it."""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from rare_al.strategies.v1.common import margin_uncertainty, normalize, nearest_distance

def _top_margin(clf, Xpool, batch, mult=5):
    u = margin_uncertainty(clf, Xpool)
    m = min(len(Xpool), max(batch, mult*batch))
    top = np.argsort(-u, kind="stable")[:m]
    return u, top

def uherding(clf, Xpool, batch, rng, Xl=None, candidate_mult=5, **kwargs):
    u, top = _top_margin(clf, Xpool, batch, candidate_mult)
    Xt = Xpool[top]
    if len(Xt) <= batch:
        return top
    chosen = [0]
    min_d = np.full(len(Xt), np.inf)
    un = normalize(u[top])
    for _ in range(1, batch):
        d = np.sqrt(((Xt - Xt[chosen[-1]])**2).sum(axis=1))
        min_d = np.minimum(min_d, d)
        score = 0.55*un + 0.45*normalize(min_d)
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return top[np.asarray(chosen)]

def kmeans_diversity(clf, Xpool, batch, rng, candidate_mult=5, **kwargs):
    _, top = _top_margin(clf, Xpool, batch, candidate_mult)
    Xt = Xpool[top]
    k = min(batch, len(Xt), len(np.unique(Xt, axis=0)))
    if k <= 1:
        return top[:batch]
    km = KMeans(n_clusters=k, n_init=10, random_state=int(rng.randint(2**31-1))).fit(Xt)
    selected = []
    for center in km.cluster_centers_:
        order = np.argsort(((Xt-center)**2).sum(axis=1))
        for j in order:
            if int(j) not in selected:
                selected.append(int(j)); break
    return top[np.asarray(selected[:batch])]

def coreset(clf, Xpool, batch, rng, candidate_mult=5, **kwargs):
    _, top = _top_margin(clf, Xpool, batch, candidate_mult)
    Xt = Xpool[top]
    if len(Xt) <= batch:
        return top
    chosen = [int(rng.randint(len(Xt)))]
    mind = np.full(len(Xt), np.inf)
    while len(chosen) < batch:
        d = np.sqrt(((Xt - Xt[chosen[-1]])**2).sum(axis=1))
        mind = np.minimum(mind, d)
        mind[chosen] = -np.inf
        chosen.append(int(np.argmax(mind)))
    return top[np.asarray(chosen)]

def typiclust(clf, Xpool, batch, rng, Xl=None, neighbors=5, **kwargs):
    """Tabular adaptation of TypiClust acquisition logic.

    Standardized input features are treated as the representation.
    Labeled + unlabeled data are clustered; clusters already containing
    labeled samples are considered covered.
    """
    Xl = np.asarray(Xl)
    Xall = np.vstack([Xl, Xpool])
    n_l = len(Xl)
    k = min(batch, len(Xpool))
    n_clusters = min(max(k, n_l + k), len(Xall), len(np.unique(Xall, axis=0)))
    if n_clusters <= 1:
        return np.arange(k)

    km = KMeans(
        n_clusters=n_clusters, n_init=10, random_state=int(rng.randint(2**31-1))
    ).fit(Xall)
    covered = set(km.labels_[:n_l].tolist())
    ulab = km.labels_[n_l:]
    ids, sizes = np.unique(ulab, return_counts=True)
    cluster_order = [int(c) for c in ids[np.argsort(-sizes)] if int(c) not in covered]

    selected = []
    for cluster_id in cluster_order:
        if len(selected) >= k:
            break
        members = np.flatnonzero(ulab == cluster_id)
        if len(members) == 1:
            selected.append(int(members[0]))
            continue
        Xc = Xpool[members]
        kk = min(neighbors, len(members)-1)
        nn = NearestNeighbors(n_neighbors=kk+1).fit(Xc)
        dist, _ = nn.kneighbors(Xc)
        typicality = 1.0 / (dist[:,1:].mean(axis=1) + 1e-12)
        selected.append(int(members[np.argmax(typicality)]))

    if len(selected) < k:
        used = set(selected)
        nn = NearestNeighbors(n_neighbors=min(neighbors+1, len(Xpool))).fit(Xpool)
        dist, _ = nn.kneighbors(Xpool)
        typicality = (
            1.0/(dist[:,1:].mean(axis=1)+1e-12)
            if dist.shape[1] > 1 else np.ones(len(Xpool))
        )
        for i in np.argsort(-typicality):
            if int(i) not in used:
                selected.append(int(i)); used.add(int(i))
                if len(selected) >= k:
                    break
    return np.asarray(selected[:k], int)

def dcf(clf, Xpool, batch, rng, candidate_mult=5, **kwargs):
    """Definitive DCF implementation used by the consolidated repository.

    Top-(candidate_mult*batch) uncertainty gate; KMeans into feasible clusters;
    choose the most uncertain point from each cluster; uncertainty fill.
    """
    u, top = _top_margin(clf, Xpool, batch, candidate_mult)
    Xt = Xpool[top]
    k = min(batch, len(Xt), len(np.unique(Xt, axis=0)))
    if k <= 1:
        return top[:batch]
    km = KMeans(n_clusters=k, n_init=5, random_state=int(rng.randint(2**31-1))).fit(Xt)
    selected = []
    for cluster_id in range(k):
        members = np.flatnonzero(km.labels_ == cluster_id)
        if len(members):
            selected.append(int(top[members[np.argmax(u[top][members])]]))
    if len(selected) < batch:
        used = set(selected)
        for idx in top:
            idx = int(idx)
            if idx not in used:
                selected.append(idx); used.add(idx)
                if len(selected) >= batch:
                    break
    return np.asarray(selected[:batch], int)

def adaptive_repulsion(clf, Xpool, batch, rng, Xl=None, candidate_mult=5, **kwargs):
    u, top = _top_margin(clf, Xpool, batch, candidate_mult)
    Xt = Xpool[top]
    un = normalize(u[top])
    selected = [0]
    while len(selected) < min(batch, len(Xt)):
        d_batch = nearest_distance(Xt, Xt[selected])
        d_lab = nearest_distance(Xt, Xl) if Xl is not None else np.zeros(len(Xt))
        score = 0.75*un + 0.25*(0.5*normalize(d_batch)+0.5*normalize(d_lab))
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))
    return top[np.asarray(selected)]
