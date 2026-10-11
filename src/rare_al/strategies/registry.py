"""Strategy registry with metadata used by the reporting layer (Table 5 of the paper).

status: v13 = evaluated in the TMLR version (verbatim port); new = added for ESWA;
adapted = tabular adaptation of a method designed for deep representations.
"""
from __future__ import annotations

from rare_al.strategies import active_search, adaptive, basic, rca, rare_targeted
from rare_al.strategies.base import FunctionStrategy
from rare_al.strategies.v1 import committee, coverage, geometry, rareaware, uncertainty

_V1 = {
    "margin": uncertainty.margin,
    "entropy": uncertainty.entropy_select,
    "least_confident": uncertainty.least_confident,
    "random": uncertainty.random_select,
    "robust_uncertainty": uncertainty.robust_uncertainty,
    "uherding": geometry.uherding,
    "kmeans_diversity": geometry.kmeans_diversity,
    "coreset": geometry.coreset,
    "typiclust": geometry.typiclust,
    "probcover": coverage.probcover,
    "committee": committee.committee,
    "constrained_cover": coverage.constrained_cover,
    "rareguard": rareaware.rareguard,
    "dcf": geometry.dcf,
    "adaptive_repulsion": geometry.adaptive_repulsion,
}

_CLASSES = {
    "margin_rtb": basic.MarginRandomTies,
    "committee_learner": basic.CommitteeSameLearner,
    "most_likely_positive": active_search.MostLikelyPositive,
    "ens": active_search.ENS,
    "similar_flqmi": rare_targeted.SimilarFLQMI,
    "galaxy": rare_targeted.Galaxy,
    "direct": rare_targeted.Direct,
    "anchoral": rare_targeted.AnchorAL,
    "class_balanced": rare_targeted.ClassBalanced,
    "tailor": adaptive.Tailor,
    "albl": adaptive.ALBL,
    "rca": rca.RCA,
}

REGISTRY = sorted(set(_V1) | set(_CLASSES))

# name: (family, status, display name, bib key)
METADATA = {
    "margin": ("reference", "v13", "Margin", "lewis1994sequential"),
    "margin_rtb": ("reference", "new", "Margin (random ties)", ""),
    "entropy": ("uncertainty", "v13", "Entropy", ""),
    "least_confident": ("uncertainty", "v13", "Least confident", ""),
    "random": ("passive", "v13", "Random", ""),
    "robust_uncertainty": ("uncertainty", "v13", "Robust uncertainty", ""),
    "committee": ("disagreement", "v13", "Committee (LR members)", "seung1992qbc"),
    "committee_learner": ("disagreement", "new", "Committee (same learner)", "seung1992qbc"),
    "coreset": ("representativeness", "v13", "CoreSet", "sener2018coreset"),
    "kmeans_diversity": ("representativeness", "v13", "K-means diversity", "zhdanov2019diverse"),
    "typiclust": ("representativeness", "v13, adapted", "TypiClust", "hacohen2022typiclust"),
    "probcover": ("representativeness", "v13, adapted", "ProbCover", "yehuda2022probcover"),
    "uherding": ("hybrid", "v13", "U-Herding", "bae2025uherding"),
    "dcf": ("hybrid", "v13, re-impl.", "DCF", "arya2026lcd"),
    "similar_flqmi": ("rare_targeted", "new", "SIMILAR (FLQMI)", "kothawade2021similar"),
    "galaxy": ("rare_targeted", "new, adapted", "GALAXY", "zhang2022galaxy"),
    "direct": ("rare_targeted", "new, adapted", "DIRECT", "nuggehalli2025direct"),
    "anchoral": ("rare_targeted", "new, adapted", "AnchorAL", "lesci2024anchoral"),
    "class_balanced": ("rare_targeted", "new, adapted", "Class-balanced AL", "bengar2022classbalanced"),
    "most_likely_positive": ("active_search", "new", "Most-likely-positive", "attenberg2010search"),
    "ens": ("active_search", "new", "ENS", "jiang2017ens"),
    "albl": ("adaptive", "new", "ALBL", "hsu2015albl"),
    "tailor": ("adaptive", "new, adapted", "TAILOR", "zhang2023tailor"),
    "rca": ("framework_instance", "new", "RCA", ""),
    "constrained_cover": ("probe", "v13", "Constrained coverage", ""),
    "rareguard": ("probe", "v13", "RareGuard", ""),
    "adaptive_repulsion": ("probe", "v13", "Adaptive repulsion", ""),
}


def build(name: str, **params):
    """Instantiate a strategy (one instance per trajectory)."""
    if name in _V1:
        return FunctionStrategy(_V1[name], **params)
    if name in _CLASSES:
        return _CLASSES[name](**params)
    raise KeyError(f"Unknown strategy: {name}")


def family(name: str) -> str:
    return METADATA.get(name, ("other",))[0]


def display(name: str) -> str:
    return METADATA.get(name, (None, None, name))[2]
