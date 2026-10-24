"""Dataset registry for the ESWA real-data benchmark.

Each entry documents the application domain, the rare event, the expected raw
file layout, the split mode and, where documented, the misclassification costs.
Fields marked ``verify`` must be checked against the dataset documentation before
the final runs (prevalences are nominal values from the documentation).

``origin`` distinguishes natural rare events from one-vs-rest recodings of
balanced multiclass problems and thresholded targets -- a distinction the paper
now reports explicitly for the imbalanced-learn stratum.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetSpec:
    id: str
    domain: str
    rare_event: str
    loader: str
    folder: str = ""
    params: dict = field(default_factory=dict)
    split: str = "stratified"               # stratified | temporal | provided
    nominal_prevalence: float | None = None
    origin: str = "natural"                 # natural | ovr_recoding | thresholded
    costs: dict | None = None               # {"cost_fp": .., "cost_fn": ..}
    capacity: float | None = None           # investigation capacity (fraction flagged)
    reference: str = ""
    source: str = ""
    verify: str = ""


REGISTRY: dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec) -> None:
    REGISTRY[spec.id] = spec


# ------------------------------------------------------------------ finance
register(DatasetSpec(
    id="creditcard", domain="finance", rare_event="fraudulent card transaction",
    loader="csv", folder="creditcard",
    params=dict(file="creditcard.csv", target="Class", positive="1", time="Time"),
    split="temporal", nominal_prevalence=0.00172, capacity=0.005,
    reference="dalpozzolo2018credit",
    source="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (ODbL)"))

register(DatasetSpec(
    id="baf_base", domain="finance", rare_event="fraudulent bank-account application",
    loader="csv", folder="baf",
    params=dict(file="Base.csv", target="fraud_bool", positive="1", time="month"),
    split="temporal", nominal_prevalence=0.011, capacity=0.02, reference="jesus2022baf",
    source="https://github.com/feedzai/bank-account-fraud (CC BY-NC-SA 4.0)",
    verify="prevalence; large file (1M rows): subsampled"))

register(DatasetSpec(
    id="taiwan_bankruptcy", domain="finance", rare_event="company bankruptcy",
    loader="csv", folder="taiwan_bankruptcy",
    params=dict(file="data.csv", target="Bankrupt?", positive="1"),
    nominal_prevalence=0.032, reference="liang2016bankruptcy",
    source="UCI ML Repository: Taiwanese Bankruptcy Prediction (CC BY 4.0)"))

for _year in range(1, 6):
    register(DatasetSpec(
        id=f"polish_bankruptcy_y{_year}", domain="finance", rare_event="company bankruptcy",
        loader="arff", folder="polish_bankruptcy",
        params=dict(file=f"{_year}year.arff", target="class", positive="1"),
        nominal_prevalence=None, reference="zieba2016bankruptcy",
        source="UCI ML Repository: Polish companies bankruptcy (CC BY 4.0)",
        verify="prevalence per year (approx. 2-7%)"))

# ------------------------------------------------------------------ industry
register(DatasetSpec(
    id="aps_failure", domain="industry", rare_event="air-pressure system failure",
    loader="train_test_csv", folder="aps_failure",
    params=dict(files=["aps_failure_training_set.csv", "aps_failure_test_set.csv"],
                target="class", positive="pos", na_values=["na"], header_contains="class,"),
    split="provided", nominal_prevalence=0.0167, costs={"cost_fp": 10.0, "cost_fn": 500.0},
    capacity=0.03, source="UCI ML Repository: APS Failure at Scania Trucks (GPL 3)",
    verify="cost values 10/500 and header offset in the raw files"))

register(DatasetSpec(
    id="ai4i2020", domain="industry", rare_event="machine failure",
    loader="csv", folder="ai4i2020",
    params=dict(file="ai4i2020.csv", target="Machine failure", positive="1",
                drop=("UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF")),
    nominal_prevalence=0.0339, reference="matzka2020ai4i",
    source="UCI ML Repository: AI4I 2020 Predictive Maintenance (CC BY 4.0)",
    verify="failure-mode columns are dropped to avoid target leakage"))

register(DatasetSpec(
    id="secom", domain="industry", rare_event="wafer failure", loader="secom", folder="secom",
    nominal_prevalence=0.066,
    source="UCI ML Repository: SECOM (CC BY 4.0)"))

# ------------------------------------------------------------------ medicine
register(DatasetSpec(
    id="isic2020_emb", domain="medicine", rare_event="melanoma",
    loader="npz", folder="isic2020", params=dict(file="isic2020_embeddings.npz"),
    nominal_prevalence=0.0176, reference="rotemberg2021isic",
    source="ISIC 2020 Challenge; embeddings from a frozen encoder (scripts/make_isic_embeddings.py)",
    verify="encoder choice and version; licence CC BY-NC"))

# --------------------------------------------------------------- cybersecurity
_UNSW = ["Worms", "Shellcode", "Backdoor", "Analysis"]
for _cat in _UNSW:
    register(DatasetSpec(
        id=f"unsw_nb15_{_cat.lower()}", domain="cybersecurity",
        rare_event=f"{_cat} attack vs normal traffic", loader="train_test_csv", folder="unsw_nb15",
        params=dict(files=["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"],
                    target="attack_cat", positive=_cat, drop=("id", "label"),
                    filter_fn=(lambda c: lambda d: d["attack_cat"].astype(str).str.strip()
                               .isin(["Normal", c]))(_cat)),
        split="provided", reference="moustafa2015unsw",
        source="UNSW-NB15 (UNSW Canberra; research use)",
        verify="prevalence of each category after filtering"))

_CIC = {"heartbleed": "heartbleed", "infiltration": "infiltration", "sql_injection": "sql injection"}
for _key, _pat in _CIC.items():
    register(DatasetSpec(
        id=f"cicids2017_{_key}", domain="cybersecurity", rare_event=f"{_key} vs benign traffic",
        loader="folder_csv", folder="cicids2017",
        params=dict(pattern="*.csv", label_col="Label",
                    keep_fn=(lambda p: lambda s: s.astype(str).str.upper().eq("BENIGN")
                             | s.astype(str).str.lower().str.contains(p))(_pat),
                    target_fn=(lambda p: lambda s: s.astype(str).str.lower().str.contains(p))(_pat)),
        reference="sharafaldin2018cicids", source="CIC-IDS2017 (UNB CIC)",
        verify="very large: benign majority subsampled with preserved prevalence"))

# -------------------------------------------------------- imbalanced-learn stratum
_IMBLEARN_ORIGIN = {
    "ecoli": "ovr_recoding", "optical_digits": "ovr_recoding", "satimage": "ovr_recoding",
    "pen_digits": "ovr_recoding", "abalone": "ovr_recoding", "sick_euthyroid": "natural",
    "spectrometer": "thresholded", "car_eval_34": "ovr_recoding", "isolet": "ovr_recoding",
    "us_crime": "thresholded", "yeast_ml8": "ovr_recoding", "scene": "ovr_recoding",
    "libras_move": "ovr_recoding", "thyroid_sick": "natural", "coil_2000": "natural",
    "arrhythmia": "ovr_recoding", "solar_flare_m0": "thresholded", "oil": "natural",
    "car_eval_4": "ovr_recoding", "wine_quality": "thresholded", "letter_img": "ovr_recoding",
    "yeast_me2": "ovr_recoding", "webpage": "natural", "ozone_level": "natural",
    "mammography": "natural", "protein_homo": "natural", "abalone_19": "thresholded",
}
_MEDICAL = {"sick_euthyroid", "thyroid_sick", "mammography", "arrhythmia"}
for _name, _origin in _IMBLEARN_ORIGIN.items():
    register(DatasetSpec(
        id=f"imblearn_{_name}", domain="medicine" if _name in _MEDICAL else "moderate_stratum",
        rare_event="minority class", loader="imblearn", params=dict(name=_name),
        origin=_origin, reference="lemaitre2017imbalanced",
        source="imbalanced-learn benchmark collection (Zenodo)",
        verify="origin labels (natural / one-vs-rest / thresholded)"))


def register_adbench(names, folder="adbench", domain="cross_domain"):
    """ADBench classical datasets, stored as <folder>/<name>.npz with arrays X, y."""
    for n in names:
        register(DatasetSpec(id=f"adbench_{n}", domain=domain, rare_event="anomaly",
                             loader="npz", folder=folder, params=dict(file=f"{n}.npz"),
                             reference="han2022adbench", source="ADBench (Han et al., 2022)"))


def register_custom(entry: dict) -> DatasetSpec:
    """Register a dataset described in a configuration file (generic CSV/NPZ)."""
    spec = DatasetSpec(**entry)
    register(spec)
    return spec


def get(dataset_id: str) -> DatasetSpec:
    if dataset_id not in REGISTRY:
        raise KeyError(f"Unknown dataset '{dataset_id}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[dataset_id]
