"""File loaders for the real-data benchmark.

All loaders return a dict ``{"X", "y", "t" (optional), "is_test" (optional),
"feature_names"}`` with ``y`` in {0,1} (1 = rare class). Raw files are expected
under ``data/raw/<folder>/`` (see docs/DATASETS.md for download sources); nothing
is downloaded automatically except the imbalanced-learn collection.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def tabular_to_numeric(df: pd.DataFrame, max_categories: int = 30) -> pd.DataFrame:
    """Convert a raw feature frame to numeric: numeric coercion + one-hot for categoricals."""
    out = []
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_bool_dtype(col):
            out.append(col.astype(float).rename(c))
            continue
        if pd.api.types.is_numeric_dtype(col):
            out.append(col.astype(float).rename(c))
            continue
        num = pd.to_numeric(col, errors="coerce")
        if num.notna().mean() > 0.95:
            out.append(num.rename(c))
            continue
        col = col.astype(str)
        top = col.value_counts().index[:max_categories]
        col = col.where(col.isin(top), "__other__")
        out.append(pd.get_dummies(col, prefix=c, dtype=float))
    return pd.concat(out, axis=1)


def _finish(df, target_mask, t=None, is_test=None, drop=()):
    feats = df.drop(columns=[c for c in drop if c in df.columns])
    feats = tabular_to_numeric(feats)
    keep = feats.columns[feats.nunique(dropna=True) > 1]   # drop constant columns
    feats = feats[keep]
    res = {"X": feats.to_numpy(float), "y": np.asarray(target_mask, int),
           "feature_names": list(feats.columns)}
    if t is not None:
        res["t"] = np.asarray(t, float)
    if is_test is not None:
        res["is_test"] = np.asarray(is_test, bool)
    return res


def load_csv(folder, file, target, positive, time=None, drop=(), sep=",", na_values=None,
             header_contains=None, **_):
    path = Path(folder) / file
    skip = 0
    if header_contains:  # some files start with license text (e.g. APS)
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if header_contains in line:
                    skip = i
                    break
    df = pd.read_csv(path, sep=sep, na_values=na_values, skiprows=skip, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    y = (df[target].astype(str).str.strip() == str(positive)).astype(int)
    t = df[time].to_numpy() if time else None
    return _finish(df, y, t=t, drop=[target, *drop] + ([time] if time else []))


def load_train_test_csv(folder, files, target, positive, drop=(), na_values=None,
                        header_contains=None, filter_fn=None, target_fn=None, **_):
    frames = []
    for i, f in enumerate(files):
        path = Path(folder) / f
        skip = 0
        if header_contains:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                for j, line in enumerate(fh):
                    if header_contains in line:
                        skip = j
                        break
        d = pd.read_csv(path, na_values=na_values, skiprows=skip, low_memory=False)
        d.columns = [str(c).strip() for c in d.columns]
        d["__is_test__"] = i == 1
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if filter_fn is not None:
        df = df[filter_fn(df)].reset_index(drop=True)
    y = target_fn(df) if target_fn else (df[target].astype(str).str.strip() == str(positive))
    return _finish(df, y.astype(int), is_test=df["__is_test__"].to_numpy(),
                   drop=[target, "__is_test__", *drop])


def load_folder_csv(folder, pattern, label_col, keep_fn, target_fn, drop=(), **_):
    """Concatenate all CSVs of a folder (e.g. CIC-IDS2017 daily files)."""
    paths = sorted(Path(folder).glob(pattern))
    if not paths:                                  # empty/absent folder -> skip cleanly
        raise FileNotFoundError(f"no files matching {pattern!r} in {folder}")
    frames = []
    for p in paths:
        d = pd.read_csv(p, low_memory=False, encoding="latin-1")
        d.columns = [str(c).strip() for c in d.columns]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[keep_fn(df[label_col])].reset_index(drop=True)
    return _finish(df, target_fn(df[label_col]).astype(int), drop=[label_col, *drop])


def load_arff(folder, file, target, positive, **_):
    from scipy.io import arff
    data, _meta = arff.loadarff(Path(folder) / file)
    df = pd.DataFrame(data)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
    y = (df[target].astype(str).str.strip() == str(positive)).astype(int)
    return _finish(df, y, drop=[target])


def load_secom(folder, **_):
    folder = Path(folder)
    X = pd.read_csv(folder / "secom.data", sep=r"\s+", header=None, na_values="NaN")
    lab = pd.read_csv(folder / "secom_labels.data", sep=r"\s+", header=None)
    y = (lab[0] == 1).astype(int)             # 1 = fail (rare), -1 = pass
    # the timestamp may be one quoted field (col 1) or split into date+time (cols 1,2)
    t = None
    if lab.shape[1] > 1:
        tstr = lab.iloc[:, 1:].astype(str).agg(" ".join, axis=1)
        ts = pd.to_datetime(tstr, dayfirst=True, errors="coerce")
        if ts.notna().all():
            t = ts.astype("int64").to_numpy() // 10**9
    return _finish(X, y, t=t)


def load_npz(folder, file, **_):
    z = np.load(Path(folder) / file, allow_pickle=False)
    y = np.asarray(z["y"]).astype(int).ravel()
    X = np.asarray(z["X"], float)
    return {"X": X, "y": (y == 1).astype(int),
            "feature_names": [f"x{i}" for i in range(X.shape[1])]}


def load_imblearn(name, **_):
    from imblearn.datasets import fetch_datasets
    data = fetch_datasets(filter_data=(name,))[name]
    y = np.asarray(data.target).ravel()
    labels, counts = np.unique(y, return_counts=True)
    rare_label = labels[np.argmin(counts)]
    X = np.asarray(data.data, float)
    return {"X": X, "y": (y == rare_label).astype(int),
            "feature_names": [f"x{i}" for i in range(X.shape[1])]}


def load_rare27_xlsx(folder, file, **_):
    """Offline moderate stratum: a Rare27 export (``feature_*`` columns + a
    ``target_binary`` column, 1 = minority). Avoids the imbalanced-learn network
    fetch when the xlsx are already on disk (e.g. native/rare27/)."""
    path = Path(folder) / file
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    feats = [c for c in df.columns if c.startswith("feature_")]
    if not feats:                                  # fall back: all but target cols
        feats = [c for c in df.columns if c not in ("target_binary", "target_original")]
    y = df["target_binary"].astype(int).to_numpy()
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return {"X": X, "y": (y == 1).astype(int), "feature_names": feats}


LOADERS = {
    "csv": load_csv,
    "train_test_csv": load_train_test_csv,
    "folder_csv": load_folder_csv,
    "arff": load_arff,
    "secom": load_secom,
    "npz": load_npz,
    "imblearn": load_imblearn,
    "rare27_xlsx": load_rare27_xlsx,
}
