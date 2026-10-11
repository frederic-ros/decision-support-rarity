#!/usr/bin/env python3
"""Fetch the cross-domain ESWA datasets into data/raw/<folder>/ with the exact
filenames the loaders expect, so `rare-al prepare-data` can cache them.

What is automatic vs manual
---------------------------
* UCI sets (taiwan / polish bankruptcy, ai4i2020, aps_failure, secom) download
  and unpack automatically -- no account needed.
* Kaggle sets (creditcard, baf) and UNSW-NB15 download automatically IF the
  `kaggle` CLI is installed and configured (`pip install kaggle`, put
  kaggle.json in ~/.kaggle); otherwise the script prints the one-click page.
* CIC-IDS2017 (corrected LYCOS-IDS2017) and ISIC-2020 embeddings are manual
  (licence / your own extraction) -- the script prints exactly where to drop them.

At the end it verifies every expected file and prints a checklist, then tells you
to run `rare-al prepare-data`.

Usage
-----
    python scripts/fetch_domains.py                    # fetch everything it can
    python scripts/fetch_domains.py --raw-dir data/raw
    python scripts/fetch_domains.py --only secom aps_failure
    python scripts/fetch_domains.py --list             # just show the plan
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

UCI = "https://archive.ics.uci.edu/static/public"

# key -> plan. `expect` = basenames that must end up in data/raw/<folder>/.
# `pick` maps an expected basename to the glob(s) used to find it inside the
# downloaded archive (case-insensitive, searched recursively).
DATASETS = {
    # ---- UCI (automatic, no account) ----------------------------------------
    "taiwan_bankruptcy": {
        "folder": "taiwan_bankruptcy", "method": "uci",
        "url": f"{UCI}/572/taiwanese+bankruptcy+prediction.zip",
        "expect": ["data.csv"], "pick": {"data.csv": ["*.csv"]},
        "page": "https://archive.ics.uci.edu/dataset/572",
    },
    "polish_bankruptcy": {
        "folder": "polish_bankruptcy", "method": "uci",
        "url": f"{UCI}/365/polish+companies+bankruptcy+data.zip",
        "expect": ["1year.arff", "2year.arff", "3year.arff", "4year.arff", "5year.arff"],
        "pick": {f"{y}year.arff": [f"{y}year.arff"] for y in range(1, 6)},
        "page": "https://archive.ics.uci.edu/dataset/365",
    },
    "ai4i2020": {
        "folder": "ai4i2020", "method": "uci",
        "url": f"{UCI}/601/ai4i+2020+predictive+maintenance+dataset.zip",
        "expect": ["ai4i2020.csv"], "pick": {"ai4i2020.csv": ["ai4i2020.csv", "*.csv"]},
        "page": "https://archive.ics.uci.edu/dataset/601",
    },
    "aps_failure": {
        "folder": "aps_failure", "method": "uci",
        "url": f"{UCI}/421/aps+failure+at+scania+trucks.zip",
        "expect": ["aps_failure_training_set.csv", "aps_failure_test_set.csv"],
        "pick": {"aps_failure_training_set.csv": ["*training*set*.csv", "*train*.csv"],
                 "aps_failure_test_set.csv": ["*test*set*.csv", "*test*.csv"]},
        "page": "https://archive.ics.uci.edu/dataset/421",
    },
    "secom": {
        "folder": "secom", "method": "uci",
        "url": f"{UCI}/179/secom.zip",
        "expect": ["secom.data", "secom_labels.data"],
        "pick": {"secom.data": ["secom.data"], "secom_labels.data": ["secom_labels.data"]},
        "page": "https://archive.ics.uci.edu/dataset/179",
    },
    # ---- Kaggle (automatic only with the kaggle CLI) ------------------------
    "creditcard": {
        "folder": "creditcard", "method": "kaggle",
        "kaggle": "mlg-ulb/creditcardfraud", "expect": ["creditcard.csv"],
        "page": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
    },
    "baf": {
        "folder": "baf", "method": "kaggle",
        "kaggle": "sgpjesus/bank-account-fraud-dataset-neurips-2022",
        "expect": ["Base.csv"], "pick": {"Base.csv": ["Base.csv", "*ase.csv"]},
        "page": "https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022",
    },
    "unsw_nb15": {
        "folder": "unsw_nb15", "method": "kaggle",
        "kaggle": "mrwellsdavid/unsw-nb15",
        "expect": ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"],
        "pick": {"UNSW_NB15_training-set.csv": ["*training-set*.csv", "*training*set*.csv"],
                 "UNSW_NB15_testing-set.csv": ["*testing-set*.csv", "*test*set*.csv"]},
        "page": "https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15",
    },
    # ---- manual (licence / your own files) ----------------------------------
    "cicids2017": {
        "folder": "cicids2017", "method": "manual",
        "expect": ["*.csv"],
        "note": "Use the CORRECTED LYCOS-IDS2017 CSVs (not the original CIC CSVs). "
                "Drop every day's *.csv straight into the folder.",
        "page": "https://lycos-ids.univ-lemans.fr/",
    },
    "isic2020": {
        "folder": "isic2020", "method": "manual",
        "expect": ["isic2020_embeddings.npz"],
        "note": "Your pre-extracted ISIC-2020 embeddings, saved as an .npz with "
                "arrays X (float) and y (int, 1=melanoma).",
        "page": "https://challenge2020.isic-archive.com/",
    },
}


def _download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "rare-al/2.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _find(root: Path, patterns):
    for pat in patterns:
        hits = sorted(root.rglob(pat))
        hits = [h for h in hits if h.is_file()]
        if hits:
            return hits[0]
    # case-insensitive fallback
    low = {p.name.lower(): p for p in root.rglob("*") if p.is_file()}
    for pat in patterns:
        if pat.lower() in low:
            return low[pat.lower()]
    return None


def _place_from_archive(archive: Path, out: Path, pick):
    """Extract `archive`, then copy each expected file (resolved via its globs)
    into `out`. Nested zips are expanded once."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        with zipfile.ZipFile(archive) as z:
            z.extractall(tdp)
        for inner in list(tdp.rglob("*.zip")):          # one level of nested zips
            try:
                with zipfile.ZipFile(inner) as z:
                    z.extractall(inner.parent / (inner.stem + "_x"))
            except zipfile.BadZipFile:
                pass
        placed, missing = [], []
        for name, globs in pick.items():
            src = _find(tdp, globs)
            if src is None:
                missing.append(name)
                continue
            shutil.copyfile(src, out / name)
            placed.append(name)
    return placed, missing


def fetch_uci(spec, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        zp = Path(td) / "d.zip"
        print(f"    downloading {spec['url']}")
        _download(spec["url"], zp)
        placed, missing = _place_from_archive(zp, out, spec["pick"])
    if placed:
        print(f"    placed: {', '.join(placed)}")
    if missing:
        print(f"    NOT found inside the archive: {', '.join(missing)} "
              f"-- check the layout at {spec['page']}")
    return not missing


def fetch_kaggle(spec, out: Path):
    if shutil.which("kaggle") is None:
        print(f"    kaggle CLI not found -> MANUAL: {spec['page']}")
        print(f"      (pip install kaggle, add ~/.kaggle/kaggle.json, then re-run)")
        return False
    out.mkdir(parents=True, exist_ok=True)
    import subprocess
    print(f"    kaggle datasets download -d {spec['kaggle']}")
    r = subprocess.run(["kaggle", "datasets", "download", "-d", spec["kaggle"],
                        "-p", str(out), "--unzip"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    kaggle failed: {r.stderr.strip().splitlines()[-1] if r.stderr else '?'}")
        print(f"      MANUAL: {spec['page']}")
        return False
    if "pick" in spec:                                   # normalise filenames
        placed, missing = [], []
        for name, globs in spec["pick"].items():
            if (out / name).exists():
                placed.append(name); continue
            src = _find(out, globs)
            if src is not None and src.name != name:
                shutil.copyfile(src, out / name); placed.append(name)
            elif src is None:
                missing.append(name)
        if missing:
            print(f"    downloaded but could not map: {', '.join(missing)}")
    return True


def verify(spec, out: Path):
    ok = []
    for e in spec["expect"]:
        if any(out.glob(e)) if "*" in e else (out / e).exists():
            ok.append(True)
        else:
            ok.append(False)
    return all(ok), [e for e, k in zip(spec["expect"], ok) if not k]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", default="data/domain")
    ap.add_argument("--only", nargs="+", default=None, choices=list(DATASETS))
    ap.add_argument("--list", action="store_true", help="show the plan and exit")
    args = ap.parse_args(argv)

    raw = Path(args.raw_dir)
    keys = args.only or list(DATASETS)

    if args.list:
        for k in keys:
            s = DATASETS[k]
            print(f"{k:20s} [{s['method']:6s}] -> {raw / s['folder']}  ({', '.join(s['expect'])})")
        return

    manual, failed = [], []
    for k in keys:
        s = DATASETS[k]
        out = raw / s["folder"]
        print(f"\n== {k}  ({s['method']}) -> {out}")
        done, miss = verify(s, out) if out.exists() else (False, s["expect"])
        if done:
            print("    already present, skipping")
            continue
        try:
            if s["method"] == "uci":
                fetch_uci(s, out)
            elif s["method"] == "kaggle":
                if not fetch_kaggle(s, out):
                    manual.append(k)
            else:
                out.mkdir(parents=True, exist_ok=True)
                print(f"    MANUAL: {s['note']}")
                print(f"      source: {s['page']}")
                print(f"      drop files into: {out}")
                manual.append(k)
        except Exception as exc:
            print(f"    ERROR: {type(exc).__name__}: {exc}")
            print(f"      get it by hand at {s['page']}")
            failed.append(k)

    print("\n================ checklist ================")
    allok = True
    for k in keys:
        s = DATASETS[k]
        out = raw / s["folder"]
        done, miss = verify(s, out) if out.exists() else (False, s["expect"])
        mark = "OK " if done else ".. "
        allok &= done
        tail = "" if done else f"  (missing: {', '.join(miss)})"
        print(f"  [{mark}] {k:20s} {out}{tail}")
    print("===========================================")
    if manual:
        print(f"Manual still needed: {', '.join(manual)}")
    if allok:
        print("All set -> run:  rare-al prepare-data --config configs/eswa/full.json")
    else:
        print("Fetch/drop the remaining files, then run `rare-al prepare-data`.")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
