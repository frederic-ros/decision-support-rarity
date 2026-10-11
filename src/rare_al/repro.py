"""Deterministic seeding and environment manifests (unchanged from v1 for compatibility)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def stable_seed(master_seed, *parts) -> int:
    """Hash-based seed. Identical to v1 so that v1 states can be regenerated exactly."""
    key = "|".join(map(str, (master_seed, *parts)))
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def environment_manifest() -> dict:
    versions = {}
    for name in ["numpy", "pandas", "scipy", "sklearn", "imblearn", "matplotlib",
                 "statsmodels", "lightgbm", "xgboost", "tabpfn"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = "unavailable"
    from rare_al import __version__
    return {
        "rare_al": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": commit,
        "versions": versions,
        "thread_env": {k: os.environ.get(k) for k in
                       ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]},
    }
