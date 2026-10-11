"""Fabricate tiny fixtures so the smoke config runs offline, on CPU, in ~2 min.

Writes:
  * data/image_cache/cifar10/{dinov2_vits14,resnet18}.npz  -- 10-class blobs
  * data/cache_smoke/{imblearn_mammography,imblearn_oil}.npz + manifest.json

These are NOT real data -- they only exercise the full pipeline end to end. For
real results run `rare-al prepare-data` + `rare-al extract-images`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def make_image_cache():
    root = ROOT / "data/image_cache/cifar10"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for backbone, dim in [("dinov2_vits14", 384), ("resnet18", 512)]:
        n = 5000
        y = rng.integers(0, 10, n)
        centers = rng.normal(0, 2.5, (10, dim))
        X = centers[y] + rng.normal(0, 1.0, (n, dim))
        np.savez_compressed(root / f"{backbone}.npz", X=X.astype(np.float32), y=y)
    print("image cache:", root)


def make_real_cache():
    cache = ROOT / "data/cache_smoke"
    cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    man = []
    for did, ir, dim in [("imblearn_mammography", 40, 6), ("imblearn_oil", 20, 8)]:
        n = 2500
        n_rare = n // (ir + 1)
        y = np.r_[np.zeros(n - n_rare, int), np.ones(n_rare, int)]
        X = rng.normal(0, 1, (n, dim))
        X[y == 1] += rng.normal(1.2, 0.5, dim)
        np.savez_compressed(cache / f"{did}.npz", X=X.astype(np.float32), y=y)
        man.append({"dataset": did, "domain": "medicine", "rare_event": "minority",
                    "origin": "synthetic-stub", "split": "stratified",
                    "path": str(cache / f"{did}.npz"), "n_original": n,
                    "n_rare_original": int(y.sum()), "ir_nominal": (n - y.sum()) / y.sum(),
                    "n_eval": n, "n_rare_eval": int(y.sum()), "d": dim,
                    "ir_eval": (n - y.sum()) / y.sum(), "costs": None, "capacity": None,
                    "reference": "", "source": "", "verify": "", "ir_compression": 1.0})
    (cache / "manifest.json").write_text(json.dumps(man, indent=2, default=str))
    print("real cache:", cache)


if __name__ == "__main__":
    make_image_cache()
    make_real_cache()
    print("smoke fixtures ready")
