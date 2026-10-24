"""Build the cached, rarity-preserving versions of the real datasets + manifest."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rare_al.data import loaders, registry
from rare_al.data.subsample import stratified_subsample, target_size
from rare_al.repro import ensure_dir, stable_seed


def prepare(dataset_ids, raw_dir, cache_dir, master_seed, n_max=8000, min_rare=100,
            n_hard_max=60000, force=False, rare27_dir=None):
    cache_dir = ensure_dir(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    manifest = {m["dataset"]: m for m in json.loads(manifest_path.read_text())} \
        if manifest_path.exists() else {}
    for did in dataset_ids:
        out = cache_dir / f"{did}.npz"
        if out.exists() and did in manifest and not force:
            continue
        spec = registry.get(did)
        folder = Path(raw_dir) / spec.folder
        loader, params = spec.loader, dict(spec.params)
        # offline moderate stratum: use a local Rare27 xlsx if one is provided
        if rare27_dir and did.startswith("imblearn_"):
            name = did[len("imblearn_"):]
            xlsx = Path(rare27_dir) / f"{name}.xlsx"
            if xlsx.exists():
                loader, params, folder = "rare27_xlsx", {"file": xlsx.name}, Path(rare27_dir)
        try:
            data = loaders.LOADERS[loader](folder=folder, **params)
        except FileNotFoundError as exc:
            print(f"[prepare] SKIP {did}: raw file missing ({exc})", flush=True)
            continue
        X, y = data["X"], data["y"]
        n, n_rare = len(y), int(y.sum())
        size = target_size(n, n_rare, n_max, min_rare, n_hard_max)
        strata = y * 2 + data["is_test"].astype(int) if "is_test" in data else None
        keep = stratified_subsample(y, size, stable_seed(master_seed, "subsample", did), strata)
        arrays = {"X": X[keep], "y": y[keep]}
        for extra in ("t", "is_test"):
            if extra in data:
                arrays[extra] = data[extra][keep]
        np.savez_compressed(out, **arrays)
        ye = arrays["y"]
        manifest[did] = {
            "dataset": did, "domain": spec.domain, "rare_event": spec.rare_event,
            "origin": spec.origin, "split": spec.split, "path": str(out),
            "n_original": n, "n_rare_original": n_rare,
            "ir_nominal": (n - n_rare) / max(n_rare, 1),
            "n_eval": int(len(ye)), "n_rare_eval": int(ye.sum()), "d": int(X.shape[1]),
            "ir_eval": float((len(ye) - ye.sum()) / max(ye.sum(), 1)),
            "costs": spec.costs, "capacity": spec.capacity,
            "reference": spec.reference, "source": spec.source, "verify": spec.verify,
        }
        manifest[did]["ir_compression"] = manifest[did]["ir_nominal"] / max(manifest[did]["ir_eval"], 1e-12)
        print(f"[prepare] {did}: n={n}->{len(ye)} IR={manifest[did]['ir_nominal']:.1f}"
              f"->{manifest[did]['ir_eval']:.1f}", flush=True)
        # write incrementally so a later failure never loses finished datasets
        manifest_path.write_text(json.dumps(list(manifest.values()), indent=2, default=str))
    manifest_path.write_text(json.dumps(list(manifest.values()), indent=2, default=str))
    return manifest


def load_prepared(cache_dir, did):
    p = Path(cache_dir) / f"{did}.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `rare-al prepare-data` first.")
    z = np.load(p)
    return {k: z[k] for k in z.files}


def read_manifest(cache_dir):
    p = Path(cache_dir) / "manifest.json"
    return {m["dataset"]: m for m in json.loads(p.read_text())} if p.exists() else {}
