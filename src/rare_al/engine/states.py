"""State specifications.

A *state* fixes one dataset realization and one state seed. States are plain
dicts (picklable) and are materialized inside worker processes.
"""
from __future__ import annotations

from rare_al.data.synthetic import make_regime, n_for_ir
from rare_al.repro import stable_seed


def synthetic_states(cfg, section="synthetic_grid", experiment="synthetic"):
    """Controlled grid. Seeds are v1-identical for the v1 grid parameters."""
    p = cfg[section]
    min_rare = p.get("min_rare", 20)
    for regime in p["regimes"]:
        for ir in p["irs"]:
            for d in p["dims"]:
                for sep in p["seps"]:
                    for s in range(p.get("seed_offset", 0), p.get("seed_offset", 0) + p["seeds"]):
                        n = p["n_samples"] if p["n_samples"] != "auto" else n_for_ir(ir, min_rare)
                        seed = stable_seed(cfg["master_seed"], "synthetic-state", regime, ir, d, sep, s)
                        yield {"experiment": experiment, "kind": "synthetic",
                               "dataset": f"syn_{regime}", "domain": "synthetic",
                               "regime": regime, "ir": ir, "d": d, "sep": sep, "seed": s,
                               "n_samples": n, "min_rare": min_rare, "state_seed": seed,
                               "split": "stratified",
                               "state_id": f"{experiment}|{regime}|{ir}|{d}|{sep}|{s}"}


def rarity_states(cfg, section="rarity_stress", experiment="rarity"):
    """Rarity-stress experiment (v1-identical seeds)."""
    p = cfg[section]
    min_rare = p.get("min_rare", 20)
    for ir in p["irs"]:
        for s in range(p["seeds"]):
            n = p["n_samples"] if p["n_samples"] != "auto" else n_for_ir(ir, min_rare)
            seed = stable_seed(cfg["master_seed"], "rarity-state", p["regime"], p["n_samples"],
                               p["dim"], p["sep"], ir, s)
            yield {"experiment": experiment, "kind": "synthetic", "dataset": f"syn_{p['regime']}",
                   "domain": "synthetic", "regime": p["regime"], "ir": ir, "d": p["dim"],
                   "sep": p["sep"], "seed": s, "n_samples": n, "min_rare": min_rare,
                   "state_seed": seed, "split": "stratified",
                   "state_id": f"{experiment}|{p['regime']}|{ir}|{s}"}


def real_states(cfg, manifest, section="real_benchmark", experiment="real"):
    p = cfg[section]
    ids = p["datasets"] if p["datasets"] != "all" else sorted(manifest)
    for did in ids:
        if did not in manifest:
            print(f"[states] {did} not prepared -- skipped", flush=True)
            continue
        m = manifest[did]
        seed_name = did[len("imblearn_"):] if did.startswith("imblearn_") else did  # v1 seeds
        for s in range(p["seeds"]):
            seed = stable_seed(cfg["master_seed"], "real-state", seed_name, s)
            yield {"experiment": experiment, "kind": "real", "dataset": did,
                   "domain": m["domain"], "origin": m.get("origin"), "ir": m["ir_eval"],
                   "ir_nominal": m["ir_nominal"], "d": m["d"], "seed": s, "state_seed": seed,
                   "split": p.get("split_override", {}).get(did, m["split"]),
                   "costs": m.get("costs"), "capacity": m.get("capacity"),
                   "cache_dir": cfg["data"]["cache_dir"],
                   "state_id": f"{experiment}|{did}|{s}"}


def image_states(cfg, section="image_benchmark", experiment="image"):
    """Image tasks (reviewer requirement: two embeddings).

    Each (dataset, backbone, positive_class, target_ir, seed) is a state carrying
    the embedding-cache location and the ImageTaskSpec parameters, so it is
    materialized -- from cached embeddings -- inside the worker and flows through
    the identical engine as the tabular states.
    """
    p = cfg[section]
    cache_dirs = p["embedding_cache_dirs"]
    backbones = p.get("backbones", ["dinov2_vits14", "resnet18"])
    max_pool = p.get("max_pool", 10_000)
    min_rare = p.get("min_rare", cfg.get("active_learning", {}).get("min_rare", 50))
    default_ir = p.get("default_ir", 99)
    for ds in p["datasets"]:
        name = ds["dataset"]
        rarity = ds.get("rarity", "one_vs_rest")
        for backbone in backbones:
            for pc in ds.get("positive_classes", [0]):
                irs = ds.get("target_irs", [default_ir]) if rarity == "one_vs_rest" else [float("nan")]
                for ir in irs:
                    for s in range(p["seeds"]):
                        seed = stable_seed(cfg["master_seed"], "image-state",
                                           name, backbone, str(pc), str(ir), s)
                        yield {
                            "experiment": experiment, "kind": "image",
                            "dataset": name, "domain": ds.get("domain", "image"),
                            "backbone": backbone, "positive_class": str(pc),
                            "rarity": rarity, "ir_target": ir, "seed": s,
                            "state_seed": seed, "split": "stratified",
                            "min_rare": min_rare, "max_pool": max_pool,
                            "cache_dirs": list(cache_dirs),
                            "state_id": f"{experiment}|{name}|{backbone}|{pc}|{ir}|{s}",
                        }


def materialize(state):
    """Return (X, y, t, is_test) for a state."""
    if state["kind"] == "synthetic":
        X, y = make_regime(state["regime"], state["n_samples"], state["d"], state["ir"],
                           state["sep"], state["state_seed"], min_rare=state["min_rare"])
        return X, y, None, None
    if state["kind"] == "image":
        from rare_al.data.images import ImageTaskSpec, make_task
        import numpy as np
        spec = ImageTaskSpec(dataset=state["dataset"], backbone=state["backbone"],
                             rarity=state.get("rarity", "one_vs_rest"),
                             positive_class=state["positive_class"],
                             target_ir=(state["ir_target"] if state["ir_target"] == state["ir_target"]
                                        else 99.0),
                             min_rare=state.get("min_rare", 50),
                             max_pool=state.get("max_pool", 10_000))
        X, y, meta = make_task(spec, state["cache_dirs"],
                               rng=np.random.default_rng(state["state_seed"]))
        if not meta.get("min_rare_satisfied", True):
            raise ValueError(f"image task {state['state_id']} has only {meta['n_rare']} rare "
                             f"(< min_rare); increase max_pool or lower target_ir")
        return X, y, None, None
    from rare_al.data.prepare import load_prepared
    z = load_prepared(state["cache_dir"], state["dataset"])
    return z["X"], z["y"], z.get("t"), z.get("is_test")
