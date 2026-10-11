"""Command-line entry point for rare_al 2.0.

Subcommands
-----------
prepare-data    download / cache the tabular + domain datasets and build the manifest
extract-images  extract the two embeddings (DINOv2, ResNet-18) for each image set
run             run experiments (synthetic / rarity / real / image / oracle / observability)
analyze         build every table and figure from the results

Typical full pipeline (see runners/):
    rare-al prepare-data    --config configs/eswa/full.json
    rare-al extract-images  --config configs/eswa/full.json      # GPU + network
    rare-al run             --config configs/eswa/full.json
    rare-al analyze         --config configs/eswa/full.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rare_al.config import load_config, validate


# --------------------------------------------------------------------- helpers
def _dataset_ids(cfg):
    from rare_al.data import registry
    want = cfg.get("real_benchmark", {}).get("datasets", "all")
    allids = sorted(registry.REGISTRY)
    return allids if want == "all" else [d for d in want if d in registry.REGISTRY]


def _states_for(cfg, which, manifest):
    from rare_al.engine import states as S
    out = []
    if "synthetic" in which and "synthetic_grid" in cfg:
        out += list(S.synthetic_states(cfg))
    if "rarity" in which and "rarity_stress" in cfg:
        out += list(S.rarity_states(cfg))
    if "real" in which and "real_benchmark" in cfg:
        out += list(S.real_states(cfg, manifest))
    if "image" in which and "image_benchmark" in cfg:
        out += list(S.image_states(cfg))
    return out


# --------------------------------------------------------------------- commands
def cmd_prepare_data(args):
    cfg = load_config(args.config)
    from rare_al.data.prepare import prepare
    data = cfg["data"]
    ids = _dataset_ids(cfg)
    print(f"[prepare-data] {len(ids)} datasets -> {data['cache_dir']}", flush=True)
    prepare(ids, raw_dir=args.raw_dir or data.get("raw_dir", "data/raw"),
            cache_dir=data["cache_dir"], master_seed=cfg["master_seed"],
            n_max=data.get("real_n_max", 8000), min_rare=data.get("min_rare", 100),
            force=args.force, rare27_dir=args.rare27_dir or data.get("rare27_dir"))


def cmd_extract_images(args):
    cfg = load_config(args.config)
    img = cfg["image_benchmark"]
    from rare_al.data.images import extract_embeddings
    out_dir = args.out or img["embedding_cache_dirs"][0]
    data_root = args.data_root or img.get("data_root", "data/images_raw")
    names = {d["dataset"] for d in img["datasets"]}
    backbones = img.get("backbones", ["dinov2_vits14", "resnet18"])
    for name in sorted(names):
        for backbone in backbones:
            p = Path(out_dir) / name / f"{backbone}.npz"
            if p.exists() and not args.force:
                print(f"[extract-images] cached {name}/{backbone}", flush=True)
                continue
            print(f"[extract-images] {name}/{backbone} ...", flush=True)
            try:
                _, shape = extract_embeddings(name, backbone, out_dir, data_root=data_root)
                print(f"[extract-images] wrote {name}/{backbone} {shape}", flush=True)
            except Exception as exc:
                print(f"[extract-images] FAILED {name}/{backbone}: {exc}", flush=True)


def cmd_run(args):
    cfg = load_config(args.config)
    validate(cfg)
    from rare_al.data.prepare import read_manifest
    from rare_al.engine.runner import run
    from rare_al.experiments import oracle as oracle_exp
    manifest = read_manifest(cfg["data"]["cache_dir"])
    which = args.only or ["synthetic", "rarity", "real", "image"]
    results = Path(cfg["results_dir"])
    from rare_al.engine import states as S
    for section in which:
        if section == "oracle":
            if "oracle" in cfg:
                osts = list(S.synthetic_states(cfg, section="oracle", experiment="oracle"))
                print(f"[run] oracle: {len(osts)} states", flush=True)
                oracle_exp.run(cfg, osts, results / "oracle")
            continue
        if section == "observability":
            if "observability" in cfg and (results / "oracle" / "oracle_states.csv").exists():
                from rare_al.experiments import observability
                try:
                    observability.run(cfg, results / "oracle", results / "observability",
                                      learner=cfg.get("oracle", {}).get("classifier"))
                except Exception as exc:
                    print(f"[run] observability skipped: {type(exc).__name__}: {exc}", flush=True)
            else:
                print("[run] observability skipped: run oracle first", flush=True)
            continue
        sts = _states_for(cfg, [section], manifest)
        if not sts:
            print(f"[run] no states for '{section}' (section absent or unprepared)", flush=True)
            continue
        print(f"[run] {section}: {len(sts)} states", flush=True)
        run(cfg, sts, results / section)


def cmd_validate(args):
    """Out-of-sample (leave-one-domain-out) validation of the decision procedure."""
    import pandas as pd
    cfg = load_config(args.config)
    results = Path(cfg["results_dir"])
    traj = results / "real" / "trajectories.csv"
    if not traj.exists():
        print(f"[validate] {traj} not found; run the real benchmark first.", flush=True)
        return
    from rare_al.analysis.load import load_trajectories
    from rare_al.analysis.paired import final_rows, paired
    from rare_al.analysis import validation as V

    df = load_trajectories(traj)
    df = df[df.budget == "main"] if "budget" in df else df
    p = paired(final_rows(df), cfg.get("reference_strategy", "margin"))

    # observable (initial-label) descriptors per state, cached
    desc_path = results / "real" / "descriptors.csv"
    if desc_path.exists() and not args.recompute_descriptors:
        desc = pd.read_csv(desc_path)
    else:
        from rare_al.analysis.descriptors import state_descriptors
        from rare_al.data.prepare import read_manifest
        from rare_al.engine.states import real_states
        manifest = read_manifest(cfg["data"]["cache_dir"])
        states = list(real_states(cfg, manifest))
        print(f"[validate] computing observable descriptors for {len(states)} states ...", flush=True)
        desc = state_descriptors(states, cfg)
        desc.to_csv(desc_path, index=False)

    loo = V.leave_one_domain_out(p, desc, reference=cfg.get("reference_strategy", "margin"),
                                 tau=args.tau, k=args.k)
    summ, meta = V.summary(loo)
    out = args.out or (results / "assets")
    (Path(out) / "stats").mkdir(parents=True, exist_ok=True)
    (Path(out) / "tables").mkdir(parents=True, exist_ok=True)
    loo.to_csv(Path(out) / "stats" / "validation_loo.csv", index=False)
    summ.to_csv(Path(out) / "stats" / "validation_summary.csv", index=False)

    name = {"margin": "Always Margin", "mean_rank": "Mean-rank pick",
            "risk_global": "Risk-aware (global)", "risk_cond": "Risk-aware (conditional)",
            "oracle_best": "Best in hindsight"}
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Selection rule & $\Delta$AUPRC [95\% CI] & Win vs Margin & Worst 10\% \\", r"\midrule"]
    for _, r in summ.iterrows():
        lines.append(f"{name.get(r['rule'], r['rule'])} & "
                     f"{r['mean_gain']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] & "
                     f"{100*r['win_rate_vs_margin']:.0f}\\% & {r['worst10']:+.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (Path(out) / "tables" / "tab_validation.tex").write_text("\n".join(lines), encoding="utf-8")

    print("\n[validate] held-out performance by selection rule:")
    print(summ.to_string(index=False))
    print(f"\n[validate] risk-aware(conditional) - mean-rank: "
          f"mean {meta['risk_cond_vs_mean_rank_mean']:+.4f} AUPRC, "
          f"Wilcoxon p={meta['wilcoxon_p']:.3g} over {meta['n_pairs']} held-out datasets")
    print(f"[validate] assets -> {out}")


def cmd_analyze(args):
    cfg = load_config(args.config)
    from rare_al.reporting.assets import build_all
    out = args.out or (Path(cfg["results_dir"]) / "assets")
    report = build_all(out, v13=args.v13, runs=cfg["results_dir"],
                       cache_dir=cfg["data"]["cache_dir"], fast=args.fast)
    print(f"[analyze] {len(report)} assets -> {out}", flush=True)


# --------------------------------------------------------------------- parser
def build_parser():
    ap = argparse.ArgumentParser(prog="rare-al", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-data", help="cache tabular/domain datasets + manifest")
    p.add_argument("--config", required=True)
    p.add_argument("--raw-dir", dest="raw_dir", default=None)
    p.add_argument("--rare27-dir", dest="rare27_dir", default=None,
                   help="folder of Rare27 xlsx (offline moderate stratum, e.g. native/rare27)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("extract-images", help="extract the two embeddings per image set (GPU)")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None, help="embedding cache dir (default: first in config)")
    p.add_argument("--data-root", dest="data_root", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_extract_images)

    p = sub.add_parser("run", help="run experiments")
    p.add_argument("--config", required=True)
    p.add_argument("--only", nargs="+", default=None,
                   choices=["synthetic", "rarity", "real", "image", "oracle", "observability"],
                   help="restrict to these experiments (default: synthetic rarity real image)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze", help="build tables + figures")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--v13", default=None, help="path to a v13 archive for provenance overlay")
    p.add_argument("--fast", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("validate-guidelines",
                       help="leave-one-domain-out out-of-sample validation of the decision procedure")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--tau", type=float, default=0.10, help="material-loss tolerance alpha")
    p.add_argument("--k", type=int, default=5, help="nearest training datasets for the conditional rule")
    p.add_argument("--recompute-descriptors", action="store_true")
    p.set_defaults(func=cmd_validate)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
