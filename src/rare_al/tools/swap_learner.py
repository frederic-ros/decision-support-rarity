"""Build a results tree in which one learner's rows are replaced by another's.

Revision (LR intercept). The reporting pipeline (``rare-al analyze``) names the linear
learner ``logreg``. To rebuild every main table and figure with the corrected linear
learner (``logreg_lbfgs``: same model, unpenalized intercept) without touching the
reporting code, this tool copies ``--base`` (e.g. results/campaign) to ``--out``, drops
the rows of ``--drop`` (``logreg``, liblinear) from every CSV with a ``learner`` column,
appends the rows of ``--learner`` (``logreg_lbfgs``) found under ``--add`` renamed to
``--drop``, and records the substitution in ``SWAP.json`` and in a ``learner_impl`` column.

States that the add tree has and the base tree lacks for ``--drop`` (for instance image
datasets embedded after the campaign) are left out by default, so that every learner is
compared on the same states (``--keep-extra-states`` keeps them).

Rows of noise-replicate aliases (``margin_rep*``) are left out of the merged tree by
default (``--keep-replicates`` keeps them, for the revision tree).

A section is ``sub`` or ``base_sub:add_sub`` when the two trees differ, e.g.
``noise_real/real:real`` takes the lbfgs Margin replicates of ``add/real`` for
``base/noise_real/real``. Base and add may be the same tree (oracle files that already
hold both learners).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

SECTIONS = ("synthetic", "rarity", "real", "image", "oracle", "oracle_campaign", "observability")


def _swap_csv(base_csv: Path | None, add_csv: Path | None, out_csv: Path, drop: str, learner: str,
              keep_replicates: bool, keep_extra_states: bool = False) -> dict:
    parts, info = [], {"file": str(out_csv)}
    base_states = None
    if base_csv is not None and base_csv.exists():
        b = pd.read_csv(base_csv, low_memory=False)
        if "learner" in b and "state_id" in b and (b.learner == drop).any() and out_csv.name != "errors.csv":
            base_states = set(b.loc[b.learner == drop, "state_id"])
        if "learner" in b:
            info["base_dropped"] = int((b.learner == drop).sum())
            b = b[~b.learner.isin([drop, learner])].copy()     # base and add may be the same file
            b["learner_impl"] = b.learner
        parts.append(b)
    if add_csv is not None and add_csv.exists():
        a = pd.read_csv(add_csv, low_memory=False)
        if "learner" not in a and parts:          # learner-free file (e.g. descriptors): keep base
            info["kept"] = "base"
        elif "learner" not in a:
            parts.append(a)
        else:
            a = a[a.learner == learner].copy()
            if base_states is not None and not keep_extra_states and "state_id" in a:
                extra = set(a.state_id) - base_states       # e.g. datasets added since the campaign
                if extra:
                    info["extra_states_dropped"] = len(extra)
                    a = a[a.state_id.isin(base_states)]
            if not keep_replicates and "strategy" in a:
                a = a[~a.strategy.astype(str).str.startswith("margin_rep")]
            a["learner_impl"] = learner
            a["learner"] = drop
            info["added"] = int(len(a))
            parts.append(a)
    if not parts:
        return info
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True, sort=False).to_csv(out_csv, index=False)
    return info


def swap_learner(base, add, out, drop="logreg", learner="logreg_lbfgs", sections=SECTIONS,
                 keep_replicates=False, require_complete=True, clean=True, keep_extra_states=False):
    """Returns the list of per-file summaries; raises if a section of ``--add`` lacks
    states that ``--base`` has for ``drop`` (unless ``require_complete`` is False)."""
    base, add, out = Path(base), Path(add), Path(out)
    if clean and out.exists():
        shutil.rmtree(out)
    log = []
    for spec in sections:
        sec, _, asec = spec.partition(":")
        bdir, adir = base / sec, add / (asec or sec)
        if not bdir.exists() and not adir.exists():
            continue
        names = sorted({p.name for d in (bdir, adir) if d.exists() for p in d.glob("*.csv")})
        for name in names:
            bcsv, acsv = bdir / name, adir / name
            if require_complete and name == "trajectories.csv" and bcsv.exists():
                if not acsv.exists():
                    raise FileNotFoundError(f"{acsv} missing: run the {learner} config for '{sec}'")
                bs = set(pd.read_csv(bcsv, usecols=["learner", "state_id"]).query("learner == @drop").state_id)
                a = pd.read_csv(acsv, usecols=["learner", "state_id"])
                miss = bs - set(a[a.learner == learner].state_id)
                if miss:
                    raise RuntimeError(f"{sec}: {len(miss)} of {len(bs)} {drop} states have no {learner} run "
                                       f"(e.g. {sorted(miss)[:2]}); finish the run or pass --allow-partial")
            log.append(_swap_csv(bcsv if bcsv.exists() else None, acsv if acsv.exists() else None,
                                 out / sec / name, drop, learner, keep_replicates, keep_extra_states))
        for p in (bdir.glob("*") if bdir.exists() else []):       # non-CSV assets (e.g. caches)
            if p.is_file() and p.suffix != ".csv":
                (out / sec).mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out / sec / p.name)
    out.mkdir(parents=True, exist_ok=True)
    rec = out / "SWAP.json"
    hist = json.loads(rec.read_text()) if (rec.exists() and not clean) else []
    hist = hist if isinstance(hist, list) else [hist]
    hist.append({"base": str(base), "add": str(add), "dropped": drop, "replaced_by": learner,
                 "sections": list(sections), "files": log})
    rec.write_text(json.dumps(hist, indent=2))
    return log
