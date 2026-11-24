r"""Keep the last round of every trajectory, for a compact transfer of large runs.

Usage (from the repository root):
    python scripts/revision/extract_final.py results/revision/sensitivity
Writes results/revision/sensitivity_final/<sub>/trajectories.csv (and copies errors.csv),
then the archive results/revision/sensitivity_final.zip.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

KEYS = ["state_id", "learner", "budget", "start", "strategy"]

src = Path(sys.argv[1])
dst = src.parent / (src.name + "_final")
if dst.exists():
    shutil.rmtree(dst)
for f in sorted(src.rglob("trajectories.csv")):
    rel = f.parent.relative_to(src)
    d = pd.concat(pd.read_csv(f, chunksize=300000, low_memory=False), ignore_index=True)
    n_dup = int(d.duplicated(KEYS + ["round"]).sum())
    d = d.drop_duplicates(KEYS + ["round"])
    last = d[d["round"] == d.groupby(KEYS)["round"].transform("max")]
    out = dst / rel
    out.mkdir(parents=True, exist_ok=True)
    last.to_csv(out / "trajectories.csv", index=False)
    if (f.parent / "errors.csv").exists():
        shutil.copy2(f.parent / "errors.csv", out / "errors.csv")
    print(f"{f}: {len(d)} rows -> {len(last)} trajectories, {d.state_id.nunique()} states, "
          f"learners {sorted(d.learner.unique())}, duplicates removed {n_dup}")
shutil.make_archive(str(dst), "zip", root_dir=src.parent, base_dir=dst.name)
print("zip:", str(dst) + ".zip")
