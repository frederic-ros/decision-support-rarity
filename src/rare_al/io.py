"""Append-only CSV checkpoints (resumable runs)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def append_rows(rows, path) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if path.exists():
        header = pd.read_csv(path, nrows=0).columns.tolist()
        for c in header:
            if c not in frame:
                frame[c] = pd.NA
        extra = [c for c in frame.columns if c not in header]
        if extra:  # schema grew: rewrite file with union of columns
            old = pd.read_csv(path)
            pd.concat([old, frame], ignore_index=True).to_csv(path, index=False)
            return
        frame = frame[header]
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_keys(path, columns) -> set:
    path = Path(path)
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=columns)
    return set(map(tuple, frame[columns].astype(str).to_numpy()))
