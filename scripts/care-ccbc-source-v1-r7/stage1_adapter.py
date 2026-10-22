#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from common import PROJECT


def load_core():
    candidates = [
        PROJECT / "scripts/care-droid-formal-diagnostics-v1/stage1_core.py",
        PROJECT / "scripts/care-p31-stage1-rf-selection-source-v1/stage1_core.py",
    ]
    for path in candidates:
        if path.is_file():
            sys.path.insert(0, str(path.parent))
            import stage1_core
            return stage1_core
    raise RuntimeError("BLOCKED_BY_CONTRACT: official P31 Stage-I core missing")


def strict_nested_committee(x, y, seed: int, jobs: int = 4):
    core = load_core()
    x = np.asarray(x, float); y = np.asarray(y, int)
    output = np.full((len(y), 4), np.nan)
    splitter = StratifiedKFold(4, shuffle=True, random_state=int(seed))
    for split_index, (fit, hold) in enumerate(splitter.split(x, y), 1):
        output[hold] = core.fit_calibrated_committee(
            x[fit], y[fit], x[hold], int(seed) + 100000 * split_index,
            inner_folds=3, jobs=jobs,
        )
    if not np.isfinite(output).all():
        raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete strict nested committee")
    rf_index = list(core.MEMBERS).index("RF")
    rf = output[:, rf_index]
    di = core.mean_opposition(output, rf)
    return output, rf, di

