#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import tempfile
from pathlib import Path
from modeling import fit_calibrated_rf
import run_validation

rng = np.random.RandomState(1)
x = rng.normal(size=(80, 6))
y = np.tile([0, 1], 40)
groups = np.repeat(np.arange(40), 2)
probability = fit_calibrated_rf(x[:60], y[:60], x[60:], 11, groups_train=groups[:60])
assert probability.shape == (20,)
assert np.isfinite(probability).all()
assert np.all((probability > 0) & (probability < 1))

n = 100
fold = np.repeat(np.arange(1, 6), 20)
labels = np.tile([0, 1], 50)
stage1_pred = labels.copy(); stage1_pred[::7] = 1 - stage1_pred[::7]
stage1 = np.where(stage1_pred == 1, .7, .3)
contexts = {outer: stage1.copy() for outer in range(1, 6)}
features = {
    "COMPACT_MEAN_STD_MAX": rng.normal(size=(n, 129)),
    "COMPACT_MEAN": rng.normal(size=(n, 43)),
    "COMPACT_STD": rng.normal(size=(n, 43)),
    "COMPACT_MAX_ABS": rng.normal(size=(n, 43)),
    "COUNT_ONLY": rng.normal(size=(n, 6)),
}
count = rng.normal(size=(n, 6))
route = np.zeros(n, bool)
for outer in range(1, 6): route[np.flatnonzero(fold == outer)[:5]] = True
ids = np.asarray([f"{index:064x}" for index in range(n)])
with tempfile.TemporaryDirectory() as directory:
    run_validation.OUTPUT = Path(directory)
    result = run_validation.run_compact_protocol(
        "SYNTHETIC", ids, labels, fold, stage1, contexts, features, count, route=route
    )
    assert len(result) == 6 and result[0] and result[1] and result[2] and result[3]
    assert all(np.isfinite(values).all() for values in result[4].values())
print("CARE_GROUP_COMPACT_VALIDATION_INTEGRATION_SELFTEST_PASS")

