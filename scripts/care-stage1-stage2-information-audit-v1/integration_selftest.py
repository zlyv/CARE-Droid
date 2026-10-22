#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import run_audit

rng = np.random.RandomState(17)
n = 100
fold = np.repeat(np.arange(1, 6), 20)
y = np.tile([0, 1], 50)
p31 = rng.gamma(1.5, 0.2, size=(n, 31))
bags = [rng.normal(size=(1 + (index % 4), 105)) for index in range(n)]
for bag in bags:
    bag[:, 43:74] = np.abs(bag[:, 43:74])
representations = run_audit.aggregate_representations(p31, bags)
stage1_pred = y.copy()
stage1_pred[::7] = 1 - stage1_pred[::7]
stage1_probability = np.where(stage1_pred == 1, 0.7, 0.3)
route = np.zeros(n, bool)
for held in range(1, 6):
    route[np.flatnonzero(fold == held)[:5]] = True
delta = rng.normal(scale=0.2, size=n)
ids = [f"{index:064x}" for index in range(n)]

with tempfile.TemporaryDirectory() as directory:
    run_audit.OUTPUT = Path(directory)
    aggregate, fold_rows, nulls, predictions = run_audit.stage2_probe(
        representations, y, fold, stage1_probability, route, delta, ids
    )
    assert aggregate and fold_rows and nulls
    assert all(np.isfinite(value).all() for value in predictions.values())
    assert (Path(directory) / "stage2_information_ceiling.csv").is_file()
    assert (Path(directory) / "stage2_nearest_neighbor_collisions.csv").is_file()

print("CARE_INFORMATION_AUDIT_INTEGRATION_SELFTEST_PASS")

