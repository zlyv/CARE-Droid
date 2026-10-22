#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path("artifacts/care-litelocal-p31-c43-directionality-audit-v1")
P31 = Path("artifacts/care-litelocal-p31-c43-source-v1")
FAIL = Path("artifacts/care-litelocal-p31-c43-failure-audit-v1")
CACHE = Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz")
REQUIRED = (
    P31 / "preflight/p31_reference/source_predictions.csv",
    P31 / "arms/P31_C43_RQ/source_predictions.csv",
    FAIL / "routed_errors_mechanism.csv",
    CACHE,
)


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    missing = [str(path) for path in REQUIRED if not path.is_file() or path.stat().st_size < 100]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty locked inputs:\n- " + "\n- ".join(missing))
    reference = read(REQUIRED[0])
    errors = read(REQUIRED[2])
    if len(reference) != 799 or len(errors) != 46:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source/error cohort changed")
    route = [row for row in reference if int(float(row["P31_DI25__routed"])) == 1]
    if len(route) != 200:
        raise RuntimeError("BLOCKED_BY_CONTRACT: P31-DI25 route changed")
    with np.load(CACHE, allow_pickle=False) as saved:
        required = {"sha256", "host_values", "host_offsets", "real_host_count"}
        if not required.issubset(saved.files):
            raise RuntimeError("BLOCKED_BY_CONTRACT: Compact43 cache keys changed")
        if (len(saved["sha256"]) != 799 or saved["host_values"].shape[1] != 105 or
                saved["host_offsets"].shape != (800,)):
            raise RuntimeError("BLOCKED_BY_CONTRACT: Compact43 cache shape changed")
    ROOT.mkdir(parents=True, exist_ok=True)
    print("P31_C43_DIRECTIONALITY_AUDIT_PREFLIGHT_PASS")
    print("source_rows=799 routed_rows=200 routed_errors=46")
    print("host_layout=Compact43+Role31+Relation31 width=105")
    print("backgrounds=outer_train_only class_symmetric APK_equal_weighted role_conditioned")
    print("training_invoked=false inference_invoked=false future_rows_loaded=0 ego_consumed=false")


if __name__ == "__main__":
    main()
