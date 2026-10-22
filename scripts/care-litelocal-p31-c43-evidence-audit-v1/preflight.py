#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

P31 = Path("artifacts/care-litelocal-p31-c43-source-v1")
FAIL = Path("artifacts/care-litelocal-p31-c43-failure-audit-v1")
CACHE = Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz")
REQUIRED = (
    P31 / "preflight/p31_reference/source_predictions.csv",
    P31 / "arms/P31_C43_RQ/source_predictions.csv",
    Path("artifacts/care-litelocal-p31-c43-direct-source-v1/arms/P31_C43_DIRECT/source_predictions.csv"),
    FAIL / "routed_errors_mechanism.csv",
    CACHE,
)


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    missing = [str(path) for path in REQUIRED
               if not path.is_file() or path.stat().st_size < 100]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty locked input:\n- " +
                           "\n- ".join(missing))
    reference = read(REQUIRED[0])
    errors = read(REQUIRED[3])
    if len(reference) != 799 or len(errors) != 46:
        raise RuntimeError("BLOCKED_BY_CONTRACT: locked cohort changed")
    with np.load(CACHE, allow_pickle=False) as values:
        keys = list(values.files)
        if "sha256" not in keys:
            raise RuntimeError("BLOCKED_BY_CONTRACT: compact cache lacks sha256")
        if len(values["sha256"]) != 799:
            raise RuntimeError("BLOCKED_BY_CONTRACT: compact cache cohort changed")
        shapes = {key: list(values[key].shape) for key in keys}
    print("P31_C43_EVIDENCE_AUDIT_PREFLIGHT_PASS")
    print("source_rows=799 routed_rows=200 routed_errors=46")
    print("compact_cache_keys=" + ",".join(keys))
    print("training_invoked=false inference_invoked=false future_rows_loaded=0")


if __name__ == "__main__":
    main()
