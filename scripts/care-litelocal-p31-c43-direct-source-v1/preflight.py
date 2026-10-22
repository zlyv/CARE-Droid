#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path("artifacts/care-litelocal-p31-c43-direct-source-v1")
P31_ROOT = Path("artifacts/care-litelocal-p31-c43-source-v1")
REQUIRED = (
    P31_ROOT / "preflight/p31_reference/source_predictions.csv",
    P31_ROOT / "arms/P31_C43_RQ/source_predictions.csv",
    P31_ROOT / "results/hard_audit.json",
    Path("artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"),
    Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),
    Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz"),
    Path("scripts/care-litelocal-p31-c43-source-v1/nested_p31.py"),
    Path("scripts/care-di25-compact43-relationquery-source-v1/compact_models.py"),
    Path("scripts/care-di25-compact43-relationquery-source-v1/compact_background.py"),
)


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    missing = [str(path) for path in REQUIRED
               if not path.is_file() or path.stat().st_size < 100]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- " +
                           "\n- ".join(missing))
    rows = read(REQUIRED[0])
    fold = np.asarray([int(float(row["fold"])) for row in rows])
    route = np.asarray([int(float(row["P31_DI25__routed"])) for row in rows])
    if len(rows) != 799 or len({row["sha256"].casefold() for row in rows}) != 799:
        raise RuntimeError("BLOCKED_BY_CONTRACT: P31 source cohort changed")
    if int(route.sum()) != 200 or [int(route[fold == value].sum())
                                   for value in range(1, 6)] != [40] * 5:
        raise RuntimeError("BLOCKED_BY_CONTRACT: P31-DI25 changed")
    runtime = (Path(__file__).parent / "run_direct.py").read_text(encoding="utf-8")
    forbidden = ("future_interz.npz", "within_year_router_inputs.csv",
                 "future_regions_2018_2021.npz")
    if any(token in runtime for token in forbidden):
        raise RuntimeError("BLOCKED_BY_CONTRACT: future input in source runner")
    print("P31_C43_DIRECT_SOURCE_PREFLIGHT_PASS")
    print("source_rows=799 year=2017 route_P31_DI25=200")
    print("stage2=C43_RELATION_QUERY output=direct_classification_logit")
    print("loss=standard_BCE fixkeep=false keep_consistency=false")
    print("stage2_train_scope=all_outer_train outer_train_stage1=nested_inner_OOF")
    print("future_rows_loaded=0 ego_consumed=false")


if __name__ == "__main__":
    main()
