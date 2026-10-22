#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("artifacts/care-litelocal-p31-c43-source-v1")
REQUIRED = (
    Path("artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"),
    Path("artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv"),
    ROOT / "preflight/p31_reference/source_predictions.csv",
    Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),
    Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz"),
    Path("scripts/care-di25-compact43-relationquery-source-v1/compact_models.py"),
    Path("scripts/care-di25-compact43-relationquery-source-v1/compact_background.py"),
    Path("scripts/care-di25-graphglobal-path-source-v1/source_label_contract.py"),
    Path("scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py"),
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
    reference = read(ROOT / "preflight/p31_reference/source_predictions.csv")
    ids = [row["sha256"].strip().casefold() for row in reference]
    fold = np.asarray([int(float(row["fold"])) for row in reference])
    route = np.asarray([int(float(row["P31_DI25__routed"])) for row in reference])
    if len(reference) != 799 or len(set(ids)) != 799:
        raise RuntimeError("BLOCKED_BY_CONTRACT: P31 reference cohort changed")
    if int(route.sum()) != 200 or any(int(route[fold == value].sum()) != 40
                                      for value in range(1, 6)):
        raise RuntimeError("BLOCKED_BY_CONTRACT: P31 route changed")
    with np.load(REQUIRED[0], allow_pickle=False) as values:
        if values["role_prevalence"].shape != (799, 31):
            raise RuntimeError("BLOCKED_BY_CONTRACT: P31 feature shape changed")
    forbidden = (
        "future_interz.npz", "future_regions_2018_2021.npz",
        "endpoint_centered_tensor_cache_2018_2021.jsonl.gz",
        "within_year_router_inputs.csv",
    )
    runtime = "\n".join(
        (Path(__file__).parent / name).read_text(encoding="utf-8")
        for name in ("run_p31_c43.py", "nested_p31.py", "build_p31_reference.py")
    )
    if any(token in runtime for token in forbidden):
        raise RuntimeError("BLOCKED_BY_CONTRACT: future/within-year dependency in runtime")
    contract = json.loads((ROOT / "preflight/p31_contract.json").read_text())
    if contract["stage1_representation"] != "P31 only (31-D sensitive-role method prevalence)":
        raise RuntimeError("BLOCKED_BY_CONTRACT: Stage-I is not P31")
    print("P31_C43_SOURCE_PREFLIGHT_PASS")
    print("source_rows=799 year=2017 fold_sizes=1:160,2:160,3:160,4:160,5:159")
    print("stage1=P31_MEAN_COMMITTEE route_P31_DI25=200 route_per_fold=40,40,40,40,40")
    print("stage2=C43_RELATION_QUERY stage2_train_scope=all_outer_train")
    print("outer_train_stage1=nested_inner_OOF future_rows_loaded=0 ego_consumed=false")


if __name__ == "__main__":
    main()
