#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from common import (
    FORMAL_FREEZE, ROOT, SEALED_FEATURE_CANDIDATES,
    SOURCE_A0_CACHE, SOURCE_N, SOURCE_P31, SOURCE_REFERENCE, SCRIPTS,
    canonicalize_fold_assignments, discover_base_relation_cache,
    discover_endpoint_cache, discover_fold_assignment_rows,
    discover_future_inter_cache, discover_p31_cache, dump, first_existing,
    load_future_cohort, read_csv, write_csv,
)


def require(path, label):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {label}: {path}")


def main():
    for path, label in (
        (FORMAL_FREEZE, "formal freeze manifest"), (SOURCE_P31, "source P31 cache"),
        (SOURCE_REFERENCE, "P31-RF reference"), (SOURCE_A0_CACHE, "formal A0 cache"),
        (SCRIPTS / "care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py", "C43 v4"),
        (SCRIPTS / "care-p31rf-c43-haig-host-ablation-source-v1/haig_models.py", "A0 model"),
        (SCRIPTS / "care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py", "A0 background"),
    ):
        require(path, label)
    freeze = json.loads(FORMAL_FREEZE.read_text(encoding="utf-8"))
    if not freeze.get("freeze_complete") or not freeze.get("reproduction_pass"):
        raise RuntimeError("BLOCKED_BY_CONTRACT: CARE_DROID_FORMAL_V1 is not frozen/reproduced")
    source_rows = read_csv(SOURCE_REFERENCE)
    if len(source_rows) != SOURCE_N or sum(int(r["A25__routed"]) for r in source_rows) != 200:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen P31-RF/DI25 source reference changed")
    with np.load(SOURCE_A0_CACHE, allow_pickle=False) as z:
        if len(z["sha256"]) != SOURCE_N or np.asarray(z["host_values"]).shape[1] != 169:
            raise RuntimeError("BLOCKED_BY_CONTRACT: A0_ROLE_ONLY cache changed")
    _, cohort = load_future_cohort()
    fold_sources, discovered_fold_rows = discover_fold_assignment_rows()
    assignments, year_mismatches = canonicalize_fold_assignments(raw_rows=discovered_fold_rows)
    p31 = discover_p31_cache(os.environ.get("FUTURE_P31_CACHE"))
    relation = discover_base_relation_cache(os.environ.get("C43_FUTURE_REGION_CACHE"))
    endpoint = discover_endpoint_cache(os.environ.get("ENDPOINT_CACHE"))
    inter = discover_future_inter_cache(os.environ.get("FUTURE_INTER_CACHE"))
    inter_mode = "reuse_discovered_cache" if inter is not None else "build_label_free_from_endpoint_cache"
    if inter is None:
        inter = ROOT / "cache/future_interz.npz"
    features = first_existing(SEALED_FEATURE_CANDIDATES, "label-sealed 2017+future feature table")
    ROOT.mkdir(parents=True, exist_ok=True)
    canonical_folds = ROOT / "canonical_fold_assignments.csv"
    write_csv(canonical_folds, assignments)
    dump(ROOT / "fold_year_alignment_audit.json", {
        "sources": [str(path) for path in fold_sources], "rows": len(assignments),
        "authoritative_year_source": "frozen label-free future cohort keyed by SHA",
        "fold_ids_modified": False, "year_mismatch_count": len(year_mismatches),
        "year_mismatches": year_mismatches,
    })
    resolved = {
        "formal_freeze": str(FORMAL_FREEZE), "source_reference": str(SOURCE_REFERENCE),
        "source_a0_cache": str(SOURCE_A0_CACHE), "future_p31_cache": str(p31),
        "future_base_relation_cache": str(relation), "sealed_features": str(features),
        "endpoint_cache": str(endpoint),
        "future_inter_cache": str(inter), "future_inter_mode": inter_mode,
        "fold_assignments": str(canonical_folds),
        "fold_assignment_sources": [str(path) for path in fold_sources],
        "fold_year_mismatch_count": len(year_mismatches),
        "future_rows": len(cohort), "future_rows_loaded_for_selection": 0,
        "protocol": "frozen post-selection diagnostic only",
    }
    dump(ROOT / "resolved_inputs.json", resolved)
    dump(ROOT / "protocol.json", {
        "formal_method": "CARE_DROID_FORMAL_V1", "stage1": "calibrated P31 Random Forest only",
        "router": "RF-anchored LR/RF/XGB/SVM MeanOpposition, Top25%",
        "stage2": "A0 Role C43-RQ", "max_delta": 2.25,
        "protocols": ["frozen_2017_to_2018_2021", "independent_same_year_5fold"],
        "future_used_for_selection": False, "threshold": .5, "route_budget": .25,
        "full100_is_diagnostic": True,
    })
    print("CARE_DROID_FORMAL_DIAGNOSTICS_PREFLIGHT_PASS")
    print(f"future_p31_cache={p31}")
    print(f"future_base_relation_cache={relation}")
    print(f"endpoint_cache={endpoint}")
    print(f"future_inter_cache={inter}")
    print(f"future_inter_mode={inter_mode}")
    print("fold_assignment_sources=" + ";".join(map(str, fold_sources)))
    print(f"fold_year_mismatch_count={len(year_mismatches)} fold_ids_modified=false")
    print(f"sealed_features={features}")
    print("future_used_for_selection=false formal_method_unchanged=true")


if __name__ == "__main__":
    main()
