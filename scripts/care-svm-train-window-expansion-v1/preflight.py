#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import (BASELINE_SOURCE, BASELINE_STAGE2, ENDPOINT_SOURCE, FORMAL_DIAG, LABELS, OUTPUT,
    SOURCE_A0_CACHE, SOURCE_P31, WITHIN_META, dump_json, file_record, read_csv)


def main():
    resolved_path = FORMAL_DIAG / "resolved_inputs.json"
    required = {"source_p31": SOURCE_P31, "source_endpoint": ENDPOINT_SOURCE, "source_a0_cache": SOURCE_A0_CACHE,
        "w17_stage2_baseline": BASELINE_STAGE2, "w17_source_baseline": BASELINE_SOURCE,
        "future_labels": LABELS, "2018_training_metadata": WITHIN_META, "formal_resolved_inputs": resolved_path,
        "future_s7": FORMAL_DIAG / "cache/future_s7.npz", "all_role_cache": FORMAL_DIAG / "cache/a0_role_2017_2021.npz"}
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing inputs:\n- " + "\n- ".join(missing))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    for key in ("sealed_features", "endpoint_cache", "future_inter_cache", "future_p31_cache"):
        path = Path(resolved[key])
        if not path.is_file():
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: resolved {key} missing: {path}")
        required[f"resolved_{key}"] = path
    metadata = [row for row in read_csv(WITHIN_META) if int(row["year"]) == 2018]
    labels = read_csv(LABELS); baseline = read_csv(BASELINE_STAGE2)
    if len(metadata) != 793 or len(baseline) != 1989 or len(labels) != 1989:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: cohort changed 2018={len(metadata)} baseline={len(baseline)} labels={len(labels)}")
    with np.load(SOURCE_P31, allow_pickle=False) as archive:
        if np.asarray(archive["role_prevalence"]).shape != (799, 31):
            raise RuntimeError("BLOCKED_BY_CONTRACT: source P31 changed")
    before = [file_record(path, name) for name, path in required.items()]
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", {"assets": before})
    protocol = {"experiment": "CARE-Droid SVM train-window expansion v1", "question": "Does adding labeled 2018 alleviate benign-side temporal correction failure?",
        "arms": {"W17": "train 2017; frozen existing reference", "W18": "train 2018 only; recency/control with comparable n",
            "W1718": "train pooled 2017+2018; sample-size and diversity arm"},
        "test_years": [2019, 2020, 2021], "stage1": "calibrated P31-SVM; probability only from SVM",
        "router": "SVM-anchored MeanOpposition Top25% independently per target year", "stage2": "unchanged A0 Role C43 Relation-Query",
        "max_delta": 2.25, "target_labels_used_for_fit": False, "hyperparameters_selected_on_target": False,
        "interpretation": "W18 vs W17 diagnoses recency/distribution; W1718 beating both supports added sample diversity/size.",
        "future_status": "diagnostic; these years were already observed in prior development"}
    dump_json(OUTPUT / "protocol.json", protocol)
    dump_json(OUTPUT / "preflight.json", {"status": "PASS", "source_2017_rows": 799, "train_2018_rows": 793,
        "target_rows": 1196, "target_year_counts": {"2019": 399, "2020": 397, "2021": 400},
        "formal_stage1_changed": False, "formal_stage2_changed": False, "target_labels_used_for_fit": False})
    print("SVM_TRAIN_WINDOW_PREFLIGHT_PASS train17=799 train18=793 target=1196", flush=True)


if __name__ == "__main__":
    main()
