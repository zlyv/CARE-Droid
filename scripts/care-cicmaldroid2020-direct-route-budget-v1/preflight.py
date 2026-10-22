#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from common import COHORTS, DATASET, FORMAL_DIAG, OUTPUT, PROJECT, SOURCE_P31, SOURCE_RESOLVED, SOURCE_SVM, TRAIN_WINDOW, dump_json, file_record, read_json


def main():
    output_root = (PROJECT / "outputs").resolve()
    if not OUTPUT.is_relative_to(output_root):
        raise RuntimeError("BLOCKED_BY_CONTRACT: output must stay below project outputs/: " + str(OUTPUT))
    required = {"dataset": DATASET, "source_p31": SOURCE_P31, "source_resolved": SOURCE_RESOLVED,
                "source_svm": SOURCE_SVM, "formal_resolved": FORMAL_DIAG / "resolved_inputs.json",
                "train_window_results": TRAIN_WINDOW / "train_window_results.csv",
                "stage1_core": PROJECT / "scripts/care-svm-train-window-expansion-v1/stage1_core.py",
                "model_adapter": PROJECT / "scripts/care-p31-classifier-family-stage2-v1/model_adapter.py",
                "direct_stage2_variant": PROJECT / "scripts/care-svm-c43-remove-bounded-correction-v1/model_variants.py",
                "v4_model": PROJECT / "scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py",
                "formal_region_cache": FORMAL_DIAG / "cache/a0_role_2017_2021.npz",
                "formal_future_s7": FORMAL_DIAG / "cache/future_s7.npz",
                "feature_schema": PROJECT / "artifacts/p1-features/feature_schema.json"}
    for cohort, spec in COHORTS.items():
        required["dataset_archive_" + cohort] = DATASET / "archives" / spec["archive"]
    formal = read_json(required["formal_resolved"])
    for key in ("sealed_features", "endpoint_cache", "future_inter_cache", "future_p31_cache"):
        if key in formal:
            required["formal_" + key] = Path(formal[key])
    missing = {key: str(path) for key, path in required.items() if not path.exists()}
    if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing required assets " + repr(missing))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # The four dataset archives total tens of GB.  Their immutable size/mtime is
    # recorded by archive_inventory.py; formal-asset integrity hashing is kept
    # to CARE-Droid code/config/cache files so preflight and final integrity do
    # not each reread the entire external corpus.
    records = [file_record(path, key) for key, path in required.items()
               if path.is_file() and not key.startswith("dataset_archive_")]
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", records)
    dump_json(OUTPUT / "preflight.json", {"status": "PASS", "dataset": str(DATASET),
        "frozen_method": "W1718 calibrated P31-SVM -> frozen SVM MeanOpposition ranking -> Top25/35/45/50 -> U2 direct C43 Stage-II (no bounded residual)",
        "dataset_scope": "user-supplied Benign/Adware/Banking/Riskware archives; SMS archive absent",
        "target_labels_permitted_for_fit": False, "required_assets": {key: str(path) for key, path in required.items()}})
    print("CICMALDROID2020_DIRECT_BUDGET_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__": main()

