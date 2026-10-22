#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import (ACTIVE_ARMS, ARMS, MEMBERS, FORMAL_DIAG, FUTURE_LABELS, FUTURE_REFERENCE, FUTURE_RF,
    OUTPUT, SOURCE_A0, SOURCE_A0_CACHE, SOURCE_MEMBER, SOURCE_P31, SOURCE_RF_REFERENCE, WITHIN_RF,
    SOURCE_L0_STAGE2, FUTURE_L0_STAGE2, WITHIN_L0_STAGE2,
    PRIOR_STAGE2_TABLE,
    classification_metrics, dump_json, file_record, norm, read_csv, read_json)
from transforms import EPSILON, feature_names, validate_p31


def main():
    resolved = FORMAL_DIAG / "resolved_inputs.json"
    required = {
        "source_member_oof": SOURCE_MEMBER, "source_p31": SOURCE_P31,
        "source_formal_a0": SOURCE_A0, "source_a0_cache": SOURCE_A0_CACHE,
        "source_rf_reference": SOURCE_RF_REFERENCE, "formal_resolved_inputs": resolved,
        "prior_source_raw_lr_stage2": SOURCE_L0_STAGE2,
        "prior_frozen_raw_lr_stage2": FUTURE_L0_STAGE2,
        "prior_within_raw_lr_stage2": WITHIN_L0_STAGE2,
        "prior_classifier_family_stage2_table": PRIOR_STAGE2_TABLE,
        "future_member_reference": FUTURE_REFERENCE, "future_labels": FUTURE_LABELS,
        "future_formal_rf": FUTURE_RF, "within_formal_rf_2018_2019": WITHIN_RF[0],
        "within_formal_rf_2020_2021": WITHIN_RF[1],
        "v4_runner": Path("/home/boyuanjie/projects/tasa-droid/scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py"),
        "a0_background": Path("/home/boyuanjie/projects/tasa-droid/scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py"),
        "a0_model": Path("/home/boyuanjie/projects/tasa-droid/scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_models.py"),
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file() or not path.stat().st_size]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing inputs:\n- " + "\n- ".join(missing))
    source = read_csv(SOURCE_MEMBER)
    expected_columns = {"sha256", "label", "fold"} | {f"p_P31__{name}" for name in MEMBERS}
    if len(source) != 799 or not source or not expected_columns <= set(source[0]):
        raise RuntimeError("BLOCKED_BY_CONTRACT: source member OOF contract changed")
    sha = [norm(row["sha256"]) for row in source]
    y = np.asarray([int(float(row["label"])) for row in source], int)
    folds = np.asarray([int(float(row["fold"])) for row in source], int)
    rf = np.asarray([float(row["p_P31__RF"]) for row in source], float)
    metric = classification_metrics(y, rf)
    if len(set(sha)) != 799 or [int(np.sum(folds == fold)) for fold in range(1, 6)] != [160, 160, 160, 160, 159]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source SHA/fold contract changed")
    if abs(metric["BA"] - .9273809523809524) > 1e-12 or metric["errors"] != 58:
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal RF Stage-I no longer reproduces")
    with np.load(SOURCE_P31, allow_pickle=False) as archive:
        source_p31 = np.asarray(archive["role_prevalence"], float)
        p31_keys = sorted(archive.files)
    normalized = validate_p31(source_p31)
    if source_p31.shape != (799, 31):
        raise RuntimeError("BLOCKED_BY_CONTRACT: source P31 shape changed")
    future = read_csv(FUTURE_REFERENCE); labels = read_csv(FUTURE_LABELS)
    if len(future) != 1989 or len(labels) != 1989:
        raise RuntimeError("BLOCKED_BY_CONTRACT: future cohort changed")
    resolved_value = read_json(resolved)
    dynamic_paths = {
        "sealed_features": Path(resolved_value["sealed_features"]),
        "endpoint_cache": Path(resolved_value["endpoint_cache"]),
        "future_p31_cache": Path(resolved_value["future_p31_cache"]),
        "future_inter_cache": Path(resolved_value["future_inter_cache"]),
        "future_a0_cache": FORMAL_DIAG / "cache/a0_role_2017_2021.npz",
        "future_s7_cache": FORMAL_DIAG / "cache/future_s7.npz",
    }
    dynamic_missing = [f"{name}: {path}" for name, path in dynamic_paths.items() if not path.is_file() or not path.stat().st_size]
    if dynamic_missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing resolved inputs:\n- " + "\n- ".join(dynamic_missing))
    assets = [file_record(path, name) for name, path in {**required, **dynamic_paths}.items()]
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", {"assets": assets})
    protocol = {
        "experiment": "P31-LR temporally robust compositional static representations plus identical Stage-II v1",
        "arms": list(ACTIVE_ARMS), "all_predeclared_arms": list(ARMS), "primary_source": "2017 fixed five-fold strict OOF",
        "stage1": "the classifier is calibrated logistic regression in every arm; only the label-free P31 transform changes",
        "representations": {arm: feature_names(arm) for arm in ARMS},
        "pseudocount": EPSILON,
        "routing": "the arm-specific LR anchors an arm-specific LR/RF/XGB/SVM MeanOpposition; deterministic Top25% inside every evaluation fold/year",
        "stage2": "same A0 Role C43 Relation-Query architecture and capacity for every arm",
        "stage2_training": "all outer-train APKs; Stage-I context supplied by inner OOF for the same arm",
        "max_delta": 2.25, "threshold": .5,
        "L0_RAW": "reuse frozen formal raw P31-LR Stage-I and C43 source predictions after exact reproduction checks",
        "future": "frozen 2017-to-2018..2021 diagnostic only; never used for selection",
        "same_year": "independent fixed five-fold OOF; not temporal generalization",
        "formal_stage1_changed": False, "future_selection_forbidden": True,
    }
    dump_json(OUTPUT / "protocol.json", protocol)
    dump_json(OUTPUT / "preflight.json", {"status": "PASS", "source_rows": 799,
        "future_rows": 1989, "formal_rf_stage1": metric, "p31_keys": p31_keys,
        "p31_row_sum_min_after_closure": float(normalized.sum(1).min()),
        "p31_row_sum_max_after_closure": float(normalized.sum(1).max()),
        "p31_zero_mass_rows": int(np.sum(source_p31.sum(1) <= 0)),
        "assets": {k: str(v) for k, v in dynamic_paths.items()},
        "future_used_for_selection": False, "formal_method_changed": False})
    print("P31_LR_ROBUST_STATIC_PREFLIGHT_PASS source=799 future=1989 formal_RF_reference=true", flush=True)


if __name__ == "__main__":
    main()
