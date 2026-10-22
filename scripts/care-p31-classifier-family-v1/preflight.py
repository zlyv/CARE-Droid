#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import (ARMS, FUTURE_LABELED, FUTURE_MEMBER, MEMBERS, OUTPUT, REFERENCE_AUDIT, RESOLVED,
    SOURCE_MEMBER, SOURCE_P31, WITHIN_INPUTS, YEAR_COUNTS, dump_json, file_record, metric, norm, read_csv, read_json)


def main() -> None:
    paths = {"source_member_oof": SOURCE_MEMBER, "source_p31": SOURCE_P31, "future_member_label_free": FUTURE_MEMBER,
        "future_diagnostic_labels": FUTURE_LABELED, "formal_resolved_inputs": RESOLVED, "future_reference_audit": REFERENCE_AUDIT,
        "within_2018_2019": WITHIN_INPUTS[0], "within_2020_2021": WITHIN_INPUTS[1]}
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing inputs:\n- " + "\n- ".join(missing))
    source = read_csv(SOURCE_MEMBER); required = {"sha256", "label", "fold"} | {f"p_P31__{name}" for name in MEMBERS}
    if len(source) != 799 or not required <= set(source[0]):
        raise RuntimeError("BLOCKED_BY_CONTRACT: source member OOF schema changed")
    y = np.asarray([int(float(row["label"])) for row in source], int); fold = np.asarray([int(float(row["fold"])) for row in source], int)
    rf = np.asarray([float(row["p_P31__RF"]) for row in source], float)
    rf_metric = metric(y, rf)
    if [int(np.sum(fold == f)) for f in range(1, 6)] != [160, 160, 160, 160, 159] or (int(np.sum(y == 0)), int(np.sum(y == 1))) != (400, 399):
        raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort/folds changed")
    if abs(rf_metric["BA"] - .9273809523809524) > 1e-10 or rf_metric["errors"] != 58:
        raise RuntimeError("BLOCKED_BY_CONTRACT: RF no longer reproduces formal Stage-I")
    future = read_csv(FUTURE_MEMBER); labeled = read_csv(FUTURE_LABELED)
    if len(future) != 1989 or len(labeled) != 1989 or not {f"p_P31__{name}" for name in MEMBERS} <= set(future[0]):
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen future member table changed")
    labels = {norm(row["sha256"]): int(float(row["label"])) for row in labeled}
    if set(labels) != {norm(row["sha256"]) for row in future}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen future label alignment failed")
    within = read_csv(WITHIN_INPUTS[0]) + read_csv(WITHIN_INPUTS[1])
    counts = {year: sum(int(float(row["year"])) == year for row in within) for year in YEAR_COUNTS}
    if len(within) != 1989 or counts != YEAR_COUNTS or {norm(row["sha256"]) for row in within} != set(labels):
        raise RuntimeError("BLOCKED_BY_CONTRACT: same-year cohort changed")
    reference_audit = read_json(REFERENCE_AUDIT)
    if reference_audit.get("future_labels_loaded") is not False:
        raise RuntimeError("BLOCKED_BY_CONTRACT: future member reference was not label-free")
    resolved = read_json(RESOLVED); future_p31 = __import__("pathlib").Path(resolved["future_p31_cache"])
    if not future_p31.is_file():
        raise RuntimeError("BLOCKED_BY_CONTRACT: resolved future P31 cache missing: " + str(future_p31))
    with np.load(future_p31, allow_pickle=False) as archive:
        ids = {norm(value) for value in archive["sha256"]}; matrix = np.asarray(archive["role_prevalence"])
    if matrix.ndim != 2 or matrix.shape[1] != 31 or not set(labels) <= ids:
        raise RuntimeError("BLOCKED_BY_CONTRACT: future P31 cache coverage/schema changed")
    assets = [file_record(path, name) for name, path in paths.items()] + [file_record(future_p31, "future_p31_cache")]
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", {"assets": assets})
    protocol = {"experiment": "P31 calibrated classifier-family comparison v1", "members": list(MEMBERS),
        "control": "MEAN4", "feature": "frozen P31 role_prevalence[31]", "threshold": .5,
        "calibration": "inner-fold sigmoid/Platt calibration for every member",
        "source": "2017 fixed five-fold OOF; primary comparison domain",
        "frozen_future": "train 2017, predict 2018-2021; diagnostic only",
        "same_year": "independent fixed five-fold OOF inside each year; not temporal generalization",
        "router_audit": "self-anchored MeanOpposition Top25%; classification comparison remains primary",
        "future_used_for_selection": False, "formal_stage1_changed": False}
    dump_json(OUTPUT / "protocol.json", protocol)
    dump_json(OUTPUT / "preflight.json", {"status": "PASS", "source_rows": 799, "future_rows": 1989,
        "within_rows": 1989, "year_counts": counts, "formal_rf": rf_metric, "future_member_predictions_label_free": True,
        "future_p31_cache": str(future_p31), "formal_stage1_changed": False})
    print("P31_CLASSIFIER_FAMILY_PREFLIGHT_PASS source=799 future=1989 formal_RF_reproduced=true", flush=True)


if __name__ == "__main__":
    main()

