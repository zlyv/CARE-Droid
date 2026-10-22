#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import (FUNCTION_INVENTORY, OUTPUT, SOURCE_P31, SOURCE_STAGE2,
                    asset_record, dump_json, norm, read_csv)


def main():
    required = ((SOURCE_STAGE2, "frozen SVM+C43 source predictions"),
                (SOURCE_P31, "frozen source P31"),
                (FUNCTION_INVENTORY, "completed static function-family inventory"))
    for path, _ in required:
        if not path.is_file():
            raise RuntimeError("BLOCKED_BY_CONTRACT: missing " + str(path))
    rows = [row for row in read_csv(SOURCE_STAGE2) if norm(row.get("arm")) == "svm"]
    inventory = read_csv(FUNCTION_INVENTORY)
    if len(rows) != 799 or len(inventory) != 799:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: cohort changed svm={len(rows)} inventory={len(inventory)}")
    sha = [norm(row["sha256"]) for row in rows]
    if set(sha) != {norm(row["sha256"]) for row in inventory}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: function inventory SHA mismatch")
    y = np.asarray([int(row["label"]) for row in rows]); base = np.asarray([float(row["stage1_prob"]) for row in rows])
    route = np.asarray([int(row["routed"]) for row in rows], bool); pred = base >= .5; error = pred != y
    if int(route.sum()) != 200 or int(np.sum(route & error)) != 84:
        raise RuntimeError("BLOCKED_BY_CONTRACT: SVM route changed")
    if int(np.sum(route & pred & (y == 0))) != 43 or int(np.sum(route & (~pred) & (y == 1))) != 41:
        raise RuntimeError("BLOCKED_BY_CONTRACT: routed SVM FP/FN composition changed")
    manifest = [asset_record(path, purpose) for path, purpose in required]
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", manifest)
    dump_json(OUTPUT / "protocol.json", {
        "name": "CARE-Droid SVM Positive Benign Functional Certificate v1",
        "stage1": "calibrated P31-SVM", "router": "SVM-anchored MeanOpposition DI25 Top25%",
        "old_stage2": "A0 Role C43-RQ bounded correction max_delta=2.25",
        "positive_benign_evidence": "observed non-sensitive code-function family positively lifts an observed Role above benign outer-train background",
        "forbidden_benign_evidence": ["low anomaly", "absence of malware evidence", "1-malware_score", "raw count"],
        "certificate": "weighted_positive_lift * explained_role_mass; zero when context absent/unknown/not above background",
        "primary_probe": "FP-vs-TP among routed SVM-predicted-malware APKs",
        "gate": {"AUC>=0.65": True, "AUC>matched_shuffle_95pct+0.03": True,
                 "AUC>count_only+0.03": True, "majority_evaluable_folds": True,
                 "max_abs_size_correlation<0.70": True},
        "integration": "forbidden unless every evidence gate passes",
        "future": "forbidden unless source evidence and source integration gates pass",
        "future_rows_loaded": 0, "seed": 20261009})
    dump_json(OUTPUT / "preflight.json", {"status": "PASS", "source_rows": 799, "routed": 200,
        "routed_FP": 43, "routed_FN": 41, "future_rows_loaded": 0,
        "positive_evidence_only": True, "low_anomaly_is_benign_evidence": False})
    print("SVM_POSITIVE_BENIGN_PREFLIGHT_PASS source=799 routed=200 routed_FP=43 routed_FN=41")


if __name__ == "__main__":
    main()

