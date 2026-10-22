#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import OUTPUT, SOURCE_MEMBERS, SOURCE_STAGE2, dump, file_record, norm, read_csv


def main():
    for path in (SOURCE_STAGE2, SOURCE_MEMBERS):
        if not path.is_file():
            raise RuntimeError("BLOCKED_BY_CONTRACT: missing required frozen input " + str(path))
    all_stage2 = read_csv(SOURCE_STAGE2); svm = [row for row in all_stage2 if norm(row.get("arm")) == "svm"]
    members = read_csv(SOURCE_MEMBERS)
    if len(svm) != 799 or len(members) != 799:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: cohort changed svm={len(svm)} members={len(members)}")
    svm.sort(key=lambda row: norm(row["sha256"])); members.sort(key=lambda row: norm(row["sha256"]))
    if [norm(row["sha256"]) for row in svm] != [norm(row["sha256"]) for row in members]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: SHA alignment mismatch")
    required = ("p_P31__LR", "p_P31__RF", "p_P31__XGB", "p_P31__SVM")
    missing = [name for name in required if name not in members[0]]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing member probability columns " + repr(missing))
    base = np.asarray([float(row["stage1_prob"]) for row in svm])
    p_svm = np.asarray([float(row["p_P31__SVM"]) for row in members])
    error = float(np.max(np.abs(base - p_svm)))
    if error > 1e-12:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: SVM probability mismatch max_error={error}")
    manifest = [file_record(SOURCE_STAGE2, "frozen SVM+C43 source predictions"),
                file_record(SOURCE_MEMBERS, "frozen calibrated member OOF probabilities")]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dump(OUTPUT / "baseline_asset_manifest_before.json", manifest)
    dump(OUTPUT / "protocol.json", {
        "name": "CARE-Droid SVM symmetric consensus selective correction v1",
        "source": "2017 fixed five-fold OOF only",
        "stage1": "calibrated P31-SVM",
        "router": "SVM-anchored MeanOpposition DI25 Top25%",
        "candidate": "A0 Role C43-RQ bounded correction max_delta=2.25",
        "primary_rule": "for a C43 class flip, accept iff at least 2 of calibrated LR/RF/XGB predict the proposed class",
        "control_rule": "accept iff all 3 calibrated LR/RF/XGB predict the proposed class",
        "nonflip_policy": "retain the C43 probability refinement because it cannot create fixed/broken label changes",
        "matched_random": "2000 permutations preserving fold x proposed class x accepted-flip count",
        "predeclared_success_criteria": {
            "majority_net>S1_net": True,
            "majority_broken<S1_broken": True,
            "fixed_retention>=0.75": True,
            "noninferior_folds>=3": True,
            "primary_trust_AUC>=0.70": True,
            "majority_net>matched_random_95pct": True
        },
        "next_gate": "strict nested learned selection and future diagnostics are forbidden unless every source criterion passes",
        "future_rows_loaded": 0,
        "seed": 20261009
    })
    dump(OUTPUT / "preflight.json", {"status": "PASS", "source_rows": 799, "future_rows_loaded": 0,
        "svm_probability_max_error": error, "mainline": "P31-SVM / SVM-anchored DI25 / A0 C43-RQ / max_delta=2.25",
        "rule": "one symmetric majority-support rule; no FP/FN-specific definitions"})
    print("SVM_CONSENSUS_SELECTIVE_CORRECTION_PREFLIGHT_PASS source=799 future_rows_loaded=0")


if __name__ == "__main__":
    main()
