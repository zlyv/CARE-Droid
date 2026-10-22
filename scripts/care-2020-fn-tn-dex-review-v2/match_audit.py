#!/usr/bin/env python3
"""Diagnostic-only structural matching of the frozen 2020 routed FN/TN cohort."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path("/home/boyuanjie/projects/tasa-droid")
OUT = ROOT / "outputs/care-2020-fn-tn-dex-review-v2"
REF = ROOT / "outputs/care_crossyear_layerwise_diagnosis_v2/per_apk_error_contract.csv"
EMB = ROOT / "outputs/care_crossyear_layerwise_diagnosis_v2/embeddings/T3_C1/validation.npz"


def main() -> None:
    if OUT.exists():
        raise RuntimeError("Independent output version already exists")
    OUT.mkdir(parents=True)
    with REF.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["tag"] == "T3_C1" and r["year"] == "2020"
                and r["routed"] == "1" and r["stage1_prediction"] == "0"]
    fn = [r for r in rows if r["category"] == "UNFIXED_FN" and r["label"] == "1"]
    tn = [r for r in rows if r["category"] == "STABLE_CORRECT" and r["label"] == "0"]
    if len(fn) != 24 or len(tn) != 48:
        raise RuntimeError(f"Frozen cohort mismatch {len(fn)}/{len(tn)}")
    with np.load(EMB, allow_pickle=False) as z:
        ids = list(z["ids"].astype(str))
        p31 = np.asarray(z["P0_P31_SIZE"][:, :31], float)
        host_n = np.asarray(z["host_n"], float)
    index = {sid: i for i, sid in enumerate(ids)}
    if any(r["sha256"] not in index for r in fn + tn):
        raise RuntimeError("CSV/embedding SHA alignment failed")
    scale = np.std(p31, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    standardized = p31 / scale
    unit = standardized / np.maximum(np.linalg.norm(standardized, axis=1, keepdims=True), 1e-12)
    n, m = len(fn), len(tn)
    cost = np.full((n, m + n), 100.0)
    details = {}
    for i, a in enumerate(fn):
        ia = index[a["sha256"]]
        for j, b in enumerate(tn):
            ib = index[b["sha256"]]
            pdiff = abs(float(a["stage1_probability"]) - float(b["stage1_probability"]))
            ratio = max(host_n[ia], host_n[ib]) / max(min(host_n[ia], host_n[ib]), 1.0)
            cosine = float(1 - np.dot(unit[ia], unit[ib]))
            # Diagnostic comparability contract, set after seeing the v1 size mismatch.
            eligible = bool(pdiff <= 0.15 and ratio <= 2.0 and cosine <= 0.25)
            details[i, j] = (pdiff, ratio, cosine, eligible)
            if eligible:
                cost[i, j] = pdiff / 0.15 + abs(np.log(ratio)) / np.log(2) + cosine / 0.25
        cost[i, m + i] = 10.0  # explicit unmatched case; no forced poor match
    rr, cc = linear_sum_assignment(cost)
    paired = []
    unmatched = []
    for i, j in zip(rr, cc):
        if j >= m or cost[i, j] >= 10:
            unmatched.append(fn[i]["sha256"])
            continue
        pdiff, ratio, cosine, eligible = details[i, j]
        paired.append({"fn_sha256": fn[i]["sha256"], "tn_sha256": tn[j]["sha256"],
                       "stage1_probability_difference": pdiff,
                       "host_count_ratio": ratio, "p31_standardized_cosine_distance": cosine,
                       "fn_direct_probability": float(fn[i]["expert_probability"]),
                       "matching_cost": float(cost[i, j])})
    paired.sort(key=lambda r: (r["matching_cost"], r["fn_sha256"]))
    fields = ["fn_sha256", "tn_sha256", "stage1_probability_difference", "host_count_ratio",
              "p31_standardized_cosine_distance", "fn_direct_probability", "matching_cost"]
    with (OUT / "matched_pairs_restricted.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(paired)
    decision = {
        "status": "STRUCTURAL_MATCHING_AUDIT_COMPLETE",
        "cohort": {"unfixed_fn": n, "correct_tn": m},
        "eligible_fn_with_any_tn": sum(any(details[i, j][3] for j in range(m)) for i in range(n)),
        "one_to_one_eligible_pairs": len(paired),
        "unmatched_fn": len(unmatched),
        "comparability_contract": "Stage-I probability difference <=0.15, Host-count ratio <=2, standardized P31 cosine distance <=0.25",
        "labels_used_for_diagnostic_case_control_selection": True,
        "labels_used_for_model_or_threshold_selection": False,
        "model_training": False, "dynamic_execution": False,
        "interpretation_boundary": "These thresholds define a diagnostic matching sensitivity analysis, not a classifier or an estimated treatment effect.",
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("STRUCTURAL_MATCHING_AUDIT_DONE " + json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
