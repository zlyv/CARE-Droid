#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from common import (DIRECT_METHOD, OUTPUT, PROJECT, YEARS, cap_route, discover, dump_json,
                    ensure_output, file_record, norm, read_csv, sha256_file, threshold_higher,
                    top_k, write_csv)


def main():
    ensure_output(); OUTPUT.mkdir(parents=True, exist_ok=True); assets = discover()
    protected = [file_record(assets[key], purpose) for key, purpose in (
        ("source", "frozen source Direct-C43 full-cohort OOF predictions"),
        ("architecture", "frozen Direct-C43 architecture"),
        ("future", "frozen label-free future Direct-C43 predictions"),
        ("future_labels", "frozen future labels; not read during policy freeze"))]
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", protected)

    source = read_csv(assets["source"]); future = read_csv(assets["future"])
    source_sha = np.asarray([norm(row["sha256"]) for row in source]); source_fold = np.asarray([int(float(row["fold"])) for row in source])
    source_score = np.asarray([float(row["A25__score"]) for row in source]); global_tau = threshold_higher(source_score)
    source_fixed25 = top_k(source_score, source_sha, source_fold, .25)
    source_lofo_raw = np.zeros(len(source), bool); thresholds = {}
    for fold in sorted(set(source_fold.tolist())):
        train = source_fold != fold; valid = ~train; tau = threshold_higher(source_score[train]); thresholds[str(fold)] = tau
        source_lofo_raw[valid] = source_score[valid] >= tau
    source_lofo_cap25 = cap_route(source_lofo_raw, source_score, source_sha, source_fold, .25)

    future_sha = np.asarray([norm(row["sha256"]) for row in future]); year = np.asarray([int(row["year"]) for row in future])
    future_score = np.asarray([float(row["router_score"]) for row in future]); future_fixed25 = top_k(future_score, future_sha, year, .25)
    future_raw = future_score >= global_tau; future_cap25 = cap_route(future_raw, future_score, future_sha, year, .25)

    source_rows = []
    for i, row in enumerate(source):
        source_rows.append({"sha256": source_sha[i], "fold": int(source_fold[i]), "label": int(float(row["label"])),
            "stage1_prob": float(row["Global24__prob"]), "router_score": float(source_score[i]),
            "expert_prob": float(row[DIRECT_METHOD + "__expert_prob"]), "fixed25": int(source_fixed25[i]),
            "lofo_q75_raw": int(source_lofo_raw[i]), "lofo_q75_cap25": int(source_lofo_cap25[i])})
    future_rows = []
    for i, row in enumerate(future):
        future_rows.append({"sha256": future_sha[i], "year": int(year[i]), "stage1_prob": float(row["stage1_prob"]),
            "router_score": float(future_score[i]), "expert_prob": float(row["direct_c43_prob"]),
            "fixed25": int(future_fixed25[i]), "q75_raw": int(future_raw[i]), "q75_cap25": int(future_cap25[i])})
    write_csv(OUTPUT / "source_route_assignments.csv", source_rows)
    write_csv(OUTPUT / "future_route_assignments_label_free.csv", future_rows)

    protocol = {"name": "Elastic MeanOpposition Routing v1", "stage1": "calibrated P31-SVM",
        "router_score": "frozen four-member SVM-anchored MeanOpposition", "expert": "frozen Direct-C43",
        "source_reference_quantile": .75, "quantile_method": "higher", "source_global_threshold": global_tau,
        "source_evaluation_thresholds_leave_one_fold_out": thresholds,
        "candidate_policy": "route score >= source threshold, capped at Top25% independently per deployment batch",
        "raw_threshold_arm_is_diagnostic_only": True, "tie_break": "score descending then SHA ascending",
        "threshold_uses_labels": False, "capacity_fraction": .25, "future_used_for_selection": False,
        "future_labels_loaded_during_freeze": False, "source_frontier_not_used_to_choose_quantile": True,
        "fixed15_not_selected": True, "benefit_gate": False, "sample_level_gate": False}
    dump_json(OUTPUT / "protocol.json", protocol)
    protocol_hash = sha256_file(OUTPUT / "protocol.json")
    (OUTPUT / "protocol_hash.txt").write_text(protocol_hash + "\n", encoding="utf-8")
    dump_json(OUTPUT / "freeze_status.json", {"status": "FROZEN", "protocol_hash": protocol_hash,
        "source_global_threshold": global_tau, "future_rows": len(future), "future_labels_loaded": False,
        "future_route_counts": {str(value): {"fixed25": int(np.sum(future_fixed25 & (year == value))),
            "q75_raw": int(np.sum(future_raw & (year == value))), "q75_cap25": int(np.sum(future_cap25 & (year == value)))} for value in YEARS}})
    print("ELASTIC_MEANOPPOSITION_POLICY_FROZEN hash=" + protocol_hash)


if __name__ == "__main__": main()
