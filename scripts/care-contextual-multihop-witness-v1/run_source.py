#!/usr/bin/env python3
from __future__ import annotations

import json
import math

import numpy as np

from common import OUTPUT, SEED, dump_json, metrics, parse_source_formal, source_assets, write_csv
from modeling import ARMS, correction_probability, crossfit_scores, direction_summary, load_feature_matrix


def correction_audit(rows, probability, delta):
    y = np.asarray([row["label"] for row in rows], int)
    base = np.asarray([row["stage1_prob"] for row in rows], float)
    route = np.asarray([row["routed"] for row in rows], bool)
    error = ((base >= .5) != y) & route
    signed = np.asarray(delta, float) * np.where(y == 1, 1.0, -1.0)
    final_ok = (np.asarray(probability, float) >= .5) == y
    return {"routed_stage1_errors": int(error.sum()),
            "fixed_direction_outcome": int(np.sum(error & final_ok)),
            "correct_direction_but_too_weak": int(np.sum(error & (signed > 1e-12) & ~final_ok)),
            "wrong_direction": int(np.sum(error & (signed < -1e-12))),
            "insufficient_zero_delta": int(np.sum(error & (np.abs(delta) <= 1e-12)))}


def main() -> None:
    rows = parse_source_formal(source_assets()["formal"]); folds = np.asarray([row["fold"] for row in rows], int)
    y = np.asarray([row["label"] for row in rows], int); base = np.asarray([row["stage1_prob"] for row in rows], float)
    route = np.asarray([row["routed"] for row in rows], bool); feature_dir = OUTPUT / "cache/features/source"
    direction, predictions, end_results, fold_rows, end_fold_rows = {}, [], {}, [], []
    score_by_arm = {}
    for ordinal, arm in enumerate(ARMS):
        x, fields = load_feature_matrix(rows, feature_dir, arm)
        score, local_folds = crossfit_scores(rows, x, folds, SEED + ordinal * 1000)
        summary = direction_summary(rows, score, local_folds); summary["feature_count"] = len(fields)
        direction[arm] = summary; score_by_arm[arm] = score
        probability, delta = correction_probability(rows, score)
        end_results[arm] = {**metrics(y, base, probability, route), **correction_audit(rows, probability, delta)}
        for fold in range(1, 6):
            mask = folds == fold
            end_fold_rows.append({"arm": arm, "fold": fold,
                                  **metrics(y[mask], base[mask], probability[mask], route[mask])})
        for item in local_folds:
            fold_rows.append({"arm": arm, **item})
        for index, row in enumerate(rows):
            predictions.append({"sha256": row["sha256"], "fold": row["fold"], "label": row["label"], "arm": arm,
                                "stage1_prob": row["stage1_prob"], "routed": row["routed"], "direction_score": score[index],
                                "bounded_delta": delta[index], "final_prob": probability[index]})
        print(f"CMW_SOURCE_ARM_DONE arm={arm} FN_AUC={summary['FN_vs_TN_AUC']:.6f} FP_AUC={summary['FP_vs_TP_AUC']:.6f}", flush=True)

    real = direction["REAL"]
    null_fn = max(direction[name]["FN_vs_TN_AUC"] for name in ("ROOT_SHUFFLE", "CALLSITE_SHUFFLE", "COUNT_ONLY"))
    null_fp = max(direction[name]["FP_vs_TP_AUC"] for name in ("ROOT_SHUFFLE", "CALLSITE_SHUFFLE", "COUNT_ONLY"))
    direction_criteria = {"FN_AUC>=0.60": real["FN_vs_TN_AUC"] >= .60,
                          "FP_AUC>=0.60": real["FP_vs_TP_AUC"] >= .60,
                          "FN_real>=best_null+0.03": real["FN_vs_TN_AUC"] >= null_fn + .03,
                          "FP_real>=best_null+0.03": real["FP_vs_TP_AUC"] >= null_fp + .03,
                          "FN_real>=one_hop+0.02": real["FN_vs_TN_AUC"] >= direction["ONE_HOP"]["FN_vs_TN_AUC"] + .02,
                          "FP_real>=one_hop+0.02": real["FP_vs_TP_AUC"] >= direction["ONE_HOP"]["FP_vs_TP_AUC"] + .02,
                          "FN_consistent_folds>=3": real["FN_consistent_folds"] >= 3,
                          "FP_consistent_folds>=3": real["FP_consistent_folds"] >= 3}
    formal = metrics(y, base, np.asarray([row["formal_prob"] for row in rows], float), route)
    stage1 = metrics(y, base, base, np.zeros(len(rows), bool))
    real_end = end_results["REAL"]
    noninferior = 0
    for fold in range(1, 6):
        mask = folds == fold
        witness_p, _ = correction_probability([rows[i] for i in np.where(mask)[0]], score_by_arm["REAL"][mask])
        m = metrics(y[mask], base[mask], witness_p, route[mask])
        f = metrics(y[mask], base[mask], np.asarray([rows[i]["formal_prob"] for i in np.where(mask)[0]]), route[mask])
        noninferior += int(m["net"] >= f["net"])
    end_criteria = {"net>formal_+4": real_end["net"] > formal["net"], "broken<=formal_7": real_end["broken"] <= formal["broken"],
                    "fixed_FN>0": real_end["fixed_FN"] > 0, "fixed_FP>0": real_end["fixed_FP"] > 0,
                    "noninferior_folds>=3": noninferior >= 3}
    supported = all(direction_criteria.values()) and all(end_criteria.values())
    decision = "SUPPORTED_FOR_FROZEN_DIAGNOSTICS" if supported else "NOT_SUPPORTED"
    dump_json(OUTPUT / "direction_probe_results.json", {"arms": direction, "criteria": direction_criteria, "future_rows_loaded": 0})
    dump_json(OUTPUT / "source_end_to_end_results.json", {"stage1": stage1, "formal_A0": formal, "arms": end_results,
              "criteria": end_criteria, "noninferior_folds": noninferior, "future_rows_loaded": 0})
    write_csv(OUTPUT / "source_fold_direction_results.csv", fold_rows)
    write_csv(OUTPUT / "source_fold_end_to_end_results.csv", end_fold_rows)
    write_csv(OUTPUT / "source_predictions.csv", predictions)
    protocol = {"name": "Context-Sensitive Multi-Hop Behavioral Witness v1", "source_year": 2017,
                "stage1": "P31-RF", "router": "RF-anchored MeanOpposition DI25 Top25%", "localizer": "frozen A0 C43 attention",
                "max_depth": 8, "downstream_depth": 3, "context_k": 3, "contexts_per_node_category": 16,
                "model": "StandardScaler + class-balanced L2 LogisticRegression C=0.25",
                "correction": "2.25*tanh(signed positive logit(direction probability)); routed only",
                "ablations": ["one-hop", "path-only"],
                "nulls": ["root-label shuffle", "degree-preserving callsite-target shuffle", "count-only"],
                "future_rows_loaded_during_selection": 0, "source_decision": decision}
    dump_json(OUTPUT / "protocol.json", protocol)
    import hashlib
    digest = hashlib.sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest()
    (OUTPUT / "protocol_hash.txt").write_text(digest + "\n", encoding="utf-8")
    dump_json(OUTPUT / "source_decision.json", {"status": decision, "direction_criteria": direction_criteria,
              "end_to_end_criteria": end_criteria, "formal_A0": formal, "REAL": real_end, "protocol_hash": digest,
              "future_rows_loaded": 0})
    print(f"CMW_SOURCE_DECISION={decision}"); print(json.dumps({"direction": real, "formal": formal, "real_end": real_end}, indent=2), flush=True)
    if not supported:
        dump_json(OUTPUT / "final_decision.json", {"status": "NOT_SUPPORTED", "stopped_at": "SOURCE_DIRECTION_OR_END_TO_END_GATE",
                  "future_rows_loaded": 0, "reason": "REAL did not beat both-side null and formal end-to-end gates."})
        raise SystemExit(21)


if __name__ == "__main__":
    main()
