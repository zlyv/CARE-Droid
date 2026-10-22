#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import (ARMS, COUNT_FIELDS, FEATURE_DIR, OUTPUT, SEED, binary_auc,
                    bootstrap_auc_difference, dump_json, finite, parse_formal,
                    read_gzip_json, write_csv)


def load_features(rows: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, list[str]]]:
    compact = []
    for index, row in enumerate(rows, 1):
        record = read_gzip_json(FEATURE_DIR / f"{row['sha256']}.json.gz")
        compact.append(record["variants"])
        if index % 100 == 0 or index == len(rows):
            print(f"CMW_POSTGATE_FEATURE_LOAD {index}/{len(rows)}", flush=True)
    output = {}
    for arm in ARMS:
        if arm in {"REAL", "ONE_HOP", "ROOT_SHUFFLE", "CALLSITE_SHUFFLE"}:
            variant = arm
            fields = sorted(key for key in compact[0][variant] if key not in COUNT_FIELDS)
        elif arm == "COUNT_ONLY":
            variant = "REAL"; fields = list(COUNT_FIELDS)
        else:
            variant = "REAL"
            fields = sorted(key for key in compact[0][variant]
                            if any(token in key for token in ("inverse_distance", "reachable", "user_share", "auto_share", "depth_")))
        matrix = np.asarray([[float(record[variant].get(field, 0.0)) for field in fields] for record in compact], float)
        if arm == "COUNT_ONLY":
            matrix = np.log1p(np.maximum(matrix, 0.0))
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"nonfinite matrix arm={arm}")
        output[arm] = (matrix, fields)
    return output


def fit_score(x_train, y_train, x_test, seed):
    y_train = np.asarray(y_train, int)
    if len(np.unique(y_train)) < 2:
        return np.full(len(x_test), float(np.mean(y_train)) if len(y_train) else .5)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=.25, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed))
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def crossfit(rows, matrix, seed):
    folds = np.asarray([row["fold"] for row in rows], int)
    labels = np.asarray([row["label"] for row in rows], int)
    stage1 = np.asarray([row["stage1_pred"] for row in rows], int)
    routed = np.asarray([row["routed"] for row in rows], bool)
    score = np.full(len(rows), np.nan); fold_rows = []
    for fold in range(1, 6):
        train_all, validation_all = folds != fold, folds == fold
        for side in (0, 1):
            train = train_all & (stage1 == side); validation = validation_all & (stage1 == side)
            target_train = (labels[train] == (1 if side == 0 else 0)).astype(int)
            local = fit_score(matrix[train], target_train, matrix[validation], seed + fold * 100 + side)
            score[validation] = local
            target_validation = (labels[validation] == (1 if side == 0 else 0)).astype(int)
            routed_validation = routed[validation]
            fold_rows.append({"fold": fold, "side": "FN_vs_TN" if side == 0 else "FP_vs_TP",
                              "n": int(validation.sum()), "positive": int(target_validation.sum()),
                              "AUC": binary_auc(target_validation, local),
                              "routed_n": int(routed_validation.sum()),
                              "routed_positive": int(target_validation[routed_validation].sum()),
                              "routed_AUC": binary_auc(target_validation[routed_validation], local[routed_validation])})
    if not np.isfinite(score).all():
        raise RuntimeError("OOF score incomplete")
    return score, fold_rows


def side_arrays(rows, score, side, routed_only=False):
    labels = np.asarray([row["label"] for row in rows], int)
    prediction = np.asarray([row["stage1_pred"] for row in rows], int)
    mask = prediction == side
    if routed_only:
        mask &= np.asarray([row["routed"] for row in rows], bool)
    target = (labels[mask] == (1 if side == 0 else 0)).astype(int)
    return mask, target, np.asarray(score)[mask]


def main() -> None:
    rows = parse_formal(); matrices = load_features(rows)
    all_scores, fold_output, arm_summary, prediction_rows = {}, [], {}, []
    for ordinal, arm in enumerate(ARMS):
        matrix, fields = matrices[arm]
        score, local_folds = crossfit(rows, matrix, SEED + ordinal * 1000)
        all_scores[arm] = score
        for item in local_folds:
            fold_output.append({"arm": arm, **item})
        _, fn_target, fn_score = side_arrays(rows, score, 0)
        _, fp_target, fp_score = side_arrays(rows, score, 1)
        _, routed_fn_target, routed_fn_score = side_arrays(rows, score, 0, routed_only=True)
        _, routed_fp_target, routed_fp_score = side_arrays(rows, score, 1, routed_only=True)
        arm_summary[arm] = {"feature_count": len(fields), "FN_vs_TN_AUC": binary_auc(fn_target, fn_score),
                            "FP_vs_TP_AUC": binary_auc(fp_target, fp_score),
                            "routed_FN_vs_TN_AUC": binary_auc(routed_fn_target, routed_fn_score),
                            "routed_FP_vs_TP_AUC": binary_auc(routed_fp_target, routed_fp_score),
                            "FN_n": len(fn_target), "FN_positive": int(fn_target.sum()),
                            "FP_n": len(fp_target), "FP_positive": int(fp_target.sum()),
                            "routed_FN_n": len(routed_fn_target), "routed_FN_positive": int(routed_fn_target.sum()),
                            "routed_FP_n": len(routed_fp_target), "routed_FP_positive": int(routed_fp_target.sum()),
                            "FN_folds_above_random": sum(finite(item["AUC"]) and float(item["AUC"]) > .5 for item in local_folds if item["side"] == "FN_vs_TN"),
                            "FP_folds_above_random": sum(finite(item["AUC"]) and float(item["AUC"]) > .5 for item in local_folds if item["side"] == "FP_vs_TP")}
        for index, row in enumerate(rows):
            prediction_rows.append({"sha256": row["sha256"], "fold": row["fold"], "label": row["label"],
                                    "stage1_prob": row["stage1_prob"], "stage1_pred": row["stage1_pred"],
                                    "routed": row["routed"], "arm": arm, "direction_score": score[index]})
        print(f"CMW_POSTGATE_ARM_DONE arm={arm} FN_AUC={arm_summary[arm]['FN_vs_TN_AUC']:.6f} FP_AUC={arm_summary[arm]['FP_vs_TP_AUC']:.6f}", flush=True)

    comparisons = {}
    for comparator in ("ONE_HOP", "ROOT_SHUFFLE", "CALLSITE_SHUFFLE", "COUNT_ONLY", "PATH_ONLY"):
        comparisons[comparator] = {}
        for side, name, routed_only in ((0, "FN_vs_TN", False), (1, "FP_vs_TP", False),
                                        (0, "ROUTED_FN_vs_TN", True), (1, "ROUTED_FP_vs_TP", True)):
            mask, target, real_score = side_arrays(rows, all_scores["REAL"], side, routed_only=routed_only)
            comparator_score = all_scores[comparator][mask]
            result = bootstrap_auc_difference(target, real_score, comparator_score,
                                              SEED + 10000 + ARMS.index(comparator) * 10 + side)
            side_name = "FN_vs_TN" if side == 0 else "FP_vs_TP"
            eligible = [item for item in fold_output if item["arm"] in {"REAL", comparator} and item["side"] == side_name]
            by = {(item["arm"], item["fold"]): item for item in eligible}
            metric_key = "routed_AUC" if routed_only else "AUC"
            result["real_better_folds"] = sum(
                finite(by[("REAL", fold)][metric_key]) and finite(by[(comparator, fold)][metric_key])
                and float(by[("REAL", fold)][metric_key]) > float(by[(comparator, fold)][metric_key])
                for fold in range(1, 6))
            comparisons[comparator][name] = result

    best_null_fn = max(arm_summary[name]["FN_vs_TN_AUC"] for name in ("ROOT_SHUFFLE", "CALLSITE_SHUFFLE", "COUNT_ONLY"))
    best_null_fp = max(arm_summary[name]["FP_vs_TP_AUC"] for name in ("ROOT_SHUFFLE", "CALLSITE_SHUFFLE", "COUNT_ONLY"))
    best_null_routed_fn = max(arm_summary[name]["routed_FN_vs_TN_AUC"] for name in ("ROOT_SHUFFLE", "CALLSITE_SHUFFLE", "COUNT_ONLY"))
    best_null_routed_fp = max(arm_summary[name]["routed_FP_vs_TP_AUC"] for name in ("ROOT_SHUFFLE", "CALLSITE_SHUFFLE", "COUNT_ONLY"))
    real = arm_summary["REAL"]; one = arm_summary["ONE_HOP"]
    criteria = {"REAL_FN_AUC>=0.60": real["FN_vs_TN_AUC"] >= .60,
                "REAL_FP_AUC>=0.60": real["FP_vs_TP_AUC"] >= .60,
                "REAL_routed_FN_AUC>=0.55": real["routed_FN_vs_TN_AUC"] >= .55,
                "REAL_routed_FP_AUC>=0.55": real["routed_FP_vs_TP_AUC"] >= .55,
                "REAL_routed_FN>=ONE_HOP+0.02": real["routed_FN_vs_TN_AUC"] >= one["routed_FN_vs_TN_AUC"] + .02,
                "REAL_routed_FP>=ONE_HOP+0.02": real["routed_FP_vs_TP_AUC"] >= one["routed_FP_vs_TP_AUC"] + .02,
                "REAL_routed_FN>=BEST_NULL+0.03": real["routed_FN_vs_TN_AUC"] >= best_null_routed_fn + .03,
                "REAL_routed_FP>=BEST_NULL+0.03": real["routed_FP_vs_TP_AUC"] >= best_null_routed_fp + .03,
                "REAL_FN>=ONE_HOP+0.02": real["FN_vs_TN_AUC"] >= one["FN_vs_TN_AUC"] + .02,
                "REAL_FP>=ONE_HOP+0.02": real["FP_vs_TP_AUC"] >= one["FP_vs_TP_AUC"] + .02,
                "REAL_FN>=BEST_NULL+0.03": real["FN_vs_TN_AUC"] >= best_null_fn + .03,
                "REAL_FP>=BEST_NULL+0.03": real["FP_vs_TP_AUC"] >= best_null_fp + .03,
                "REAL_FN_above_random_in>=3folds": real["FN_folds_above_random"] >= 3,
                "REAL_FP_above_random_in>=3folds": real["FP_folds_above_random"] >= 3,
                "REAL_beats_ONE_HOP_FN_in>=3folds": comparisons["ONE_HOP"]["FN_vs_TN"]["real_better_folds"] >= 3,
                "REAL_beats_ONE_HOP_FP_in>=3folds": comparisons["ONE_HOP"]["FP_vs_TP"]["real_better_folds"] >= 3}
    supported = all(criteria.values())
    status = "MULTIHOP_DIRECTION_SIGNAL_PRESENT" if supported else "MULTIHOP_DIRECTION_NOT_SUPPORTED"
    protocol = {"name": "CMW post-gate source sufficiency audit v1", "source": 2017,
                "future_rows_loaded": 0, "features": "frozen CMW v1 label-free features",
                "arms": list(ARMS), "model": "fixed StandardScaler + balanced L2 LogisticRegression C=0.25",
                "evaluation": "fixed five-fold side-conditional OOF", "bootstrap_repeats": 2000,
                "boundary": "diagnostic cannot revise the frozen v1 coverage-gate decision"}
    dump_json(OUTPUT / "protocol.json", protocol)
    protocol_hash = hashlib.sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest()
    (OUTPUT / "protocol_hash.txt").write_text(protocol_hash + "\n", encoding="utf-8")
    dump_json(OUTPUT / "direction_results.json", {"arms": arm_summary, "comparisons": comparisons,
              "criteria": criteria, "status": status, "future_rows_loaded": 0})
    write_csv(OUTPUT / "fold_results.csv", fold_output)
    write_csv(OUTPUT / "source_predictions.csv", prediction_rows)
    dump_json(OUTPUT / "final_decision.json", {"status": status, "criteria": criteria,
              "protocol_hash": protocol_hash, "future_rows_loaded": 0,
              "v1_decision_unchanged": "NOT_FEASIBLE", "end_to_end_run": False,
              "future_diagnostics_run": False})
    print("CMW_POSTGATE_SOURCE_DECISION=" + status); print(json.dumps({"arms": arm_summary, "criteria": criteria}, indent=2), flush=True)


if __name__ == "__main__":
    main()
