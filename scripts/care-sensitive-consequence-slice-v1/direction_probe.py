#!/usr/bin/env python3
from __future__ import annotations

import math

import numpy as np

from common import OUTPUT, SEED, assets, binary_auc, dump_json, finite, parse_formal, read_gzip_json, write_csv

ARMS = ("REAL", "LOCAL_ONLY", "ONE_HOP", "EFFECT_SHUFFLE", "EDGE_SHUFFLE", "COUNT_ONLY", "PATH_ONLY")


def matrices(rows):
    folder = OUTPUT / "cache/features/source"
    records = [read_gzip_json(folder / f"{row['sha256']}.json.gz") for row in rows]
    output = {}
    for arm in ARMS:
        fields = sorted(records[0]["variants"][arm])
        matrix = np.asarray([[float(record["variants"][arm].get(field, 0.0)) for field in fields] for record in records], float)
        if arm == "COUNT_ONLY":
            matrix = np.log1p(np.maximum(matrix, 0.0))
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"nonfinite matrix arm={arm}")
        output[arm] = (matrix, fields)
    return output


def fit(train_x, train_y, test_x, seed):
    train_y = np.asarray(train_y, int)
    if len(np.unique(train_y)) < 2:
        return np.full(len(test_x), float(np.mean(train_y)) if len(train_y) else .5)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.25, class_weight="balanced", solver="liblinear",
        max_iter=2000, random_state=seed))
    model.fit(train_x, train_y); return model.predict_proba(test_x)[:, 1]


def crossfit(rows, matrix, seed):
    labels = np.asarray([row["label"] for row in rows], int); prediction = np.asarray([row["stage1_pred"] for row in rows], int)
    folds = np.asarray([row["fold"] for row in rows], int); routed = np.asarray([row["routed"] for row in rows], bool)
    scores = np.full(len(rows), np.nan); metrics = []
    for fold in range(1, 6):
        for side in (0, 1):
            train = (folds != fold) & (prediction == side); test = (folds == fold) & (prediction == side)
            train_y = (labels[train] == (1 if side == 0 else 0)).astype(int)
            local = fit(matrix[train], train_y, matrix[test], seed + fold * 100 + side); scores[test] = local
            test_y = (labels[test] == (1 if side == 0 else 0)).astype(int); local_route = routed[test]
            metrics.append({"fold": fold, "side": "FN_vs_TN" if side == 0 else "FP_vs_TP", "n": int(test.sum()),
                "positive": int(test_y.sum()), "AUC": binary_auc(test_y, local), "routed_n": int(local_route.sum()),
                "routed_positive": int(test_y[local_route].sum()), "routed_AUC": binary_auc(test_y[local_route], local[local_route])})
    if not np.isfinite(scores).all():
        raise RuntimeError("OOF scores incomplete")
    return scores, metrics


def summarize(rows, scores, folds):
    labels = np.asarray([row["label"] for row in rows], int); pred = np.asarray([row["stage1_pred"] for row in rows], int)
    routed = np.asarray([row["routed"] for row in rows], bool)
    result = {}
    for side, name in ((0, "FN"), (1, "FP")):
        mask = pred == side; target = (labels[mask] == (1 if side == 0 else 0)).astype(int)
        rmask = mask & routed; rtarget = (labels[rmask] == (1 if side == 0 else 0)).astype(int)
        result[f"{name}_vs_{'TN' if side == 0 else 'TP'}_AUC"] = binary_auc(target, scores[mask])
        result[f"routed_{name}_vs_{'TN' if side == 0 else 'TP'}_AUC"] = binary_auc(rtarget, scores[rmask])
        side_name = "FN_vs_TN" if side == 0 else "FP_vs_TP"
        result[f"{name}_folds_above_random"] = sum(finite(item["AUC"]) and float(item["AUC"]) > .5 for item in folds if item["side"] == side_name)
        result[f"routed_{name}_folds_above_random"] = sum(finite(item["routed_AUC"]) and float(item["routed_AUC"]) > .5 for item in folds if item["side"] == side_name)
    return result


def main() -> None:
    rows = parse_formal(assets()["formal"]); source = matrices(rows); summaries = {}; fold_rows = []; predictions = []
    for ordinal, arm in enumerate(ARMS):
        matrix, fields = source[arm]; scores, folds = crossfit(rows, matrix, SEED + ordinal * 1000)
        summary = summarize(rows, scores, folds); summary["feature_count"] = len(fields); summaries[arm] = summary
        fold_rows.extend({"arm": arm, **item} for item in folds)
        predictions.extend({"sha256": row["sha256"], "fold": row["fold"], "label": row["label"], "stage1_pred": row["stage1_pred"],
            "routed": row["routed"], "arm": arm, "direction_score": float(scores[index])} for index, row in enumerate(rows))
        print(f"SCS_ARM_DONE arm={arm} FN={summary['FN_vs_TN_AUC']:.6f} FP={summary['FP_vs_TP_AUC']:.6f} routed_FN={summary['routed_FN_vs_TN_AUC']:.6f} routed_FP={summary['routed_FP_vs_TP_AUC']:.6f}", flush=True)
    real = summaries["REAL"]; nulls = ("EFFECT_SHUFFLE", "EDGE_SHUFFLE", "COUNT_ONLY")
    best_fn = max(summaries[arm]["FN_vs_TN_AUC"] for arm in nulls); best_fp = max(summaries[arm]["FP_vs_TP_AUC"] for arm in nulls)
    best_rfn = max(summaries[arm]["routed_FN_vs_TN_AUC"] for arm in nulls); best_rfp = max(summaries[arm]["routed_FP_vs_TP_AUC"] for arm in nulls)
    criteria = {"REAL_FN>=0.60": real["FN_vs_TN_AUC"] >= .60, "REAL_FP>=0.60": real["FP_vs_TP_AUC"] >= .60,
        "REAL_routed_FN>=0.55": real["routed_FN_vs_TN_AUC"] >= .55, "REAL_routed_FP>=0.55": real["routed_FP_vs_TP_AUC"] >= .55,
        "REAL_FN>=BEST_NULL+0.03": real["FN_vs_TN_AUC"] >= best_fn + .03,
        "REAL_FP>=BEST_NULL+0.03": real["FP_vs_TP_AUC"] >= best_fp + .03,
        "REAL_routed_FN>=BEST_NULL+0.02": real["routed_FN_vs_TN_AUC"] >= best_rfn + .02,
        "REAL_routed_FP>=BEST_NULL+0.02": real["routed_FP_vs_TP_AUC"] >= best_rfp + .02,
        "REAL_FN_above_random_in>=3folds": real["FN_folds_above_random"] >= 3,
        "REAL_FP_above_random_in>=3folds": real["FP_folds_above_random"] >= 3,
        "REAL_beats_ONE_HOP_FN": real["FN_vs_TN_AUC"] >= summaries["ONE_HOP"]["FN_vs_TN_AUC"] + .02,
        "REAL_beats_ONE_HOP_FP": real["FP_vs_TP_AUC"] >= summaries["ONE_HOP"]["FP_vs_TP_AUC"] + .02}
    fn_supported = all(criteria[key] for key in ("REAL_FN>=0.60", "REAL_routed_FN>=0.55", "REAL_FN>=BEST_NULL+0.03", "REAL_routed_FN>=BEST_NULL+0.02", "REAL_FN_above_random_in>=3folds"))
    fp_supported = all(criteria[key] for key in ("REAL_FP>=0.60", "REAL_routed_FP>=0.55", "REAL_FP>=BEST_NULL+0.03", "REAL_routed_FP>=BEST_NULL+0.02", "REAL_FP_above_random_in>=3folds"))
    multihop_supported = criteria["REAL_beats_ONE_HOP_FN"] and criteria["REAL_beats_ONE_HOP_FP"]
    status = "SUPPORTED_FOR_BOUNDED_CORRECTION" if fn_supported and fp_supported and multihop_supported else "PARTIALLY_SUPPORTED" if fn_supported != fp_supported else "NOT_SUPPORTED"
    result = {"status": status, "arms": summaries, "criteria": criteria, "fn_supported": fn_supported,
        "fp_supported": fp_supported, "pruned_multihop_supported": multihop_supported,
        "future_rows_loaded": 0, "end_to_end_run": False,
        "interpretation": "This probe evaluates call-reachability consequence slices, not data flow."}
    dump_json(OUTPUT / "direction_probe.json", result); dump_json(OUTPUT / "final_decision.json", result)
    write_csv(OUTPUT / "source_fold_results.csv", fold_rows); write_csv(OUTPUT / "source_predictions.csv", predictions)
    print("SCS_SOURCE_DECISION=" + status, flush=True)


if __name__ == "__main__":
    main()
