#!/usr/bin/env python3
from __future__ import annotations

import math

import numpy as np

from common import OUTPUT, SEED, binary_auc, dump_json, finite, parse_formal, spearman, write_csv
from features import COUNT_FIELDS, counts, load_record, matrix, normalized, raw_semantic

ARMS = ("COUNT_ONLY", "REAL_RAW", "NORM_REAL", "NORM_RESIDUAL", "COUNT_PLUS_NORM", "NORM_EFFECT_SHUFFLE", "NORM_EDGE_SHUFFLE")


def fit_score(train_x, train_y, test_x, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.25, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed))
    model.fit(train_x, train_y); return model.predict_proba(test_x)[:, 1]


def residualize(train_sem, test_sem, train_count, test_count):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(train_count, train_sem)
    return train_sem - model.predict(train_count), test_sem - model.predict(test_count)


def build_matrices(records):
    count_x, count_fields = matrix([counts(record) for record in records]); count_x = np.log1p(np.maximum(count_x, 0.0))
    raw_x, raw_fields = matrix([raw_semantic(record) for record in records])
    norm_x, norm_fields = matrix([normalized(record, "REAL") for record in records])
    effect_x, effect_fields = matrix([normalized(record, "EFFECT_SHUFFLE") for record in records])
    edge_x, edge_fields = matrix([normalized(record, "EDGE_SHUFFLE") for record in records])
    if norm_fields != effect_fields or norm_fields != edge_fields:
        raise RuntimeError("matched normalized feature schema mismatch")
    return {"COUNT_ONLY": count_x, "REAL_RAW": raw_x, "NORM_REAL": norm_x,
            "COUNT_PLUS_NORM": np.hstack([count_x, norm_x]), "NORM_EFFECT_SHUFFLE": effect_x,
            "NORM_EDGE_SHUFFLE": edge_x}, {"count": count_fields, "raw": raw_fields, "norm": norm_fields}


def nearest_pairs(test_indices, target, counts_x, train_counts, fold):
    positive = [index for index in test_indices if target[index] == 1]; negative = [index for index in test_indices if target[index] == 0]
    mean = train_counts.mean(axis=0); std = train_counts.std(axis=0); std[std < 1e-9] = 1.0
    standardized = (counts_x - mean) / std; available = set(negative); pairs = []
    for positive_index in sorted(positive):
        if not available:
            break
        negative_index = min(available, key=lambda index: (float(np.linalg.norm(standardized[positive_index] - standardized[index])), index))
        distance = float(np.linalg.norm(standardized[positive_index] - standardized[negative_index])); available.remove(negative_index)
        pairs.append({"fold": fold, "fp_index": positive_index, "tp_index": negative_index, "distance": distance})
    return pairs


def main() -> None:
    rows = parse_formal(); records = [load_record(row["sha256"]) for row in rows]; matrices, schemas = build_matrices(records)
    labels = np.asarray([row["label"] for row in rows], int); prediction = np.asarray([row["stage1_pred"] for row in rows], int)
    folds = np.asarray([row["fold"] for row in rows], int); routed = np.asarray([row["routed"] for row in rows], bool)
    target = (labels == 0).astype(int); side = prediction == 1; count_x = matrices["COUNT_ONLY"]; norm_x = matrices["NORM_REAL"]
    scores = {arm: np.full(len(rows), np.nan) for arm in ARMS}; fold_rows = []; pairs = []
    for fold in range(1, 6):
        train = side & (folds != fold); test = side & (folds == fold); test_indices = np.flatnonzero(test)
        residual_train, residual_test = residualize(norm_x[train], norm_x[test], count_x[train], count_x[test])
        local = {arm: matrices[arm] for arm in ARMS if arm != "NORM_RESIDUAL"}
        for ordinal, arm in enumerate(ARMS):
            if arm == "NORM_RESIDUAL":
                train_x, test_x = residual_train, residual_test
            else:
                train_x, test_x = local[arm][train], local[arm][test]
            score = fit_score(train_x, target[train], test_x, SEED + fold * 100 + ordinal); scores[arm][test] = score
            local_route = routed[test]
            fold_rows.append({"arm": arm, "fold": fold, "n": int(test.sum()), "FP": int(target[test].sum()),
                "AUC": binary_auc(target[test], score), "routed_n": int(local_route.sum()),
                "routed_FP": int(target[test][local_route].sum()), "routed_AUC": binary_auc(target[test][local_route], score[local_route])})
        pairs.extend(nearest_pairs(test_indices, target, count_x, count_x[train], fold))

    for arm in ARMS:
        if not np.isfinite(scores[arm][side]).all():
            raise RuntimeError(f"incomplete OOF score arm={arm}")
    summaries = {}
    for arm in ARMS:
        summaries[arm] = {"FP_vs_TP_AUC": binary_auc(target[side], scores[arm][side]),
            "routed_FP_vs_TP_AUC": binary_auc(target[side & routed], scores[arm][side & routed]),
            "folds_above_random": sum(finite(row["AUC"]) and float(row["AUC"]) > .5 for row in fold_rows if row["arm"] == arm),
            "routed_folds_above_random": sum(finite(row["routed_AUC"]) and float(row["routed_AUC"]) > .5 for row in fold_rows if row["arm"] == arm)}

    pair_rows = []; matched_results = {}
    selected_indices = []
    for pair in pairs:
        selected_indices.extend([pair["fp_index"], pair["tp_index"]])
        pair_rows.append({"fold": pair["fold"], "fp_sha256": rows[pair["fp_index"]]["sha256"], "tp_sha256": rows[pair["tp_index"]]["sha256"], "standardized_count_distance": pair["distance"]})
    selected = np.asarray(selected_indices, int); matched_target = target[selected]
    for arm in ARMS:
        matched_results[arm] = {"matched_AUC": binary_auc(matched_target, scores[arm][selected]), "n": len(selected), "pairs": len(pairs)}

    count_audit = []
    for index, field in enumerate(schemas["count"]):
        raw_values = np.expm1(count_x[:, index]); fp_values = raw_values[side & (target == 1)]; tp_values = raw_values[side & (target == 0)]
        count_audit.append({"variable": field, "FP_mean": float(fp_values.mean()), "FP_median": float(np.median(fp_values)),
            "TP_mean": float(tp_values.mean()), "TP_median": float(np.median(tp_values)),
            "spearman_with_FP": spearman(raw_values[side], target[side]),
            "univariate_AUC": max(binary_auc(target[side], raw_values[side]), binary_auc(target[side], -raw_values[side]))})

    shortcut_rows = []
    for field_index, field in enumerate(schemas["norm"]):
        correlations = [abs(spearman(norm_x[:, field_index], count_x[:, count_index])) for count_index in range(count_x.shape[1])]
        shortcut_rows.append({"feature": field, "max_abs_spearman_with_log_counts": max(correlations)})
    max_shortcut = max(row["max_abs_spearman_with_log_counts"] for row in shortcut_rows)
    real = summaries["NORM_RESIDUAL"]; null_names = ("NORM_EFFECT_SHUFFLE", "NORM_EDGE_SHUFFLE")
    best_null = max(summaries[name]["FP_vs_TP_AUC"] for name in null_names)
    best_routed_null = max(summaries[name]["routed_FP_vs_TP_AUC"] for name in null_names)
    best_matched_null = max(matched_results[name]["matched_AUC"] for name in null_names)
    criteria = {"NORM_RESIDUAL_FP_AUC>=0.65": real["FP_vs_TP_AUC"] >= .65,
        "NORM_RESIDUAL_routed_FP_AUC>=0.60": real["routed_FP_vs_TP_AUC"] >= .60,
        "NORM_RESIDUAL>=BEST_NULL+0.03": real["FP_vs_TP_AUC"] >= best_null + .03,
        "NORM_RESIDUAL_routed>=BEST_NULL+0.03": real["routed_FP_vs_TP_AUC"] >= best_routed_null + .03,
        "NORM_RESIDUAL_matched_AUC>=0.60": matched_results["NORM_RESIDUAL"]["matched_AUC"] >= .60,
        "NORM_RESIDUAL_matched>=BEST_NULL+0.03": matched_results["NORM_RESIDUAL"]["matched_AUC"] >= best_matched_null + .03,
        "NORM_RESIDUAL_above_random_in>=3folds": real["folds_above_random"] >= 3,
        "normalized_feature_max_abs_count_corr<0.80": max_shortcut < .80,
        "COUNT_PLUS_NORM_not_worse_than_COUNT_ONLY": summaries["COUNT_PLUS_NORM"]["FP_vs_TP_AUC"] >= summaries["COUNT_ONLY"]["FP_vs_TP_AUC"] - .01}
    status = "FP_SEMANTIC_RESIDUAL_SUPPORTED" if all(criteria.values()) else "FP_SEMANTIC_RESIDUAL_NOT_SUPPORTED"
    result = {"status": status, "arms": summaries, "matched": matched_results, "criteria": criteria,
        "matched_pairs": len(pairs), "max_normalized_feature_count_correlation": max_shortcut,
        "future_rows_loaded": 0, "end_to_end_run": False}
    dump_json(OUTPUT / "fp_residual_results.json", result); dump_json(OUTPUT / "final_decision.json", result)
    write_csv(OUTPUT / "source_fold_results.csv", fold_rows); write_csv(OUTPUT / "matched_pairs.csv", pair_rows)
    write_csv(OUTPUT / "count_direction_audit.csv", count_audit); write_csv(OUTPUT / "shortcut_audit.csv", shortcut_rows)
    prediction_rows = []
    for index, row in enumerate(rows):
        if side[index]:
            prediction_rows.append({"sha256": row["sha256"], "fold": row["fold"], "label": row["label"], "routed": row["routed"],
                **{f"{arm}_score": float(scores[arm][index]) for arm in ARMS}})
    write_csv(OUTPUT / "source_predictions.csv", prediction_rows)
    print("FPR_SOURCE_DECISION=" + status); print(result, flush=True)


if __name__ == "__main__":
    main()
