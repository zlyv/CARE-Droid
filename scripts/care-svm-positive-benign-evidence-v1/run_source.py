#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import (FUNCTION_INVENTORY, MAX_DELTA, OUTPUT, SEED, SOURCE_P31, SOURCE_STAGE2,
                    binary_auc, dump_json, logit, metrics, norm, read_csv, sigmoid, spearman, write_csv)
from evidence import (context_matrix, fit_positive_lift, oof_certificates,
                      role_arrays, transform_certificate)


def auc_or_none(y, score):
    value = binary_auc(y, score)
    return None if not np.isfinite(value) else float(value)


def matched_null_auc(y, score, fold, base, context_count, repetitions=2000):
    rng = np.random.default_rng(SEED); values = []
    base_bin = np.minimum((np.clip(base, .5, .999999) - .5) * 6, 2).astype(int)
    count_bin = np.minimum(np.asarray(context_count, int), 2)
    for _ in range(repetitions):
        shuffled = np.asarray(score, float).copy()
        for current in sorted(set(fold.tolist())):
            for confidence in range(3):
                for count in range(3):
                    index = np.flatnonzero((fold == current) & (base_bin == confidence) & (count_bin == count))
                    if len(index) > 1:
                        shuffled[index] = shuffled[rng.permutation(index)]
        value = binary_auc(y, shuffled)
        if np.isfinite(value): values.append(float(value))
    return np.asarray(values, float)


def main():
    rows = sorted([row for row in read_csv(SOURCE_STAGE2) if norm(row.get("arm")) == "svm"], key=lambda row: norm(row["sha256"]))
    ids = [norm(row["sha256"]) for row in rows]; y = np.asarray([int(row["label"]) for row in rows])
    fold = np.asarray([int(row["fold"]) for row in rows]); base = np.asarray([float(row["stage1_prob"]) for row in rows])
    candidate = np.asarray([float(row["stage2_prob"]) for row in rows]); route = np.asarray([int(row["routed"]) for row in rows], bool)
    context, raw_count = context_matrix(ids, read_csv(FUNCTION_INVENTORY))
    role, role_weight, role_active = role_arrays(ids, SOURCE_P31)
    evidence, model_audit = oof_certificates(y, fold, context, role_weight, role_active)

    predicted_malware = route & (base >= .5); target = (y[predicted_malware] == 0).astype(int)
    score = evidence["certificate"][predicted_malware]; local_fold = fold[predicted_malware]
    local_base = base[predicted_malware]; local_count = evidence["active_context_count"][predicted_malware]
    primary_auc = auc_or_none(target, score)
    count_score = local_count + np.sum(role_active[predicted_malware], axis=1) / 31.0
    count_auc = auc_or_none(target, count_score)
    null = matched_null_auc(target, score, local_fold, local_base, local_count)
    null95 = float(np.quantile(null, .95))

    fold_rows = []
    for current in range(1, 6):
        selected = predicted_malware & (fold == current); local_y = (y[selected] == 0).astype(int)
        fold_rows.append({"fold": current, "n": int(selected.sum()), "FP": int(local_y.sum()),
            "TP": int(len(local_y) - local_y.sum()), "certificate_FP_vs_TP_AUC": auc_or_none(local_y, evidence["certificate"][selected]),
            "count_only_FP_vs_TP_AUC": auc_or_none(local_y, evidence["active_context_count"][selected] + np.sum(role_active[selected], axis=1) / 31.0)})
    evaluable = [row["certificate_FP_vs_TP_AUC"] for row in fold_rows if row["certificate_FP_vs_TP_AUC"] is not None]
    correlation = {"invoke_context_count": spearman(evidence["certificate"], raw_count.sum(axis=1)),
        "active_context_count": spearman(evidence["certificate"], evidence["active_context_count"]),
        "active_role_count": spearman(evidence["certificate"], role_active.sum(axis=1)),
        "p31_mass": spearman(evidence["certificate"], role.sum(axis=1))}
    max_corr = max(abs(value or 0.0) for value in correlation.values())
    criteria = {"certificate_AUC>=0.65": primary_auc is not None and primary_auc >= .65,
        "certificate>matched_shuffle_95pct+0.03": primary_auc is not None and primary_auc > null95 + .03,
        "certificate>count_only+0.03": primary_auc is not None and count_auc is not None and primary_auc > count_auc + .03,
        "majority_evaluable_folds>0.5": sum(value > .5 for value in evaluable) >= max(2, (len(evaluable) + 1) // 2),
        "max_abs_size_correlation<0.70": max_corr < .70}
    supported = all(criteria.values())

    # Source integration is deliberately label-free on the held-out fold.  The threshold is
    # the outer-train benign 75th percentile, not a threshold selected on validation outcomes.
    threshold_by_fold = {}; qualify = np.zeros(len(y), bool)
    for current in range(1, 6):
        train_benign = (fold != current) & (y == 0)
        # Refit on outer-train only and derive the reference distribution there.
        # No score whose construction used the current outer validation fold is
        # allowed to determine its threshold.
        threshold_model = fit_positive_lift(context, role_active, train_benign)
        train_values = transform_certificate(
            threshold_model, context[train_benign], role_weight[train_benign]
        )["certificate"]
        threshold = max(.05, float(np.quantile(train_values, .75)))
        threshold_by_fold[str(current)] = threshold
        qualify[fold == current] = evidence["certificate"][fold == current] >= threshold
    base_logit = logit(base); candidate_delta = np.clip(logit(candidate) - base_logit, -MAX_DELTA, MAX_DELTA)
    malware_only_delta = np.where(route & (base < .5), np.maximum(candidate_delta, 0.0), 0.0)
    benign_only_delta = np.where(route & (base >= .5) & qualify, -MAX_DELTA, 0.0)
    finals = {"S0_STAGE1_SVM": base, "S1_FORMAL_C43": candidate}
    if supported:
        finals.update({
            "T1_C43_MALWARE_SUPPORT_ONLY": sigmoid(base_logit + malware_only_delta),
            "T2_POSITIVE_BENIGN_CERTIFICATE_ONLY": sigmoid(base_logit + benign_only_delta),
            "T3_BIDIRECTIONAL_POSITIVE_EVIDENCE": sigmoid(base_logit + malware_only_delta + benign_only_delta),
        })
    result_rows = []; result_folds = []
    for arm, final in finals.items():
        result_rows.append({"arm": arm, **metrics(y, base, final, route)})
        for current in range(1, 6):
            selected = fold == current
            result_folds.append({"arm": arm, "fold": current, **metrics(y[selected], base[selected], final[selected], route[selected])})
    prediction_rows = []
    for index, sid in enumerate(ids):
        prediction_rows.append({"sha256": sid, "fold": int(fold[index]), "label": int(y[index]),
            "stage1_prob": float(base[index]), "routed": int(route[index]), "old_c43_prob": float(candidate[index]),
            "positive_benign_certificate": float(evidence["certificate"][index]),
            "weighted_positive_support": float(evidence["weighted_support"][index]),
            "explained_role_mass": float(evidence["explained_mass"][index]),
            "positive_evidence_threshold": threshold_by_fold[str(int(fold[index]))],
            "positive_evidence_qualified": int(qualify[index]),
            **{arm + "__prob": float(final[index]) for arm, final in finals.items()}})

    if supported:
        s1 = next(row for row in result_rows if row["arm"] == "S1_FORMAL_C43")
        t3 = next(row for row in result_rows if row["arm"] == "T3_BIDIRECTIONAL_POSITIVE_EVIDENCE")
        s1_fold = {row["fold"]: row for row in result_folds if row["arm"] == "S1_FORMAL_C43"}
        t3_fold = {row["fold"]: row for row in result_folds if row["arm"] == "T3_BIDIRECTIONAL_POSITIVE_EVIDENCE"}
        integration_criteria = {"evidence_gate_passed": True,
            "T3_fixed_FP>=S1_fixed_FP": t3["fixed_FP"] >= s1["fixed_FP"],
            "T3_broken_TN<=S1_broken_TN": t3["broken_TN"] <= s1["broken_TN"],
            "T3_net>=S1_net": t3["net"] >= s1["net"],
            "T3_noninferior_folds>=3": sum(t3_fold[f]["net"] >= s1_fold[f]["net"] for f in range(1, 6)) >= 3}
        integration_pass = all(integration_criteria.values())
    else:
        integration_criteria = {"evidence_gate_passed": False, "integration_run": False,
            "reason": "positive benign evidence did not beat the frozen source controls"}
        integration_pass = False
    status = ("SOURCE_INTEGRATION_SUPPORTED_FOR_FROZEN_FUTURE" if integration_pass else
              "POSITIVE_BENIGN_EVIDENCE_NOT_CONFIRMED" if not supported else
              "POSITIVE_EVIDENCE_PRESENT_BUT_INTEGRATION_NOT_BETTER")

    write_csv(OUTPUT / "fold_probe_results.csv", fold_rows); write_csv(OUTPUT / "source_results.csv", result_rows)
    write_csv(OUTPUT / "source_fold_results.csv", result_folds); write_csv(OUTPUT / "source_predictions.csv", prediction_rows)
    write_csv(OUTPUT / "matched_null_auc.csv", [{"iteration": i, "AUC": value} for i, value in enumerate(null)])
    dump_json(OUTPUT / "model_audit.json", model_audit)
    dump_json(OUTPUT / "evidence_probe.json", {"status": "PASS" if supported else "FAIL", "FP": int(target.sum()),
        "TP": int(len(target) - target.sum()), "certificate_FP_vs_TP_AUC": primary_auc,
        "count_only_FP_vs_TP_AUC": count_auc, "matched_shuffle_95pct": null95,
        "correlations": correlation, "criteria": criteria, "future_rows_loaded": 0,
        "semantic_boundary": "zero/weak anomaly is never benign evidence; only observed positive context-to-Role lift contributes"})
    dump_json(OUTPUT / "source_decision.json", {"status": status, "evidence_criteria": criteria,
        "integration_criteria": integration_criteria, "threshold_by_fold": threshold_by_fold,
        "continue_to_frozen_future": integration_pass, "future_rows_loaded": 0})
    print("SVM_POSITIVE_BENIGN_SOURCE_DONE")
    print(f"POSITIVE_CERTIFICATE_FP_VS_TP_AUC={primary_auc:.6f} COUNT_ONLY_AUC={count_auc:.6f} NULL95={null95:.6f}")
    for row in result_rows:
        print(f"{row['arm']}: BA={row['BA']:.6f} fixed={row['fixed']} broken={row['broken']} net={row['net']} fixed_FP={row['fixed_FP']} fixed_FN={row['fixed_FN']}")
    print("SVM_POSITIVE_BENIGN_SOURCE_DECISION=" + status)


if __name__ == "__main__":
    main()

