#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import (FORMAL, FORMAL_METHOD, MAX_DELTA, MEMBER, OUTPUT, binary_auc,
    correction_metrics, dump_json, logit, norm, read_csv, sigmoid, write_csv)


def finite(value):
    return None if not np.isfinite(value) else float(value)


def main():
    members = {norm(row["sha256"]): row for row in read_csv(MEMBER)}
    formal = sorted(read_csv(FORMAL), key=lambda row: norm(row["sha256"]))
    sha = np.asarray([norm(row["sha256"]) for row in formal]); y = np.asarray([int(float(row["label"])) for row in formal])
    fold = np.asarray([int(float(row["fold"])) for row in formal]); route = np.asarray([int(float(row["A25__routed"])) for row in formal], bool)
    base = np.asarray([float(row["Global24__prob"]) for row in formal]); formal_final = np.asarray([float(row[FORMAL_METHOD + "__prob"]) for row in formal])
    if int(route.sum()) != 200:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: fixed RF route changed {route.sum()}")
    matrix = np.asarray([[float(members[sid][f"p_P31__{name}"]) for name in ("LR", "RF", "XGB", "SVM")] for sid in sha])
    if np.max(np.abs(matrix[:, 1] - base)) > 1e-12:
        raise RuntimeError("BLOCKED_BY_CONTRACT: RF member probability differs from formal Stage-I")

    rf_logit = logit(base); other_logits = logit(matrix[:, [0, 2, 3]])
    consensus = np.median(other_logits, axis=1); committee_delta = np.clip(consensus - rf_logit, -MAX_DELTA, MAX_DELTA)
    c43_delta = np.where(route, np.clip(logit(formal_final) - rf_logit, -MAX_DELTA, MAX_DELTA), 0.0)
    formula_delta = np.where(route, np.clip(c43_delta + committee_delta, -MAX_DELTA, MAX_DELTA), 0.0)
    committee_only_delta = np.where(route, committee_delta, 0.0)
    primary_final = sigmoid(rf_logit + formula_delta); committee_final = sigmoid(rf_logit + committee_only_delta)

    arms = {
        "S0_STAGE1_RF": base,
        "U0_FORMAL_C43": formal_final,
        "U1_COMMITTEE_ONLY_FIXED": committee_final,
        "U2_C43_PLUS_COMMITTEE_FIXED_PRIMARY": primary_final,
    }
    summaries = []
    for arm, probability in arms.items():
        summaries.append({"arm": arm, **correction_metrics(y, base, probability, route)})
    write_csv(OUTPUT / "routed_200_results.csv", summaries)

    rows = []
    for i in range(len(y)):
        rows.append({"sha256": sha[i], "fold": int(fold[i]), "label": int(y[i]), "routed": int(route[i]),
            "p_RF": float(base[i]), "p_LR": float(matrix[i, 0]), "p_XGB": float(matrix[i, 2]), "p_SVM": float(matrix[i, 3]),
            "c43_delta": float(c43_delta[i]), "committee_delta": float(committee_delta[i]),
            "primary_delta": float(formula_delta[i]), "formal_probability": float(formal_final[i]),
            "primary_probability": float(primary_final[i]),
            "stage1_correct": int((base[i] >= .5) == y[i]),
            "formal_fixed": int(((base[i] >= .5) != y[i]) and ((formal_final[i] >= .5) == y[i])),
            "formal_broken": int(((base[i] >= .5) == y[i]) and ((formal_final[i] >= .5) != y[i])),
            "primary_fixed": int(((base[i] >= .5) != y[i]) and ((primary_final[i] >= .5) == y[i])),
            "primary_broken": int(((base[i] >= .5) == y[i]) and ((primary_final[i] >= .5) != y[i]))})
    write_csv(OUTPUT / "source_predictions.csv", rows)

    # Direction sufficiency is evaluated only; it is not used to tune the fixed formula.
    predicted_malware = route & (base >= .5); predicted_benign = route & (base < .5)
    fp_y = (y[predicted_malware] == 0).astype(int); fn_y = (y[predicted_benign] == 1).astype(int)
    pooled_probe = {"FP_vs_TP_AUC_benign_support": finite(binary_auc(fp_y, -committee_delta[predicted_malware])),
                    "FN_vs_TN_AUC_malware_support": finite(binary_auc(fn_y, committee_delta[predicted_benign])),
                    "predicted_malware_routed_n": int(predicted_malware.sum()),
                    "predicted_benign_routed_n": int(predicted_benign.sum())}
    folds = []
    for f in range(1, 6):
        pm = predicted_malware & (fold == f); pb = predicted_benign & (fold == f)
        folds.append({"fold": f, "routed": int(np.sum(route & (fold == f))),
            "FP_vs_TP_AUC": finite(binary_auc((y[pm] == 0).astype(int), -committee_delta[pm])),
            "FN_vs_TN_AUC": finite(binary_auc((y[pb] == 1).astype(int), committee_delta[pb]))})
    write_csv(OUTPUT / "direction_probe_folds.csv", folds)

    # Matched null: keep the fixed routed cohort and marginal distributions; permute committee evidence
    # within each fold and RF-predicted side. No null is used to alter the primary formula.
    rng = np.random.default_rng(20261009); null_rows = []
    for repeat in range(200):
        shuffled = committee_delta.copy()
        for f in range(1, 6):
            for side in (0, 1):
                idx = np.flatnonzero(route & (fold == f) & ((base >= .5) == side))
                shuffled[idx] = shuffled[rng.permutation(idx)]
        delta = np.where(route, np.clip(c43_delta + shuffled, -MAX_DELTA, MAX_DELTA), 0.0)
        metrics = correction_metrics(y, base, sigmoid(rf_logit + delta), route)
        null_rows.append({"repeat": repeat, "fixed": metrics["fixed"], "broken": metrics["broken"], "net": metrics["net"], "BA": metrics["BA"]})
    write_csv(OUTPUT / "matched_shuffle_results.csv", null_rows)

    formal_metric = next(row for row in summaries if row["arm"] == "U0_FORMAL_C43")
    primary_metric = next(row for row in summaries if row["arm"] == "U2_C43_PLUS_COMMITTEE_FIXED_PRIMARY")
    null_net = np.asarray([row["net"] for row in null_rows], float)
    criteria = {"more_fixed_than_formal": primary_metric["fixed"] > formal_metric["fixed"],
                "net_greater_than_formal": primary_metric["net"] > formal_metric["net"],
                "broken_not_greater_than_formal": primary_metric["broken"] <= formal_metric["broken"],
                "net_above_shuffle_95pct": primary_metric["net"] > float(np.quantile(null_net, .95))}
    if all(criteria.values()):
        status = "SUPPORTED_FOR_STRICT_NESTED_FOLLOWUP"
    elif criteria["more_fixed_than_formal"]:
        status = "MORE_FIXED_BUT_NOT_NET_BETTER"
    else:
        status = "NOT_SUPPORTED_BY_ROUTED_200_PROBE"
    decision = {"status": status, "criteria": criteria, "formal": formal_metric, "primary": primary_metric,
        "direction_probe": pooled_probe, "shuffle_net_95pct": float(np.quantile(null_net, .95)),
        "formula_was_fit_on_labels": False, "fixed_route_rows": 200, "future_rows_loaded": 0,
        "interpretation": "This is a parameter-free routed-200 source probe. It does not authorize future diagnostics or formal replacement."}
    dump_json(OUTPUT / "final_decision.json", decision)
    print("UNIFIED_ROUTED_RESIDUAL_200_DONE", flush=True)
    for row in summaries:
        print(f"{row['arm']}: BA={row['BA']:.6f} fixed={row['fixed']} broken={row['broken']} net={row['net']} "
              f"fixed_FP={row['fixed_FP']} fixed_FN={row['fixed_FN']} broken_TN={row['broken_TN']} broken_TP={row['broken_TP']}", flush=True)
    print(f"FP_VS_TP_AUC={pooled_probe['FP_vs_TP_AUC_benign_support']} FN_VS_TN_AUC={pooled_probe['FN_vs_TN_AUC_malware_support']}", flush=True)
    print("UNIFIED_ROUTED_RESIDUAL_DECISION=" + status, flush=True)


if __name__ == "__main__":
    main()
