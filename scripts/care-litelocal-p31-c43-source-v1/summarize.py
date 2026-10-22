#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path("artifacts/care-litelocal-p31-c43-source-v1")
ARM = ROOT / "arms/P31_C43_RQ"
METHOD = "P31Mean__DI25__C43RelationQuery__Bounded"


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def metric(y, probability):
    prediction = probability >= .5
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "BA": float(balanced_accuracy_score(y, prediction)),
        "AUC": float(roc_auc_score(y, probability)),
        "F1": float(f1_score(y, prediction)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "errors": int(np.sum(prediction != y)),
    }


def main():
    reference = sorted(read(ROOT / "preflight/p31_reference/source_predictions.csv"),
                       key=lambda row: row["sha256"].casefold())
    predictions = sorted(read(ARM / "source_predictions.csv"),
                         key=lambda row: row["sha256"].casefold())
    if [row["sha256"].casefold() for row in reference] != [
            row["sha256"].casefold() for row in predictions]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: prediction cohort mismatch")
    y = np.asarray([int(float(row["label"])) for row in reference])
    fold = np.asarray([int(float(row["fold"])) for row in reference])
    base = np.asarray([float(row["P31_MEAN_COMMITTEE__prob"]) for row in reference])
    route = np.asarray([int(float(row["P31_DI25__routed"])) == 1 for row in reference])
    observed_base = np.asarray([float(row["Global24__prob"]) for row in predictions])
    observed_route = np.asarray([int(float(row["A25__routed"])) == 1
                                 for row in predictions])
    if np.max(np.abs(base - observed_base)) > 1e-12 or np.any(route != observed_route):
        raise RuntimeError("BLOCKED_BY_CONTRACT: P31 base or route changed")
    probability_column = METHOD + "__prob"
    expert_column = METHOD + "__expert_prob"
    if probability_column not in predictions[0] or expert_column not in predictions[0]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: C43-RQ probability missing")
    final = np.asarray([float(row[probability_column]) for row in predictions])
    expert = np.asarray([float(row[expert_column]) for row in predictions])
    if np.max(np.abs(final[~route] - base[~route])) > 1e-12:
        raise RuntimeError("BLOCKED_BY_CONTRACT: correction applied outside P31-DI25")
    base_correct = (base >= .5) == y
    final_correct = (final >= .5) == y
    fixed = int(np.sum(~base_correct & final_correct))
    broken = int(np.sum(base_correct & ~final_correct))
    n01, n10 = fixed, broken
    exact_p = float(binomtest(min(n01, n10), n01 + n10, .5).pvalue) if n01 + n10 else 1.0
    base_metric, final_metric = metric(y, base), metric(y, final)
    fit = read(ARM / "fit_audit.csv")
    parameters = {int(float(row["parameter_n"])) for row in fit}
    if len(fit) != 15 or parameters != {32292}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: fit/parameter audit failed")
    aggregate = [
        {"method": "P31_MEAN_COMMITTEE", **base_metric,
         "routed": 0, "fixed": 0, "broken": 0, "net": 0},
        {"method": METHOD, **final_metric, "routed": int(route.sum()),
         "fixed": fixed, "broken": broken, "net": fixed - broken},
    ]
    per_fold = []
    for value in range(1, 6):
        selected = fold == value
        fold_base_correct = base_correct[selected]
        fold_final_correct = final_correct[selected]
        per_fold.append({
            "fold": value, "method": "P31_MEAN_COMMITTEE",
            **metric(y[selected], base[selected]), "routed": 0,
            "fixed": 0, "broken": 0, "net": 0,
        })
        fold_fixed = int(np.sum(~fold_base_correct & fold_final_correct))
        fold_broken = int(np.sum(fold_base_correct & ~fold_final_correct))
        per_fold.append({
            "fold": value, "method": METHOD,
            **metric(y[selected], final[selected]),
            "routed": int(route[selected].sum()), "fixed": fold_fixed,
            "broken": fold_broken, "net": fold_fixed - fold_broken,
        })
    paired_rows = []
    for index, row in enumerate(reference):
        status = ("same_correct" if base_correct[index] and final_correct[index]
                  else "same_wrong" if not base_correct[index] and not final_correct[index]
                  else "P31_wrong_C43_correct" if final_correct[index]
                  else "P31_correct_C43_wrong")
        paired_rows.append({
            "sha256": row["sha256"], "fold": int(fold[index]),
            "label": int(y[index]), "P31_prob": float(base[index]),
            "P31_DI25_routed": int(route[index]),
            "C43_expert_prob": float(expert[index]),
            "final_prob": float(final[index]), "status": status,
        })
    results = ROOT / "results"
    write(results / "aggregate_metrics.csv", aggregate)
    write(results / "per_fold_metrics.csv", per_fold)
    write(results / "paired_changes.csv", paired_rows)
    audit = {
        "source_rows": 799, "year": 2017, "route_P31_DI25": 200,
        "route_difference_count": int(np.sum(route != observed_route)),
        "base_probability_max_abs_diff": float(np.max(np.abs(base - observed_base))),
        "stage1": "P31 Mean Committee", "stage2": "C43_RELATION_QUERY",
        "stage2_training_scope": "all outer-train APKs",
        "outer_train_stage1": "nested inner-OOF",
        "parameter_count": 32292, "max_delta": 2.25,
        "future_rows_loaded": 0, "ego_consumed": False,
        "McNemar_n01_P31_wrong_C43_correct": n01,
        "McNemar_n10_P31_correct_C43_wrong": n10,
        "McNemar_exact_p": exact_p,
    }
    (results / "hard_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = [
        "CARE-Droid P31 Mean Committee -> P31-DI25 -> C43-RQ source v1", "",
        "Strict 2017 five-fold OOF. P31 only; FUSED63 is not used.",
        "Stage-II outer-train probabilities are nested inner-OOF. C43-RQ is trained on all outer-train APKs.",
        "Global24 is routed-expert context only; it is neither Stage-I nor the Router.",
        "No future row or Ego tensor is loaded.", "",
        (f"P31_MEAN_COMMITTEE: BA={base_metric['BA']:.6f} AUC={base_metric['AUC']:.6f} "
         f"F1={base_metric['F1']:.6f} errors={base_metric['errors']}"),
        (f"P31_DI25: routed=200/799 routed_errors={int(np.sum(~base_correct & route))} "
         f"error_recall={np.sum(~base_correct & route)/max(np.sum(~base_correct),1):.6f} "
         f"precision={np.sum(~base_correct & route)/200:.6f}"),
        (f"END_TO_END: BA={final_metric['BA']:.6f} AUC={final_metric['AUC']:.6f} "
         f"F1={final_metric['F1']:.6f} errors={final_metric['errors']}"),
        (f"CORRECTION: fixed={fixed} broken={broken} net={fixed-broken} "
         f"delta_BA={final_metric['BA']-base_metric['BA']:+.6f}"),
        f"PAIRED: n01={n01} n10={n10} exact_p={exact_p:.6f}", "",
        "Interpretation boundary: source-side integration only; do not select from future results.",
    ]
    (results / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("P31_C43_SOURCE_SUMMARY_PASS")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
