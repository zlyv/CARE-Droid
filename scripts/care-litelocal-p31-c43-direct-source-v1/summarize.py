#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path("artifacts/care-litelocal-p31-c43-direct-source-v1")
P31_ROOT = Path("artifacts/care-litelocal-p31-c43-source-v1")
DIRECT_DIR = ROOT / "arms/P31_C43_DIRECT"
DIRECT_METHOD = "P31Mean__DI25__C43RelationQuery__Direct"
BOUNDED_METHOD = "P31Mean__DI25__C43RelationQuery__Bounded"


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


def paired(left_correct, right_correct):
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    total = left_only + right_only
    p_value = float(binomtest(min(left_only, right_only), total, .5).pvalue) if total else 1.0
    return left_only, right_only, p_value


def main():
    reference = sorted(read(P31_ROOT / "preflight/p31_reference/source_predictions.csv"),
                       key=lambda row: row["sha256"].casefold())
    bounded_rows = sorted(read(P31_ROOT / "arms/P31_C43_RQ/source_predictions.csv"),
                          key=lambda row: row["sha256"].casefold())
    direct_rows = sorted(read(DIRECT_DIR / "source_predictions.csv"),
                         key=lambda row: row["sha256"].casefold())
    identifiers = [row["sha256"].casefold() for row in reference]
    if identifiers != [row["sha256"].casefold() for row in bounded_rows] or identifiers != [
            row["sha256"].casefold() for row in direct_rows]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: prediction cohorts differ")
    y = np.asarray([int(float(row["label"])) for row in reference])
    fold = np.asarray([int(float(row["fold"])) for row in reference])
    base = np.asarray([float(row["P31_MEAN_COMMITTEE__prob"]) for row in reference])
    route = np.asarray([int(float(row["P31_DI25__routed"])) == 1 for row in reference])
    direct_base = np.asarray([float(row["Global24__prob"]) for row in direct_rows])
    direct_route = np.asarray([int(float(row["A25__routed"])) == 1 for row in direct_rows])
    if np.max(np.abs(base - direct_base)) > 1e-12 or np.any(route != direct_route):
        raise RuntimeError("BLOCKED_BY_CONTRACT: direct arm changed P31 or route")
    bounded = np.asarray([float(row[BOUNDED_METHOD + "__prob"])
                          for row in bounded_rows])
    direct = np.asarray([float(row[DIRECT_METHOD + "__prob"])
                         for row in direct_rows])
    direct_expert = np.asarray([float(row[DIRECT_METHOD + "__expert_prob"])
                                for row in direct_rows])
    if np.max(np.abs(direct[~route] - base[~route])) > 1e-12:
        raise RuntimeError("BLOCKED_BY_CONTRACT: direct expert applied outside route")
    base_correct = (base >= .5) == y
    bounded_correct = (bounded >= .5) == y
    direct_correct = (direct >= .5) == y
    routed_errors = route & ~base_correct
    routed_correct = route & base_correct

    def correction(candidate_correct):
        fixed = int(np.sum(~base_correct & candidate_correct))
        broken = int(np.sum(base_correct & ~candidate_correct))
        return fixed, broken, fixed - broken

    bounded_fix = correction(bounded_correct)
    direct_fix = correction(direct_correct)
    base_metric = metric(y, base)
    bounded_metric = metric(y, bounded)
    direct_metric = metric(y, direct)
    aggregate = [
        {"method": "P31_MEAN_COMMITTEE", **base_metric, "routed": 0,
         "fixed": 0, "broken": 0, "net": 0},
        {"method": BOUNDED_METHOD, **bounded_metric, "routed": 200,
         "fixed": bounded_fix[0], "broken": bounded_fix[1], "net": bounded_fix[2]},
        {"method": DIRECT_METHOD, **direct_metric, "routed": 200,
         "fixed": direct_fix[0], "broken": direct_fix[1], "net": direct_fix[2]},
    ]
    per_fold = []
    for value in range(1, 6):
        selected = fold == value
        for name, probability, correct in (
            ("P31_MEAN_COMMITTEE", base, base_correct),
            (BOUNDED_METHOD, bounded, bounded_correct),
            (DIRECT_METHOD, direct, direct_correct),
        ):
            fixed = int(np.sum(selected & ~base_correct & correct)) if name != "P31_MEAN_COMMITTEE" else 0
            broken = int(np.sum(selected & base_correct & ~correct)) if name != "P31_MEAN_COMMITTEE" else 0
            per_fold.append({
                "fold": value, "method": name,
                **metric(y[selected], probability[selected]),
                "routed": int(route[selected].sum()) if name != "P31_MEAN_COMMITTEE" else 0,
                "fixed": fixed, "broken": broken, "net": fixed - broken,
            })
    direct_only, bounded_only, direct_vs_bounded_p = paired(direct_correct, bounded_correct)
    direct_vs_base_fixed, base_vs_direct, direct_vs_base_p = paired(direct_correct, base_correct)
    paired_rows = []
    for index, row in enumerate(reference):
        paired_rows.append({
            "sha256": row["sha256"], "fold": int(fold[index]), "label": int(y[index]),
            "P31_DI25_routed": int(route[index]), "P31_prob": float(base[index]),
            "bounded_prob": float(bounded[index]),
            "direct_expert_prob": float(direct_expert[index]),
            "direct_final_prob": float(direct[index]),
            "P31_correct": int(base_correct[index]),
            "bounded_correct": int(bounded_correct[index]),
            "direct_correct": int(direct_correct[index]),
        })
    fit = read(DIRECT_DIR / "fit_audit.csv")
    parameters = {int(float(row["parameter_n"])) for row in fit}
    if len(fit) != 15 or parameters != {32292}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: direct fit/parameter audit failed")
    routed_error_fixed = int(np.sum(routed_errors & direct_correct))
    routed_safe = int(np.sum(routed_correct & direct_correct))
    results = ROOT / "results"
    write(results / "aggregate_metrics.csv", aggregate)
    write(results / "per_fold_metrics.csv", per_fold)
    write(results / "paired_predictions.csv", paired_rows)
    audit = {
        "source_rows": 799, "year": 2017, "route_P31_DI25": 200,
        "route_difference_count": int(np.sum(route != direct_route)),
        "base_probability_max_abs_diff": float(np.max(np.abs(base - direct_base))),
        "stage2_representation": "C43_RELATION_QUERY unchanged",
        "stage2_output": "direct raw classification logit",
        "bounded_formula_used": False, "standard_BCE": True,
        "parameter_count": 32292, "outer_train_stage1": "nested inner-OOF",
        "future_rows_loaded": 0, "ego_consumed": False,
        "routed_stage1_errors": int(routed_errors.sum()),
        "direct_fixed_routed_errors": routed_error_fixed,
        "direct_routed_error_recall": routed_error_fixed / int(routed_errors.sum()),
        "routed_stage1_correct": int(routed_correct.sum()),
        "direct_preserved_routed_correct": routed_safe,
        "direct_broken_routed_correct": int(routed_correct.sum()) - routed_safe,
        "direct_vs_bounded_direct_only_correct": direct_only,
        "direct_vs_bounded_bounded_only_correct": bounded_only,
        "direct_vs_bounded_exact_p": direct_vs_bounded_p,
        "direct_vs_base_direct_only_correct": direct_vs_base_fixed,
        "direct_vs_base_base_only_correct": base_vs_direct,
        "direct_vs_base_exact_p": direct_vs_base_p,
    }
    (results / "hard_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    direct_precision = direct_fix[0] / max(direct_fix[0] + direct_fix[1], 1)
    lines = [
        "CARE-Droid P31-DI25 C43-RQ direct-classification source v1", "",
        "Strict 2017 five-fold OOF; same P31, route, C43 representation, folds and three seeds as the bounded control.",
        "Direct replaces the routed prediction with sigmoid(raw C43 head). It does not add P31 logit to the output and does not apply tanh/bound.",
        "Direct is trained with standard BCE on all outer-train APKs; outer-train P31 inputs are nested inner-OOF.",
        "No future row or Ego tensor is loaded.", "",
        (f"P31: BA={base_metric['BA']:.6f} AUC={base_metric['AUC']:.6f} "
         f"F1={base_metric['F1']:.6f} errors={base_metric['errors']}"),
        (f"BOUNDED: BA={bounded_metric['BA']:.6f} AUC={bounded_metric['AUC']:.6f} "
         f"F1={bounded_metric['F1']:.6f} fixed={bounded_fix[0]} broken={bounded_fix[1]} net={bounded_fix[2]}"),
        (f"DIRECT: BA={direct_metric['BA']:.6f} AUC={direct_metric['AUC']:.6f} "
         f"F1={direct_metric['F1']:.6f} fixed={direct_fix[0]} broken={direct_fix[1]} net={direct_fix[2]} "
         f"correction_precision={direct_precision:.6f}"),
        (f"DIRECT_ROUTED_ERRORS: corrected={routed_error_fixed}/{int(routed_errors.sum())} "
         f"recall={routed_error_fixed/max(int(routed_errors.sum()),1):.6f}"),
        (f"DIRECT_ROUTED_CORRECT: preserved={routed_safe}/{int(routed_correct.sum())} "
         f"broken={int(routed_correct.sum())-routed_safe}"),
        (f"DIRECT_VS_BOUNDED: direct_only_correct={direct_only} bounded_only_correct={bounded_only} "
         f"paired_net={direct_only-bounded_only} exact_p={direct_vs_bounded_p:.6f}"), "",
        "Interpretation boundary: source-side development evidence only; inspect Broken and fold consistency before any future run.",
    ]
    (results / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("P31_C43_DIRECT_SOURCE_SUMMARY_PASS")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
