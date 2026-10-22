#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import ROOT, YEARS, YEAR_COUNTS, load_future_labels, metrics, norm, read_csv, write_csv, dump

METHODS = (
    ("A0_P31_RF_STAGE1", "stage1_prob", "none"),
    ("A2_DI25_ROLE_C43", "di25_final_prob", "di25"),
    ("A3_FULL100_ROLE_C43", "full100_prob", "all"),
)


def metric_table(protocol, rows):
    output = []
    for scope, selected in [(str(year), [r for r in rows if int(r["year"]) == year]) for year in YEARS] + [("pooled", rows)]:
        y = np.asarray([int(r["label"]) for r in selected])
        base = np.asarray([float(r["stage1_prob"]) for r in selected])
        saved_route = np.asarray([int(r["routed"]) for r in selected], bool)
        stage_error = (base >= .5) != y
        for method, column, route_kind in METHODS:
            p = np.asarray([float(r[column]) for r in selected])
            route = np.zeros(len(y), bool) if route_kind == "none" else (saved_route if route_kind == "di25" else np.ones(len(y), bool))
            row = {"protocol": protocol, "scope": scope, "n": len(y), "method": method, **metrics(y, base, p, route)}
            if route_kind == "di25":
                captured = int(np.sum(stage_error & route))
                row.update({"routed_stage1_errors": captured,
                    "router_error_recall": float(captured / max(stage_error.sum(), 1)),
                    "router_precision": float(captured / max(route.sum(), 1)),
                    "accepted_risk": float(np.sum(stage_error & ~route) / max((~route).sum(), 1))})
            output.append(row)
    return output


def fold_table(protocol, rows):
    output = []
    for year in YEARS:
        annual = [r for r in rows if int(r["year"]) == year]
        for fold in sorted({int(r["fold"]) for r in annual}):
            selected = [r for r in annual if int(r["fold"]) == fold]
            y = np.asarray([int(r["label"]) for r in selected]); base = np.asarray([float(r["stage1_prob"]) for r in selected])
            route_saved = np.asarray([int(r["routed"]) for r in selected], bool)
            for method, column, route_kind in METHODS:
                p = np.asarray([float(r[column]) for r in selected])
                route = np.zeros(len(y), bool) if route_kind == "none" else (route_saved if route_kind == "di25" else np.ones(len(y), bool))
                output.append({"protocol": protocol, "year": year, "fold": fold, "n": len(y), "method": method, **metrics(y, base, p, route)})
    return output


def main():
    frozen = read_csv(ROOT / "frozen/future_predictions_label_free.csv")
    label_path, labels = load_future_labels()
    frozen_rows = []
    for row in frozen:
        sid = norm(row["sha256"])
        if sid not in labels:
            raise RuntimeError("BLOCKED_BY_CONTRACT: frozen prediction lacks diagnostic label")
        frozen_rows.append({**row, "label": labels[sid]})
    if len(frozen_rows) != 1989:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen future row count changed")
    within_rows = []
    for path in (ROOT / "within_2018_2019/predictions.csv", ROOT / "within_2020_2021/predictions.csv"):
        within_rows.extend(read_csv(path))
    if len(within_rows) != 1989 or {norm(r["sha256"]) for r in within_rows} != {norm(r["sha256"]) for r in frozen_rows}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: within-year cohort mismatch")
    aggregate = metric_table("frozen_2017_to_future", frozen_rows) + metric_table("independent_same_year_5fold", within_rows)
    folds = fold_table("independent_same_year_5fold", within_rows)
    write_csv(ROOT / "aggregate_metrics.csv", aggregate); write_csv(ROOT / "within_year_fold_metrics.csv", folds)
    write_csv(ROOT / "frozen_predictions_with_labels.csv", frozen_rows)
    lookup = {(r["protocol"], r["scope"], r["method"]): r for r in aggregate}
    lines = ["CARE-Droid Formal v1 frozen-future and within-year diagnostics", "",
        "Frozen method: P31-RF -> RF-anchored MeanOpposition DI25 -> A0 Role C43-RQ -> bounded correction (max_delta=2.25).",
        "The LR/RF/XGB/SVM committee is routing-only; Stage-I probability is calibrated RF only.",
        "2018-2021 were not used for method, threshold, budget, feature, epoch, or architecture selection.", ""]
    for protocol, title in (("frozen_2017_to_future", "FROZEN 2017 -> 2018-2021"), ("independent_same_year_5fold", "INDEPENDENT SAME-YEAR FIVE-FOLD OOF")):
        lines.append(title + (" (diagnostic only):" if protocol.startswith("frozen") else " (not temporal generalization):"))
        for scope in [str(y) for y in YEARS] + ["pooled"]:
            s = lookup[(protocol, scope, "A0_P31_RF_STAGE1")]
            d = lookup[(protocol, scope, "A2_DI25_ROLE_C43")]
            f = lookup[(protocol, scope, "A3_FULL100_ROLE_C43")]
            lines.append(f"{scope}: Stage1_BA={float(s['BA']):.6f} DI25_BA={float(d['BA']):.6f} DI25_AUC={float(d['AUC']):.6f} DI25_F1={float(d['F1']):.6f} routed={int(d['routed'])} fixed={int(d['fixed'])} broken={int(d['broken'])} net={int(d['net'])} Full100_BA={float(f['BA']):.6f} Full100_net={int(f['net'])}")
        lines.append("")
    lines += ["Evidence boundary:",
        "Frozen future is post-selection temporal diagnosis; it cannot retune the frozen method.",
        "Same-year OOF uses labels from each corresponding year and is not temporal generalization.",
        "A3 Full100 is a diagnostic work-budget control, not a newly selected method.",
        "FORMAL_METHOD_CHANGED=false", "CARE_DROID_FORMAL_DIAGNOSTICS_DONE"]
    report = "\n".join(lines) + "\n"
    (ROOT / "REPORT.txt").write_text(report, encoding="utf-8")
    summary = {"formal_method": "CARE_DROID_FORMAL_V1", "formal_method_changed": False,
        "frozen_future": [r for r in aggregate if r["protocol"] == "frozen_2017_to_future"],
        "same_year": [r for r in aggregate if r["protocol"] == "independent_same_year_5fold"],
        "future_labels_path": str(label_path), "future_used_for_selection": False,
        "same_year_is_temporal_generalization": False}
    dump(ROOT / "final_summary.json", summary)
    print(report, end="")


if __name__ == "__main__":
    main()
