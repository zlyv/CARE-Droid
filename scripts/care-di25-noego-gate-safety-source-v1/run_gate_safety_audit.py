#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from gate_safety_core import BROKEN_COST, apply_rule, choose_threshold, counts
from preflight import load_and_validate, main as preflight

OUT = Path("artifacts/care-di25-noego-gate-safety-source-v1")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def metric(y, p):
    pred = p >= .5
    return {"BA": float(balanced_accuracy_score(y, pred)), "AUC": float(roc_auc_score(y, p)),
            "F1": float(f1_score(y, pred, zero_division=0)),
            "TN": int(np.sum((y == 0) & (~pred))), "FP": int(np.sum((y == 0) & pred)),
            "FN": int(np.sum((y == 1) & (~pred))), "TP": int(np.sum((y == 1) & pred))}


def summarize(name, y, p0, p, route, active, limited):
    fixed, broken = counts(y, p0, p); errors = route & ((p0 >= .5) != y)
    return {"method": name, **metric(y, p), "routed": int(route.sum()),
            "active": int(active.sum()), "routed_errors": int(errors.sum()),
            "gate_error_recall": float(np.sum(active & errors) / max(1, errors.sum())),
            "fixed": fixed, "broken": broken, "net": fixed - broken,
            "correction_precision": float(fixed / max(1, fixed + broken)),
            "bound_limited": int(limited.sum())}


def main():
    preflight(); router, gates = load_and_validate()
    score_by_sha = {r["sha256"].strip().lower(): float(r["score_S3"]) for r in gates}
    ids = [r["sha256"].strip().lower() for r in router]
    y = np.asarray([int(float(r["label"])) for r in router]); folds = np.asarray([int(float(r["fold"])) for r in router])
    p0 = np.asarray([float(r["p_stage1"]) for r in router]); route = np.asarray([int(float(r["route_DI"])) == 1 for r in router])
    q = np.zeros(799, dtype=np.float64)
    for i, sha in enumerate(ids):
        if route[i]: q[i] = score_by_sha[sha]

    fixed_p, fixed_active, fixed_limited = apply_rule(p0, q, route, .5)
    selected_p = p0.copy(); selected_active = np.zeros(799, dtype=bool); selected_limited = np.zeros(799, dtype=bool)
    threshold_rows, curve_rows = [], []
    for fold in range(1, 6):
        selection = folds != fold; validation = folds == fold
        best, curve = choose_threshold(y[selection], p0[selection], q[selection], route[selection])
        pred, active, limited = apply_rule(p0[validation], q[validation], route[validation], best["threshold"])
        selected_p[validation] = pred; selected_active[validation] = active; selected_limited[validation] = limited
        vf, vb = counts(y[validation], p0[validation], pred)
        threshold_rows.append({"target_fold": fold, "selection_rows": int(selection.sum()),
            "selection_routed": int(route[selection].sum()), "selected_threshold": best["threshold"],
            "selection_fixed": best["fixed"], "selection_broken": best["broken"],
            "selection_utility": best["utility"], "validation_fixed": vf, "validation_broken": vb,
            "validation_net": vf - vb, "validation_active": int(active.sum())})
        for row in curve: curve_rows.append({"target_fold": fold, **row})

    summaries = [
        {"method": "Global24", **metric(y, p0), "routed": 0, "active": 0, "routed_errors": 0,
         "gate_error_recall": 0.0, "fixed": 0, "broken": 0, "net": 0,
         "correction_precision": 0.0, "bound_limited": 0},
        summarize("S3_fixed_threshold_0p50", y, p0, fixed_p, route, fixed_active, fixed_limited),
        summarize("S3_crossfold_safety_threshold", y, p0, selected_p, route, selected_active, selected_limited),
    ]
    predictions = []
    threshold_map = {r["target_fold"]: r["selected_threshold"] for r in threshold_rows}
    for i, sha in enumerate(ids):
        predictions.append({"sha256": sha, "year": 2017, "fold": int(folds[i]), "label": int(y[i]),
            "route_DI": int(route[i]), "p_stage1": float(p0[i]), "score_S3": float(q[i]),
            "selected_threshold": float(threshold_map[int(folds[i])]),
            "active_fixed_0p50": int(fixed_active[i]), "p_fixed_0p50": float(fixed_p[i]),
            "active_safety": int(selected_active[i]), "p_safety": float(selected_p[i])})
    write_csv(OUT / "metrics.csv", summaries); write_csv(OUT / "threshold_by_fold.csv", threshold_rows)
    write_csv(OUT / "selection_curves.csv", curve_rows); write_csv(OUT / "oof_predictions.csv", predictions)
    protocol = {"source_year": 2017, "future_rows_loaded": 0, "ego_consumed": False,
        "broken_cost": BROKEN_COST, "candidate_thresholds": "0.50..1.00 step 0.01",
        "tie_break": "fewer broken, more fixed, higher threshold",
        "status": "development cross-fold diagnostic; not a fully nested unbiased estimate",
        "target_fold_labels_used_for_its_threshold": False,
        "caveat": "other-fold OOF predictors may have been trained with target-fold rows; authorize a full nested rerun before freezing"}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    lines = ["CARE-Droid strict-2017 no-Ego S3 gate-safety diagnostic",
             "Threshold utility is Fixed - 1.5*Broken. Target-fold labels are excluded from threshold selection.",
             "This is a development cross-fold estimate, not a fully nested final result.", ""]
    for row in summaries:
        lines.append(f"{row['method']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} "
                     f"active={row['active']} fixed={row['fixed']} broken={row['broken']} net={row['net']} "
                     f"precision={row['correction_precision']:.6f} error_recall={row['gate_error_recall']:.6f}")
    lines.append(""); lines.extend(f"fold {r['target_fold']}: threshold={r['selected_threshold']:.2f} validation_fixed={r['validation_fixed']} validation_broken={r['validation_broken']} validation_net={r['validation_net']}" for r in threshold_rows)
    lines += ["", "Advance only if safety selection materially reduces Broken without erasing Net; then rerun the full representation/gate pipeline with strictly nested threshold selection."]
    (OUT / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines)); print("GATE_SAFETY_SOURCE_DIAGNOSTIC_DONE")


if __name__ == "__main__": main()

