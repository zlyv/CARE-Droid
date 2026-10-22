#!/usr/bin/env python3
"""Detailed A25 correction/opportunity audit for the two frozen experts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


METHODS = {
    "A25__AnomalyBaseline": "Global24__A25__CARE_RGCondAttn_Dedup_FixKeep",
    "A25__MalBenContrast": "Global24__A25__CARE_MalBenHostV2_FixKeep",
}


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def aligned(left, right):
    index = {row["sha256"].strip().lower(): row for row in right}
    if len(index) != len(right): raise RuntimeError("duplicate prediction IDs")
    output = [index.get(row["sha256"].strip().lower()) for row in left]
    if any(row is None for row in output) or len(index) != len(left):
        raise RuntimeError("prediction cohorts are not identical")
    return output


def metric(y, p):
    pred = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"BA": balanced_accuracy_score(y, pred), "AUC": roc_auc_score(y, p),
            "F1": f1_score(y, pred, zero_division=0), "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def delta_stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {key: float("nan") for key in ("mean", "median", "P10", "P90", "mean_abs")}
    return {"mean": np.mean(values), "median": np.median(values),
            "P10": np.percentile(values, 10), "P90": np.percentile(values, 90),
            "mean_abs": np.mean(np.abs(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anomaly", type=Path, required=True)
    parser.add_argument("--malben", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    anomaly = read_csv(args.anomaly); malben = aligned(anomaly, read_csv(args.malben))
    for left, right in zip(anomaly, malben):
        if int(left["label"]) != int(right["label"]) or int(left["fold"]) != int(right["fold"]):
            raise RuntimeError("aligned expert label/fold mismatch")
        if abs(float(left["Global24__prob"]) - float(right["Global24__prob"])) > 1e-10:
            raise RuntimeError("experts do not share the same frozen Global24 probability")
        if int(float(left["A25__routed"])) != int(float(right["A25__routed"])):
            raise RuntimeError("experts do not share the same frozen A25 route")
    sources = {"A25__AnomalyBaseline": anomaly, "A25__MalBenContrast": malben}
    y = np.asarray([int(row["label"]) for row in anomaly])
    folds = np.asarray([int(row["fold"]) for row in anomaly])
    p1 = np.asarray([float(row["Global24__prob"]) for row in anomaly])
    route = np.asarray([int(float(row["A25__routed"])) for row in anomaly], dtype=bool)
    if len(y) != 799 or len(set(row["sha256"].lower() for row in anomaly)) != 799:
        raise RuntimeError("analysis requires the exact unique 799-APK source cohort")
    if not set(np.unique(folds)).issubset({1, 2, 3, 4, 5}):
        raise RuntimeError("unexpected source folds")
    if int(np.sum(route)) != 200 or any(int(np.sum(route & (folds == fold))) != 40 for fold in np.unique(folds)):
        raise RuntimeError("A25 is not the frozen per-fold Top-25% route")
    fold_rows = []; correction_rows = []; fpfn_rows = []; repair_rows = []
    merged = [{"sha256": anomaly[i]["sha256"], "label": y[i], "fold": folds[i],
               "Global24__prob": p1[i], "A25__routed": int(route[i])} for i in range(799)]
    for display, stem in METHODS.items():
        rows = sources[display]
        final = np.asarray([float(row[stem + "__prob"]) for row in rows])
        expert = np.asarray([float(row[stem + "__expert_prob"]) for row in rows])
        delta = np.asarray([float(row[stem + "__bounded_delta"]) for row in rows])
        stage1_pred = p1 >= 0.5; final_pred = final >= 0.5; expert_pred = expert >= 0.5
        stage1_correct = stage1_pred == y; final_correct = final_pred == y
        fixed = (~stage1_correct) & final_correct
        broken = stage1_correct & (~final_correct)
        repairable = (~stage1_correct) & (expert_pred == y)
        harmful = stage1_correct & (expert_pred != y)
        for i in range(799):
            merged[i][display + "__expert_prob"] = expert[i]
            merged[i][display + "__prob"] = final[i]
            merged[i][display + "__delta_logit"] = delta[i]
        groups = [("aggregate", np.ones(len(y), dtype=bool))] + [
            (str(fold), folds == fold) for fold in sorted(np.unique(folds))]
        for fold_name, selected in groups:
            m = metric(y[selected], final[selected])
            routed = selected & route
            fold_rows.append({"method": display, "fold": fold_name, **m,
                              "fixed": int(np.sum(fixed & selected)),
                              "broken": int(np.sum(broken & selected)),
                              "net": int(np.sum(fixed & selected) - np.sum(broken & selected)),
                              "routed_count": int(np.sum(routed)),
                              "routed_rate": float(np.mean(route[selected]))})
            fpfn_rows.append({
                "method": display, "fold": fold_name,
                "fixed_FP": int(np.sum(fixed & selected & (y == 0))),
                "fixed_FN": int(np.sum(fixed & selected & (y == 1))),
                "broken_to_FP": int(np.sum(broken & selected & (y == 0))),
                "broken_to_FN": int(np.sum(broken & selected & (y == 1))),
            })
            total_rep = int(np.sum(repairable & selected)); rep_route = int(np.sum(repairable & routed))
            total_harm = int(np.sum(harmful & selected)); harm_route = int(np.sum(harmful & routed))
            repair_rows.append({
                "method": display, "fold": fold_name,
                "total_stage1_errors": int(np.sum((~stage1_correct) & selected)),
                "repairable_total": total_rep, "repairable_routed": rep_route,
                "RER_router_expert": rep_route / total_rep if total_rep else float("nan"),
                "actually_fixed": int(np.sum(fixed & selected)),
                "harmful_total": total_harm, "harmful_routed": harm_route,
                "actually_broken": int(np.sum(broken & selected)),
            })
            categories = {
                "all_routed": routed,
                "stage1_correct": routed & stage1_correct,
                "stage1_wrong": routed & (~stage1_correct),
                "true_benign": routed & (y == 0),
                "true_malware": routed & (y == 1),
            }
            for category, mask in categories.items():
                correction_rows.append({"method": display, "fold": fold_name,
                                        "category": category, "count": int(np.sum(mask)),
                                        **delta_stats(delta[mask])})
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "fold_metrics.csv", fold_rows)
    write_csv(args.output / "correction_analysis.csv", correction_rows)
    write_csv(args.output / "fp_fn_analysis.csv", fpfn_rows)
    write_csv(args.output / "repairability_analysis.csv", repair_rows)
    write_csv(args.output / "oof_predictions.csv", merged)
    aggregates = [row for row in fold_rows if row["fold"] == "aggregate"]
    lines = ["CARE-Droid Host-only source comparison v2", "", "Strict 2017 OOF only; future rows are not loaded."]
    for row in aggregates:
        lines.append(f"{row['method']}: BA={float(row['BA']):.6f} AUC={float(row['AUC']):.6f} F1={float(row['F1']):.6f} fixed={row['fixed']} broken={row['broken']} net={row['net']}")
    lines += ["", "RER is reported as RER(router, expert), never as a router-only property."]
    (args.output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
