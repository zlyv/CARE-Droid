#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import BUDGETS, OUTPUT, dump_json, metrics, norm, read_csv, write_csv


def ivalue(row, key):
    return int(float(row[key]))


def evaluate_scope(rows, mask, scope, result_rows):
    y = np.asarray([int(row["label"]) for row in rows], int)[mask]
    base = np.asarray([float(row["stage1_prob"]) for row in rows], float)[mask]
    base_pred = base >= .5; stage1_error = base_pred != y
    baseline = metrics(y, base); baseline.update({
        "scope": scope, "arm": "S0_STAGE1_SVM", "budget_pct": 0, "routed": 0,
        "routed_stage1_errors": 0, "routed_stage1_correct": 0,
        "route_error_recall": 0.0, "route_precision": float("nan"), "fixed": 0,
        "broken": 0, "net": 0, "fixed_FN": 0, "fixed_FP": 0,
        "broken_TN": 0, "broken_TP": 0, "class_flips": 0,
        "intervention_precision": float("nan"), "fix_rate_routed_errors": float("nan"),
        "damage_rate_routed_correct": float("nan"), "stage2_asset_coverage": float("nan"),
    }); result_rows.append(baseline)
    for budget in BUDGETS:
        route = np.asarray([ivalue(row, f"route_{budget}") for row in rows], bool)[mask]
        available = np.asarray([ivalue(row, "stage2_asset_available") for row in rows], bool)[mask]
        final = np.asarray([float(row[f"prob_{budget}"]) for row in rows], float)[mask]
        pred = final >= .5; before_ok = base_pred == y; after_ok = pred == y
        fixed = (~before_ok) & after_ok; broken = before_ok & (~after_ok)
        routed_errors = int(np.sum(route & stage1_error)); routed_correct = int(np.sum(route & (~stage1_error)))
        fixed_n = int(fixed.sum()); broken_n = int(broken.sum())
        result = metrics(y, final); result.update({
            "scope": scope, "arm": f"D{budget}_DIRECT_STAGE2", "budget_pct": budget,
            "routed": int(route.sum()), "routed_stage1_errors": routed_errors,
            "routed_stage1_correct": routed_correct,
            "route_error_recall": routed_errors / max(int(stage1_error.sum()), 1),
            "route_precision": routed_errors / max(int(route.sum()), 1),
            "fixed": fixed_n, "broken": broken_n, "net": fixed_n - broken_n,
            "fixed_FN": int(np.sum(fixed & (y == 1))), "fixed_FP": int(np.sum(fixed & (y == 0))),
            "broken_TN": int(np.sum(broken & (y == 0))), "broken_TP": int(np.sum(broken & (y == 1))),
            "class_flips": int(np.sum(base_pred != pred)),
            "intervention_precision": fixed_n / max(fixed_n + broken_n, 1),
            "fix_rate_routed_errors": fixed_n / max(routed_errors, 1),
            "damage_rate_routed_correct": broken_n / max(routed_correct, 1),
            "stage2_asset_coverage": int(np.sum(route & available)) / max(int(route.sum()), 1),
        }); result_rows.append(result)


def main():
    exclusion = {}
    exclusion_path = OUTPUT / "p31_exclusion_audit.json"
    if exclusion_path.is_file():
        import json
        exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
    labels = {norm(row["sha256"]): row for row in read_csv(OUTPUT / "dataset_manifest_label_sealed.csv")}
    stage2_path = OUTPUT / "stage2_predictions_label_free.csv"
    stage2_available = stage2_path.is_file()
    predictions = read_csv(stage2_path if stage2_available else OUTPUT / "stage1_predictions_label_free.csv")
    rows = []
    for row in predictions:
        sid = norm(row["sha256"]); label_row = labels[sid]; y = int(label_row["label"])
        base = float(row["stage1_prob"]); expert = float(row.get("expert_prob", base))
        available = ivalue(row, "stage2_asset_available") if "stage2_asset_available" in row else 0
        output = {**row, "label": y, "cohort": label_row["cohort"],
                  "stage1_pred": int(base >= .5), "stage2_asset_available": available,
                  "direct_stage2_prob": expert}
        for budget in BUDGETS:
            routed = ivalue(row, f"route_{budget}")
            final = expert if routed and available else base
            pred = int(final >= .5); before = int(base >= .5)
            output[f"prob_{budget}"] = final; output[f"pred_{budget}"] = pred
            output[f"fixed_{budget}"] = int(before != y and pred == y)
            output[f"broken_{budget}"] = int(before == y and pred != y)
        rows.append(output)
    write_csv(OUTPUT / "budget_predictions.csv", rows)

    cohorts = np.asarray([row["cohort"] for row in rows], str)
    scopes = {
        "inclusive_riskware_as_malware": np.ones(len(rows), bool),
        "strict_excluding_riskware": cohorts != "Riskware",
    }
    result_rows = []
    for scope, mask in scopes.items():
        evaluate_scope(rows, mask, scope, result_rows)
    write_csv(OUTPUT / "budget_results.csv", result_rows)

    cohort_rows = []
    for cohort in ("Benign", "Banking", "Riskware"):
        mask = cohorts == cohort
        if not mask.any(): continue
        y = np.asarray([int(row["label"]) for row in rows], int)[mask]
        base = np.asarray([float(row["stage1_prob"]) for row in rows], float)[mask]
        base_pred = base >= .5; stage1_error = base_pred != y
        for budget in BUDGETS:
            route = np.asarray([ivalue(row, f"route_{budget}") for row in rows], bool)[mask]
            final = np.asarray([float(row[f"prob_{budget}"]) for row in rows], float)[mask]
            pred = final >= .5; fixed = stage1_error & (pred == y); broken = (~stage1_error) & (pred != y)
            part = metrics(y, final); part.update({
                "cohort": cohort, "budget_pct": budget, "routed": int(route.sum()),
                "stage1_errors": int(stage1_error.sum()),
                "routed_stage1_errors": int(np.sum(route & stage1_error)),
                "route_error_recall": int(np.sum(route & stage1_error)) / max(int(stage1_error.sum()), 1),
                "fixed": int(fixed.sum()), "broken": int(broken.sum()),
                "net": int(fixed.sum() - broken.sum())})
            cohort_rows.append(part)
    write_csv(OUTPUT / "cohort_budget_results.csv", cohort_rows)

    status = "DIRECT_STAGE2_ROUTE_BUDGET_SWEEP_COMPLETE" if stage2_available else "STAGE1_COMPLETE_STAGE2_PENDING"
    inclusive = [row for row in result_rows if row["scope"] == "inclusive_riskware_as_malware"]
    dump_json(OUTPUT / "final_decision.json", {
        "status": status, "architecture": "U2_DIRECT_STAGE2_NO_BOUNDED_RESIDUAL",
        "budgets_are_external_diagnostics_not_formal_selection": True,
        "target_labels_used_for_fit": False, "target_rows": len(rows),
        "stage1_errors_inclusive": int(inclusive[0]["errors"]),
        "riskware_policy": "included as malware in primary table; also excluded in strict secondary table",
        "sampling": "deterministic stratified sample capped at 500 APKs per supplied cohort",
        "p31_exclusion_policy": "failed extraction APKs excluded from all downstream denominators; no zero imputation",
        "p31_original_rows": exclusion.get("original_rows"),
        "p31_excluded_rows": exclusion.get("excluded_rows", 0),
        "p31_success_rate": exclusion.get("success_rate"),
        "p31_failure_by_cohort": exclusion.get("failure_by_cohort", {}),
        "dataset_limit": "SMS absent and incomplete Adware explicitly skipped; not the complete CICMalDroid2020 benchmark",
        "formal_method_changed": False})
    lines = ["# CICMalDroid2020 direct C43 route-budget diagnostic", "", f"Status: **{status}**.", "",
             "Frozen 2017+2018 P31-SVM, one shared MeanOpposition ranking and one shared direct C43 model.",
             "No bounded residual, tanh correction, max-delta integration or budget-specific retraining.",
             "SMS is absent and the incomplete Adware archive is skipped; Riskware is reported both included and excluded.",
             f"P31 extraction failures are excluded without zero imputation: {exclusion.get('excluded_rows', 0)} of {exclusion.get('original_rows', len(rows))} sampled APKs.", ""]
    lines.insert(6, "This is a fixed-seed stratified sample of at most 500 APKs per supplied cohort, not a full-corpus result.")
    for scope in scopes:
        lines += [f"## {scope}", "", "| Budget | Routed | Error recall | Precision | BA | AUC | F1 | Fixed | Broken | Net |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in [x for x in result_rows if x["scope"] == scope and x["budget_pct"]]:
            lines.append("| {budget_pct}% | {routed} | {route_error_recall:.4f} | {route_precision:.4f} | {BA:.4f} | {AUC:.4f} | {F1:.4f} | {fixed} | {broken} | {net} |".format(**row))
        lines.append("")
    lines += ["External labels were unsealed only in evaluation.py and cannot select a formal CARE-Droid budget.", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("CIC_DIRECT_ROUTE_BUDGET_EVALUATION_DONE status=" + status, flush=True)


if __name__ == "__main__":
    main()
