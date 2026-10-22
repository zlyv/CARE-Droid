#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import BUDGETS, OUTPUT, dump_json, metrics, norm, read_csv, write_csv


def ivalue(row, key):
    return int(float(row[key]))


def main():
    labels = {norm(row["sha256"]): row for row in read_csv(OUTPUT / "dataset_manifest_label_sealed.csv")}
    stage2_path = OUTPUT / "stage2_predictions_label_free.csv"
    stage2_available = stage2_path.is_file()
    predictions = read_csv(stage2_path if stage2_available else OUTPUT / "stage1_predictions_label_free.csv")
    rows = []
    for row in predictions:
        sid = norm(row["sha256"]); y = int(labels[sid]["label"]); base = float(row["stage1_prob"])
        expert = float(row.get("expert_prob", base))
        available = ivalue(row, "stage2_asset_available") if "stage2_asset_available" in row else 0
        output = {**row, "label": y, "care_overlap": labels[sid]["care_overlap"],
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

    y = np.asarray([int(row["label"]) for row in rows], int)
    base = np.asarray([float(row["stage1_prob"]) for row in rows], float)
    base_pred = base >= .5; stage1_error = base_pred != y
    result_rows = []
    baseline = metrics(y, base); baseline.update({
        "arm": "S0_STAGE1_SVM", "budget_pct": 0, "routed": 0,
        "routed_stage1_errors": 0, "routed_stage1_correct": 0,
        "route_error_recall": 0.0, "route_precision": float("nan"),
        "fixed": 0, "broken": 0, "net": 0, "fixed_FN": 0, "fixed_FP": 0,
        "broken_TN": 0, "broken_TP": 0, "class_flips": 0,
        "intervention_precision": float("nan"), "fix_rate_routed_errors": float("nan"),
        "damage_rate_routed_correct": float("nan"), "stage2_asset_coverage": float("nan"),
    }); result_rows.append(baseline)

    cohort_rows = []
    for budget in BUDGETS:
        route = np.asarray([ivalue(row, f"route_{budget}") for row in rows], bool)
        available = np.asarray([ivalue(row, "stage2_asset_available") for row in rows], bool)
        final = np.asarray([float(row[f"prob_{budget}"]) for row in rows], float)
        pred = final >= .5; before_ok = base_pred == y; after_ok = pred == y
        fixed = (~before_ok) & after_ok; broken = before_ok & (~after_ok)
        routed_errors = int(np.sum(route & stage1_error)); routed_correct = int(np.sum(route & (~stage1_error)))
        fixed_n = int(fixed.sum()); broken_n = int(broken.sum())
        result = metrics(y, final); result.update({
            "arm": f"D{budget}_DIRECT_STAGE2", "budget_pct": budget, "routed": int(route.sum()),
            "routed_stage1_errors": routed_errors, "routed_stage1_correct": routed_correct,
            "route_error_recall": routed_errors / int(stage1_error.sum()),
            "route_precision": routed_errors / int(route.sum()),
            "fixed": fixed_n, "broken": broken_n, "net": fixed_n - broken_n,
            "fixed_FN": int(np.sum(fixed & (y == 1))), "fixed_FP": int(np.sum(fixed & (y == 0))),
            "broken_TN": int(np.sum(broken & (y == 0))), "broken_TP": int(np.sum(broken & (y == 1))),
            "class_flips": int(np.sum(base_pred != pred)),
            "intervention_precision": fixed_n / max(fixed_n + broken_n, 1),
            "fix_rate_routed_errors": fixed_n / max(routed_errors, 1),
            "damage_rate_routed_correct": broken_n / max(routed_correct, 1),
            "stage2_asset_coverage": int(np.sum(route & available)) / max(int(route.sum()), 1),
        }); result_rows.append(result)
        for cohort in ("benign", "malradar", "new"):
            mask = np.asarray([row["cohort"] == cohort for row in rows], bool)
            part = metrics(y[mask], final[mask]); part.update({
                "cohort": cohort, "arm": result["arm"], "budget_pct": budget,
                "routed": int(np.sum(route & mask)), "routed_stage1_errors": int(np.sum(route & stage1_error & mask)),
                "fixed": int(np.sum(fixed & mask)), "broken": int(np.sum(broken & mask)),
                "net": int(np.sum(fixed & mask) - np.sum(broken & mask)),
            }); cohort_rows.append(part)

    write_csv(OUTPUT / "budget_results.csv", result_rows)
    write_csv(OUTPUT / "cohort_budget_results.csv", cohort_rows)
    status = "DIRECT_STAGE2_ROUTE_BUDGET_SWEEP_COMPLETE" if stage2_available else "STAGE1_COMPLETE_STAGE2_PENDING"
    dump_json(OUTPUT / "final_decision.json", {
        "status": status, "architecture": "U2_DIRECT_STAGE2_NO_BOUNDED_RESIDUAL",
        "budgets_are_diagnostic_not_selected": True, "target_labels_used_for_fit": False,
        "target_rows": len(rows), "stage1_errors": int(stage1_error.sum()),
        "budget_counts": {str(row["budget_pct"]): int(row["routed"]) for row in result_rows if row["budget_pct"]},
        "formal_method_changed": False,
        "interpretation": "MalEval labels evaluate a frozen ranking and direct Stage-II; they cannot select the formal route budget.",
    })
    lines = [
        "# MalEval2026 direct C43 route-budget diagnostic", "",
        f"Status: **{status}**.", "",
        "One frozen SVM MeanOpposition ranking and one U2 direct Stage-II model are shared by every budget.",
        "There is no bounded residual, tanh correction, max-delta integration, or budget-specific retraining.", "",
        "| Budget | Routed | Route error recall | Route precision | BA | AUC | F1 | Fixed | Broken | Net | Intervention precision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows[1:]:
        lines.append("| {budget_pct}% | {routed} | {route_error_recall:.4f} | {route_precision:.4f} | {BA:.4f} | {AUC:.4f} | {F1:.4f} | {fixed} | {broken} | {net} | {intervention_precision:.4f} |".format(**row))
    lines += ["", "External labels are used only for this post-hoc diagnostic table. No budget is promoted to the formal method by this experiment.", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("MALEVAL_DIRECT_ROUTE_BUDGET_EVALUATION_DONE status=" + status, flush=True)
    for row in result_rows: print(row, flush=True)


if __name__ == "__main__": main()
