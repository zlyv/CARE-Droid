#!/usr/bin/env python3
import csv
from pathlib import Path

from common import OUTPUT, ensure_output, read_json


def rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle: return list(csv.DictReader(handle))


def main():
    decision = read_json(OUTPUT / "final_decision.json")
    preflight = read_json(OUTPUT / "preflight.json")
    categories = rows(OUTPUT / "routed_error_category_summary.csv")
    folds = rows(OUTPUT / "routed_error_fold_summary.csv")
    scales = rows(OUTPUT / "descriptive_scale_sweep.csv")
    all_counts = {row["category"]: int(row["count"]) for row in categories if row["scope"] == "ALL"}
    lines = [
        "# CARE-Droid SVM + C43 captured-but-unfixed audit v1", "",
        "## Frozen baseline", "",
        f"- SVM Stage-I: BA={preflight['stage1']['BA']:.6f}, errors={preflight['stage1']['errors']}.",
        f"- SVM + formal C43: BA={preflight['formal_c43']['BA']:.6f}, fixed={preflight['formal_c43']['fixed']}, broken={preflight['formal_c43']['broken']}, net={preflight['formal_c43']['net']}.",
        f"- Routed composition: {decision['routed_errors']} Stage-I errors and {decision['routed_correct']} Stage-I-correct samples.", "",
        "## Routed Stage-I error anatomy", "",
        "| Category | Count |", "|---|---:|",
    ]
    for key in ("FIXED", "CORRECT_DIRECTION_TOO_WEAK", "WRONG_DIRECTION", "NEAR_ZERO", "STAGE1_MARGIN_EXCEEDS_MAX_DELTA"):
        lines.append(f"| {key} | {all_counts.get(key, 0)} |")
    lines += ["", "## Per-fold anatomy", "", "| Fold | Routed errors | Fixed | Correct but weak | Wrong direction | Near zero | Margin > cap |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in folds:
        lines.append(f"| {row['fold']} | {row['routed_errors']} | {row['fixed']} | {row['correct_too_weak']} | {row['wrong_direction']} | {row['near_zero']} | {row['margin_exceeds_cap']} |")
    lines += ["", "## Descriptive global scale sweep", "",
              "This is retrospective anatomy, not an authorized model or hyperparameter selection.", "",
              "| Scale | BA | Fixed | Broken | Net |", "|---:|---:|---:|---:|---:|"]
    for row in scales:
        lines.append(f"| {float(row['scale']):.2f} | {float(row['BA']):.6f} | {row['fixed']} | {row['broken']} | {row['net']} |")
    lines += ["", "## Decision", "", f"**{decision['status']}**", "",
              f"Strict nested magnitude test authorized: **{decision['continue_to_strict_nested_magnitude_test']}**.",
              "Future rows loaded: **0**.", f"Baseline integrity: **{decision['baseline_integrity']}**.", ""]
    ensure_output(OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("SVM_C43_UNFIXED_FINAL_STATUS=" + decision["status"], flush=True)
    print(f"routed_errors={decision['routed_errors']} fixed={decision['fixed']} unresolved={decision['unresolved']}", flush=True)
    print(f"correct_weak={decision['correct_direction_too_weak']} wrong_or_zero_or_cap={decision['unresolved']-decision['correct_direction_too_weak']}", flush=True)
    print("future_rows_loaded=0 baseline_integrity=" + decision["baseline_integrity"], flush=True)
    print("CARE_SVM_C43_UNFIXED_AUDIT_PIPELINE_DONE", flush=True)


if __name__ == "__main__": main()
