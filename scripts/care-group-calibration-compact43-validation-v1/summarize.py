#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from common import OUTPUT, dump, read_csv, sha256_file


def value(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def main():
    group = read_csv(OUTPUT / "group_stage1_results.csv")
    group_fold = read_csv(OUTPUT / "group_stage1_fold_results.csv")
    compact = read_csv(OUTPUT / "compact43_strict_results.csv")
    compact_fold = read_csv(OUTPUT / "compact43_strict_fold_results.csv")
    shuffle = read_csv(OUTPUT / "compact43_matched_shuffle.csv")
    bootstrap = read_csv(OUTPUT / "compact43_paired_bootstrap.csv")
    group_by = {row["arm"]: row for row in group}
    raw = group_by["GROUP_RF_RAW"]
    calibrated = group_by["GROUP_RF_PLATT"]
    stage1_calibration_supported = (value(calibrated, "BA") >= value(raw, "BA") + .03 and
                                    value(calibrated, "AUC") >= value(raw, "AUC") - .01)
    fixed = {(row["representation"], row["scope"]): row for row in compact if row["protocol"] == "FIXED_FOLDS"}
    real = fixed[("COMPACT_MEAN_STD_MAX", "DI25_ROUTED")]
    residual = fixed[("COMPACT_COUNT_RESIDUAL", "DI25_ROUTED")]
    count = fixed[("COUNT_ONLY", "DI25_ROUTED")]
    current = fixed[("CURRENT_C43_DELTA_FROZEN_REFERENCE", "DI25_ROUTED")]
    null_rows = [row for row in shuffle if row["protocol"] == "FIXED_FOLDS" and row["scope"] == "DI25_ROUTED"]
    null_fp = float(np.mean([value(row, "FP_vs_TP_AUC") for row in null_rows]))
    null_fn = float(np.mean([value(row, "FN_vs_TN_AUC") for row in null_rows]))
    fold_rows = [row for row in compact_fold if row["protocol"] == "FIXED_FOLDS" and
                 row["representation"] == "COMPACT_MEAN_STD_MAX" and row["scope"] == "DI25_ROUTED"]
    fp_valid = [row for row in fold_rows if math.isfinite(value(row, "FP_vs_TP_AUC"))]
    fn_valid = [row for row in fold_rows if math.isfinite(value(row, "FN_vs_TN_AUC"))]
    fp_support = sum(value(row, "FP_vs_TP_AUC") > .5 for row in fp_valid)
    fn_support = sum(value(row, "FN_vs_TN_AUC") > .5 for row in fn_valid)
    boot_index = {(row["protocol"], row["comparison"], row["control"]): row for row in bootstrap}
    fixed_fp_vs_current = boot_index[("FIXED_FOLDS", "FP_vs_TP", "CURRENT_C43_DELTA")]
    fixed_fp_vs_count = boot_index[("FIXED_FOLDS", "FP_vs_TP", "COUNT_ONLY")]
    fixed_signal = (
        value(real, "FP_vs_TP_AUC") >= value(current, "FP_vs_TP_AUC") + .03 and
        value(real, "FP_vs_TP_AUC") >= value(count, "FP_vs_TP_AUC") + .05 and
        value(real, "FP_vs_TP_AUC") >= null_fp + .05 and
        value(fixed_fp_vs_current, "difference_ci_low") > 0 and
        value(fixed_fp_vs_count, "difference_ci_low") > 0 and
        value(real, "FN_vs_TN_AUC") >= value(count, "FN_vs_TN_AUC") - .02 and
        value(residual, "FP_vs_TP_AUC") >= value(count, "FP_vs_TP_AUC") + .03 and
        len(fp_valid) >= 3 and fp_support > len(fp_valid) / 2 and
        len(fn_valid) >= 3 and fn_support > len(fn_valid) / 2
    )
    group_real = next(row for row in compact if row["protocol"] == "GROUP_FOLDS" and row["representation"] == "COMPACT_MEAN_STD_MAX" and row["scope"] == "ALL")
    group_count = next(row for row in compact if row["protocol"] == "GROUP_FOLDS" and row["representation"] == "COUNT_ONLY" and row["scope"] == "ALL")
    group_fp_vs_count = boot_index[("GROUP_FOLDS", "FP_vs_TP", "COUNT_ONLY")]
    group_fn_vs_count = boot_index[("GROUP_FOLDS", "FN_vs_TN", "COUNT_ONLY")]
    group_signal = (value(group_real, "FP_vs_TP_AUC") >= value(group_count, "FP_vs_TP_AUC") + .03 and
                    value(group_real, "FN_vs_TN_AUC") >= value(group_count, "FN_vs_TN_AUC") - .02 and
                    value(group_fp_vs_count, "difference_ci_low") > 0 and
                    value(group_fn_vs_count, "difference_ci_low") > -.02)
    if fixed_signal and group_signal:
        decision = "COMPACT43_DIRECTION_CONFIRMED_FOR_NEXT_SOURCE_INTEGRATION"
    elif fixed_signal:
        decision = "COMPACT43_DIRECTION_FIXED_FOLD_ONLY_NOT_GROUP_ROBUST"
    else:
        decision = "COMPACT43_DIRECTION_NOT_CONFIRMED"
    before = json.loads((OUTPUT / "baseline_asset_manifest_before.json").read_text())
    after, changed = [], []
    for item in before:
        path = Path(item["path"])
        record = {**item, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                  "sha256": sha256_file(path)}
        after.append(record)
        if record["sha256"] != item["sha256"] or record["bytes"] != item["bytes"]:
            changed.append({"path": str(path), "before": item["sha256"], "after": record["sha256"]})
    dump(OUTPUT / "baseline_asset_manifest_after.json", after)
    dump(OUTPUT / "baseline_integrity_check.json", {"status": "PASS" if not changed else "FAIL", "changed": changed})
    if changed:
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal asset changed")
    summary = {
        "decision": decision, "group_safe_stage1_calibration_supported": bool(stage1_calibration_supported),
        "fixed_fold_compact_signal": bool(fixed_signal), "group_fold_compact_signal": bool(group_signal),
        "group_stage1": group, "compact43": compact,
        "diagnostics": {"fixed_real_FP": value(real, "FP_vs_TP_AUC"), "fixed_current_FP": value(current, "FP_vs_TP_AUC"),
                        "fixed_count_FP": value(count, "FP_vs_TP_AUC"), "fixed_null_FP_mean": null_fp,
                        "fixed_real_FN": value(real, "FN_vs_TN_AUC"), "fixed_count_FN": value(count, "FN_vs_TN_AUC"),
                        "fixed_null_FN_mean": null_fn, "fixed_residual_FP": value(residual, "FP_vs_TP_AUC"),
                        "fixed_FP_vs_current_difference_CI": [value(fixed_fp_vs_current, "difference_ci_low"), value(fixed_fp_vs_current, "difference_ci_high")],
                        "fixed_FP_vs_count_difference_CI": [value(fixed_fp_vs_count, "difference_ci_low"), value(fixed_fp_vs_count, "difference_ci_high")],
                        "group_FP_vs_count_difference_CI": [value(group_fp_vs_count, "difference_ci_low"), value(group_fp_vs_count, "difference_ci_high")],
                        "group_FN_vs_count_difference_CI": [value(group_fn_vs_count, "difference_ci_low"), value(group_fn_vs_count, "difference_ci_high")],
                        "FP_supporting_folds": fp_support, "FP_valid_folds": len(fp_valid),
                        "FN_supporting_folds": fn_support, "FN_valid_folds": len(fn_valid)},
        "formal_integrity": "PASS", "future_rows_loaded": 0, "end_to_end_correction_run": False,
    }
    dump(OUTPUT / "final_summary.json", summary)
    report = [
        "# Group-safe RF calibration and strict-nested Compact43 validation", "",
        f"- Final decision: **{decision}**",
        f"- Group-safe Stage-I calibration supported: **{stage1_calibration_supported}**",
        f"- Compact43 fixed-fold signal: **{fixed_signal}**",
        f"- Compact43 group-fold signal: **{group_signal}**",
        "- Future rows loaded: 0", "- End-to-end correction run: no", "- Formal integrity: PASS", "",
        "This is a confirmatory diagnostic. No threshold, router, max_delta, or formal prediction was modified.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("CARE_GROUP_COMPACT_VALIDATION_COMPLETE")
    print("decision=" + decision)
    print("group_safe_stage1_calibration_supported=" + str(stage1_calibration_supported).lower())
    print("formal_integrity=PASS future_rows_loaded=0")
    print("final_summary=" + str(OUTPUT / "final_summary.json"))


if __name__ == "__main__":
    main()

