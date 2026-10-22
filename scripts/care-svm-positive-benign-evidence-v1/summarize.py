#!/usr/bin/env python3
from __future__ import annotations

from common import OUTPUT, dump_json, read_csv, read_json


def main():
    decision = read_json(OUTPUT / "source_decision.json"); probe = read_json(OUTPUT / "evidence_probe.json")
    results = read_csv(OUTPUT / "source_results.csv"); integrity = read_json(OUTPUT / "baseline_integrity_check.json")
    lines = ["# CARE-Droid SVM Positive Benign Functional Certificate v1", "",
        "This experiment never treats low anomaly, absence of malware evidence, or 1-malware-score as benign evidence.",
        "A certificate is positive only when an observed non-sensitive code-function family positively explains an observed sensitive Role under benign outer-train statistics.", "",
        "## Evidence probe", "",
        f"- FP-vs-TP AUC: {probe['certificate_FP_vs_TP_AUC']}",
        f"- count-only AUC: {probe['count_only_FP_vs_TP_AUC']}",
        f"- matched shuffle 95th percentile: {probe['matched_shuffle_95pct']}",
        f"- evidence gate: {probe['status']}", "", "## Source integration", "",
        "| Arm | BA | fixed | broken | net | fixed FP | fixed FN | broken TN | broken TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['arm']} | {float(row['BA']):.6f} | {row['fixed']} | {row['broken']} | {row['net']} | {row['fixed_FP']} | {row['fixed_FN']} | {row['broken_TN']} | {row['broken_TP']} |")
    lines += ["", "## Decision", "", f"**{decision['status']}**", "",
        f"Frozen future authorized: {decision['continue_to_frozen_future']}.",
        "Future extraction is intentionally not run unless both the positive-evidence and source-integration gates pass.",
        f"Baseline integrity: {integrity['status']}."]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump_json(OUTPUT / "final_decision.json", {**decision, "baseline_integrity": integrity["status"],
        "next_action": "build frozen 2017-to-2018-2021 package" if decision["continue_to_frozen_future"] else
        "stop; positive benign evidence did not justify future testing"})
    print("CARE_SVM_POSITIVE_BENIGN_PIPELINE_DONE")
    print("decision=" + decision["status"])
    print("report=" + str(OUTPUT / "REPORT.md"))


if __name__ == "__main__":
    main()

