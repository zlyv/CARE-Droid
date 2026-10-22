#!/usr/bin/env python3
from common import OUTPUT, read_csv, read_json


def main():
    decision = read_json(OUTPUT / "final_decision.json"); gate = read_json(OUTPUT / "direction_gate.json")
    probes = read_csv(OUTPUT / "direction_probe_results.csv"); corrections = read_csv(OUTPUT / "source_correction_results.csv")
    lines = ["# CARE-Droid Signed Host MIL v1", "", "## Frozen protocol", "",
        "2017 source-only fixed five-fold OOF; calibrated P31-SVM; SVM MeanOpposition Top25%; per-Host C43; future_rows_loaded=0.", "",
        "Host rows are MIL instances. The 799 APKs—not approximately 95,000 Hosts—are the independent supervision units.", "",
        "## Direction sufficiency", ""]
    for row in probes:
        lines.append(f"- {row['arm']}: FP-vs-TP AUC={row['FP_vs_TP_AUC']}; FN-vs-TN AUC={row['FN_vs_TN_AUC']}")
    lines += ["", f"Direction gate: **{gate['status']}**.",
        f"Matched-null FP AUC 95th percentile: {gate['matched_null_FP_AUC_95pct']}.",
        f"Top-3 deletion mean: {gate['deletion_top3_mean']}; random deletion mean: {gate['deletion_random3_mean']}.",
        "", "## Correction", ""]
    for row in corrections:
        lines.append(f"- {row['arm']}: BA={row['BA']}; fixed={row['fixed']}; broken={row['broken']}; net={row['net']}; fixed_FP={row['fixed_FP']}; fixed_FN={row['fixed_FN']}")
    lines += ["", "## Interpretation boundary", "",
        "Learned benign evidence is class-support evidence, not proof that a Host implements a legitimate user-visible function. Host deletion is a model-level counterfactual test, not dynamic program causality.", "",
        "## Final decision", "", f"**{decision['status']}**", "",
        "No future year was loaded or used for selection. Frozen future diagnostics are not authorized unless the source gate and correction integration both pass.", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("CARE_SIGNED_HOST_MIL_SOURCE_PIPELINE_DONE", flush=True)
    print("final_decision=" + str(OUTPUT / "final_decision.json"), flush=True)
    print("report=" + str(OUTPUT / "REPORT.md"), flush=True)


if __name__ == "__main__": main()

