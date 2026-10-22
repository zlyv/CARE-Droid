#!/usr/bin/env python3
from __future__ import annotations

from common import ARMS, OUTPUT, read_csv, read_json


def main() -> None:
    decision = read_json(OUTPUT / "final_decision.json")
    source = {row["arm"]: row for row in read_csv(OUTPUT / "source_metrics.csv")}
    frozen_rows = read_csv(OUTPUT / "frozen_future_metrics.csv"); within_rows = read_csv(OUTPUT / "within_year_metrics.csv")
    frozen = {(row["scope"], row["arm"]): row for row in frozen_rows}; within = {(row["scope"], row["arm"]): row for row in within_rows}
    lines = ["# CARE-Droid P31 calibrated classifier-family comparison v1", "",
        "Every arm uses exactly the same frozen 31-dimensional P31 representation and threshold 0.5.",
        "LR/RF/XGB/SVM are calibrated independently; MEAN4 is a control. No Stage-II result is mixed into this comparison.", "",
        "## 2017 fixed-fold OOF — primary source comparison", "",
        "| Arm | BA | AUC | F1 | errors | FN | FP | error recall@25% |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        row = source[arm]; lines.append(f"| {arm} | {float(row['BA']):.6f} | {float(row['AUC']):.6f} | {float(row['F1']):.6f} | {row['errors']} | {row['FN']} | {row['FP']} | {float(row['error_recall25']):.6f} |")
    lines += ["", "## Frozen 2017 → 2018–2021 — diagnostic only", ""]
    for scope in ("2018", "2019", "2020", "2021", "pooled"):
        lines.append(f"### {scope}"); lines.append("")
        for arm in ARMS:
            row = frozen[(scope, arm)]; lines.append(f"- {arm}: BA={float(row['BA']):.6f}, AUC={float(row['AUC']):.6f}, F1={float(row['F1']):.6f}, errors={row['errors']}, error_recall25={float(row['error_recall25']):.6f}")
        lines.append("")
    lines += ["## Independent same-year five-fold OOF — not temporal generalization", ""]
    for scope in ("2018", "2019", "2020", "2021", "pooled"):
        lines.append(f"### {scope}"); lines.append("")
        for arm in ARMS:
            row = within[(scope, arm)]; route = f", error_recall25={float(row['error_recall25']):.6f}" if row.get("error_recall25", "") != "" else ""
            lines.append(f"- {arm}: BA={float(row['BA']):.6f}, AUC={float(row['AUC']):.6f}, F1={float(row['F1']):.6f}, errors={row['errors']}{route}")
        lines.append("")
    lines += ["## Decision boundary", "", f"- Source metric leader: **{decision['source_metric_leader']}**.",
        f"- Candidates satisfying source majority-fold and paired-significance checks: {decision['source_candidates_beating_RF_with_majority_and_p_lt_0.05']}.",
        "- Future and same-year results were not used to select or change the formal Stage-I.",
        "- FORMAL_STAGE1_CHANGED=false"]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("CARE_P31_CLASSIFIER_FAMILY_PIPELINE_DONE", flush=True)
    print("source_leader=" + decision["source_metric_leader"], flush=True)
    print("report=" + str(OUTPUT / "report.md"), flush=True)


if __name__ == "__main__":
    main()

