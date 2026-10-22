#!/usr/bin/env python3
from __future__ import annotations

import json

from common import (BASELINE_SOURCE, OUTPUT, WINDOWS, dump_json, file_record, read_csv)


def main():
    before = json.loads((OUTPUT / "baseline_asset_manifest_before.json").read_text(encoding="utf-8"))["assets"]
    after = [file_record(item["absolute_path"], item["purpose"]) for item in before]
    changed = []
    prior = {item["absolute_path"]: item for item in before}
    for item in after:
        old = prior[item["absolute_path"]]
        if item["sha256"] != old["sha256"] or item["size_bytes"] != old["size_bytes"]:
            changed.append({"path": item["absolute_path"], "before": old, "after": item})
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", {"assets": after})
    dump_json(OUTPUT / "baseline_integrity_check.json", {"status": "PASS" if not changed else "FAIL", "changed": changed})
    if changed:
        raise RuntimeError("BLOCKED_BY_CONTRACT: baseline assets changed")
    decision = json.loads((OUTPUT / "evaluation_decision.json").read_text(encoding="utf-8"))
    rows = read_csv(OUTPUT / "train_window_results.csv")
    source = read_csv(BASELINE_SOURCE)
    lines = ["# CARE-Droid SVM train-window expansion v1", "", "## Research question", "",
        "This is a frozen diagnostic of whether adding labeled 2018 data alleviates temporal failure. It does not change the formal method.",
        "W18 is a similar-size recency control; only W1718 beating both W17 and W18 supports a sample-diversity/size explanation.", "",
        "## 2017 source reference", "", f"- Existing SVM+C43 source predictions: {len(source)} rows.", "",
        "## 2019-2021 temporal results", "",
        "| Train window | Test | Stage-I BA | Stage-II BA | Gain pp | fixed | broken | net | fixed FP | broken TN | FP-side net | fixed FN | broken TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    order = {"W17": 0, "W18": 1, "W1718": 2}; scope_order = {"2019": 0, "2020": 1, "2021": 2, "pooled": 3}
    for row in sorted(rows, key=lambda r: (order[r["window"]], scope_order[r["scope"]])):
        lines.append(f"| {row['train_years']} | {row['scope']} | {float(row['Stage1_BA']):.6f} | {float(row['Stage2_BA']):.6f} | {float(row['gain_pp']):+.4f} | {row['Stage2_fixed']} | {row['Stage2_broken']} | {row['Stage2_net']} | {row['Stage2_fixed_FP']} | {row['Stage2_broken_TN']} | {row['Stage2_fp_side_net']} | {row['Stage2_fixed_FN']} | {row['Stage2_broken_TP']} |")
    lines += ["", "## Decision", "", f"**{decision['diagnosis']}**", "",
        "2019-2021 have already been observed during development, so this is a diagnostic rather than an untouched temporal test.",
        "No target-year label was used during Stage-I or Stage-II fitting. Formal baseline assets remained unchanged."]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump_json(OUTPUT / "final_decision.json", {"status": "COMPARISON_COMPLETE", "diagnosis": decision["diagnosis"],
        "formal_method_changed": False, "target_labels_used_for_fit": False, "baseline_integrity": "PASS",
        "next_rule": "Do not adopt W1718 unless it beats both W17 and W18 across pooled and most target years, with improved FP-side net."})
    print("CARE_SVM_TRAIN_WINDOW_EXPANSION_PIPELINE_DONE")
    print("diagnosis=" + decision["diagnosis"])
    print("report=" + str(OUTPUT / "REPORT.md"))


if __name__ == "__main__":
    main()
