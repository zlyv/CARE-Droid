#!/usr/bin/env python3
from __future__ import annotations

import json

from common import OUTPUT, dump, file_record


def main():
    before = json.loads((OUTPUT / "baseline_asset_manifest_before.json").read_text(encoding="utf-8"))
    after = []; changed = []
    for old in before:
        new = file_record(__import__("pathlib").Path(old["absolute_path"]), old["purpose"]); after.append(new)
        if old["sha256"] != new["sha256"] or old["size_bytes"] != new["size_bytes"]:
            changed.append({"absolute_path": old["absolute_path"], "before": old["sha256"], "after": new["sha256"]})
    dump(OUTPUT / "baseline_asset_manifest_after.json", after)
    dump(OUTPUT / "baseline_integrity_check.json", {"status": "PASS" if not changed else "FAIL", "changed": changed})
    if changed:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen input changed")
    decision = json.loads((OUTPUT / "source_decision.json").read_text(encoding="utf-8"))
    control = json.loads((OUTPUT / "matched_random_control.json").read_text(encoding="utf-8"))
    import csv
    with (OUTPUT / "source_results.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        results = list(csv.DictReader(handle))
    lines = ["# CARE-Droid SVM symmetric consensus selective-correction audit", "",
        "## Boundary", "", "This is a source-only audit of one symmetric rule. It does not define separate FP and FN stories,",
        "does not retrain Stage-I/C43, and does not load future rows. The three non-SVM calibrated classifiers",
        "are used only as direction witnesses for a proposed C43 class flip.", "", "## Source results", ""]
    for row in results:
        lines.append(f"- {row['arm']}: BA={float(row['BA']):.6f}, fixed={row['fixed']}, broken={row['broken']}, net={row['net']}")
    lines += ["", "## Matched random", "", f"- Majority net: {control['majority_net']}",
        f"- Random net 95th percentile: {control['random_net_95pct']:.3f}",
        f"- One-sided permutation p: {control['one_sided_p_net']:.6f}", "", "## Decision", "",
        f"**{decision['status']}**", "", "A strict nested gate and frozen future diagnostics are not authorized unless all source criteria pass.",
        "Formal input integrity: PASS. Future rows loaded: 0."]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump(OUTPUT / "final_summary.json", {"decision": decision, "matched_random": control,
        "source_results": results, "formal_integrity": "PASS", "future_rows_loaded": 0})
    print("CARE_SVM_CONSENSUS_SELECTIVE_CORRECTION_PIPELINE_DONE")
    print("decision=" + decision["status"])
    print("report=" + str(OUTPUT / "report.md"))


if __name__ == "__main__":
    main()
