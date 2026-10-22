#!/usr/bin/env python3
from __future__ import annotations

import json

from common import OUTPUT, dump, read_csv, sha256_file


def main():
    baseline = json.loads((OUTPUT / "baseline_reproduction.json").read_text())
    decision = json.loads((OUTPUT / "information_ceiling_decision.json").read_text())
    stage1 = read_csv(OUTPUT / "stage1_capacity_ladder.csv")
    stage2 = read_csv(OUTPUT / "stage2_information_ceiling.csv")
    before = json.loads((OUTPUT / "baseline_asset_manifest_before.json").read_text())
    after = []
    changed = []
    for item in before:
        path = item["path"]
        from pathlib import Path
        current = Path(path)
        record = {**item, "sha256": sha256_file(current), "bytes": current.stat().st_size,
                  "mtime_ns": current.stat().st_mtime_ns}
        after.append(record)
        if record["sha256"] != item["sha256"] or record["bytes"] != item["bytes"]:
            changed.append({"path": path, "before": item["sha256"], "after": record["sha256"]})
    dump(OUTPUT / "baseline_asset_manifest_after.json", after)
    dump(OUTPUT / "baseline_integrity_check.json", {"status": "PASS" if not changed else "FAIL", "changed": changed})
    if changed:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen baseline asset changed")
    summary = {
        "status": "COMPLETE", "decision": decision["decision"],
        "baseline_reproduction": baseline, "stage1_capacity_ladder": stage1,
        "stage2_information_ceiling": stage2,
        "formal_integrity": "PASS", "future_rows_loaded": 0,
        "interpretation_boundary": (
            "Strong 2017-to-2021 performance argues against a simple calendar-year shortcut, "
            "but does not by itself exclude persistent near-duplicate/family/SDK/source shortcuts. "
            "This source audit does not use future results for selection."
        ),
    }
    dump(OUTPUT / "final_summary.json", summary)
    lines = [
        "# CARE-Droid Stage-I robustness and Stage-II information-ceiling audit",
        "", "## Outcome", "", f"- Decision: **{decision['decision']}**",
        "- Frozen baseline reproduction: PASS",
        f"- Stage-I BA: {baseline['stage1']['BA']:.6f}; errors: {baseline['stage1_errors']}",
        f"- DI25: {baseline['captured_errors']}/{baseline['stage1_errors']} errors in {baseline['route_rows']} routed APKs",
        f"- Formal A0: fixed={baseline['fixed']}, broken={baseline['broken']}, net={baseline['net']}",
        "- Future rows loaded for this audit: 0", "- Formal asset integrity: PASS", "",
        "## Interpretation boundary", "",
        "Good 2017→2018–2021 Stage-I performance is real evidence against a simple year-specific shortcut. "
        "It does not exclude stable cross-year shortcuts such as near-duplicate families, persistent SDKs, "
        "collection-source artifacts, or size/sparsity signals. The grouped and capacity audits quantify the "
        "parts that can be tested from frozen P31 assets; absence of verified family metadata is not silently "
        "relabelled as a family-disjoint test.", "",
        "## Research boundary", "",
        "This run trains probes only. It does not modify the RF, DI25, C43, max_delta, labels, or any formal artifact. "
        "No probe output is automatically integrated into correction.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("CARE_INFORMATION_AUDIT_COMPLETE")
    print("decision=" + decision["decision"])
    print("formal_integrity=PASS future_rows_loaded=0")
    print("report=" + str(OUTPUT / "report.md"))
    print("final_summary=" + str(OUTPUT / "final_summary.json"))


if __name__ == "__main__":
    main()

