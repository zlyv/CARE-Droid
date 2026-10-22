#!/usr/bin/env python3
from __future__ import annotations

import json

from common import OUTPUT


def load(name):
    path = OUTPUT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    coverage = load("coverage_gate.json"); direction = load("direction_probe.json"); decision = load("final_decision.json") or {"status": "INCOMPLETE"}
    lines = ["# CARE-Droid Sensitive Consequence Slice v1", "", f"Final status: **{decision.get('status')}**.", "",
        "This experiment is entry-free. It projects frozen C43 Hosts onto bounded coarse-effect call paths and removes methods outside those witness paths.",
        "Call reachability is not data flow, execution proof, or intent proof.", "", "## Frozen protocol", "",
        "- P31-RF, RF-anchored DI25 Top25%, fixed five folds and A0 C43 localization are unchanged.",
        "- Source is 2017 only; future_rows_loaded=0.", "- No bounded-correction or end-to-end model is run in this package.", ""]
    if coverage:
        lines += ["## Label-free coverage and pruning", "", f"- status: {coverage['status']}",
            f"- Host exact alignment: {coverage['host_exact_alignment']:.6f}",
            f"- routed APK effect coverage: {coverage['routed_apk_effect_coverage']:.6f}",
            f"- routed attention effect mass: {coverage['routed_attention_effect_mass']:.6f}",
            f"- routed median slice reduction: {coverage['routed_median_slice_reduction']:.6f}", ""]
    if direction:
        lines += ["## Source-only direction sufficiency", ""]
        for arm, value in direction["arms"].items():
            lines.append(f"- {arm}: FN={value['FN_vs_TN_AUC']:.6f}, FP={value['FP_vs_TP_AUC']:.6f}, routed_FN={value['routed_FN_vs_TN_AUC']:.6f}, routed_FP={value['routed_FP_vs_TP_AUC']:.6f}")
        lines += ["", f"- malware-side supported: {direction['fn_supported']}", f"- benign-side supported: {direction['fp_supported']}",
            f"- pruned multihop beats one-hop on both sides: {direction['pruned_multihop_supported']}", ""]
    lines += ["## Decision boundary", "", "A negative source result is final for this representation version. It does not authorize deeper paths, larger models, future tuning, or describing reachability as data flow."]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("CARE_SCS_FINAL_STATUS=" + str(decision.get("status")), flush=True)


if __name__ == "__main__":
    main()
