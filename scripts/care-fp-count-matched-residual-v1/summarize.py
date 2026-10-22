#!/usr/bin/env python3
from __future__ import annotations

import json

from common import OUTPUT


def main() -> None:
    result = json.loads((OUTPUT / "final_decision.json").read_text(encoding="utf-8"))
    lines = ["# CARE-Droid FP Count-Matched Semantic Residual Audit v1", "", f"Final status: **{result['status']}**.", "",
        "This source-only diagnostic tests whether normalized Host-to-effect semantics retain FP-vs-TP information after controlling graph/Host/effect counts.",
        "It does not modify P31-RF, DI25, C43, max_delta, routing, or final predictions. future_rows_loaded=0.", "", "## Results", ""]
    for arm, values in result["arms"].items():
        matched = result["matched"][arm]["matched_AUC"]
        lines.append(f"- {arm}: FP={values['FP_vs_TP_AUC']:.6f}, routed_FP={values['routed_FP_vs_TP_AUC']:.6f}, count-matched_FP={matched:.6f}")
    lines += ["", f"- matched pairs: {result['matched_pairs']}",
        f"- maximum normalized-feature/count correlation: {result['max_normalized_feature_count_correlation']:.6f}", "", "## Decision", ""]
    if result["status"] == "FP_SEMANTIC_RESIDUAL_SUPPORTED":
        lines.append("Normalized semantic dispersion/concentration contains independent benign-side direction evidence and may proceed to a separately frozen bounded-correction ablation.")
    else:
        lines.append("The apparent FP signal is not independently supported after count control and matched nulls. Do not use raw size or continue tuning this representation.")
    lines += ["", "Call reachability remains distinct from data flow or execution proof."]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("CARE_FPR_FINAL_STATUS=" + result["status"], flush=True)


if __name__ == "__main__":
    main()
