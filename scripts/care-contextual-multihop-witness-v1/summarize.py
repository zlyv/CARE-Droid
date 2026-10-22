#!/usr/bin/env python3
from __future__ import annotations

import json
from common import OUTPUT


def main() -> None:
    final_path = OUTPUT / "final_decision.json"
    decision = json.loads(final_path.read_text(encoding="utf-8")) if final_path.is_file() else {"status": "INCOMPLETE"}
    coverage = json.loads((OUTPUT / "coverage_gate.json").read_text(encoding="utf-8")) if (OUTPUT / "coverage_gate.json").is_file() else None
    source = json.loads((OUTPUT / "source_decision.json").read_text(encoding="utf-8")) if (OUTPUT / "source_decision.json").is_file() else None
    diagnostic = json.loads((OUTPUT / "diagnostic_summary.json").read_text(encoding="utf-8")) if (OUTPUT / "diagnostic_summary.json").is_file() else None
    lines = ["# CARE-Droid Context-Sensitive Multi-Hop Witness v1", "", f"Final status: **{decision.get('status')}**.", "",
             "The experiment preserves P31-RF, RF-anchored DI25 Top25%, A0 C43 localization and max_delta=2.25.",
             "Future and same-year diagnostics are executed only after the frozen 2017 source gate passes.", ""]
    if coverage:
        lines += ["## Label-free coverage", "", f"- status: {coverage['status']}",
                  f"- Host exact alignment: {coverage['host_exact_alignment']:.6f}",
                  f"- routed contextual attention mass: {coverage['routed_context_attention_mass']:.6f}",
                  f"- routed directed/splittable attention mass: {coverage['routed_directed_attention_mass']:.6f}", ""]
    if source:
        lines += ["## 2017 source decision", "", f"- status: {source['status']}",
                  f"- formal A0: fixed={source['formal_A0']['fixed']} broken={source['formal_A0']['broken']} net={source['formal_A0']['net']}",
                  f"- REAL: fixed={source['REAL']['fixed']} broken={source['REAL']['broken']} net={source['REAL']['net']}", ""]
    if diagnostic:
        lines += ["## Frozen future and within-year diagnostics", "",
                  f"- frozen source status: {diagnostic['source_status']}",
                  "- future_used_for_selection: false",
                  "- a negative source decision cannot be rescued by these diagnostics."]
        lookup = {(row["protocol"], row["scope"], row["arm"]): row for row in diagnostic["metrics"]}
        for protocol in ("FROZEN_2017_TO_FUTURE", "INDEPENDENT_SAME_YEAR_5FOLD"):
            formal = lookup[(protocol, "POOLED", "FORMAL_A0")]
            real = lookup[(protocol, "POOLED", "REAL")]
            lines += [f"- {protocol} formal: BA={formal['BA']:.6f}, fixed={formal['fixed']}, broken={formal['broken']}, net={formal['net']}",
                      f"- {protocol} REAL: BA={real['BA']:.6f}, fixed={real['fixed']}, broken={real['broken']}, net={real['net']}"]
        lines.append("")
    lines += ["## Interpretation boundary", "", "Call-graph reachability is not data flow. Context-sensitive call-site witnesses are static execution-context evidence only.",
              "A negative gate is final for this representation version; it does not authorize deeper paths or a larger network."]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("CARE_CMW_FINAL_STATUS=" + str(decision.get("status")), flush=True)


if __name__ == "__main__":
    main()
