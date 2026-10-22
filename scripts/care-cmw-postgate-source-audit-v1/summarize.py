#!/usr/bin/env python3
from __future__ import annotations

from common import OUTPUT, read_json


def main() -> None:
    decision = read_json(OUTPUT / "final_decision.json")
    results = read_json(OUTPUT / "direction_results.json")
    lines = ["# CMW post-gate source sufficiency audit v1", "",
             f"Final status: **{decision['status']}**.", "",
             "This is a new source-only diagnostic over frozen v1 features. It does not lower the 0.20 coverage gate and does not revise the v1 NOT_FEASIBLE decision.", "",
             "## Direction results", ""]
    for arm, row in results["arms"].items():
        lines.append(f"- {arm}: FN-vs-TN AUC={row['FN_vs_TN_AUC']:.6f}; FP-vs-TP AUC={row['FP_vs_TP_AUC']:.6f}; routed FN AUC={row['routed_FN_vs_TN_AUC']:.6f}; routed FP AUC={row['routed_FP_vs_TP_AUC']:.6f}; FN folds>0.5={row['FN_folds_above_random']}/5; FP folds>0.5={row['FP_folds_above_random']}/5")
    lines += ["", "## Decision criteria", ""]
    lines.extend(f"- {key}: {value}" for key, value in results["criteria"].items())
    lines += ["", "Future rows loaded: 0. End-to-end correction was not run."]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("CMW_POSTGATE_FINAL_STATUS=" + decision["status"], flush=True)


if __name__ == "__main__":
    main()
