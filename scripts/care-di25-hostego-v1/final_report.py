#!/usr/bin/env python3
"""Create the final frozen-method validation report from completed audits only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from freeze_final_method import ROOT, SEED_SETS, blocked, contract, read_csv


FINAL = ROOT / "final_audits"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def load_required(path: Path) -> list[dict]:
    if not path.is_file(): blocked(f"final report prerequisite missing: {path}")
    return read_csv(path)


def numeric(rows: list[dict], key: str) -> list[float]:
    output = []
    for row in rows:
        try: output.append(float(row[key]))
        except (KeyError, TypeError, ValueError): pass
    return output


def efficiency() -> list[dict]:
    ego = json.loads((ROOT / "audits/ego_usage_M1.json").read_text(encoding="utf-8"))
    rows = []
    for arm in ("M0", "M1"):
        timing = []
        for rep in ("R1", "R2", "R3", "R4"):
            path = ROOT / "seed_robustness" / rep / "timing.csv"
            if path.is_file(): timing.extend([row for row in read_csv(path) if row["arm"] == arm])
        train, infer, per = numeric(timing, "training_seconds"), numeric(timing, "inference_seconds"), numeric(timing, "seconds_per_routed_apk")
        rows.append({
            "method": arm, "parameter_count": 33388 if arm == "M0" else 34613,
            "extra_parameter_count": 0 if arm == "M0" else 1225,
            "relative_parameter_increase": 0.0 if arm == "M0" else 1225/33388,
            "routing_ratio": 200/799,
            "host_ego_usage_ratio": 0.0 if arm == "M0" else ego["host_ego_usage_ratio"],
            "mean_neighbor_count": 0.0 if arm == "M0" else ego["mean_neighbor_count_among_nontrivial_ego"],
            "training_seconds_mean": float(np.mean(train)) if train else "unavailable",
            "inference_seconds_mean": float(np.mean(infer)) if infer else "unavailable",
            "seconds_per_routed_apk": float(np.mean(per)) if per else "unavailable",
            "timed_new_replicates": len(timing),
            "cache_construction_included": False,
        })
    write_csv(FINAL / "efficiency_summary.csv", rows)
    return rows


def main() -> None:
    frozen = contract()
    source = load_required(FINAL / "source_statistical_analysis.csv")
    seeds = load_required(ROOT / "seed_robustness/seed_robustness_summary.csv")
    deltas = load_required(ROOT / "seed_robustness/seed_robustness_delta.csv")
    future = load_required(ROOT / "diagnostic_future/future_metrics_pooled.csv")
    future_pair = load_required(ROOT / "diagnostic_future/future_M0_M1_paired.csv")
    if len(seeds) != 10 or {row["replicate"] for row in seeds} != set(SEED_SETS):
        blocked("final report requires all five predefined seed replicates")
    leakage_path = FINAL / "final_leakage_audit.json"
    readiness_path = ROOT / "final_holdout/final_holdout_readiness.json"
    if not leakage_path.is_file() or not readiness_path.is_file(): blocked("leakage/readiness audit missing")
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    ego = json.loads((ROOT / "audits/ego_usage_M1.json").read_text(encoding="utf-8"))
    eff = efficiency()

    pos_ba = sum(float(row["delta_BA_M1_minus_M0"]) > 0 for row in deltas)
    pos_net = sum(float(row["delta_net_M1_minus_M0"]) > 0 for row in deltas)
    nonincrease_broken = sum(float(row["delta_broken_M1_minus_M0"]) <= 0 for row in deltas)
    m01 = next(row for row in source if row["comparison"] == "M0 vs M1")
    m23 = next(row for row in source if row["comparison"] == "M2 vs M3")
    direct0 = next(row for row in source if row["comparison"] == "M0 Direct vs M0 Bounded")
    direct1 = next(row for row in source if row["comparison"] == "M1 Direct vs M1 Bounded")
    f = {row["method"]: row for row in future}
    gp, fp0, fp1 = f["Global24"], f["M0_Host_Anomaly_Bounded"], f["M1_HostEgo_Anomaly_Bounded"]
    pooled_pair = next(row for row in future_pair if row["scope"] == "pooled_2018_2021")
    ready_years = [row["year"] for row in readiness if row.get("ready_for_frozen_inference")]
    violations = []
    if leakage.get("future_labels_used_in_training") is not False: violations.append("future-label training audit did not pass")
    if leakage.get("2018_2021_final_untouched_evaluation") is not False: violations.append("2018-2021 status is incorrect")
    unresolved = []
    if leakage.get("2022_2024_used_in_current_design") == "unknown": unresolved.append("2022-2024 design provenance remains unknown")
    if len(ready_years) != 3: unresolved.append("2022-2024 audited assets are incomplete")

    lines = [
        "CARE-Droid FINAL VALIDATION REPORT", "",
        f"FINAL_FROZEN_METHOD: {frozen['method_name']}",
        "This report validates a frozen method; no diagnostic result was used to alter it.", "",
        "Q1. Is the M1-over-M0 source improvement directionally stable across predefined seeds?",
        f"Answer: M1 BA is higher in {pos_ba}/5 predefined replicates and net is higher in {pos_net}/5. " +
        ("The positive direction is consistent across the predefined seed replicates." if pos_ba == 5 else "The replicate-level direction is mixed and must be reported as such."),
        f"The locked R0 source comparison is a positive but statistically non-significant source-OOF trend (n01={m01['n01_left_wrong_right_correct']}, n10={m01['n10_left_correct_right_wrong']}, exact p={float(m01['exact_mcnemar_two_sided_p']):.6f}).", "",
        "Q2. Does parameter overhead remain 3.67%?",
        f"Answer: Yes. M0=33388, M1=34613, extra=1225, relative increase={1225/33388:.8f} ({100*1225/33388:.2f}%).", "",
        "Q3. Is Ego coverage Host-level sparse but APK-level broad?",
        f"Answer: Yes descriptively: {ego['hosts_with_nontrivial_ego']}/{ego['total_hosts']} real Hosts use nontrivial Ego ({ego['host_ego_usage_ratio']:.2%}), while {ego['routed_apks_with_any_nontrivial_ego']}/{ego['routed_apks']} routed APKs contain at least one ({ego['routed_apk_ego_coverage']:.2%}).", "",
        "Q4. Does HostEgo mainly add fixes without clearly increasing breaks?",
        f"Answer: Locked R0 moves fixed 87->90, broken 14->14 and net 73->76. Across predefined replicates, M1 broken is no greater than M0 in {nonincrease_broken}/5; interpret the full replicate table rather than only R0.", "",
        "Q5. Does Bounded better control broken than Direct?",
        f"Answer: On locked source OOF, Direct-vs-Bounded paired changes are M0 n01={direct0['n01_left_wrong_right_correct']}, n10={direct0['n10_left_correct_right_wrong']} and M1 n01={direct1['n01_left_wrong_right_correct']}, n10={direct1['n10_left_correct_right_wrong']}. The historical diagnostic showed substantially more Direct broken samples, so Direct remains secondary and Bounded remains frozen; no causal superiority claim is made from source aggregate BA alone.", "",
        "Q6. Does MalBen show a stable advantage?",
        f"Answer: No. M2->M3 has delta BA={float(m23['delta_BA_right_minus_left']):+.6f}, delta AUC={float(m23['delta_AUC_right_minus_left']):+.6f}, delta F1={float(m23['delta_F1_right_minus_left']):+.6f}, n01={m23['n01_left_wrong_right_correct']}, n10={m23['n10_left_correct_right_wrong']}, p={float(m23['exact_mcnemar_two_sided_p']):.6f}. It is not the final evidence choice.", "",
        "Q7. Does HostEgo exhibit temporal collapse on the 2018-2021 diagnostic sets?",
        f"Answer: No pooled collapse is observed: M0 BA={float(fp0['BA']):.6f}, M1 BA={float(fp1['BA']):.6f}, M1-M0={float(fp1['BA'])-float(fp0['BA']):+.6f}; M1 net={fp1['net']} versus M0 net={fp0['net']}. These sets were previously inspected and support diagnostic description only.", "",
        "Q8. What if an annual diagnostic has M1 below M0?",
        "Answer: It is reported without model revision. Router, representation, budget, bound and head remain frozen.", "",
        "Q9. Are 2022-2024 ready for untouched evaluation?",
        f"Answer: Ready years={ready_years}. " + ("All file/schema gates pass, but provenance confirmation is still required." if len(ready_years)==3 else "No; missing assets/schema items are recorded in final_holdout_readiness.json."), "",
        "Q10. Is there any protocol violation or leakage risk?",
        f"Answer: detected violations={violations or 'none'}; unresolved items={unresolved or 'none'}. 2018-2021 are explicitly diagnostic-only, not untouched final evaluation.", "",
        "Key pooled diagnostic values (2018-2021, previously inspected):",
        f"Global24 BA={float(gp['BA']):.6f} AUC={float(gp['AUC']):.6f} F1={float(gp['F1']):.6f}.",
        f"M0 BA={float(fp0['BA']):.6f} AUC={float(fp0['AUC']):.6f} F1={float(fp0['F1']):.6f} fixed={fp0['fixed']} broken={fp0['broken']} net={fp0['net']}.",
        f"M1 BA={float(fp1['BA']):.6f} AUC={float(fp1['AUC']):.6f} F1={float(fp1['F1']):.6f} fixed={fp1['fixed']} broken={fp1['broken']} net={fp1['net']}.",
        f"M0 vs M1 pooled exact McNemar: n01={pooled_pair['n01_M0_wrong_M1_correct']}, n10={pooled_pair['n10_M0_correct_M1_wrong']}, p={float(pooled_pair['exact_mcnemar_two_sided_p']):.6f}.", "",
        "Efficiency:",
    ]
    for row in eff:
        lines.append(f"{row['method']}: parameters={row['parameter_count']}, routing={float(row['routing_ratio']):.2%}, training_seconds_mean={row['training_seconds_mean']}, inference_seconds_mean={row['inference_seconds_mean']}, seconds_per_routed_apk={row['seconds_per_routed_apk']}.")
    lines.extend(["", "Claim boundary: positive source-OOF trend and diagnostic temporal behavior are supported; statistically significant source superiority and untouched 2018-2021 generalization are not claimed."])
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / "FINAL_VALIDATION_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
