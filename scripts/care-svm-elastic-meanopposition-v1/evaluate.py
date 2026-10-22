#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from common import (OUTPUT, YEARS, discover, dump_json, file_record, metrics, mcnemar, norm,
                    read_csv, top_k, write_csv)


def evaluate_scope(rows, labels, group_name, scope, policies):
    selected = np.asarray([str(row[group_name]) == str(scope) for row in rows]) if scope != "pooled" else np.ones(len(rows), bool)
    y = np.asarray([labels[norm(row["sha256"])] for row in rows], int); base = np.asarray([float(row["stage1_prob"]) for row in rows])
    expert = np.asarray([float(row["expert_prob"]) for row in rows]); result = []
    for name, route in policies.items(): result.append({"scope": str(scope), "arm": name, **metrics(y[selected], base[selected], expert[selected], route[selected])})
    return result


def main():
    assets = discover(); frozen = json.loads((OUTPUT / "freeze_status.json").read_text(encoding="utf-8"))
    protocol_hash = (OUTPUT / "protocol_hash.txt").read_text(encoding="utf-8").strip()
    if frozen["protocol_hash"] != protocol_hash: raise RuntimeError("BLOCKED_BY_CONTRACT: protocol hash mismatch")

    source = read_csv(OUTPUT / "source_route_assignments.csv")
    source_labels = {norm(row["sha256"]): int(row["label"]) for row in source}
    source_policies = {"S0_FIXED25": np.asarray([int(row["fixed25"]) for row in source], bool),
        "E0_LOFO_Q75_RAW": np.asarray([int(row["lofo_q75_raw"]) for row in source], bool),
        "E1_LOFO_Q75_CAP25": np.asarray([int(row["lofo_q75_cap25"]) for row in source], bool)}
    source_rows = []
    for scope in [1, 2, 3, 4, 5, "pooled"]: source_rows += evaluate_scope(source, source_labels, "fold", scope, source_policies)
    source_ref = next(row for row in source_rows if row["scope"] == "pooled" and row["arm"] == "S0_FIXED25")
    if (source_ref["fixed"], source_ref["broken"], source_ref["net"]) != (49, 17, 32):
        raise RuntimeError("BLOCKED_BY_CONTRACT: source Direct-C43 reference mismatch")
    source_cap = next(row for row in source_rows if row["scope"] == "pooled" and row["arm"] == "E1_LOFO_Q75_CAP25")
    source_fold_deltas = []
    for fold in range(1, 6):
        ref = next(row for row in source_rows if row["scope"] == str(fold) and row["arm"] == "S0_FIXED25")
        cap = next(row for row in source_rows if row["scope"] == str(fold) and row["arm"] == "E1_LOFO_Q75_CAP25")
        source_fold_deltas.append({"fold": fold, "delta_net": cap["net"] - ref["net"], "delta_routed": cap["routed"] - ref["routed"]})
    write_csv(OUTPUT / "source_metrics.csv", source_rows); write_csv(OUTPUT / "source_fold_deltas.csv", source_fold_deltas)

    labels = {norm(row["sha256"]): int(row["label"]) for row in read_csv(assets["future_labels"])}
    future = read_csv(OUTPUT / "future_route_assignments_label_free.csv")
    if len(labels) < 1989 or any(norm(row["sha256"]) not in labels for row in future):
        raise RuntimeError("BLOCKED_BY_CONTRACT: future label alignment failed")
    future_score = np.asarray([float(row["router_score"]) for row in future]); future_sha = np.asarray([norm(row["sha256"]) for row in future])
    future_year = np.asarray([int(row["year"]) for row in future])
    fixed25 = np.asarray([int(row["fixed25"]) for row in future], bool); raw = np.asarray([int(row["q75_raw"]) for row in future], bool)
    cap = np.asarray([int(row["q75_cap25"]) for row in future], bool)
    raw_rate = float(raw.mean()); cap_rate = float(cap.mean())
    future_policies = {"S0_FIXED25": fixed25, "E0_Q75_RAW": raw, "E1_Q75_CAP25": cap,
        "C0_FIXED_EQUAL_RAW_COST": top_k(future_score, future_sha, future_year, raw_rate),
        "C1_FIXED_EQUAL_CAP_COST": top_k(future_score, future_sha, future_year, cap_rate)}
    future_rows = []
    for scope in list(YEARS) + ["pooled"]: future_rows += evaluate_scope(future, labels, "year", scope, future_policies)
    write_csv(OUTPUT / "future_metrics.csv", future_rows)

    y = np.asarray([labels[norm(row["sha256"])] for row in future]); base = np.asarray([float(row["stage1_prob"]) for row in future])
    expert = np.asarray([float(row["expert_prob"]) for row in future]); comparisons = []
    for candidate_name, baseline_name in (("E0_Q75_RAW", "C0_FIXED_EQUAL_RAW_COST"), ("E1_Q75_CAP25", "C1_FIXED_EQUAL_CAP_COST"), ("E1_Q75_CAP25", "S0_FIXED25")):
        candidate = future_policies[candidate_name]; baseline = future_policies[baseline_name]
        candidate_ok = (np.where(candidate, expert, base) >= .5) == y; baseline_ok = (np.where(baseline, expert, base) >= .5) == y
        a, b, p = mcnemar(candidate_ok, baseline_ok)
        comparisons.append({"candidate": candidate_name, "baseline": baseline_name, "candidate_only_correct": a,
            "baseline_only_correct": b, "paired_net": a - b, "mcnemar_exact_p": p,
            "candidate_routed": int(candidate.sum()), "baseline_routed": int(baseline.sum())})
    write_csv(OUTPUT / "paired_comparisons.csv", comparisons)

    future_ref = next(row for row in future_rows if row["scope"] == "pooled" and row["arm"] == "S0_FIXED25")
    future_cap = next(row for row in future_rows if row["scope"] == "pooled" and row["arm"] == "E1_Q75_CAP25")
    equal_cap = next(row for row in future_rows if row["scope"] == "pooled" and row["arm"] == "C1_FIXED_EQUAL_CAP_COST")
    per_year_noninferior = 0
    for year in YEARS:
        ref = next(row for row in future_rows if row["scope"] == str(year) and row["arm"] == "S0_FIXED25")
        candidate = next(row for row in future_rows if row["scope"] == str(year) and row["arm"] == "E1_Q75_CAP25")
        per_year_noninferior += int(candidate["net"] >= ref["net"])
    source_gate = (source_cap["net"] >= 30 and source_cap["broken"] <= 17 and
                   sum(row["delta_net"] >= 0 for row in source_fold_deltas) >= 4)
    future_equal_cost_gate = (future_cap["net"] > equal_cap["net"] and future_cap["routed"] == equal_cap["routed"])
    future_reference_gate = (future_cap["net"] >= future_ref["net"] and future_cap["routed"] < future_ref["routed"] and per_year_noninferior >= 3)
    if source_gate and future_equal_cost_gate and future_reference_gate:
        status = "ELASTIC_CONTRACTION_FROZEN_DIAGNOSTIC_SUPPORTED"
        next_action = "validate the frozen policy on an untouched external cohort and measure actual runtime"
    elif source_gate:
        status = "SOURCE_FEASIBLE_BUT_ELASTIC_TEMPORAL_ADVANTAGE_NOT_CONFIRMED"
        next_action = "retain fixed Top25%; do not promote the elastic controller"
    else:
        status = "ELASTIC_THRESHOLD_NOT_SUPPORTED"
        next_action = "retain fixed Top25%; stop elastic routing"
    decision = {"status": status, "protocol_hash": protocol_hash, "source_gate": source_gate,
        "future_equal_cost_gate": future_equal_cost_gate, "future_reference_gate": future_reference_gate,
        "source_fixed25": source_ref, "source_elastic_cap25": source_cap,
        "future_fixed25": future_ref, "future_elastic_cap25": future_cap, "future_equal_cap_cost": equal_cap,
        "future_years_net_noninferior": per_year_noninferior, "future_used_for_selection": False,
        "fixed15_selected": False, "actual_runtime_measured": False, "next_action": next_action}
    dump_json(OUTPUT / "final_decision.json", decision)

    before = json.loads((OUTPUT / "baseline_asset_manifest_before.json").read_text(encoding="utf-8")); after = []
    for row in before: after.append(file_record(Path(row["absolute_path"]), row["purpose"]))
    changed = [row["absolute_path"] for row, current in zip(before, after) if row["sha256"] != current["sha256"]]
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", after)
    dump_json(OUTPUT / "baseline_integrity_check.json", {"status": "PASS" if not changed else "FAIL", "asset_count": len(after), "changed": changed})
    if changed: raise RuntimeError("BLOCKED_BY_CONTRACT: protected asset changed")

    lines = ["# CARE-Droid Elastic MeanOpposition Routing v1", "", f"Final status: **{status}**.", "",
        "The policy was frozen from the label-free 2017 MeanOpposition score distribution before future labels were loaded.",
        "It is a batch-level resource rule, not a per-APK benefit gate. The deployment candidate can only contract from Top25%.", "",
        "## Source strict leave-one-fold-out threshold", "",
        f"- Fixed25: routed={source_ref['routed']}, fixed={source_ref['fixed']}, broken={source_ref['broken']}, net={source_ref['net']}, BA={source_ref['BA']:.6f}.",
        f"- Q75+cap25: routed={source_cap['routed']}, fixed={source_cap['fixed']}, broken={source_cap['broken']}, net={source_cap['net']}, BA={source_cap['BA']:.6f}.", "",
        "## Frozen 2017 to 2018–2021 diagnostic", "",
        f"- Fixed25 pooled: routed={future_ref['routed']}, fixed={future_ref['fixed']}, broken={future_ref['broken']}, net={future_ref['net']}, BA={future_ref['BA']:.6f}.",
        f"- Q75+cap25 pooled: routed={future_cap['routed']}, fixed={future_cap['fixed']}, broken={future_cap['broken']}, net={future_cap['net']}, BA={future_cap['BA']:.6f}.",
        f"- Equal-cost fixed pooled: routed={equal_cap['routed']}, fixed={equal_cap['fixed']}, broken={equal_cap['broken']}, net={equal_cap['net']}, BA={equal_cap['BA']:.6f}.",
        f"- Years non-inferior in net to Fixed25: {per_year_noninferior}/4.", "",
        "## Boundary", "", "Future results are diagnostic only and did not alter the threshold, cap, model, or route score.",
        "Routed count is a cost proxy; actual runtime remains unmeasured.", f"Formal asset integrity: {'PASS' if not changed else 'FAIL'}.", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("ELASTIC_MEANOPPOSITION_STATUS=" + status)


if __name__ == "__main__": main()
