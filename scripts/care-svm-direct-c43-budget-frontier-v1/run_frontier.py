#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from common import (
    BUDGETS, DIRECT_METHOD, OUTPUT, PROJECT, asset, discover, dump_json, ensure_boundary,
    metrics, mcnemar_exact, norm, read_csv, route_at_budget, write_csv,
)


def main():
    started = time.monotonic(); ensure_boundary(); OUTPUT.mkdir(parents=True, exist_ok=True)
    assets = discover()
    protected = [asset(assets["predictions"], "frozen 799-sample Direct-C43 OOF predictions"), asset(assets["architecture"], "frozen Direct-C43 architecture")]
    before_path = OUTPUT / "baseline_asset_manifest_before.json"
    before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.is_file() else protected
    if not before_path.is_file(): dump_json(before_path, before)

    rows = read_csv(assets["predictions"]); method = DIRECT_METHOD
    sha = np.asarray([norm(row["sha256"]) for row in rows]); y = np.asarray([int(float(row["label"])) for row in rows], int)
    fold = np.asarray([int(float(row["fold"])) for row in rows], int)
    base = np.asarray([float(row["Global24__prob"]) for row in rows]); score = np.asarray([float(row["A25__score"]) for row in rows])
    expert = np.asarray([float(row[method + "__expert_prob"]) for row in rows])
    stage1_error = (base >= .5) != y
    routes = {budget: route_at_budget(score, sha, fold, budget) for budget in BUDGETS}
    reference_route = routes[.25]; reference_prob = np.where(reference_route, expert, base); reference_ok = (reference_prob >= .5) == y
    reference_metrics = metrics(y, base, expert, reference_route)
    if (reference_metrics["fixed"], reference_metrics["broken"], reference_metrics["net"]) != (49, 17, 32):
        raise RuntimeError("BLOCKED_BY_CONTRACT: 25% Direct-C43 reference mismatch")

    frontier = []
    fold_rows = []
    sample_rows = []
    metrics_by_budget = {}
    fold_net_by_budget = {}
    for budget in BUDGETS:
        route = routes[budget]; final = np.where(route, expert, base); candidate_ok = (final >= .5) == y
        result = metrics(y, base, expert, route); metrics_by_budget[budget] = result
        a, b, p = mcnemar_exact(candidate_ok, reference_ok)
        row = {
            "budget": budget, "budget_percent": int(round(100 * budget)), **result,
            "delta_routed_vs_25": result["routed"] - reference_metrics["routed"],
            "delta_captured_errors_vs_25": result["routed_stage1_errors"] - reference_metrics["routed_stage1_errors"],
            "delta_fixed_vs_25": result["fixed"] - reference_metrics["fixed"],
            "delta_broken_vs_25": result["broken"] - reference_metrics["broken"],
            "delta_net_vs_25": result["net"] - reference_metrics["net"],
            "candidate_only_correct_vs_25": a, "budget25_only_correct": b, "mcnemar_exact_p_vs_25": p,
            "actual_runtime_cost_available": False, "cost_proxy": "routed APK count",
        }
        frontier.append(row)
        local_nets = []
        for current_fold in range(1, 6):
            selected = fold == current_fold
            local = metrics(y[selected], base[selected], expert[selected], route[selected])
            ref_local = metrics(y[selected], base[selected], expert[selected], reference_route[selected])
            local_row = {"budget": budget, "budget_percent": int(round(100 * budget)), "fold": current_fold, **local,
                         "delta_net_vs_25": local["net"] - ref_local["net"], "delta_fixed_vs_25": local["fixed"] - ref_local["fixed"],
                         "delta_broken_vs_25": local["broken"] - ref_local["broken"]}
            fold_rows.append(local_row); local_nets.append(local_row["delta_net_vs_25"])
        fold_net_by_budget[budget] = local_nets
        for index in range(len(rows)):
            sample_rows.append({
                "sha256": sha[index], "fold": int(fold[index]), "label": int(y[index]),
                "budget_percent": int(round(100 * budget)), "router_score": float(score[index]),
                "stage1_prob": float(base[index]), "expert_prob": float(expert[index]),
                "routed": int(route[index]), "stage1_error": int(stage1_error[index]),
                "final_prob": float(final[index]), "final_correct": int(candidate_ok[index]),
            })

    marginal = []
    previous = None
    for budget in BUDGETS:
        route = routes[budget]
        if previous is None:
            newly = route.copy(); lower = 0.0
        else:
            newly = route & (~routes[previous]); lower = previous
        fixed = int(np.sum(newly & stage1_error & ((expert >= .5) == y)))
        broken = int(np.sum(newly & (~stage1_error) & ((expert >= .5) != y)))
        marginal.append({
            "from_budget_percent": int(round(100 * lower)), "to_budget_percent": int(round(100 * budget)),
            "newly_routed": int(newly.sum()), "newly_routed_stage1_errors": int(np.sum(newly & stage1_error)),
            "newly_routed_stage1_correct": int(np.sum(newly & (~stage1_error))),
            "marginal_fixed": fixed, "marginal_broken": broken, "marginal_net": fixed - broken,
            "marginal_route_precision": float(np.sum(newly & stage1_error) / max(newly.sum(), 1)),
        })
        previous = budget

    for row in frontier:
        dominated = any(
            other["routed"] <= row["routed"] and other["errors"] <= row["errors"] and
            (other["routed"] < row["routed"] or other["errors"] < row["errors"])
            for other in frontier
        )
        row["cost_error_pareto"] = int(not dominated)
        budget = float(row["budget"])
        row["folds_delta_net_nonnegative_vs_25"] = int(sum(value >= 0 for value in fold_net_by_budget[budget]))

    expansion_candidates = []
    for row in frontier:
        if row["budget"] <= .25 or row["budget"] > .50:
            continue
        if (row["delta_net_vs_25"] >= 5 and row["delta_fixed_vs_25"] > row["delta_broken_vs_25"] and
                row["delta_captured_errors_vs_25"] >= 5 and row["folds_delta_net_nonnegative_vs_25"] >= 4):
            expansion_candidates.append(int(row["budget_percent"]))
    contraction_candidates = []
    for row in frontier:
        if row["budget"] > .20:
            continue
        if row["net"] >= 0.90 * reference_metrics["net"] and row["broken"] <= reference_metrics["broken"]:
            contraction_candidates.append(int(row["budget_percent"]))

    write_csv(OUTPUT / "budget_frontier.csv", frontier)
    write_csv(OUTPUT / "budget_frontier_by_fold.csv", fold_rows)
    write_csv(OUTPUT / "marginal_budget_response.csv", marginal)
    write_csv(OUTPUT / "sample_budget_predictions.csv", sample_rows)
    protocol = {
        "stage1": "calibrated P31-SVM", "router": "frozen four-member SVM-anchored MeanOpposition ranking",
        "expert": "frozen strict-OOF Direct-C43 expert probability available for all 799 APKs",
        "budgets": BUDGETS, "tie_break": "score descending, SHA ascending, separately within formal fold",
        "reference_budget": 0.25, "descriptive_only": True, "future_rows_loaded": 0,
        "no_retraining": True, "actual_runtime_cost_available": False,
    }
    dump_json(OUTPUT / "protocol.json", protocol)
    status = "BUDGET_FRONTIER_SUPPORTS_NESTED_ELASTIC_TEST" if expansion_candidates or contraction_candidates else "FIXED_25_NEAR_DESCRIPTIVE_FRONTIER"
    decision = {
        "status": status, "source_only": True, "future_rows_loaded": 0,
        "reference_25": reference_metrics,
        "expansion_candidates_percent": expansion_candidates,
        "contraction_candidates_percent": contraction_candidates,
        "dynamic_controller_run": False, "fixed_source_threshold_selected": False,
        "next_action": "run a separately frozen nested elastic-vs-equal-cost test" if status.startswith("BUDGET_FRONTIER_SUPPORTS") else "stop elastic-budget work; retain fixed Top25%",
        "interpretation": "This audit changes only route count along the frozen MeanOpposition order; it does not select a deployment budget.",
    }
    dump_json(OUTPUT / "final_decision.json", decision)

    after = [asset(assets["predictions"], "frozen 799-sample Direct-C43 OOF predictions"), asset(assets["architecture"], "frozen Direct-C43 architecture")]
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", after)
    before_map = {row["absolute_path"]: row["sha256"] for row in before}; after_map = {row["absolute_path"]: row["sha256"] for row in after}
    changed = [path for path in sorted(before_map) if after_map.get(path) != before_map[path]]
    integrity = {"status": "PASS" if not changed else "FAIL", "asset_count": len(before), "changed": changed}
    dump_json(OUTPUT / "baseline_integrity_check.json", integrity)
    if integrity["status"] != "PASS":
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal asset changed")

    best = min(frontier, key=lambda row: (row["errors"], row["routed"]))
    report = f"""# SVM Direct-C43 Budget–Response Frontier v1

Final status: **{status}**.

This source-only audit reuses frozen strict-OOF expert probabilities for all 799 APKs. It changes only the number of APKs admitted along the unchanged MeanOpposition ordering. It does not retrain a model, select a threshold, run future diagnostics, or estimate runtime from route percentage.

## Frozen reference

- Top25%: captured errors={reference_metrics['routed_stage1_errors']}/97, fixed={reference_metrics['fixed']}, broken={reference_metrics['broken']}, net={reference_metrics['net']}, BA={reference_metrics['BA']:.6f}.

## Descriptive frontier

- Lowest observed error count: budget={best['budget_percent']}%, errors={best['errors']}, fixed={best['fixed']}, broken={best['broken']}, net={best['net']}.
- Expansion candidates satisfying the frozen gate: {expansion_candidates}.
- Contraction candidates satisfying the frozen gate: {contraction_candidates}.
- Actual analysis runtime was not measured here; routed APK count is reported only as a cost proxy.

## Decision boundary

Only `BUDGET_FRONTIER_SUPPORTS_NESTED_ELASTIC_TEST` authorizes a new nested comparison against a fixed equal-cost router. The present audit cannot select a deployment budget or a source score threshold.

Formal asset integrity: **{integrity['status']}**. Future rows loaded: **0**.
"""
    (OUTPUT / "REPORT.md").write_text(report, encoding="utf-8"); (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    dump_json(OUTPUT / "timing.json", {"frontier_audit_wall_seconds": time.monotonic() - started, "stage2_runtime_measured": False,
                                        "cost_proxy": "routed APK count", "future_rows_loaded": 0})
    (OUTPUT / "reproduce_commands.md").write_text(
        f"# Reproduce\n\n```bash\ncd {PROJECT}\n{sys.executable} {Path(__file__).resolve()}\n```\n", encoding="utf-8")
    print("SVM_DIRECT_C43_BUDGET_FRONTIER_STATUS=" + status, flush=True)
    for row in frontier:
        print(f"budget={row['budget_percent']:>3}% routed={row['routed']:>3} captured={row['routed_stage1_errors']:>2}/97 fixed={row['fixed']:>2} broken={row['broken']:>2} net={row['net']:>3} BA={row['BA']:.6f}", flush=True)


if __name__ == "__main__": main()
