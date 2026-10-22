#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import OUTPUT, assets, dump_json, parse_formal, read_gzip_json


def corr(left, right) -> float:
    left, right = np.asarray(left, float), np.asarray(right, float)
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    rows = parse_formal(assets()["formal"]); folder = OUTPUT / "cache/features/source"
    records = [read_gzip_json(folder / f"{row['sha256']}.json.gz") for row in rows]
    route = np.asarray([row["routed"] for row in rows], bool); folds = np.asarray([row["fold"] for row in rows], int)
    audit = [record["audit"] for record in records]
    host_count = np.asarray([value["host_count"] for value in audit], float)
    method_count = np.asarray([value["graph_method_count"] for value in audit], float)
    edge_count = np.asarray([value["graph_edge_count"] for value in audit], float)
    effect_mass = np.asarray([value["attention_effect_mass"] for value in audit], float)
    apk_coverage = np.asarray([value["any_effect_hosts"] > 0 for value in audit], bool)
    slice_ratio = np.asarray([value["slice_node_ratio"] for value in audit], float)
    aligned = sum(value["aligned_hosts"] for value in audit) / max(1, sum(value["host_count"] for value in audit))
    per_fold = []
    for fold in range(1, 6):
        mask = route & (folds == fold)
        per_fold.append({"fold": fold, "routed": int(mask.sum()), "apk_effect_coverage": float(apk_coverage[mask].mean()),
            "attention_effect_mass": float(effect_mass[mask].mean()), "median_slice_ratio": float(np.median(slice_ratio[mask]))})
    correlations = {"effect_mass_host_count": corr(effect_mass, host_count), "effect_mass_method_count": corr(effect_mass, method_count),
        "effect_mass_edge_count": corr(effect_mass, edge_count), "slice_ratio_host_count": corr(slice_ratio, host_count),
        "slice_ratio_method_count": corr(slice_ratio, method_count), "slice_ratio_edge_count": corr(slice_ratio, edge_count)}
    criteria = {"feature_rows==799": len(records) == 799, "host_exact_alignment>=0.95": aligned >= .95,
        "routed_apk_effect_coverage>=0.50": float(apk_coverage[route].mean()) >= .50,
        "routed_attention_effect_mass>=0.20": float(effect_mass[route].mean()) >= .20,
        "median_slice_reduction>=0.50": float(np.median(1 - slice_ratio[route])) >= .50,
        "at_least_4_folds_apk_coverage>=0.35": sum(value["apk_effect_coverage"] >= .35 for value in per_fold) >= 4,
        "max_abs_size_correlation<0.80": max(abs(value) for value in correlations.values()) < .80}
    status = "PASS" if all(criteria.values()) else "FAIL"
    result = {"status": status, "criteria": criteria, "source_rows": 799, "future_rows_loaded": 0,
        "host_exact_alignment": aligned, "routed_apk_effect_coverage": float(apk_coverage[route].mean()),
        "routed_attention_effect_mass": float(effect_mass[route].mean()),
        "routed_median_slice_reduction": float(np.median(1 - slice_ratio[route])), "per_fold": per_fold,
        "correlations": correlations, "labels_used_for_gate": False}
    dump_json(OUTPUT / "coverage_gate.json", result)
    print("SCS_LABEL_FREE_COVERAGE_" + status); print(result, flush=True)
    if status != "PASS":
        dump_json(OUTPUT / "final_decision.json", {"status": "NOT_FEASIBLE", "stopped_at": "LABEL_FREE_SLICE_GATE",
            "reason": "The entry-free Host-to-effect slice did not satisfy the frozen coverage, pruning, or shortcut gate.",
            "future_rows_loaded": 0, "end_to_end_run": False})
        raise SystemExit(20)


if __name__ == "__main__":
    main()
