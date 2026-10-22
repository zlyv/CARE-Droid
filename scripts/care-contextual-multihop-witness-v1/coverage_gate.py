#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from common import OUTPUT, dump_json, parse_source_formal, read_gzip_json, source_assets


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    rows = parse_source_formal(source_assets()["formal"])
    folder = OUTPUT / "cache/features/source"
    records = [read_gzip_json(folder / f"{row['sha256']}.json.gz") for row in rows]
    route = np.asarray([row["routed"] for row in rows], bool); folds = np.asarray([row["fold"] for row in rows], int)
    real = [record["variants"]["REAL"] for record in records]
    context = np.asarray([min(1.0, value["attn_user_reachable"] + value["attn_auto_reachable"] - value["attn_both"]) for value in real])
    directed = np.asarray([min(1.0, value["attn_user_exclusive"] + value["attn_auto_exclusive"] + value["attn_splittable"]) for value in real])
    critical = np.asarray([value["critical_directional"] for value in real])
    host_count = np.asarray([value["host_count"] for value in real]); method_count = np.asarray([value["graph_method_count"] for value in real])
    aligned_hosts = sum(sum(int(host["aligned"]) for host in record["hosts"]) for record in records)
    total_hosts = sum(len(record["hosts"]) for record in records)
    per_fold = []
    for fold in range(1, 6):
        mask = route & (folds == fold)
        per_fold.append({"fold": fold, "routed": int(mask.sum()), "context_attention_mass": float(context[mask].mean()),
                         "directed_attention_mass": float(directed[mask].mean()), "critical_directional_rate": float(critical[mask].mean())})
    criteria = {"feature_rows==799": len(records) == 799,
                "host_exact_alignment>=0.85": aligned_hosts / max(1, total_hosts) >= .85,
                "routed_context_attention_mass>=0.20": float(context[route].mean()) >= .20,
                "routed_directed_attention_mass>=0.20": float(directed[route].mean()) >= .20,
                "critical_directional_rate>=0.20": float(critical[route].mean()) >= .20,
                "at_least_4_folds_context>=0.15": sum(row["context_attention_mass"] >= .15 for row in per_fold) >= 4,
                "size_correlation_abs<0.80": max(abs(corr(context, host_count)), abs(corr(context, method_count)),
                                                  abs(corr(directed, host_count)), abs(corr(directed, method_count))) < .80}
    status = "PASS" if all(criteria.values()) else "FAIL"
    report = {"status": status, "criteria": criteria, "source_rows": len(records), "future_rows_loaded": 0,
              "host_exact_alignment": aligned_hosts / max(1, total_hosts),
              "routed_context_attention_mass": float(context[route].mean()),
              "routed_directed_attention_mass": float(directed[route].mean()),
              "critical_directional_rate": float(critical[route].mean()), "per_fold": per_fold,
              "correlations": {"context_host": corr(context, host_count), "context_method": corr(context, method_count),
                               "directed_host": corr(directed, host_count), "directed_method": corr(directed, method_count)}}
    dump_json(OUTPUT / "coverage_gate.json", report)
    print("CMW_LABEL_FREE_COVERAGE_" + status); print(report, flush=True)
    if status != "PASS":
        dump_json(OUTPUT / "final_decision.json", {"status": "NOT_FEASIBLE", "stopped_at": "LABEL_FREE_COVERAGE_GATE",
                  "future_rows_loaded": 0, "reason": "Context-sensitive witnesses did not satisfy the frozen coverage/shortcut gate."})
        raise SystemExit(20)


if __name__ == "__main__":
    main()
