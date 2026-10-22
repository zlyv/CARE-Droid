#!/usr/bin/env python3
from __future__ import annotations

import hashlib

from common import OUTPUT, SCS_OUTPUT, dump_json, file_record, formal_path, parse_formal, read_json


def main() -> None:
    rows = parse_formal(); wanted = {row["sha256"] for row in rows}
    missing = [sid for sid in sorted(wanted) if not (SCS_OUTPUT / "cache/features/source" / f"{sid}.json.gz").is_file()]
    if missing:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: SCS feature coverage {799-len(missing)}/799 missing_first={missing[:10]}")
    decision_path = SCS_OUTPUT / "final_decision.json"; coverage_path = SCS_OUTPUT / "coverage_gate.json"
    direction_path = SCS_OUTPUT / "direction_probe.json"
    for path in (decision_path, coverage_path, direction_path):
        if not path.is_file():
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing frozen SCS result {path}")
    decision = read_json(decision_path)
    if decision.get("status") != "PARTIALLY_SUPPORTED" or not decision.get("fn_supported") or decision.get("fp_supported"):
        raise RuntimeError("BLOCKED_BY_CONTRACT: expected frozen SCS partial decision not found")
    assets = [file_record(formal_path(), "formal_predictions"), file_record(decision_path, "SCS_final_decision"),
              file_record(coverage_path, "SCS_coverage_gate"), file_record(direction_path, "SCS_direction_probe")]
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", {"assets": assets})
    protocol = {"name": "CARE-Droid FP Count-Matched Semantic Residual Audit v1", "source_year": 2017,
        "source_rows": 799, "future_rows_loaded": 0, "evaluated_side": "Stage-I predicted-malware: FP positive vs TP negative",
        "formal_protocol_unchanged": True, "input_representation": "frozen SCS v1 source features",
        "count_variables": ["graph_method_count", "graph_edge_count", "host_count", "effect_method_count", "effect_tag_count"],
        "semantic_model": "balanced L2 LogisticRegression C=0.25", "residualizer": "outer-train StandardScaler + Ridge alpha=10",
        "arms": ["COUNT_ONLY", "REAL_RAW", "NORM_REAL", "NORM_RESIDUAL", "COUNT_PLUS_NORM", "NORM_EFFECT_SHUFFLE", "NORM_EDGE_SHUFFLE"],
        "matching": "within each validation fold, each FP matched without replacement to nearest TP in outer-train-standardized log-count space; no sample discarded",
        "boundary": "diagnostic only; no correction, future, threshold selection, or model integration"}
    dump_json(OUTPUT / "protocol.json", protocol); digest = hashlib.sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest()
    (OUTPUT / "protocol_hash.txt").write_text(digest + "\n", encoding="utf-8")
    dump_json(OUTPUT / "preflight.json", {"status": "PASS", "source_rows": 799, "feature_rows": 799,
        "predicted_malware": sum(row["stage1_pred"] == 1 for row in rows), "FP": sum(row["stage1_pred"] == 1 and row["label"] == 0 for row in rows),
        "TP": sum(row["stage1_pred"] == 1 and row["label"] == 1 for row in rows), "future_rows_loaded": 0})
    print("FPR_PREFLIGHT_PASS source_rows=799 feature_rows=799 future_rows_loaded=0", flush=True)


if __name__ == "__main__":
    main()
