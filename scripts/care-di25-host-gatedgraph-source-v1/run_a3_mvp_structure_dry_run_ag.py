#!/usr/bin/env python3
"""Audit the A3-MVP bounded graph-input structure without training.

The full graph and raw method signatures remain in memory only. The JSONL
output stores counts and a tensor hash, not node IDs, edge indexes, or raw
graph data. Training-fold D/T/one-minus-F reliability is deliberately not
filled by this script.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tasa_droid.p2.graph_representation import (  # noqa: E402
    GraphPolicy,
    build_compact_graph_summary,
)
from tasa_droid.p3.a3_mvp import build_a3_mvp_structure  # noqa: E402
from run_semantic_mapping_audit_ag import load_yaml  # noqa: E402
from run_static_coverage_audit_ag import read_csv, suppress_logs  # noqa: E402


def _suppress_loguru():
    try:
        from loguru import logger

        logger.remove()
    except Exception:
        pass


def analyze_one(row, rules, priority, edge_rules, roles, policy, include_payload=False):
    result = {
        "sha256": row.get("sha256", ""),
        "metadata_year": row.get("metadata_year", ""),
        "metadata_label": row.get("metadata_label", ""),
        "status": "FAILED",
        "error_type": "",
        "error_message": "",
    }
    apk = dalvik = analysis = graph = None
    try:
        from androguard.misc import AnalyzeAPK

        apk, dalvik, analysis = AnalyzeAPK(row["absolute_path"])
        graph = analysis.get_call_graph(no_isolated=False)
        summary = build_compact_graph_summary(
            apk,
            graph,
            rules,
            priority,
            policy,
            edge_rules,
            include_internal=True,
        )
        internal = summary.pop("_internal")
        structure = build_a3_mvp_structure(
            internal["node_records"],
            internal["edge_records"],
            internal["entry_signatures"],
            internal["path_node_signatures"] or internal["entry_signatures"],
            roles,
        )
        result.update(
            {
                "status": "OK",
                "node_count": structure["node_count"],
                "edge_count": structure["edge_count"],
                "base_node_feature_count": structure["base_node_feature_count"],
                "edge_feature_count": structure["edge_feature_count"],
                "role_vocabulary_size": structure["role_vocabulary_size"],
                "reliability_status": structure["reliability_status"],
                "tensor_sha256": structure["tensor_sha256"],
                "path_retained_count": summary["path_summary"]["retained_path_count"],
            }
        )
        if include_payload:
            result["payload"] = structure["payload"]
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:500]
    finally:
        graph = None
        analysis = None
        dalvik = None
        apk = None
        gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description="A3-MVP structure dry run")
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--priority", required=True, type=Path)
    parser.add_argument("--edge-semantics", required=True, type=Path)
    parser.add_argument("--failure-csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--K-out", type=int, default=8)
    parser.add_argument("--P-cap", type=int, default=8)
    parser.add_argument("--E-cap", type=int, default=2000)
    parser.add_argument("--max-states", type=int, default=20000)
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()

    _suppress_loguru()
    suppress_logs()
    mapping = load_yaml(args.mapping)
    priority = load_yaml(args.priority)
    edge_mapping = load_yaml(args.edge_semantics)
    rules = list(mapping.get("rules", []))
    roles = list(mapping.get("roles", []))
    edge_rules = list(edge_mapping.get("edge_rules", []))
    excluded = set()
    if args.failure_csv and args.failure_csv.exists():
        excluded = {
            row.get("sha256", "")
            for row in read_csv(args.failure_csv)
            if row.get("sha256")
        }
    rows = [row for row in read_csv(args.lineage) if row.get("sha256") not in excluded]
    rows = rows[: max(0, args.limit)]
    if not rows:
        raise SystemExit("no APK selected")
    policy = GraphPolicy(
        H=max(0, args.H),
        K_out=max(1, args.K_out),
        P_cap=max(1, args.P_cap),
        E_cap=max(1, args.E_cap),
        max_states=max(100, args.max_states),
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"A3-MVP structure dry run started: {len(rows)} APKs", flush=True)
    for index, row in enumerate(rows, start=1):
        results.append(analyze_one(row, rules, priority, edge_rules, roles, policy))
        if index % max(1, args.progress_every) == 0 or index == len(rows):
            print(f"processed {index}/{len(rows)}", flush=True)
    with (output / "a3_mvp_structure.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P3_A3_MVP_STRUCTURE_DRY_RUN",
        "formal_training_enabled": False,
        "requested_apk_count": len(rows),
        "excluded_failure_count": len(excluded),
        "status_counts": dict(Counter(row["status"] for row in results)),
        "reliability_components": ["D", "T", "one_minus_F"],
        "reliability_status": "REQUIRES_TRAINING_FOLD_COMPONENTS",
        "policy": {
            "H": policy.H,
            "K_out": policy.K_out,
            "P_cap": policy.P_cap,
            "E_cap": policy.E_cap,
        },
        "outputs": {"structure_jsonl": str(output / "a3_mvp_structure.jsonl")},
        "blocking_reasons": [
            "training_fold_reliability_components_not_generated",
            "formal_training_config_not_resolved",
        ],
        "notes": [
            "raw graphs and raw method signatures are not persisted",
            "tensor hashes verify deterministic structure without saving tensors",
            "this is an input-contract dry run, not model training",
        ],
    }
    (output / "a3_mvp_structure_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

