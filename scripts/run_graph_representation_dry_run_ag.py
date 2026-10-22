#!/usr/bin/env python3
"""Run a bounded P2 graph-representation dry run.

The complete call graph exists only while one APK is being processed.  The
output is a compact JSONL audit summary and never contains raw DEX, raw
NetworkX graphs, or the full method-signature graph.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import shutil
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
from run_semantic_mapping_audit_ag import load_yaml  # noqa: E402
from run_static_coverage_audit_ag import read_csv, suppress_logs  # noqa: E402


SCRIPT_SCHEMA_VERSION = "1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_apk(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def initial_result(row: dict) -> dict:
    return {
        "sha256": row.get("sha256", ""),
        "absolute_path": row.get("absolute_path", ""),
        "source_id": row.get("source_id", ""),
        "lineage_id": row.get("lineage_id", ""),
        "metadata_year": row.get("metadata_year", ""),
        "metadata_label": row.get("metadata_label", ""),
        "package_name": row.get("package_name", ""),
        "status": "FAILED",
        "error_type": "",
        "error_message": "",
    }


def analyze_one(
    row: dict,
    rules: list[dict],
    priority: dict,
    policy: GraphPolicy,
    edge_rules: list[dict],
) -> dict:
    result = initial_result(row)
    apk = dalvik = analysis = graph = None
    try:
        from androguard.misc import AnalyzeAPK

        apk, dalvik, analysis = AnalyzeAPK(row["absolute_path"])
        graph = analysis.get_call_graph(no_isolated=False)
        result["status"] = "OK"
        result["summary"] = build_compact_graph_summary(
            apk, graph, rules, priority, policy, edge_rules
        )
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


def apk_row(path: Path) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "sha256": sha256_apk(resolved),
        "absolute_path": str(resolved),
        "source_id": "single_apk_dry_run",
        "lineage_id": "",
        "metadata_year": "",
        "metadata_label": "",
        "package_name": "",
    }


def select_rows(args, excluded: set[str]) -> list[dict]:
    if args.apk:
        row = apk_row(args.apk)
        return [] if row["sha256"] in excluded else [row]
    rows = read_csv(args.lineage)
    rows = [row for row in rows if row.get("sha256", "") not in excluded]
    if args.sha256:
        wanted = args.sha256.casefold()
        rows = [row for row in rows if row.get("sha256", "").casefold() == wanted]
        if not rows:
            raise SystemExit(f"SHA256 not found in lineage: {args.sha256}")
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    return rows


def main():
    parser = argparse.ArgumentParser(description="P2 bounded graph representation dry run")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--apk", type=Path)
    source.add_argument("--lineage", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--priority", required=True, type=Path)
    parser.add_argument("--edge-semantics", type=Path)
    parser.add_argument("--failure-csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--H", type=int, default=2)
    parser.add_argument("--K-out", type=int, default=8)
    parser.add_argument("--P-cap", type=int, default=4)
    parser.add_argument("--E-cap", type=int, default=2000)
    parser.add_argument("--max-states", type=int, default=20000)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    args = parser.parse_args()

    if args.apk and args.sha256:
        raise SystemExit("--sha256 can only be used with --lineage")
    suppress_logs()
    mapping = load_yaml(args.mapping)
    priority = load_yaml(args.priority)
    edge_mapping = load_yaml(args.edge_semantics) if args.edge_semantics else {}
    roles = list(mapping.get("roles", []))
    rules = list(mapping.get("rules", []))
    edge_rules = list(edge_mapping.get("edge_rules", []))
    for edge_rule in edge_rules:
        if edge_rule.get("flag") not in {"icc_any", "reflection_any"}:
            raise SystemExit(
                f"unsupported semantic edge flag: {edge_rule.get('flag')}"
            )
        if not edge_rule.get("class_prefixes") or not edge_rule.get("method_names"):
            raise SystemExit(
                f"semantic edge rule requires class_prefixes and method_names: "
                f"{edge_rule.get('rule_id')}"
            )
    if not roles or not rules:
        raise SystemExit("mapping must contain non-empty roles and rules")
    excluded = set()
    if args.failure_csv and args.failure_csv.exists():
        excluded = {
            row.get("sha256", "")
            for row in read_csv(args.failure_csv)
            if row.get("sha256")
        }
    rows = select_rows(args, excluded)
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
    if shutil.disk_usage(output).free < int(max(0.1, args.min_free_gb) * 1024**3):
        raise SystemExit("free-space guard reached before dry run")

    results = []
    print(f"P2 graph dry run started: {len(rows)} APKs", flush=True)
    for index, row in enumerate(rows, start=1):
        results.append(analyze_one(row, rules, priority, policy, edge_rules))
        if index % max(1, args.progress_every) == 0 or index == len(rows):
            print(f"processed {index}/{len(rows)}", flush=True)

    jsonl_path = output / "graph_representation_dry_run.jsonl"
    write_jsonl(jsonl_path, results)
    failures = [row for row in results if row["status"] != "OK"]
    failure_fields = [
        "sha256", "absolute_path", "source_id", "lineage_id", "metadata_year",
        "metadata_label", "package_name", "status", "error_type", "error_message",
    ]
    with (output / "graph_representation_failures.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=failure_fields)
        writer.writeheader()
        writer.writerows(failures)

    schema = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "phase": "P2_GRAPH_REPRESENTATION_DRY_RUN",
        "row_format": "one_compact_summary_per_apk_jsonl",
        "raw_graph_persistence": "never",
        "raw_dex_persistence": "never",
        "path_hash": "sha256(canonical_node_signatures_and_edge_types)",
        "node_fields": [
            "canonical_node_count", "internal_node_count", "external_node_count",
            "mapped_api_node_count", "role_node_counts",
        ],
        "edge_fields": [
            "canonical_edge_count", "role_edge_counts", "edge_flag_status",
            "edge_source_counts", "edge_type_counts",
        ],
        "entry_fields": [
            "component_count", "bound_component_count", "entry_method_count",
            "alias_count", "alias_bound_count",
        ],
        "path_fields": [
            "candidate_path_count_capped", "retained_path_count",
            "retained_edge_count", "flags_truncated", "path_hashes",
        ],
        "formal_training_enabled": False,
    }
    (output / "graph_representation_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    ok_rows = [row for row in results if row["status"] == "OK"]
    summaries = [row["summary"] for row in ok_rows]
    edge_source_totals = Counter()
    edge_type_totals = Counter()
    for summary in summaries:
        edge_source_totals.update(summary.get("edge_source_counts", {}))
        edge_type_totals.update(summary.get("edge_type_counts", {}))
    audit = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P2_GRAPH_REPRESENTATION_DRY_RUN",
        "formal_training_enabled": False,
        "mapping_file": str(args.mapping.expanduser().resolve()),
        "mapping_sha256": sha256_file(args.mapping),
        "priority_file": str(args.priority.expanduser().resolve()),
        "priority_sha256": sha256_file(args.priority),
        "edge_semantics_file": (
            str(args.edge_semantics.expanduser().resolve())
            if args.edge_semantics
            else None
        ),
        "edge_semantics_sha256": (
            sha256_file(args.edge_semantics) if args.edge_semantics else None
        ),
        "edge_rule_count": len(edge_rules),
        "requested_apk_count": len(rows),
        "excluded_failure_count": len(excluded),
        "processed_apk_count": len(results),
        "status_counts": dict(Counter(row["status"] for row in results)),
        "graph_ok_count": len(ok_rows),
        "aggregate": {
            "canonical_nodes": sum(item["canonical_node_count"] for item in summaries),
            "canonical_edges": sum(item["canonical_edge_count"] for item in summaries),
            "mapped_api_nodes": sum(item["mapped_api_node_count"] for item in summaries),
            "retained_paths": sum(item["path_summary"]["retained_path_count"] for item in summaries),
            "retained_edges": sum(item["path_summary"]["retained_edge_count"] for item in summaries),
            "entry_methods": sum(item["entry_binding"]["entry_method_count"] for item in summaries),
        },
        "edge_flag_status_counts": dict(
            Counter(item["edge_flag_status"] for item in summaries)
        ),
        "edge_source_counts": dict(sorted(edge_source_totals.items())),
        "edge_type_counts": dict(sorted(edge_type_totals.items())),
        "truncated_graph_count": sum(
            bool(item["path_summary"]["flags_truncated"]) for item in summaries
        ),
        "policy": {
            "H": policy.H,
            "K_out": policy.K_out,
            "P_cap": policy.P_cap,
            "E_cap": policy.E_cap,
            "max_states": policy.max_states,
        },
        "outputs": {
            "summaries": str(jsonl_path),
            "failures": str(output / "graph_representation_failures.csv"),
            "schema": str(output / "graph_representation_schema.json"),
        },
        "blocking_reasons": [
            "first_seen_utc_not_joined",
            (
                "edge_icc_reflection_rules_require_coverage_calibration"
                if edge_rules
                else "edge_icc_reflection_metadata_not_available_or_not_calibrated"
            ),
            "truncation_hyperparameters_not_resolved_by_nested_validation",
            "resolved_formal_training_config_not_generated",
        ],
        "notes": [
            "call graphs are built only in memory and released after each APK",
            "outputs contain compact counts and bounded path hashes, not raw graph data",
            "edge flags are never guessed when the call-graph provider does not expose them",
            "this is a P2 representation dry run, not model training",
        ],
    }
    (output / "graph_representation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
