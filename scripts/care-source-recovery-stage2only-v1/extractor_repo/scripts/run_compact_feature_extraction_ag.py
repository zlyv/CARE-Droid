#!/usr/bin/env python3
"""Extract compact static/API features in one Androguard pass per APK.

This is a P1 feature-extraction stage, not formal training.  It combines
Manifest statistics, call-graph audit statistics, and the new V6 semantic role
mapping in one pass.  Graph objects and raw method signatures are never
persisted.  Feature and failure CSVs are appended per APK so an interrupted
run can be resumed without reprocessing completed samples.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import logging
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "PyYAML is required. Run this script with the Androguard environment "
        "that provides yaml.safe_load()."
    ) from exc

from run_semantic_mapping_audit_ag import (
    api_rule_matches,
    class_and_method,
    load_yaml,
    manifest_context_matches,
)
from run_static_coverage_audit_ag import (
    is_external,
    method_signature,
    read_csv,
    suppress_logs,
    write_csv,
)


SCRIPT_SCHEMA_VERSION = "1.0.0"


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_call(function, default=None):
    try:
        return function()
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def log1p(value):
    return math.log1p(max(0, safe_int(value)))


def component_count(apk, getter_name):
    values = safe_call(getattr(apk, getter_name), [])
    return len(values or [])


def manifest_features(apk):
    permissions = safe_call(apk.get_permissions, []) or []
    activities = component_count(apk, "get_activities")
    services = component_count(apk, "get_services")
    receivers = component_count(apk, "get_receivers")
    providers = component_count(apk, "get_providers")
    return {
        "manifest_parse_ok": 1,
        "manifest_permission_count": len(permissions),
        "manifest_permission_count_log": log1p(len(permissions)),
        "manifest_activity_count": activities,
        "manifest_service_count": services,
        "manifest_receiver_count": receivers,
        "manifest_provider_count": providers,
        "manifest_component_count": activities + services + receivers + providers,
        "manifest_component_count_log": log1p(activities + services + receivers + providers),
        "min_sdk": safe_int(safe_call(apk.get_min_sdk_version, 0)),
        "target_sdk": safe_int(safe_call(apk.get_target_sdk_version, 0)),
    }


def role_fields(roles):
    fields = []
    for role in roles:
        fields.extend(
            [
                f"role_node__{role}",
                f"role_edge__{role}",
                f"context_role__{role}",
            ]
        )
    return fields


def feature_fields(roles):
    return [
        "sha256", "absolute_path", "source_id", "lineage_id", "metadata_year",
        "metadata_label", "package_name", "certificate_sha256", "version_code",
        "status", "manifest_parse_ok", "manifest_permission_count",
        "manifest_permission_count_log", "manifest_activity_count",
        "manifest_service_count", "manifest_receiver_count", "manifest_provider_count",
        "manifest_component_count", "manifest_component_count_log", "min_sdk",
        "target_sdk", "method_count", "graph_node_count", "graph_edge_count",
        "internal_node_count", "internal_edge_count", "external_node_count",
        "external_edge_count", "mapped_api_node_count", "unmapped_external_node_count",
        "mapped_external_ratio", "sensitive_node_count", "sensitive_edge_count",
        "manifest_context_match_count", "error_type", "error_message",
    ] + role_fields(roles)


def initial_row(row, roles):
    result = {
        "sha256": row.get("sha256", ""),
        "absolute_path": row.get("absolute_path", ""),
        "source_id": row.get("source_id", ""),
        "lineage_id": row.get("lineage_id", ""),
        "metadata_year": row.get("metadata_year", ""),
        "metadata_label": row.get("metadata_label", ""),
        "package_name": row.get("package_name", ""),
        "certificate_sha256": row.get("certificate_sha256", ""),
        "version_code": row.get("version_code", ""),
        "status": "ERROR",
        "manifest_parse_ok": 0,
        "manifest_permission_count": 0,
        "manifest_permission_count_log": 0.0,
        "manifest_activity_count": 0,
        "manifest_service_count": 0,
        "manifest_receiver_count": 0,
        "manifest_provider_count": 0,
        "manifest_component_count": 0,
        "manifest_component_count_log": 0.0,
        "min_sdk": 0,
        "target_sdk": 0,
        "method_count": 0,
        "graph_node_count": 0,
        "graph_edge_count": 0,
        "internal_node_count": 0,
        "internal_edge_count": 0,
        "external_node_count": 0,
        "external_edge_count": 0,
        "mapped_api_node_count": 0,
        "unmapped_external_node_count": 0,
        "mapped_external_ratio": 0.0,
        "sensitive_node_count": 0,
        "sensitive_edge_count": 0,
        "manifest_context_match_count": 0,
        "error_type": "",
        "error_message": "",
    }
    for field in role_fields(roles):
        result[field] = 0
    return result


def extract_one(row, rules, roles):
    result = initial_row(row, roles)
    apk = dalvik = analysis = graph = None
    try:
        from androguard.misc import AnalyzeAPK

        apk, dalvik, analysis = AnalyzeAPK(row["absolute_path"])
        result.update(manifest_features(apk))
        result["method_count"] = len(list(analysis.get_methods()))
        graph = analysis.get_call_graph(no_isolated=False)
        result["status"] = "OK"
        result["graph_node_count"] = int(graph.number_of_nodes())
        result["graph_edge_count"] = int(graph.number_of_edges())

        nodes = list(graph.nodes)
        external = {node for node in nodes if is_external(node)}
        internal = set(nodes) - external
        result["external_node_count"] = len(external)
        result["internal_node_count"] = len(internal)
        result["external_edge_count"] = sum(
            1 for left, right in graph.edges
            if left in external or right in external
        )
        result["internal_edge_count"] = sum(
            1 for left, right in graph.edges
            if left in internal and right in internal
        )

        api_rules = [rule for rule in rules if rule.get("rule_kind", "api") == "api"]
        node_roles = defaultdict(set)
        for node in external:
            class_name, method_name = class_and_method(method_signature(node))
            for rule in api_rules:
                if api_rule_matches(rule, class_name, method_name):
                    node_roles[node].add(rule["role"])
        result["mapped_api_node_count"] = len(node_roles)
        result["unmapped_external_node_count"] = len(external - set(node_roles))
        result["mapped_external_ratio"] = len(node_roles) / max(1, len(external))
        result["sensitive_node_count"] = len(node_roles)
        for node, matched_roles in node_roles.items():
            for role in matched_roles:
                result[f"role_node__{role}"] += 1

        for left, right in graph.edges:
            edge_roles = node_roles.get(left, set()) | node_roles.get(right, set())
            if edge_roles:
                result["sensitive_edge_count"] += 1
                for role in edge_roles:
                    result[f"role_edge__{role}"] += 1

        for rule in rules:
            if rule.get("rule_kind") != "manifest_context":
                continue
            match_count = manifest_context_matches(apk, rule)
            result["manifest_context_match_count"] += match_count
            result[f"context_role__{rule['role']}"] += match_count
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


def append_csv(path, row, fields):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_completed(path):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        return {row["sha256"] for row in read_csv(path) if row.get("sha256")}
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser(description="Extract compact V6 static/API features")
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--priority", required=True, type=Path)
    parser.add_argument("--failure-csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    suppress_logs()
    mapping = load_yaml(args.mapping)
    priority = load_yaml(args.priority)
    roles = list(mapping.get("roles", []))
    rules = list(mapping.get("rules", []))
    allowed_roles = {
        role
        for tier in priority.get("tiers", {}).values()
        for role in tier.get("roles", [])
    }
    unknown_roles = sorted(
        (set(roles) | {rule.get("role") for rule in rules}) - allowed_roles
    )
    if unknown_roles:
        raise SystemExit(f"mapping contains unknown V6 roles: {unknown_roles}")

    lineage = read_csv(args.lineage)
    excluded = set()
    if args.failure_csv and args.failure_csv.exists():
        excluded = {row["sha256"] for row in read_csv(args.failure_csv)}
    lineage = [row for row in lineage if row.get("sha256") not in excluded]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".compact_feature_extraction.lock"
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(
            f"another compact extractor owns the exclusive writer lock: {lock_path}"
        ) from exc
    features_path = output / "features.csv"
    failures_path = output / "feature_failures.csv"
    fields = feature_fields(roles)
    completed = set() if args.no_resume else read_completed(features_path) | read_completed(failures_path)
    if not args.no_resume:
        feature_ids = [row.get("sha256", "").strip().lower()
                       for row in (read_csv(features_path) if features_path.exists() else [])]
        failure_ids = [row.get("sha256", "").strip().lower()
                       for row in (read_csv(failures_path) if failures_path.exists() else [])]
        duplicates = sorted({value for value in feature_ids + failure_ids
                             if value and (feature_ids + failure_ids).count(value) > 1})
        if duplicates:
            raise SystemExit(
                "resume input contains duplicate APK IDs; run the source-lineage "
                f"integrity audit before resuming (n={len(duplicates)})"
            )
    pending = [row for row in lineage if row.get("sha256") not in completed]
    free_guard_bytes = int(max(0.1, args.min_free_gb) * (1024 ** 3))
    progress_every = max(1, args.progress_every)

    mapping_sha = sha256_file(args.mapping)
    priority_sha = sha256_file(args.priority)
    schema_path = output / "feature_schema.json"
    if schema_path.exists() and not args.no_resume:
        try:
            previous_schema = json.loads(schema_path.read_text(encoding="utf-8"))
            if previous_schema.get("mapping_sha256") != mapping_sha:
                raise SystemExit(
                    "existing output uses a different semantic mapping hash; "
                    "choose a new --output directory or use --no-resume"
                )
            if previous_schema.get("priority_sha256") != priority_sha:
                raise SystemExit(
                    "existing output uses a different priority-file hash; "
                    "choose a new --output directory or use --no-resume"
                )
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid existing feature_schema.json: {exc}") from exc
    schema = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "feature_fields": fields,
        "mapping_sha256": mapping_sha,
        "priority_sha256": priority_sha,
        "excluded_static_failure_count": len(excluded),
        "formal_training_enabled": False,
        "graph_persistence": "never",
    }
    schema_path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"compact feature extraction: total={len(lineage)} "
        f"completed={len(completed)} pending={len(pending)}",
        flush=True,
    )

    status_counts = Counter()
    for index, row in enumerate(pending, start=1):
        if shutil.disk_usage(output).free < free_guard_bytes:
            raise SystemExit("free-space guard reached; rerun to resume")
        result = extract_one(row, rules, roles)
        target = features_path if result["status"] == "OK" else failures_path
        append_csv(target, result, fields)
        status_counts[result["status"]] += 1
        if index % progress_every == 0 or index == len(pending):
            print(f"processed {index}/{len(pending)}", flush=True)

    final_feature_rows = read_csv(features_path) if features_path.exists() else []
    final_failure_rows = read_csv(failures_path) if failures_path.exists() else []
    all_rows = final_feature_rows + final_failure_rows
    audit = {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P1_COMPACT_STATIC_API_FEATURE_EXTRACTION",
        "formal_training_enabled": False,
        "lineage_csv": str(args.lineage.expanduser().resolve()),
        "mapping_file": str(args.mapping.expanduser().resolve()),
        "mapping_sha256": mapping_sha,
        "priority_file": str(args.priority.expanduser().resolve()),
        "priority_sha256": priority_sha,
        "requested_apk_count": len(read_csv(args.lineage)),
        "excluded_static_failure_count": len(excluded),
        "eligible_apk_count": len(lineage),
        "feature_row_count": len(final_feature_rows),
        "failure_row_count": len(final_failure_rows),
        "status_counts": dict(Counter(row.get("status", "") for row in all_rows)),
        "mapped_api_node_total": sum(safe_int(row.get("mapped_api_node_count")) for row in final_feature_rows),
        "external_node_total": sum(safe_int(row.get("external_node_count")) for row in final_feature_rows),
        "sensitive_edge_total": sum(safe_int(row.get("sensitive_edge_count")) for row in final_feature_rows),
        "outputs": {
            "features": str(features_path),
            "failures": str(failures_path),
            "schema": str(output / "feature_schema.json"),
        },
        "blocking_reasons": [
            "first_seen_utc_not_joined",
            "sdk_docs_manifest_not_generated",
            "truncation_hyperparameters_not_resolved_by_nested_validation",
            "resolved_formal_training_config_not_generated",
        ],
        "notes": [
            "features and mapping coverage are extracted in one APK parse",
            "call graphs are built only in memory and never persisted",
            "feature CSV is append-resumable after interruption",
            "this is a compact P1 extraction stage, not formal training",
        ],
    }
    (output / "feature_extraction_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    lock_handle.close()


if __name__ == "__main__":
    main()
