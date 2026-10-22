#!/usr/bin/env python3
"""Audit the new V6 semantic mapping against the resolved APK inventory.

The call graph is created for one APK at a time and released immediately.
Only per-APK role counts and compact coverage summaries are written.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "PyYAML is required. Run this script with the Androguard environment "
        "that provides yaml.safe_load()."
    ) from exc

from run_static_coverage_audit_ag import (
    is_external,
    method_signature,
    read_csv,
    suppress_logs,
    write_csv,
)


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path):
    with path.expanduser().resolve(strict=True).open(
        "r", encoding="utf-8", errors="replace"
    ) as handle:
        return yaml.safe_load(handle)


def canonical(value):
    return str(value or "").strip().lower().replace(" ", "")


def class_and_method(signature):
    signature = canonical(signature)
    if "->" not in signature:
        return signature, ""
    class_name, tail = signature.split("->", 1)
    method_name = tail.split("(", 1)[0]
    return class_name, method_name


def api_rule_matches(rule, class_name, method_name):
    if rule.get("rule_kind", "api") != "api":
        return False
    class_prefixes = [canonical(item) for item in rule.get("class_prefixes", [])]
    class_contains = [canonical(item) for item in rule.get("class_contains", [])]
    method_names = [canonical(item) for item in rule.get("method_names", [])]
    method_prefixes = [canonical(item) for item in rule.get("method_prefixes", [])]
    if class_prefixes or class_contains:
        class_ok = any(class_name.startswith(item) for item in class_prefixes)
        class_ok = class_ok or any(item in class_name for item in class_contains)
        if not class_ok:
            return False
    if method_names or method_prefixes:
        method_ok = method_name in method_names
        method_ok = method_ok or any(method_name.startswith(item) for item in method_prefixes)
        if not method_ok:
            return False
    return bool(class_prefixes or class_contains) and bool(method_names or method_prefixes)


def manifest_context_matches(apk, rule):
    if rule.get("rule_kind") != "manifest_context":
        return 0
    wanted = {canonical(value) for value in rule.get("context_values", [])}
    if not wanted:
        return 0
    try:
        manifest = apk.get_android_manifest_xml()
        matches = 0
        for element in manifest.iter():
            tag = str(getattr(element, "tag", ""))
            if not tag.endswith("action"):
                continue
            for key, value in element.attrib.items():
                if str(key).endswith("name") and canonical(value) in wanted:
                    matches += 1
        return matches
    except Exception:
        return 0


def process_apk(row, rules, roles):
    result = {
        "sha256": row.get("sha256", ""),
        "absolute_path": row.get("absolute_path", ""),
        "metadata_year": row.get("metadata_year", ""),
        "metadata_label": row.get("metadata_label", ""),
        "package_name": row.get("package_name", ""),
        "status": "ERROR",
        "graph_node_count": 0,
        "external_node_count": 0,
        "mapped_api_node_count": 0,
        "unmapped_external_node_count": 0,
        "manifest_context_match_count": 0,
        "error_type": "",
        "error_message": "",
    }
    for role in roles:
        result[f"role__{role}"] = 0
    apk = dalvik = analysis = graph = None
    try:
        from androguard.misc import AnalyzeAPK

        apk, dalvik, analysis = AnalyzeAPK(row["absolute_path"])
        graph = analysis.get_call_graph(no_isolated=False)
        result["status"] = "OK"
        result["graph_node_count"] = int(graph.number_of_nodes())
        external_nodes = [node for node in graph.nodes if is_external(node)]
        result["external_node_count"] = len(external_nodes)
        api_rules = [rule for rule in rules if rule.get("rule_kind", "api") == "api"]
        for node in external_nodes:
            class_name, method_name = class_and_method(method_signature(node))
            matched_roles = {
                rule["role"] for rule in api_rules
                if api_rule_matches(rule, class_name, method_name)
            }
            if matched_roles:
                result["mapped_api_node_count"] += 1
                for role in matched_roles:
                    result[f"role__{role}"] += 1
            else:
                result["unmapped_external_node_count"] += 1
        for rule in rules:
            result["manifest_context_match_count"] += manifest_context_matches(apk, rule)
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
    parser = argparse.ArgumentParser(description="Audit V6 semantic mapping coverage")
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--priority", required=True, type=Path)
    parser.add_argument("--failure-csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
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
    rule_ids = [rule.get("rule_id") for rule in rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise SystemExit("mapping contains duplicate rule_id values")

    excluded = set()
    if args.failure_csv and args.failure_csv.exists():
        excluded = {row["sha256"] for row in read_csv(args.failure_csv)}
    lineage = [row for row in read_csv(args.lineage) if row.get("sha256") not in excluded]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    free_guard_bytes = int(max(0.1, args.min_free_gb) * (1024 ** 3))

    rows = []
    progress_every = max(1, args.progress_every)
    print(f"semantic mapping audit started: {len(lineage)} APKs", flush=True)
    for index, row in enumerate(lineage, start=1):
        if shutil.disk_usage(output).free < free_guard_bytes:
            raise SystemExit("free-space guard reached before audit completion")
        rows.append(process_apk(row, rules, roles))
        if index % progress_every == 0 or index == len(lineage):
            print(f"processed {index}/{len(lineage)}", flush=True)

    fields = [
        "sha256", "absolute_path", "metadata_year", "metadata_label", "package_name",
        "status", "graph_node_count", "external_node_count", "mapped_api_node_count",
        "unmapped_external_node_count", "manifest_context_match_count",
    ] + [f"role__{role}" for role in roles] + ["error_type", "error_message"]
    write_csv(output / "semantic_mapping_coverage.csv", rows, fields)
    failures = [row for row in rows if row["status"] != "OK"]
    write_csv(output / "semantic_mapping_failures.csv", failures, fields)

    api_mapped = sum(row["mapped_api_node_count"] for row in rows)
    api_external = sum(row["external_node_count"] for row in rows)
    role_totals = {
        role: sum(row[f"role__{role}"] for row in rows)
        for role in roles
    }
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P0_SEMANTIC_MAPPING_COVERAGE_AUDIT",
        "formal_training_enabled": False,
        "lineage_csv": str(args.lineage.expanduser().resolve()),
        "mapping_file": str(args.mapping.expanduser().resolve()),
        "mapping_sha256": sha256_file(args.mapping),
        "priority_file": str(args.priority.expanduser().resolve()),
        "priority_sha256": sha256_file(args.priority),
        "mapping_role_count": len(roles),
        "mapping_rule_count": len(rules),
        "unknown_roles": unknown_roles,
        "requested_apk_count": len(read_csv(args.lineage)),
        "excluded_failure_count": len(excluded),
        "processed_apk_count": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "graph_ok_count": sum(row["status"] == "OK" for row in rows),
        "external_node_count": api_external,
        "mapped_api_node_count": api_mapped,
        "api_mapping_coverage": api_mapped / api_external if api_external else 0.0,
        "role_match_totals": role_totals,
        "outputs": {
            "coverage_csv": str(output / "semantic_mapping_coverage.csv"),
            "failures_csv": str(output / "semantic_mapping_failures.csv"),
        },
        "blocking_reasons": [
            "first_seen_utc_not_joined",
            "sdk_docs_manifest_not_generated",
            "resolved_formal_training_config_not_generated",
        ],
        "notes": [
            "new V6 mapping is audited independently of prior V12 semantic features",
            "four static-parse failures are excluded from the primary audit and retained in failure analysis",
            "graphs are built only in memory and never persisted",
            "mapping coverage is a diagnostic signal, not a classifier performance result",
        ],
    }
    (output / "semantic_mapping_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
