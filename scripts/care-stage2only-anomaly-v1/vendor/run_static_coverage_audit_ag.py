#!/usr/bin/env python3
"""Audit Androguard static and call-graph coverage without persisting graphs.

The call graph is created only in memory for one APK at a time.  The output is
limited to compact per-APK statistics and a failure CSV, which is suitable for
the low-disk P0 audit stage.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path):
    with path.expanduser().resolve(strict=True).open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def suppress_logs():
    logging.disable(logging.CRITICAL)
    try:
        from loguru import logger

        logger.remove()
    except Exception:
        pass


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return 0


def object_method(node):
    try:
        getter = getattr(node, "get_method", None)
        if callable(getter):
            method = getter()
            if method is not None:
                return method
    except Exception:
        pass
    return node


def method_signature(node):
    method = object_method(node)
    parts = []
    for name in ("get_class_name", "get_name", "get_descriptor"):
        try:
            getter = getattr(method, name, None)
            parts.append(str(getter()) if callable(getter) else "")
        except Exception:
            parts.append("")
    if parts[0] or parts[1]:
        return f"{parts[0]}->{parts[1]}{parts[2]}"
    return str(method)


def class_name(node):
    method = object_method(node)
    try:
        getter = getattr(method, "get_class_name", None)
        if callable(getter):
            return str(getter())
    except Exception:
        pass
    signature = method_signature(node)
    return signature.split("->", 1)[0]


def is_external(node):
    method = object_method(node)
    try:
        getter = getattr(method, "is_external", None)
        if callable(getter) and bool(getter()):
            return True
    except Exception:
        pass
    name = class_name(node)
    return name.startswith(
        (
            "Landroid/",
            "Landroidx/",
            "Ljava/",
            "Ljavax/",
            "Lkotlin/",
            "Lorg/json/",
        )
    )


def default_sensitive_prefixes():
    return (
        "Landroid/telephony/",
        "Landroid/location/",
        "Landroid/accounts/",
        "Landroid/content/pm/",
        "Landroid/net/",
        "Ljava/net/",
        "Ljava/io/",
        "Landroid/os/",
        "Landroid/app/",
        "Landroid/provider/",
        "Landroid/media/",
        "Landroid/hardware/",
        "Landroid/webkit/",
        "Landroid/crypto/",
        "Ljava/lang/Runtime;->exec",
        "Ljava/lang/ProcessBuilder;",
    )


def load_sensitive_prefixes(path: Path | None):
    if path is None:
        return default_sensitive_prefixes(), "BUILT_IN_PREFIXES"
    try:
        text = path.expanduser().resolve(strict=True).read_text(
            encoding="utf-8", errors="replace"
        )
        # This intentionally accepts a simple YAML/JSON priority file without
        # requiring PyYAML. Every non-empty scalar-looking line is considered
        # a possible endpoint pattern; callers can also use the built-ins.
        values = []
        for line in text.splitlines():
            line = line.strip().strip("- ").strip("\"'")
            if not line or line.startswith("#") or ":" in line and line.endswith(":"):
                continue
            if len(line) >= 5 and any(token in line for token in ("/", "->", ".")):
                values.append(line)
        values = tuple(dict.fromkeys(values))
        if values:
            return values, "FILE_PATTERNS"
    except Exception:
        pass
    return default_sensitive_prefixes(), "BUILT_IN_PREFIXES_FALLBACK"


def apk_row(row, sensitive_prefixes):
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
        "static_status": "ERROR",
        "call_graph_status": "NOT_ATTEMPTED",
        "dex_count": 0,
        "class_count": 0,
        "method_count": 0,
        "graph_node_count": 0,
        "graph_edge_count": 0,
        "external_node_count": 0,
        "external_edge_count": 0,
        "sensitive_node_count": 0,
        "sensitive_edge_count": 0,
        "error_type": "",
        "error_message": "",
    }
    apk = dalvik = analysis = graph = None
    try:
        from androguard.misc import AnalyzeAPK

        apk, dalvik, analysis = AnalyzeAPK(row["absolute_path"])
        result["static_status"] = "OK"
        result["dex_count"] = len(dalvik) if isinstance(dalvik, (list, tuple)) else 1
        try:
            result["class_count"] = sum(len(vm.get_classes()) for vm in dalvik)
        except Exception:
            result["class_count"] = 0
        try:
            result["method_count"] = len(list(analysis.get_methods()))
        except Exception:
            result["method_count"] = 0

        try:
            graph = analysis.get_call_graph(no_isolated=False)
            result["call_graph_status"] = "OK"
            result["graph_node_count"] = int(graph.number_of_nodes())
            result["graph_edge_count"] = int(graph.number_of_edges())
            external_nodes = {
                node for node in graph.nodes if is_external(node)
            }
            result["external_node_count"] = len(external_nodes)
            result["external_edge_count"] = sum(
                1 for left, right in graph.edges
                if is_external(left) or is_external(right)
            )
            sensitive_nodes = {
                node for node in graph.nodes
                if any(
                    pattern in method_signature(node)
                    for pattern in sensitive_prefixes
                )
            }
            result["sensitive_node_count"] = len(sensitive_nodes)
            result["sensitive_edge_count"] = sum(
                1 for left, right in graph.edges
                if left in sensitive_nodes or right in sensitive_nodes
            )
        except Exception as exc:
            result["call_graph_status"] = "FAILED"
            result["error_type"] = type(exc).__name__
            result["error_message"] = str(exc)[:500]
    except Exception as exc:
        result["static_status"] = "FAILED"
        result["call_graph_status"] = "NOT_ATTEMPTED"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:500]
    finally:
        # The graph is deliberately never serialized. Drop all references
        # before moving to the next APK so peak memory stays bounded.
        graph = None
        analysis = None
        dalvik = None
        apk = None
        gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description="Audit Androguard static/call-graph coverage")
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sensitive-endpoints", type=Path)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    args = parser.parse_args()

    suppress_logs()
    lineage_rows = read_csv(args.lineage)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    sensitive_prefixes, sensitive_source = load_sensitive_prefixes(args.sensitive_endpoints)
    free_guard_bytes = int(max(0.1, args.min_free_gb) * (1024 ** 3))

    rows = []
    interrupted_by_disk = False
    progress_every = max(1, args.progress_every)
    print(f"static coverage audit started: {len(lineage_rows)} APKs", flush=True)
    for index, row in enumerate(lineage_rows, start=1):
        if shutil.disk_usage(output).free < free_guard_bytes:
            interrupted_by_disk = True
            print("stopped: free-space guard reached", flush=True)
            break
        rows.append(apk_row(row, sensitive_prefixes))
        if index % progress_every == 0 or index == len(lineage_rows):
            print(f"processed {index}/{len(lineage_rows)}", flush=True)

    csv_fields = [
        "sha256", "absolute_path", "source_id", "lineage_id", "metadata_year",
        "metadata_label", "package_name", "certificate_sha256", "version_code",
        "static_status", "call_graph_status", "dex_count", "class_count",
        "method_count", "graph_node_count", "graph_edge_count",
        "external_node_count", "external_edge_count", "sensitive_node_count",
        "sensitive_edge_count", "error_type", "error_message",
    ]
    write_csv(output / "static_coverage.csv", rows, csv_fields)
    failures = [row for row in rows if row["static_status"] != "OK" or row["call_graph_status"] != "OK"]
    write_csv(output / "static_coverage_failures.csv", failures, csv_fields)

    graph_ok = sum(row["call_graph_status"] == "OK" for row in rows)
    static_ok = sum(row["static_status"] == "OK" for row in rows)
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P0_STATIC_PARSE_AND_API_GRAPH_COVERAGE",
        "formal_training_enabled": False,
        "lineage_csv": str(args.lineage.expanduser().resolve()),
        "requested_apk_count": len(lineage_rows),
        "processed_apk_count": len(rows),
        "static_status_counts": dict(Counter(row["static_status"] for row in rows)),
        "call_graph_status_counts": dict(Counter(row["call_graph_status"] for row in rows)),
        "static_ok_count": static_ok,
        "call_graph_ok_count": graph_ok,
        "static_coverage": static_ok / len(rows) if rows else 0.0,
        "call_graph_coverage": graph_ok / len(rows) if rows else 0.0,
        "total_graph_nodes": sum(row["graph_node_count"] for row in rows),
        "total_graph_edges": sum(row["graph_edge_count"] for row in rows),
        "total_external_nodes": sum(row["external_node_count"] for row in rows),
        "total_external_edges": sum(row["external_edge_count"] for row in rows),
        "sensitive_endpoint_source": sensitive_source,
        "sensitive_endpoint_pattern_count": len(sensitive_prefixes),
        "disk_guard_triggered": interrupted_by_disk,
        "outputs": {
            "coverage_csv": str(output / "static_coverage.csv"),
            "failures_csv": str(output / "static_coverage_failures.csv"),
        },
        "blocking_reasons": [
            "first_seen_utc_not_joined",
            "formal_feature_schema_not_locked",
            "resolved_formal_training_config_not_generated",
        ],
        "notes": [
            "call graphs are built only in memory and are never persisted",
            "Androguard logs are suppressed; only progress and final summary are printed",
            "static_coverage.csv contains compact audit statistics, not graph data",
            "identity-unresolved APKs are outside the lineage input and remain excluded",
        ],
    }
    (output / "static_coverage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
