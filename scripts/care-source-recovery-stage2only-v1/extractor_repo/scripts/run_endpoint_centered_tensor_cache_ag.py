#!/usr/bin/env python3
"""Extract a bounded sensitive-endpoint graph cache with low disk usage.

Each APK is parsed once.  Its full Androguard call graph and raw signatures
are released immediately after a compact hashed endpoint-centred payload is
written.  Labels are never used for endpoint selection or graph construction.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import gzip
import hashlib
import json
import logging
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_semantic_mapping_audit_ag import load_yaml  # noqa: E402
from run_static_coverage_audit_ag import read_csv, suppress_logs  # noqa: E402
from tasa_droid.p2.endpoint_centered_view import (  # noqa: E402
    EndpointViewPolicy,
    build_endpoint_centered_structure,
)
from tasa_droid.p2.graph_representation import (  # noqa: E402
    GraphPolicy,
    build_compact_graph_summary,
)


def suppress_loguru() -> None:
    logging.disable(logging.INFO)
    try:
        from loguru import logger

        logger.remove()
    except Exception:
        pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role_tiers(priority: dict) -> dict[str, int]:
    result: dict[str, int] = {}
    for tier, value in (priority.get("tiers", {}) or {}).items():
        try:
            numeric_tier = int(tier)
        except (TypeError, ValueError):
            numeric_tier = 4
        for role in (value or {}).get("roles", []):
            result[str(role)] = numeric_tier
    return result


def existing_successful_shas(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status") == "OK" and row.get("sha256"):
                    result.add(str(row["sha256"]).casefold())
    except (OSError, EOFError):
        return set()
    return result


def append_cache(path: Path, row: dict) -> None:
    with gzip.open(path, "at", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def append_failure(path: Path, row: dict) -> None:
    fields = [
        "sha256", "absolute_path", "source_id", "lineage_id",
        "metadata_year", "status", "error_type", "error_message",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fields})


def analyze_one(
    row: dict[str, str],
    rules: list[dict],
    priority: dict,
    edge_rules: list[dict],
    roles: list[str],
    role_tier: dict[str, int],
    entry_policy: GraphPolicy,
    endpoint_policy: EndpointViewPolicy,
) -> dict:
    result = {
        "sha256": row.get("sha256", ""),
        "source_id": row.get("source_id", ""),
        "metadata_year": row.get("metadata_year", ""),
        "status": "FAILED",
        "error_type": "",
        "error_message": "",
    }
    apk = dalvik = analysis = graph = None
    try:
        from androguard.misc import AnalyzeAPK

        apk, dalvik, analysis = AnalyzeAPK(row["absolute_path"])
        graph = analysis.get_call_graph(no_isolated=False)
        compact = build_compact_graph_summary(
            apk,
            graph,
            rules,
            priority,
            entry_policy,
            edge_rules,
            include_internal=True,
        )
        internal = compact.pop("_internal")
        package_name = str(apk.get_package() or "")
        structure = build_endpoint_centered_structure(
            internal["node_records"],
            internal["edge_records"],
            set(internal["entry_signatures"]),
            roles,
            role_tier,
            package_name,
            endpoint_policy,
        )
        result.update({
            "status": "OK",
            "node_count": structure["node_count"],
            "edge_count": structure["edge_count"],
            "base_node_feature_count": structure["base_node_feature_count"],
            "edge_feature_count": structure["edge_feature_count"],
            "role_vocabulary_size": structure["role_vocabulary_size"],
            "tensor_sha256": structure["tensor_sha256"],
            "summary": structure["summary"],
            "payload": structure["payload"],
        })
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--priority", required=True, type=Path)
    parser.add_argument("--edge-semantics", required=True, type=Path)
    parser.add_argument("--failure-csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20, help="0 means all APKs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--upstream-hops", type=int, default=2)
    parser.add_argument("--downstream-hops", type=int, default=1)
    parser.add_argument("--K-in", type=int, default=8)
    parser.add_argument("--K-out", type=int, default=4)
    parser.add_argument("--endpoint-cap", type=int, default=128)
    parser.add_argument("--node-cap", type=int, default=2048)
    parser.add_argument("--edge-cap", type=int, default=4096)
    parser.add_argument("--reachability-max-states", type=int, default=100000)
    parser.add_argument("--min-free-gb", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    suppress_loguru()
    suppress_logs()
    mapping = load_yaml(args.mapping)
    priority = load_yaml(args.priority)
    edge_mapping = load_yaml(args.edge_semantics)
    rules = list(mapping.get("rules", []))
    roles = list(mapping.get("roles", []))
    edge_rules = list(edge_mapping.get("edge_rules", []))
    tiers = role_tiers(priority)

    excluded: set[str] = set()
    if args.failure_csv and args.failure_csv.exists():
        excluded = {
            row.get("sha256", "").casefold()
            for row in read_csv(args.failure_csv)
            if row.get("sha256")
        }
    rows = [
        row for row in read_csv(args.lineage)
        if row.get("sha256", "").casefold() not in excluded
    ]
    rows.sort(key=lambda row: row.get("sha256", "").casefold())
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".endpoint_cache_extraction.lock"
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(
            f"another endpoint-cache extractor owns the exclusive writer lock: {lock_path}"
        ) from exc
    cache_path = output / "endpoint_centered_tensor_cache.jsonl.gz"
    failure_path = output / "endpoint_centered_tensor_failures.csv"
    already_done = existing_successful_shas(cache_path) if args.resume else set()
    rows = [
        row for row in rows
        if row.get("sha256", "").casefold() not in already_done
    ]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no APK selected after exclusion/resume filtering")

    # The existing entry-path builder is invoked only to canonicalise the
    # transient graph and bind Manifest entries.  Its payload is not rewritten.
    entry_policy = GraphPolicy(H=4, K_out=8, P_cap=8, E_cap=2000, max_states=20000)
    endpoint_policy = EndpointViewPolicy(
        upstream_hops=max(0, args.upstream_hops),
        downstream_hops=max(0, args.downstream_hops),
        K_in=max(1, args.K_in),
        K_out=max(1, args.K_out),
        endpoint_cap=max(1, args.endpoint_cap),
        node_cap=max(1, args.node_cap),
        edge_cap=max(1, args.edge_cap),
        reachability_max_states=max(1, args.reachability_max_states),
    )
    statuses = Counter()
    totals = Counter()
    print(
        f"endpoint-centred cache started: {len(rows)} APKs "
        f"(resume_skipped={len(already_done)})",
        flush=True,
    )
    for index, row in enumerate(rows, 1):
        free_gb = shutil.disk_usage(output).free / (1024 ** 3)
        if free_gb < args.min_free_gb:
            raise SystemExit(
                f"disk guard stopped run: free_gb={free_gb:.3f} "
                f"< min_free_gb={args.min_free_gb:.3f}"
            )
        result = analyze_one(
            row, rules, priority, edge_rules, roles, tiers,
            entry_policy, endpoint_policy,
        )
        statuses[result["status"]] += 1
        if result["status"] == "OK":
            append_cache(cache_path, result)
            summary = result["summary"]
            totals["sensitive"] += int(summary["sensitive_endpoint_count_total"])
            totals["selected"] += int(summary["selected_endpoint_count"])
            totals["reachable"] += int(summary["entry_reachable_selected_endpoint_count"])
            totals["unreachable"] += int(summary["entry_unreachable_selected_endpoint_count"])
            totals["unknown"] += int(summary["reachability_unknown_selected_endpoint_count"])
            totals["node"] += int(result["node_count"])
            totals["edge"] += int(result["edge_count"])
            totals["endpoint_truncated"] += int(bool(summary["endpoint_truncated"]))
            totals["node_truncated"] += int(bool(summary["node_truncated"]))
            totals["edge_truncated"] += int(bool(summary["edge_truncated"]))
            totals["reachability_cap_hit"] += int(
                bool(summary["entry_reachability_state_cap_hit"])
            )
        else:
            failure_row = dict(row)
            failure_row.update(result)
            append_failure(failure_path, failure_row)
        if index % max(1, args.progress_every) == 0 or index == len(rows):
            print(
                f"processed {index}/{len(rows)} "
                f"ok={statuses['OK']} failed={statuses['FAILED']}",
                flush=True,
            )
        gc.collect()

    ok_count = statuses["OK"]
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P25_SENSITIVE_ENDPOINT_CENTERED_BOUNDED_TENSOR_CACHE",
        "formal_training_enabled": False,
        "label_independent_graph_construction": True,
        "lineage_csv": str(args.lineage.expanduser().resolve()),
        "requested_apk_count": len(rows),
        "resume_skipped_success_count": len(already_done),
        "status_counts": dict(statuses),
        "excluded_static_failure_count": len(excluded),
        "policy": {
            "upstream_hops": endpoint_policy.upstream_hops,
            "downstream_hops": endpoint_policy.downstream_hops,
            "K_in": endpoint_policy.K_in,
            "K_out": endpoint_policy.K_out,
            "endpoint_cap": endpoint_policy.endpoint_cap,
            "node_cap": endpoint_policy.node_cap,
            "edge_cap": endpoint_policy.edge_cap,
            "reachability_max_states": endpoint_policy.reachability_max_states,
        },
        "aggregate": {
            "sensitive_endpoint_count_total": totals["sensitive"],
            "selected_endpoint_count": totals["selected"],
            "entry_reachable_selected_endpoint_count": totals["reachable"],
            "entry_unreachable_selected_endpoint_count": totals["unreachable"],
            "reachability_unknown_selected_endpoint_count": totals["unknown"],
            "cached_node_count": totals["node"],
            "cached_edge_count": totals["edge"],
            "mean_cached_nodes_per_ok_apk": totals["node"] / max(1, ok_count),
            "mean_cached_edges_per_ok_apk": totals["edge"] / max(1, ok_count),
            "endpoint_truncated_apk_count": totals["endpoint_truncated"],
            "node_truncated_apk_count": totals["node_truncated"],
            "edge_truncated_apk_count": totals["edge_truncated"],
            "reachability_state_cap_hit_apk_count": totals["reachability_cap_hit"],
        },
        "cache_sha256": sha256_file(cache_path) if cache_path.exists() else None,
        "outputs": {
            "cache": str(cache_path),
            "failures": str(failure_path),
        },
        "blocking_reasons": [
            "endpoint_view_hyperparameters_require_2017_only_validation",
            "dual_view_fusion_not_implemented",
            "formal_training_config_not_resolved",
        ],
        "notes": [
            "raw DEX, raw graphs, paths, and raw method signatures are never persisted",
            "the cache contains bounded hashed IDs, role IDs, indexes, and numeric features only",
            "entry-unreachable and reachability-unknown are represented as distinct states",
            "this cache stage does not train or select a model",
        ],
    }
    (output / "endpoint_centered_tensor_cache_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    lock_handle.close()


if __name__ == "__main__":
    main()
