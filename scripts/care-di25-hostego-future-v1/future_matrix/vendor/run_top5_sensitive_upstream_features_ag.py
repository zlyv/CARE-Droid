#!/usr/bin/env python3
"""Streaming no-Entry sensitive-upstream structural feature extraction.

No graph is persisted.

For each APK:
  1. build the full Androguard call graph in memory;
  2. construct the frozen 4-hop sensitive-upstream view in memory;
  3. immediately pool it into fixed-length numeric features;
  4. append one compressed CSV row;
  5. release all graph objects.

Persisted output contains no node IDs, method signatures, edge indexes,
or serialized graph/subgraph.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import shutil
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_endpoint_centered_tensor_cache_ag import (  # noqa: E402
    append_failure,
    role_tiers,
    suppress_loguru,
)
from run_semantic_mapping_audit_ag import load_yaml  # noqa: E402
from run_static_coverage_audit_ag import read_csv, suppress_logs  # noqa: E402

from run_sensitive_upstream_tensor_cache_ag import (  # noqa: E402
    build_sensitive_upstream_view,
    canonicalize_call_graph,
)


GLOBAL_FEATURE_NAMES = [
    "up_global_endpoint_log",
    "up_global_internal_caller_log",
    "up_global_path_edge_log",
    "up_global_role_coverage",
    "up_global_edge_per_node",
    "up_global_icc_edge_ratio",
    "up_global_reflection_edge_ratio",
    "up_global_non_direct_edge_ratio",

    "up_global_depth1_caller_log",
    "up_global_depth2_caller_log",
    "up_global_depth3_caller_log",
    "up_global_depth4_caller_log",

    "up_global_depth1_caller_ratio",
    "up_global_depth2_caller_ratio",
    "up_global_depth3_caller_ratio",
    "up_global_depth4_caller_ratio",

    "up_global_shared_ge2_role_log",
    "up_global_shared_ge3_role_log",
    "up_global_shared_ge2_role_ratio",
    "up_global_shared_role_span_mean",
    "up_global_shared_role_span_max_norm",
]


def feature_names(roles: list[str]) -> list[str]:
    names = list(GLOBAL_FEATURE_NAMES)

    for role in roles:
        prefix = f"up_role__{role}__"
        names.extend([
            prefix + "endpoint_log",
            prefix + "caller_log",
            prefix + "caller_ratio",
            prefix + "mean_depth_norm",
            prefix + "deep_caller_ratio",
        ])

    return names


def existing_shas(path: Path) -> set[str]:
    if not path.exists():
        return set()

    result = set()

    try:
        with gzip.open(
            path, "rt", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)

            for row in reader:
                sha = str(row.get("sha256", "")).casefold()
                if sha:
                    result.add(sha)

    except (OSError, EOFError):
        return set()

    return result


def append_feature_row(
    path: Path,
    fields: list[str],
    row: dict,
) -> None:
    exists = path.exists() and path.stat().st_size > 0

    with gzip.open(
        path, "at", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        if not exists:
            writer.writeheader()

        writer.writerow({
            field: row.get(field, "")
            for field in fields
        })


def build_incoming(
    active_nodes: set[str],
    retained_edges: list[dict],
):
    incoming = defaultdict(list)

    for edge in retained_edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))

        if (
            source in active_nodes
            and target in active_nodes
        ):
            incoming[target].append(source)

    for target in incoming:
        incoming[target] = sorted(
            set(incoming[target])
        )

    return incoming


def reverse_distances(
    seeds: set[str],
    incoming,
    node_records,
    max_hops: int,
):
    """Minimum internal-caller distance to any seed."""

    distances = {
        seed: 0
        for seed in seeds
    }

    queue = deque(
        sorted(seeds)
    )

    while queue:
        current = queue.popleft()
        distance = distances[current]

        if distance >= max_hops:
            continue

        for caller in incoming.get(current, []):
            # Context must be application-internal.
            if node_records[caller].get("external"):
                continue

            next_distance = distance + 1
            previous = distances.get(caller)

            if (
                previous is None
                or next_distance < previous
            ):
                distances[caller] = next_distance
                queue.append(caller)

    return distances


def structural_features(
    node_records: dict,
    active_nodes: set[str],
    retained_edges: list[dict],
    roles: list[str],
    max_hops: int,
):
    role_set = set(roles)

    endpoints = {
        signature
        for signature in active_nodes
        if node_records[signature].get("roles")
    }

    internal_callers = {
        signature
        for signature in active_nodes
        if signature not in endpoints
        and not node_records[signature].get("external")
    }

    incoming = build_incoming(
        active_nodes,
        retained_edges,
    )

    # Minimum distance of each internal caller to any sensitive endpoint.
    global_distances = reverse_distances(
        endpoints,
        incoming,
        node_records,
        max_hops,
    )

    caller_min_depth = {
        signature: distance
        for signature, distance
        in global_distances.items()
        if signature in internal_callers
    }

    depth_counts = Counter(
        caller_min_depth.values()
    )

    # Role-specific caller sets and distances.
    role_endpoint_counts = Counter()
    role_caller_distances = {}
    caller_reached_roles = defaultdict(set)

    for endpoint in endpoints:
        for role in node_records[endpoint].get(
            "roles", []
        ):
            if role in role_set:
                role_endpoint_counts[role] += 1

    for role in roles:
        role_endpoints = {
            endpoint
            for endpoint in endpoints
            if role in node_records[endpoint].get(
                "roles", []
            )
        }

        distances = reverse_distances(
            role_endpoints,
            incoming,
            node_records,
            max_hops,
        )

        current = {
            signature: distance
            for signature, distance
            in distances.items()
            if signature in internal_callers
        }

        role_caller_distances[role] = current

        for caller in current:
            caller_reached_roles[caller].add(role)

    # Edge-level bounded structural statistics.
    edge_count = len(retained_edges)

    icc_count = sum(
        bool(edge.get("icc_any"))
        for edge in retained_edges
    )
    reflection_count = sum(
        bool(edge.get("reflection_any"))
        for edge in retained_edges
    )
    non_direct_count = sum(
        str(edge.get(
            "edge_type", "DIRECT_CALL"
        )) != "DIRECT_CALL"
        for edge in retained_edges
    )

    total_callers = len(internal_callers)
    total_nodes = len(active_nodes)

    endpoint_roles_present = {
        role
        for endpoint in endpoints
        for role in node_records[endpoint].get(
            "roles", []
        )
        if role in role_set
    }

    spans = [
        len(caller_reached_roles[caller])
        for caller in internal_callers
    ]

    shared_ge2 = sum(
        value >= 2
        for value in spans
    )
    shared_ge3 = sum(
        value >= 3
        for value in spans
    )

    features = {
        "up_global_endpoint_log":
            float(np.log1p(len(endpoints))),

        "up_global_internal_caller_log":
            float(np.log1p(total_callers)),

        "up_global_path_edge_log":
            float(np.log1p(edge_count)),

        "up_global_role_coverage":
            len(endpoint_roles_present)
            / max(1, len(roles)),

        "up_global_edge_per_node":
            edge_count / max(1, total_nodes),

        "up_global_icc_edge_ratio":
            icc_count / max(1, edge_count),

        "up_global_reflection_edge_ratio":
            reflection_count / max(1, edge_count),

        "up_global_non_direct_edge_ratio":
            non_direct_count / max(1, edge_count),

        "up_global_shared_ge2_role_log":
            float(np.log1p(shared_ge2)),

        "up_global_shared_ge3_role_log":
            float(np.log1p(shared_ge3)),

        "up_global_shared_ge2_role_ratio":
            shared_ge2 / max(1, total_callers),

        "up_global_shared_role_span_mean":
            float(np.mean(spans))
            if spans else 0.0,

        "up_global_shared_role_span_max_norm":
            (
                max(spans) / max(1, len(roles))
                if spans else 0.0
            ),
    }

    for depth in range(1, 5):
        count = depth_counts.get(depth, 0)

        features[
            f"up_global_depth{depth}_caller_log"
        ] = float(np.log1p(count))

        features[
            f"up_global_depth{depth}_caller_ratio"
        ] = count / max(1, total_callers)

    for role in roles:
        callers = role_caller_distances[role]
        distances = list(callers.values())

        prefix = f"up_role__{role}__"

        features[prefix + "endpoint_log"] = (
            float(np.log1p(
                role_endpoint_counts[role]
            ))
        )

        features[prefix + "caller_log"] = (
            float(np.log1p(len(callers)))
        )

        features[prefix + "caller_ratio"] = (
            len(callers)
            / max(1, total_callers)
        )

        features[prefix + "mean_depth_norm"] = (
            float(np.mean(distances))
            / max(1, max_hops)
            if distances else 0.0
        )

        features[prefix + "deep_caller_ratio"] = (
            sum(value >= 3 for value in distances)
            / max(1, len(distances))
            if distances else 0.0
        )

    return features


def analyze_one(
    row,
    rules,
    priority,
    edge_rules,
    roles,
    upstream_hops,
    k_in,
    endpoint_cap,
    node_cap,
    edge_cap,
):
    result = {
        "sha256": row.get("sha256", ""),
        "source_id": row.get("source_id", ""),
        "metadata_year": row.get("metadata_year", ""),
        "metadata_label": row.get("metadata_label", ""),
        "status": "FAILED",
        "error_type": "",
        "error_message": "",
    }

    apk = dalvik = analysis = graph = None
    node_records = None
    edge_records = None
    active_nodes = None
    retained_edges = None

    try:
        from androguard.misc import AnalyzeAPK

        apk, dalvik, analysis = AnalyzeAPK(
            row["absolute_path"]
        )

        graph = analysis.get_call_graph(
            no_isolated=False
        )

        tiers = role_tiers(priority)

        node_records, edge_records = (
            canonicalize_call_graph(
                graph,
                rules,
                tiers,
                edge_rules,
            )
        )

        (
            active_nodes,
            retained_edges,
            summary,
        ) = build_sensitive_upstream_view(
            node_records,
            edge_records,
            tiers,
            upstream_hops,
            k_in,
            endpoint_cap,
            node_cap,
            edge_cap,
        )

        values = structural_features(
            node_records,
            active_nodes,
            retained_edges,
            roles,
            upstream_hops,
        )

        result.update({
            "status": "OK",
            "retained_node_count":
                summary["retained_node_count"],
            "retained_edge_count":
                summary["retained_path_edge_count"],
            "retained_internal_caller_count":
                summary[
                    "retained_internal_caller_count"
                ],
            "max_upstream_distance":
                summary[
                    "max_retained_upstream_distance"
                ],
            "node_truncated":
                int(summary["node_truncated"]),
            "edge_truncated":
                int(summary["edge_truncated"]),
            **values,
        })

    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:500]

    finally:
        # Nothing graph-shaped survives this APK.
        retained_edges = None
        active_nodes = None
        edge_records = None
        node_records = None
        graph = None
        analysis = None
        dalvik = None
        apk = None
        gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--lineage",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--mapping",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--priority",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--edge-semantics",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--failure-csv",
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--upstream-hops",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--K-in",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--endpoint-cap",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--node-cap",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--edge-cap",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
    )

    args = parser.parse_args()

    suppress_loguru()
    suppress_logs()

    mapping = load_yaml(args.mapping)
    priority = load_yaml(args.priority)
    edge_mapping = load_yaml(
        args.edge_semantics
    )

    rules = list(
        mapping.get("rules", [])
    )

    # Freeze deterministic role order.
    roles = sorted(
        str(role)
        for role in mapping.get("roles", [])
    )

    edge_rules = list(
        edge_mapping.get("edge_rules", [])
    )

    excluded = set()

    if (
        args.failure_csv
        and args.failure_csv.exists()
    ):
        excluded = {
            row.get("sha256", "").casefold()
            for row in read_csv(
                args.failure_csv
            )
            if row.get("sha256")
        }

    rows = [
        row
        for row in read_csv(args.lineage)
        if row.get(
            "sha256", ""
        ).casefold() not in excluded
    ]

    rows.sort(
        key=lambda row:
            row.get(
                "sha256", ""
            ).casefold()
    )

    output = args.output.expanduser().resolve()
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_path = (
        output
        / "sensitive_upstream_features.csv.gz"
    )

    failure_path = (
        output
        / "sensitive_upstream_failures.csv"
    )

    already_done = (
        existing_shas(feature_path)
        if args.resume
        else set()
    )

    rows = [
        row
        for row in rows
        if row.get(
            "sha256", ""
        ).casefold()
        not in already_done
    ]

    if args.limit > 0:
        rows = rows[:args.limit]

    if not rows:
        raise SystemExit(
            "no APK selected"
        )

    numeric_fields = feature_names(roles)

    fields = [
        "sha256",
        "source_id",
        "metadata_year",
        "metadata_label",
        "status",
        "retained_node_count",
        "retained_edge_count",
        "retained_internal_caller_count",
        "max_upstream_distance",
        "node_truncated",
        "edge_truncated",
    ] + numeric_fields

    statuses = Counter()

    print(
        f"sensitive-upstream streaming features: "
        f"{len(rows)} APKs "
        f"(resume_skipped={len(already_done)})",
        flush=True,
    )

    for index, row in enumerate(rows, 1):
        free_gb = (
            shutil.disk_usage(output).free
            / (1024 ** 3)
        )

        if free_gb < args.min_free_gb:
            raise SystemExit(
                f"disk guard: "
                f"{free_gb:.3f} GB free"
            )

        result = analyze_one(
            row,
            rules,
            priority,
            edge_rules,
            roles,
            args.upstream_hops,
            args.K_in,
            args.endpoint_cap,
            args.node_cap,
            args.edge_cap,
        )

        statuses[result["status"]] += 1

        if result["status"] == "OK":
            append_feature_row(
                feature_path,
                fields,
                result,
            )
        else:
            append_failure(
                failure_path,
                {**row, **result},
            )

        if (
            index % max(
                1,
                args.progress_every,
            ) == 0
            or index == len(rows)
        ):
            print(
                f"processed {index}/{len(rows)} "
                f"ok={statuses['OK']} "
                f"failed={statuses['FAILED']}",
                flush=True,
            )

    schema = {
        "schema_version": "1.0.0",
        "view":
            "SENSITIVE_UPSTREAM_STREAMING_FEATURES",
        "feature_count": len(numeric_fields),
        "feature_fields": numeric_fields,
        "roles": roles,
        "policy": {
            "upstream_hops":
                args.upstream_hops,
            "K_in":
                args.K_in,
            "endpoint_cap":
                args.endpoint_cap,
            "node_cap":
                args.node_cap,
            "edge_cap":
                args.edge_cap,
        },
        "persistence": {
            "raw_call_graph": False,
            "bounded_subgraph": False,
            "node_ids": False,
            "edge_index": False,
            "method_signatures": False,
            "numeric_features_only": True,
        },
    }

    (
        output
        / "sensitive_upstream_feature_schema.json"
    ).write_text(
        json.dumps(
            schema,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "phase":
            "SENSITIVE_UPSTREAM_STREAMING_FEATURE_EXTRACTION",
        "status_counts":
            dict(statuses),
        "processed_apk_count":
            sum(statuses.values()),
        "persisted_graph":
            False,
        "persisted_node_ids":
            False,
        "persisted_edge_index":
            False,
        "persisted_method_signatures":
            False,
        "feature_count":
            len(numeric_fields),
        "outputs": {
            "features":
                str(feature_path),
            "schema":
                str(
                    output
                    / "sensitive_upstream_feature_schema.json"
                ),
            "failures":
                str(failure_path),
        },
    }

    (
        output
        / "sensitive_upstream_feature_audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
