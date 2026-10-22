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
from itertools import combinations
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



FROZEN_TOP5_ROLES = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)


def feature_names(_roles):
    """
    Fixed Top5 semantic caller-organization representation.

    Counts and normalized semantic-organization features are both
    persisted so that source-year ablations can separate:
      - raw amount/size evidence;
      - normalized semantic organization.
    """
    roles = list(FROZEN_TOP5_ROLES)
    names = []

    # 5 roles x 3 = 15.
    for role in roles:
        names.extend([
            f"sem_role__{role}__shared_caller_log",
            f"sem_role__{role}__shared_caller_ratio",
            f"sem_role__{role}__mean_semantic_degree_norm",
        ])

    # C(5, 2) x 2 = 20.
    for left, right in combinations(roles, 2):
        names.extend([
            (
                f"sem_pair__{left}__{right}"
                "__shared_caller_log"
            ),
            (
                f"sem_pair__{left}__{right}"
                "__jaccard"
            ),
        ])

    # 4 global semantic-organization summaries.
    names.extend([
        "sem_global__organizer_log",
        "sem_global__multi_role_ge2_ratio",
        "sem_global__multi_role_ge3_ratio",
        "sem_global__connected_active_pair_ratio",
    ])

    assert len(names) == 39, len(names)
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
    node_records,
    active_nodes,
    retained_edges,
    _roles,
    upstream_hops,
):
    """
    Project the retained Top5 reverse caller slice onto a fixed
    role-level semantic organization representation.

    Important:
    - only internal methods count as organizers/callers;
    - raw method names and API identities are not features;
    - semantic identity comes only from the frozen Top5 endpoint roles;
    - a caller may organize multiple semantic roles.
    """
    roles = list(FROZEN_TOP5_ROLES)
    role_set = set(roles)
    active_set = set(active_nodes)

    # --------------------------------------------------------
    # Retained incoming adjacency:
    #
    # source -> target is a normal call-graph edge.
    # Starting at an endpoint, traverse incoming edges to find
    # internal callers.
    # --------------------------------------------------------
    incoming = {}

    for edge in retained_edges:
        source = edge["source"]
        target = edge["target"]

        if (
            source not in active_set
            or target not in active_set
        ):
            continue

        incoming.setdefault(target, []).append(source)

    # --------------------------------------------------------
    # Semantic endpoint sets.
    #
    # We deliberately require external=True here. An internal
    # method carrying an incidental role annotation must not be
    # reinterpreted as a sensitive API endpoint.
    # --------------------------------------------------------
    endpoints_by_role = {}

    for role in roles:
        endpoints_by_role[role] = {
            signature
            for signature in active_nodes
            if node_records[signature].get("external")
            and role in node_records[signature].get(
                "roles", []
            )
        }

    # --------------------------------------------------------
    # C_r:
    # set of INTERNAL callers that can reach role r endpoints
    # in <= H reverse hops.
    # --------------------------------------------------------
    callers_by_role = {}

    for role in roles:
        endpoints = endpoints_by_role[role]

        distance = {
            endpoint: 0
            for endpoint in endpoints
        }

        queue = list(endpoints)
        head = 0

        while head < len(queue):
            current = queue[head]
            head += 1

            current_distance = distance[current]

            if current_distance >= upstream_hops:
                continue

            for caller in incoming.get(current, []):
                caller_record = node_records[caller]

                # Ordinary external/framework APIs are not
                # semantic organizers.
                if caller_record.get("external"):
                    continue

                next_distance = current_distance + 1
                previous = distance.get(caller)

                if (
                    previous is None
                    or next_distance < previous
                ):
                    distance[caller] = next_distance
                    queue.append(caller)

        callers_by_role[role] = {
            signature
            for signature, d in distance.items()
            if d > 0
            and not node_records[signature].get(
                "external"
            )
        }

    # --------------------------------------------------------
    # Semantic degree:
    #
    # number of frozen roles whose endpoints are reachable
    # downstream from this internal caller.
    # --------------------------------------------------------
    all_callers = set()

    for callers in callers_by_role.values():
        all_callers.update(callers)

    semantic_degree = {}

    for caller in all_callers:
        semantic_degree[caller] = sum(
            caller in callers_by_role[role]
            for role in roles
        )

    values = {}

    # --------------------------------------------------------
    # A. Per-role semantic organization: 15 dims.
    # --------------------------------------------------------
    role_denominator = max(1, len(roles) - 1)

    for role in roles:
        callers = callers_by_role[role]
        n = len(callers)

        shared = {
            caller
            for caller in callers
            if semantic_degree[caller] >= 2
        }

        values[
            f"sem_role__{role}__shared_caller_log"
        ] = float(
            __import__("math").log1p(len(shared))
        )

        values[
            f"sem_role__{role}__shared_caller_ratio"
        ] = (
            len(shared) / n
            if n
            else 0.0
        )

        if n:
            values[
                (
                    f"sem_role__{role}"
                    "__mean_semantic_degree_norm"
                )
            ] = sum(
                (
                    semantic_degree[caller] - 1
                ) / role_denominator
                for caller in callers
            ) / n
        else:
            values[
                (
                    f"sem_role__{role}"
                    "__mean_semantic_degree_norm"
                )
            ] = 0.0

    # --------------------------------------------------------
    # B. Pairwise role organization: 20 dims.
    # --------------------------------------------------------
    connected_active_pairs = 0
    active_pair_total = 0

    for left, right in combinations(roles, 2):
        left_callers = callers_by_role[left]
        right_callers = callers_by_role[right]

        intersection = (
            left_callers & right_callers
        )

        union = (
            left_callers | right_callers
        )

        values[
            (
                f"sem_pair__{left}__{right}"
                "__shared_caller_log"
            )
        ] = float(
            __import__("math").log1p(
                len(intersection)
            )
        )

        values[
            (
                f"sem_pair__{left}__{right}"
                "__jaccard"
            )
        ] = (
            len(intersection) / len(union)
            if union
            else 0.0
        )

        # Conditional on BOTH semantic roles actually having
        # selected endpoints. This avoids treating a missing
        # role as a disconnected semantic relation.
        if (
            endpoints_by_role[left]
            and endpoints_by_role[right]
        ):
            active_pair_total += 1

            if intersection:
                connected_active_pairs += 1

    # --------------------------------------------------------
    # C. Global semantic organization: 4 dims.
    # --------------------------------------------------------
    organizer_count = len(all_callers)

    values[
        "sem_global__organizer_log"
    ] = float(
        __import__("math").log1p(
            organizer_count
        )
    )

    if organizer_count:
        values[
            "sem_global__multi_role_ge2_ratio"
        ] = (
            sum(
                degree >= 2
                for degree in semantic_degree.values()
            )
            / organizer_count
        )

        values[
            "sem_global__multi_role_ge3_ratio"
        ] = (
            sum(
                degree >= 3
                for degree in semantic_degree.values()
            )
            / organizer_count
        )
    else:
        values[
            "sem_global__multi_role_ge2_ratio"
        ] = 0.0

        values[
            "sem_global__multi_role_ge3_ratio"
        ] = 0.0

    values[
        "sem_global__connected_active_pair_ratio"
    ] = (
        connected_active_pairs / active_pair_total
        if active_pair_total
        else 0.0
    )

    expected = set(feature_names(roles))

    if set(values) != expected:
        missing = sorted(expected - set(values))
        extra = sorted(set(values) - expected)

        raise RuntimeError(
            "semantic feature schema mismatch: "
            f"missing={missing}, extra={extra}"
        )

    return values


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
            endpoint_roles=set(FROZEN_TOP5_ROLES),
        )

        values = structural_features(
            node_records,
            active_nodes,
            retained_edges,
            FROZEN_TOP5_ROLES,
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

    available_roles = set(roles)

    missing_top5 = [
        role
        for role in FROZEN_TOP5_ROLES
        if role not in available_roles
    ]

    if missing_top5:
        raise RuntimeError(
            "Frozen Top5 role(s) missing from mapping: "
            f"{missing_top5}"
        )

    # The graph mapping may still know all semantic roles, but
    # the correction representation itself is fixed to Top5.
    roles = list(FROZEN_TOP5_ROLES)

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
