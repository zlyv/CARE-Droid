#!/usr/bin/env python3
"""Build a strictly no-Entry sensitive-API upstream call-context cache.

Definition:
  * anchors: all mapped sensitive API endpoints;
  * traverse only backwards along caller -> callee edges;
  * default maximum upstream distance: 4 hops;
  * only internal callers are retained as intermediate context;
  * no downstream expansion;
  * only edges actually traversed on upstream search are retained;
  * Manifest Entry information is never used for node selection, edge selection,
    reachability, or node features;
  * output uses the standard A3 4-base-node / 3-edge-feature cache contract.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_endpoint_centered_tensor_cache_ag import (  # noqa: E402
    append_cache,
    append_failure,
    existing_successful_shas,
    role_tiers,
    sha256_file,
    suppress_loguru,
)
from run_semantic_mapping_audit_ag import load_yaml  # noqa: E402
from run_static_coverage_audit_ag import read_csv, suppress_logs  # noqa: E402
from tasa_droid.p2.graph_representation import (  # noqa: E402
    api_rule_matches,
    canonical_signature,
    class_and_method,
    edge_descriptor,
    is_external,
    method_signature,
)
from tasa_droid.p3.a3_mvp import build_a3_mvp_structure  # noqa: E402


def stable_id(value: str) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def edge_key(edge: dict) -> tuple:
    return (
        str(edge.get("source", "")),
        str(edge.get("target", "")),
        str(edge.get("edge_type", "DIRECT_CALL")),
        bool(edge.get("icc_any")),
        bool(edge.get("reflection_any")),
    )


def iter_edges(graph):
    try:
        return graph.edges(data=True)
    except TypeError:
        return ((left, right, {}) for left, right in graph.edges)


def canonicalize_call_graph(
    graph,
    rules: list[dict],
    role_tier: dict[str, int],
    edge_rules: list[dict],
):
    """Canonicalise the full call graph WITHOUT Manifest/Entry processing."""

    raw_nodes = list(graph.nodes)
    node_records: dict[str, dict] = {}
    object_to_signature = {}

    api_rules = [
        rule
        for rule in rules
        if rule.get("rule_kind", "api") == "api"
    ]

    for node in raw_nodes:
        signature = canonical_signature(method_signature(node))
        if not signature:
            continue

        object_to_signature[node] = signature
        class_name, method_name = class_and_method(signature)
        external = is_external(node)

        roles = set()
        if external:
            roles = {
                str(rule["role"])
                for rule in api_rules
                if api_rule_matches(rule, class_name, method_name)
            }

        current = node_records.get(signature)
        if current is None:
            node_records[signature] = {
                "class_name": class_name,
                "method_name": method_name,
                "external": bool(external),
                "roles": set(roles),
            }
        else:
            current["external"] = bool(current["external"] or external)
            current["roles"].update(roles)

    edge_map: dict[tuple, dict] = {}

    for left, right, data in iter_edges(graph):
        source = (
            object_to_signature.get(left)
            or canonical_signature(method_signature(left))
        )
        target = (
            object_to_signature.get(right)
            or canonical_signature(method_signature(right))
        )

        if (
            not source
            or not target
            or source not in node_records
            or target not in node_records
        ):
            continue

        edge_type, icc_any, reflection_any, _, _ = edge_descriptor(
            data,
            target,
            edge_rules,
        )

        key = (
            source,
            target,
            edge_type,
            bool(icc_any),
            bool(reflection_any),
        )
        edge_map[key] = {
            "source": source,
            "target": target,
            "edge_type": edge_type,
            "icc_any": bool(icc_any),
            "reflection_any": bool(reflection_any),
        }

    for record in node_records.values():
        record["roles"] = sorted(record["roles"])
        record["tier"] = min(
            (
                int(role_tier.get(role, 4))
                for role in record["roles"]
            ),
            default=4,
        )

    return node_records, list(edge_map.values())


def select_endpoints(
    candidates: set[str],
    node_records: dict[str, dict],
    role_tier: dict[str, int],
    cap: int,
) -> list[str]:
    """Deterministic selection while preserving semantic-role coverage."""

    def tier(signature: str) -> int:
        return min(
            (
                int(role_tier.get(role, 4))
                for role in node_records[signature].get("roles", [])
            ),
            default=4,
        )

    ordered = sorted(
        candidates,
        key=lambda signature: (
            tier(signature),
            stable_id(signature),
        ),
    )

    all_roles = sorted(
        {
            role
            for signature in candidates
            for role in node_records[signature].get("roles", [])
        },
        key=lambda role: (
            int(role_tier.get(role, 4)),
            str(role),
        ),
    )

    selected: list[str] = []
    selected_set: set[str] = set()

    # First preserve rare-role coverage.
    for role in all_roles:
        match = next(
            (
                signature
                for signature in ordered
                if signature not in selected_set
                and role in node_records[signature].get("roles", [])
            ),
            None,
        )
        if match is not None:
            selected.append(match)
            selected_set.add(match)
            if len(selected) >= cap:
                return selected

    for signature in ordered:
        if signature in selected_set:
            continue
        selected.append(signature)
        selected_set.add(signature)
        if len(selected) >= cap:
            break

    return selected


def build_sensitive_upstream_view(
    node_records: dict[str, dict],
    edge_records: list[dict],
    role_tier: dict[str, int],
    upstream_hops: int,
    k_in: int,
    endpoint_cap: int,
    node_cap: int,
    edge_cap: int,
    endpoint_roles: set[str] | None = None,
):
    """Select internal caller chains terminating at selected sensitive APIs.

    When endpoint_roles is None, preserve the original all-sensitive behavior.
    Otherwise, only nodes carrying at least one allowed endpoint role can seed
    the reverse upstream search. Caller nodes themselves remain unrestricted.
    """

    allowed_endpoint_roles = (
        None
        if endpoint_roles is None
        else set(endpoint_roles)
    )

    sensitive_all = {
        signature
        for signature, record in node_records.items()
        if record.get("roles")
        and (
            allowed_endpoint_roles is None
            or any(
                role in allowed_endpoint_roles
                for role in record.get("roles", [])
            )
        )
    }

    endpoint_limit = min(max(1, endpoint_cap), max(1, node_cap))
    selected_endpoints = select_endpoints(
        sensitive_all,
        node_records,
        role_tier,
        endpoint_limit,
    )
    endpoint_set = set(selected_endpoints)

    # Runtime contract for selective extraction:
    # every selected endpoint must have been admitted by at least
    # one explicitly allowed semantic role.
    if allowed_endpoint_roles is not None:
        invalid_endpoint_count = sum(
            1
            for signature in selected_endpoints
            if not any(
                role in allowed_endpoint_roles
                for role in node_records[signature].get("roles", [])
            )
        )

        if invalid_endpoint_count:
            raise RuntimeError(
                "selective endpoint-role contract violated: "
                f"{invalid_endpoint_count} endpoint(s) have no "
                "allowed semantic role"
            )

    canonical_edges: dict[tuple, dict] = {}
    incoming: dict[str, list[tuple[str, tuple]]] = defaultdict(list)

    for raw_edge in edge_records:
        key = edge_key(raw_edge)
        source, target = key[0], key[1]

        if (
            source not in node_records
            or target not in node_records
        ):
            continue

        edge = {
            "source": source,
            "target": target,
            "edge_type": key[2],
            "icc_any": key[3],
            "reflection_any": key[4],
        }
        canonical_edges[key] = edge
        incoming[target].append((source, key))

    def caller_sort_key(signature: str):
        record = node_records[signature]
        tier = min(
            (
                int(role_tier.get(role, 4))
                for role in record.get("roles", [])
            ),
            default=4,
        )
        return (
            0 if not record.get("external") else 1,
            0 if record.get("roles") else 1,
            tier,
            stable_id(signature),
        )

    for target in incoming:
        incoming[target].sort(
            key=lambda item: caller_sort_key(item[0])
        )

    # Distance is measured backwards from the sensitive endpoint.
    distance = {
        endpoint: 0
        for endpoint in selected_endpoints
    }
    traversed_edge_keys: set[tuple] = set()
    queue = deque(
        (endpoint, 0)
        for endpoint in selected_endpoints
    )

    while queue:
        current, current_distance = queue.popleft()

        if current_distance >= upstream_hops:
            continue

        eligible = []

        for caller, key in incoming.get(current, []):
            caller_record = node_records[caller]

            # Critical pruning rule:
            # ordinary external/framework APIs are NOT allowed as context.
            # Sensitive endpoints themselves may remain anchors.
            if (
                caller_record.get("external")
                and caller not in endpoint_set
            ):
                continue

            eligible.append((caller, key))

        for caller, key in eligible[:k_in]:
            traversed_edge_keys.add(key)

            next_distance = current_distance + 1
            previous = distance.get(caller)

            if previous is None or next_distance < previous:
                distance[caller] = next_distance

                # Do not continue traversing through external API nodes.
                if not node_records[caller].get("external"):
                    queue.append((caller, next_distance))

    candidate_nodes = set(distance)

    def node_sort_key(signature: str):
        record = node_records[signature]
        tier = min(
            (
                int(role_tier.get(role, 4))
                for role in record.get("roles", [])
            ),
            default=4,
        )
        return (
            0 if signature in endpoint_set else 1,
            distance.get(signature, 10**9),
            0 if not record.get("external") else 1,
            tier,
            stable_id(signature),
        )

    active = sorted(
        candidate_nodes,
        key=node_sort_key,
    )[:node_cap]
    active_set = set(active)

    node_truncated = len(candidate_nodes) > len(active_set)

    # IMPORTANT:
    # keep only edges traversed during the reverse search.
    # Do NOT reconstruct the induced subgraph between selected nodes.
    path_edges = [
        canonical_edges[key]
        for key in traversed_edge_keys
        if key in canonical_edges
        and canonical_edges[key]["source"] in active_set
        and canonical_edges[key]["target"] in active_set
    ]

    path_edges.sort(
        key=lambda edge: (
            distance.get(edge["target"], 10**9),
            distance.get(edge["source"], 10**9),
            stable_id("|".join(map(str, edge_key(edge)))),
        )
    )

    edge_truncated = len(path_edges) > edge_cap
    retained_edges = path_edges[:edge_cap]

    # After edge truncation, remove caller nodes no longer connected
    # to any retained path. Sensitive endpoints remain even if isolated.
    final_nodes = set(selected_endpoints) & active_set
    for edge in retained_edges:
        final_nodes.add(edge["source"])
        final_nodes.add(edge["target"])

    active = sorted(final_nodes, key=node_sort_key)
    active_set = set(active)

    selected_role_counts = Counter(
        role
        for signature in selected_endpoints
        for role in node_records[signature].get("roles", [])
        if (
            allowed_endpoint_roles is None
            or role in allowed_endpoint_roles
        )
    )

    internal_callers = sum(
        1
        for signature in active
        if signature not in endpoint_set
        and not node_records[signature].get("external")
    )

    summary = {
        "sensitive_endpoint_count_total": len(sensitive_all),
        "selected_endpoint_count": len(selected_endpoints),
        "endpoint_truncated": len(sensitive_all) > len(selected_endpoints),
        "upstream_hops": upstream_hops,
        "K_in": k_in,
        "candidate_upstream_node_count": len(candidate_nodes),
        "retained_node_count": len(active),
        "retained_internal_caller_count": internal_callers,
        "candidate_path_edge_count": len(path_edges),
        "retained_path_edge_count": len(retained_edges),
        "node_truncated": node_truncated,
        "edge_truncated": edge_truncated,
        "max_retained_upstream_distance": max(
            (distance.get(signature, 0) for signature in active),
            default=0,
        ),
        "selected_endpoint_role_counts": dict(
            sorted(selected_role_counts.items())
        ),
        "entry_information_used": False,
        "downstream_information_used": False,
        "induced_nonpath_edges_added": False,
        "ordinary_external_context_allowed": False,
    }

    return active_set, retained_edges, summary


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

    try:
        from androguard.misc import AnalyzeAPK

        apk, dalvik, analysis = AnalyzeAPK(row["absolute_path"])
        graph = analysis.get_call_graph(no_isolated=False)

        tiers = role_tiers(priority)

        node_records, edge_records = canonicalize_call_graph(
            graph,
            rules,
            tiers,
            edge_rules,
        )

        active_nodes, retained_edges, summary = (
            build_sensitive_upstream_view(
                node_records,
                edge_records,
                tiers,
                upstream_hops,
                k_in,
                endpoint_cap,
                node_cap,
                edge_cap,
            )
        )

        # Empty entry set is deliberate:
        # fourth A3 base feature is therefore always zero.
        structure = build_a3_mvp_structure(
            node_records,
            retained_edges,
            set(),
            active_nodes,
            roles,
        )

        result.update({
            "status": "OK",
            "node_count": structure["node_count"],
            "edge_count": structure["edge_count"],
            "base_node_feature_count": structure[
                "base_node_feature_count"
            ],
            "edge_feature_count": structure[
                "edge_feature_count"
            ],
            "role_vocabulary_size": structure[
                "role_vocabulary_size"
            ],
            "reliability_status": structure[
                "reliability_status"
            ],
            "tensor_sha256": structure["tensor_sha256"],
            "summary": summary,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--priority", required=True, type=Path)
    parser.add_argument("--edge-semantics", required=True, type=Path)
    parser.add_argument("--failure-csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)

    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--upstream-hops", type=int, default=4)
    parser.add_argument("--K-in", type=int, default=8)
    parser.add_argument("--endpoint-cap", type=int, default=128)
    parser.add_argument("--node-cap", type=int, default=2048)
    parser.add_argument("--edge-cap", type=int, default=4096)

    # Accepted only so an old endpoint-cache command can be reused.
    parser.add_argument("--downstream-hops", type=int, default=0)
    parser.add_argument("--K-out", type=int, default=1)
    parser.add_argument("--reachability-max-states", type=int, default=1)

    parser.add_argument("--min-free-gb", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=5)

    args = parser.parse_args()

    if args.upstream_hops < 1:
        raise SystemExit("upstream-hops must be >= 1")

    suppress_loguru()
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
            row.get("sha256", "").casefold()
            for row in read_csv(args.failure_csv)
            if row.get("sha256")
        }

    rows = [
        row
        for row in read_csv(args.lineage)
        if row.get("sha256", "").casefold() not in excluded
    ]

    rows.sort(
        key=lambda row: row.get("sha256", "").casefold()
    )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    cache_path = (
        output / "sensitive_upstream_tensor_cache.jsonl.gz"
    )
    failure_path = (
        output / "sensitive_upstream_tensor_failures.csv"
    )

    already_done = (
        existing_successful_shas(cache_path)
        if args.resume
        else set()
    )

    rows = [
        row
        for row in rows
        if row.get("sha256", "").casefold()
        not in already_done
    ]

    if args.limit > 0:
        rows = rows[:args.limit]

    if not rows:
        raise SystemExit(
            "no APK selected after exclusion/resume filtering"
        )

    statuses = Counter()

    print(
        f"sensitive-upstream cache started: {len(rows)} APKs "
        f"(resume_skipped={len(already_done)})",
        flush=True,
    )

    for index, row in enumerate(rows, 1):
        free_gb = (
            shutil.disk_usage(output).free / (1024 ** 3)
        )

        if free_gb < args.min_free_gb:
            raise SystemExit(
                f"disk guard stopped run: "
                f"free_gb={free_gb:.3f} "
                f"< min_free_gb={args.min_free_gb:.3f}"
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
            append_cache(cache_path, result)
        else:
            append_failure(failure_path, {
                **row,
                **result,
            })

        if (
            index % max(1, args.progress_every) == 0
            or index == len(rows)
        ):
            print(
                f"processed {index}/{len(rows)} "
                f"ok={statuses['OK']} "
                f"failed={statuses['FAILED']}",
                flush=True,
            )

        gc.collect()

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "phase": "SENSITIVE_UPSTREAM_NO_ENTRY_CACHE",
        "formal_training_enabled": False,
        "lineage_csv": str(
            args.lineage.expanduser().resolve()
        ),
        "mapping": str(
            args.mapping.expanduser().resolve()
        ),
        "priority": str(
            args.priority.expanduser().resolve()
        ),
        "edge_semantics": str(
            args.edge_semantics.expanduser().resolve()
        ),
        "processed_apk_count": sum(statuses.values()),
        "status_counts": dict(statuses),
        "policy": {
            "upstream_hops": args.upstream_hops,
            "K_in": args.K_in,
            "endpoint_cap": args.endpoint_cap,
            "node_cap": args.node_cap,
            "edge_cap": args.edge_cap,
            "downstream_hops": 0,
            "entry_information_used": False,
            "ordinary_external_context_allowed": False,
            "induced_nonpath_edges_added": False,
        },
        "cache_contract": {
            "base_node_feature_count": 4,
            "edge_feature_count": 3,
            "fourth_base_feature_entry_flag": 0,
        },
        "cache_sha256": (
            sha256_file(cache_path)
            if cache_path.exists()
            else None
        ),
        "outputs": {
            "cache": str(cache_path),
            "failures": str(failure_path),
        },
    }

    (
        output / "sensitive_upstream_tensor_cache_audit.json"
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
