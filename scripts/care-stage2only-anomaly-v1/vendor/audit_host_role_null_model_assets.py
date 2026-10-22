#!/usr/bin/env python3
"""Audit whether the existing endpoint cache can support PairZ/Intra/Inter.

This is a read-only asset audit.  It does not train a classifier and does not
use labels.  For each cached APK it independently reconstructs the behavior
host--role incidence graph, checks PairRaw against the existing E1
implementation, probes a bipartite degree-preserving null model, and measures
whether retained host-to-host edges exist for a future Inter-Host view.

Copy this file into ``scripts/`` in the tasa-droid repository and run it from
the repository root.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_benefitdroid_e1_exact_pair_2017 as e1


ROLE_COUNT = 31
PAIR_COUNT = ROLE_COUNT * (ROLE_COUNT - 1) // 2


def norm_sha(value):
    return str(value or "").strip().casefold()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_cache(path, year):
    selected = []
    total_ok = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "OK" or not row.get("payload"):
                continue
            total_ok += 1
            row_year = str(row.get("metadata_year", "")).strip()
            if year != "all" and row_year != year:
                continue
            row["_line_number"] = line_number
            selected.append(row)
    selected.sort(key=lambda row: norm_sha(row.get("sha256")))
    return selected, total_ok


def payload_arrays(payload):
    node_ids = list(payload.get("node_ids") or [])
    role_rows = list(payload.get("node_role_ids") or [])
    node_base = np.asarray(payload.get("node_base_features") or [], dtype=float)
    edge_index = payload.get("edge_index") or [[], []]
    if len(edge_index) != 2:
        raise ValueError("edge_index must have two rows")
    source = np.asarray(edge_index[0], dtype=np.int64)
    target = np.asarray(edge_index[1], dtype=np.int64)
    if node_base.size == 0:
        node_base = np.zeros((0, 15), dtype=float)
    if node_base.ndim != 2 or node_base.shape[1] != 15:
        raise ValueError("node_base_features must have shape (n,15)")
    if len(node_ids) != len(role_rows) or len(node_ids) != len(node_base):
        raise ValueError("node payload row counts disagree")
    if len(source) != len(target):
        raise ValueError("edge_index source/target lengths disagree")
    if len(source):
        if int(source.min()) < 0 or int(target.min()) < 0:
            raise ValueError("negative node index")
        if int(max(source.max(), target.max())) >= len(node_ids):
            raise ValueError("edge_index points outside node array")
    roles = []
    for values in role_rows:
        clean = frozenset(
            int(value) for value in (values or [])
            if 0 <= int(value) < ROLE_COUNT
        )
        roles.append(clean)
    return node_ids, node_base, roles, source, target


def reconstruct_host_roles(payload):
    node_ids, node_base, node_roles, source, target = payload_arrays(payload)
    endpoint_nodes = set(map(int, np.flatnonzero(node_base[:, 4] > 0.5)))
    caller_targets = {}
    for src_value, dst_value in zip(source, target):
        src = int(src_value)
        dst = int(dst_value)
        if dst in endpoint_nodes:
            caller_targets.setdefault(src, set()).add(dst)
    host_roles = {}
    for host, endpoints in caller_targets.items():
        merged = frozenset().union(*(node_roles[index] for index in endpoints))
        if merged:
            host_roles[int(host)] = merged
    host_set = set(host_roles)
    host_host_edges = {
        (int(src), int(dst))
        for src, dst in zip(source, target)
        if int(src) in host_set and int(dst) in host_set and int(src) != int(dst)
    }
    host_host_role_transitions = set()
    for src, dst in host_host_edges:
        for role_a in host_roles[src]:
            for role_b in host_roles[dst]:
                host_host_role_transitions.add((int(role_a), int(role_b)))
    return {
        "node_ids": node_ids,
        "node_base": node_base,
        "node_roles": node_roles,
        "source": source,
        "target": target,
        "endpoint_nodes": endpoint_nodes,
        "host_roles": host_roles,
        "host_host_edges": host_host_edges,
        "host_host_role_transitions": host_host_role_transitions,
    }


def pair_schema():
    pairs = list(combinations(range(ROLE_COUNT), 2))
    return pairs, {pair: index for index, pair in enumerate(pairs)}


def pair_counts(host_roles, pair_index):
    result = np.zeros(PAIR_COUNT, dtype=np.float64)
    for roles in host_roles.values():
        for pair in combinations(sorted(roles), 2):
            result[pair_index[pair]] += 1.0
    return result


def incidence_edges(host_roles):
    return {
        (int(host), int(role))
        for host, roles in host_roles.items()
        for role in roles
    }


def swapped_incidence(observed, rng, target_successes, max_attempt_factor):
    edges = set(observed)
    if len(edges) < 2 or target_successes <= 0:
        return edges, 0, 0
    edge_list = list(edges)
    successes = 0
    attempts = 0
    max_attempts = max(20, int(target_successes) * int(max_attempt_factor))
    while successes < target_successes and attempts < max_attempts:
        attempts += 1
        i = int(rng.randint(len(edge_list)))
        j = int(rng.randint(len(edge_list) - 1))
        if j >= i:
            j += 1
        h1, r1 = edge_list[i]
        h2, r2 = edge_list[j]
        if h1 == h2 or r1 == r2:
            continue
        new_a = (h1, r2)
        new_b = (h2, r1)
        if new_a in edges or new_b in edges:
            continue
        old_a = edge_list[i]
        old_b = edge_list[j]
        edges.remove(old_a)
        edges.remove(old_b)
        edges.add(new_a)
        edges.add(new_b)
        edge_list[i] = new_a
        edge_list[j] = new_b
        successes += 1
    return edges, successes, attempts


def degrees(edges):
    host = {}
    role = {}
    for h, r in edges:
        host[h] = host.get(h, 0) + 1
        role[r] = role.get(r, 0) + 1
    return host, role


def edge_set_to_host_roles(edges):
    result = {}
    for host, role in edges:
        result.setdefault(int(host), set()).add(int(role))
    return result


def null_pair_z(host_roles, pair_index, null_samples, swaps_per_edge, seed):
    observed_edges = incidence_edges(host_roles)
    observed_pair = pair_counts(host_roles, pair_index)
    original_degrees = degrees(observed_edges)
    rng = np.random.RandomState(int(seed))
    samples = []
    total_successes = 0
    total_attempts = 0
    degree_preserved = True
    target_successes = max(1, int(swaps_per_edge) * len(observed_edges))
    for _ in range(int(null_samples)):
        randomized, successes, attempts = swapped_incidence(
            observed_edges, rng, target_successes, max_attempt_factor=40
        )
        total_successes += successes
        total_attempts += attempts
        degree_preserved = degree_preserved and degrees(randomized) == original_degrees
        samples.append(pair_counts(edge_set_to_host_roles(randomized), pair_index))
    if samples:
        null = np.asarray(samples, dtype=np.float64)
        mean = null.mean(axis=0)
        std = null.std(axis=0, ddof=0)
    else:
        mean = np.zeros(PAIR_COUNT, dtype=np.float64)
        std = np.zeros(PAIR_COUNT, dtype=np.float64)
    z = np.divide(
        observed_pair - mean,
        std,
        out=np.zeros_like(observed_pair),
        where=std > 0.0,
    )
    return {
        "observed": observed_pair,
        "mean": mean,
        "std": std,
        "z": z,
        "successful_swaps": int(total_successes),
        "attempted_swaps": int(total_attempts),
        "degree_preserved": bool(degree_preserved),
    }


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path,
        default=Path(
            "artifacts/p25-endpoint-cache-full-cap256/"
            "endpoint_centered_tensor_cache.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/host-role-null-model-asset-audit-v1"),
    )
    parser.add_argument(
        "--year", default="2017",
        help="metadata_year to audit; use 'all' for every successful cache row",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--null-samples", type=int, default=16)
    parser.add_argument("--swaps-per-edge", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    if args.limit < 0:
        raise SystemExit("--limit must be nonnegative")
    if args.null_samples < 1:
        raise SystemExit("--null-samples must be positive")
    if args.swaps_per_edge < 1:
        raise SystemExit("--swaps-per-edge must be positive")
    if not args.cache.exists():
        raise SystemExit("cache not found: " + str(args.cache))

    args.output.mkdir(parents=True, exist_ok=True)
    rows, total_ok = read_cache(args.cache, str(args.year))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no successful cache rows matched --year/--limit")

    pairs, pair_index = pair_schema()
    audit_rows = []
    shas = []
    pair_raw_rows = []
    pair_z_rows = []
    null_mean_rows = []
    null_std_rows = []
    exact_failures = []
    errors = []

    for index, row in enumerate(rows, 1):
        sha = norm_sha(row.get("sha256"))
        try:
            payload = row["payload"]
            asset = reconstruct_host_roles(payload)
            host_roles = asset["host_roles"]
            independent_pair = pair_counts(host_roles, pair_index)
            stats, existing_pair, _, existing_audit = e1.reconstruct_host_structure(
                payload, pairs, pair_index
            )
            existing_pair = np.asarray(existing_pair, dtype=np.float64)
            exact = bool(np.array_equal(independent_pair, existing_pair))
            if not exact:
                exact_failures.append(sha)
            null = null_pair_z(
                host_roles,
                pair_index,
                args.null_samples,
                args.swaps_per_edge,
                args.seed + index,
            )
            incidence = incidence_edges(host_roles)
            host_degrees = [len(value) for value in host_roles.values()]
            summary = row.get("summary") or {}
            attempts = null["attempted_swaps"]
            successes = null["successful_swaps"]
            audit_rows.append({
                "sha256": sha,
                "metadata_year": row.get("metadata_year", ""),
                "node_count": len(asset["node_ids"]),
                "edge_count": len(asset["source"]),
                "selected_endpoint_count": len(asset["endpoint_nodes"]),
                "behavior_host_count": len(host_roles),
                "host_role_incidence_count": len(incidence),
                "active_role_count": len({role for _, role in incidence}),
                "multi_role_host_count": sum(value >= 2 for value in host_degrees),
                "max_host_role_degree": max(host_degrees, default=0),
                "pair_raw_nonzero_dims": int(np.count_nonzero(independent_pair)),
                "pair_raw_total": float(independent_pair.sum()),
                "pairraw_exact_match_existing_e1": int(exact),
                "stats_dimension": len(stats),
                "host_host_directed_edge_count": len(asset["host_host_edges"]),
                "host_host_role_transition_count": len(
                    asset["host_host_role_transitions"]
                ),
                "endpoint_truncated": int(bool(summary.get("endpoint_truncated"))),
                "node_truncated": int(bool(summary.get("node_truncated"))),
                "edge_truncated": int(bool(summary.get("edge_truncated"))),
                "null_successful_swaps": successes,
                "null_attempted_swaps": attempts,
                "null_swap_success_rate": safe_ratio(successes, attempts),
                "null_any_swap": int(successes > 0),
                "null_degree_preserved": int(null["degree_preserved"]),
                "pairz_variable_dims": int(np.count_nonzero(null["std"] > 0.0)),
                "pairz_nonzero_dims": int(np.count_nonzero(null["z"])),
                "pairz_max_abs": float(np.max(np.abs(null["z"]))),
            })
            shas.append(sha)
            pair_raw_rows.append(independent_pair)
            pair_z_rows.append(null["z"])
            null_mean_rows.append(null["mean"])
            null_std_rows.append(null["std"])
        except Exception as exc:
            errors.append({
                "sha256": sha,
                "line_number": row.get("_line_number", ""),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            })
        if index % args.progress_every == 0 or index == len(rows):
            print("audited:", index, "/", len(rows), "errors:", len(errors), flush=True)

    write_csv(args.output / "apk_asset_audit.csv", audit_rows)
    write_csv(args.output / "audit_errors.csv", errors)
    np.savez_compressed(
        args.output / "pairz_preview.npz",
        sha256=np.asarray(shas),
        pair_raw=np.asarray(pair_raw_rows, dtype=np.float64),
        pair_z=np.asarray(pair_z_rows, dtype=np.float64),
        null_mean=np.asarray(null_mean_rows, dtype=np.float64),
        null_std=np.asarray(null_std_rows, dtype=np.float64),
    )

    n = len(audit_rows)
    pair_exact_n = sum(row["pairraw_exact_match_existing_e1"] for row in audit_rows)
    degree_ok_n = sum(row["null_degree_preserved"] for row in audit_rows)
    swappable_n = sum(row["null_any_swap"] for row in audit_rows)
    inter_n = sum(row["host_host_directed_edge_count"] > 0 for row in audit_rows)
    endpoint_truncated_n = sum(row["endpoint_truncated"] for row in audit_rows)
    node_truncated_n = sum(row["node_truncated"] for row in audit_rows)
    edge_truncated_n = sum(row["edge_truncated"] for row in audit_rows)
    variable_pairz_n = sum(row["pairz_variable_dims"] > 0 for row in audit_rows)
    summary = {
        "protocol": {
            "purpose": "read-only Host-Role/PairZ/Inter cache asset audit",
            "labels_used": False,
            "classifier_trained": False,
            "cache": str(args.cache),
            "cache_sha256": sha256_file(args.cache),
            "year_filter": str(args.year),
            "limit": int(args.limit),
            "null_samples_per_apk": int(args.null_samples),
            "successful_swaps_target_per_sample": (
                str(args.swaps_per_edge) + " x incidence_edge_count"
            ),
            "random_seed": int(args.seed),
            "null_model": (
                "binary Host-Role bipartite double-edge swap; preserves exact "
                "host degrees and role degrees"
            ),
        },
        "counts": {
            "successful_cache_rows_total": int(total_ok),
            "selected_rows": int(len(rows)),
            "audited_rows": int(n),
            "error_rows": int(len(errors)),
            "pairraw_exact_match_rows": int(pair_exact_n),
            "degree_preserved_rows": int(degree_ok_n),
            "rows_with_at_least_one_valid_swap": int(swappable_n),
            "rows_with_variable_pairz_dimension": int(variable_pairz_n),
            "rows_with_retained_host_host_edge": int(inter_n),
            "endpoint_truncated_rows": int(endpoint_truncated_n),
            "node_truncated_rows": int(node_truncated_n),
            "edge_truncated_rows": int(edge_truncated_n),
        },
        "rates": {
            "pairraw_exact_match": safe_ratio(pair_exact_n, n),
            "degree_preserved": safe_ratio(degree_ok_n, n),
            "swappable": safe_ratio(swappable_n, n),
            "variable_pairz": safe_ratio(variable_pairz_n, n),
            "inter_host_edge_available": safe_ratio(inter_n, n),
            "endpoint_truncated": safe_ratio(endpoint_truncated_n, n),
            "node_truncated": safe_ratio(node_truncated_n, n),
            "edge_truncated": safe_ratio(edge_truncated_n, n),
        },
        "verdict": {
            "pairraw_reconstruction_pass": bool(n > 0 and pair_exact_n == n),
            "intra_pairz_schema_available": bool(n > 0 and pair_exact_n == n),
            "degree_preserving_sampler_pass": bool(n > 0 and degree_ok_n == n),
            "inter_host_schema_available": bool(inter_n > 0),
            "full_experiment_ready": bool(
                n > 0
                and not errors
                and pair_exact_n == n
                and degree_ok_n == n
                and variable_pairz_n > 0
            ),
        },
        "limitations": [
            "The cache is endpoint-centred and bounded, so it is not the full APK call graph.",
            "Host-Host edges are only those retained inside the bounded induced graph.",
            "Degenerate APKs can have no valid bipartite swap; their PairZ is deterministically zero.",
            "This audit establishes data availability and reconstruction fidelity, not predictive benefit.",
        ],
        "pair_schema": {
            "role_count": ROLE_COUNT,
            "pair_count": PAIR_COUNT,
            "ordering": "itertools.combinations(range(31), 2)",
        },
        "exact_match_failure_examples": exact_failures[:20],
        "error_examples": errors[:20],
    }
    with (args.output / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (args.output / "pair_schema.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {"index": index, "role_a": a, "role_b": b}
                for index, (a, b) in enumerate(pairs)
            ],
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    print("Done:", args.output)
    print(json.dumps(summary["verdict"], ensure_ascii=False, indent=2))
    if errors or exact_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
