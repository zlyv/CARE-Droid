#!/usr/bin/env python3
"""Label-free local-region asset audit for the CARE-Droid second layer.

This script does not train a classifier and never reads malware labels.  It
checks whether the existing bounded endpoint cache can support three proposed
region granularities:

1. observed caller-method instances (called ``host`` by the existing cache);
2. unique undirected 1-hop host ego regions;
3. weakly connected host components as a conservative topology probe.

The component probe is not presented as a community detector.  Its purpose is
to determine whether enough retained host-to-host topology exists to justify a
later community algorithm.  Every candidate region is also passed through a
fixed 72-dimensional preview extractor so finite local feature construction is
audited rather than assumed.

Copy this file into ``scripts/`` in the tasa-droid repository and run it from
the repository root.  It reuses ``audit_host_role_null_model_assets.py`` from
the previous HostRole asset experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_host_role_null_model_assets as host_asset


ROLE_COUNT = int(host_asset.ROLE_COUNT)
TOPOLOGY_NAMES = [
    "log1p_region_host_count",
    "log1p_role_incidence_count",
    "active_role_ratio",
    "mean_roles_per_host",
    "multi_role_host_ratio",
    "log1p_internal_directed_edge_count",
    "log1p_boundary_directed_edge_count",
    "directed_internal_density",
    "directed_reciprocity_ratio",
    "normalized_role_entropy",
]
REGION_FEATURE_NAMES = (
    [f"role_fraction__{index:02d}" for index in range(ROLE_COUNT)]
    + [f"role_log_count__{index:02d}" for index in range(ROLE_COUNT)]
    + TOPOLOGY_NAMES
)
REGION_FEATURE_DIM = len(REGION_FEATURE_NAMES)


def parse_years(text):
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if token and token not in values:
            values.append(token)
    if not values:
        raise ValueError("--years is empty")
    return values


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def undirected_adjacency(hosts, directed_edges):
    adjacency = {int(host): set() for host in hosts}
    for source, target in directed_edges:
        source = int(source)
        target = int(target)
        if source == target or source not in adjacency or target not in adjacency:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


def weak_components(adjacency):
    remaining = set(adjacency)
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = deque([start])
        component = {start}
        while queue:
            node = queue.popleft()
            for neighbor in sorted(adjacency[node]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(frozenset(component))
    components.sort(key=lambda value: (-len(value), tuple(sorted(value))))
    return components


def unique_ego_regions(adjacency):
    regions = {
        frozenset({host, *neighbors})
        for host, neighbors in adjacency.items()
    }
    return sorted(regions, key=lambda value: (-len(value), tuple(sorted(value))))


def normalized_entropy(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probabilities = counts[counts > 0.0] / total
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return float(entropy / math.log(ROLE_COUNT)) if ROLE_COUNT > 1 else 0.0


def region_vector(nodes, host_roles, directed_edges):
    nodes = frozenset(int(value) for value in nodes)
    role_count = np.zeros(ROLE_COUNT, dtype=np.float64)
    host_role_degree = []
    for host in nodes:
        roles = host_roles.get(host, ())
        host_role_degree.append(len(roles))
        for role in roles:
            role_count[int(role)] += 1.0
    internal_edges = {
        (int(source), int(target))
        for source, target in directed_edges
        if int(source) in nodes and int(target) in nodes and int(source) != int(target)
    }
    boundary_edges = {
        (int(source), int(target))
        for source, target in directed_edges
        if (int(source) in nodes) != (int(target) in nodes)
    }
    reciprocal = sum((target, source) in internal_edges for source, target in internal_edges)
    host_n = len(nodes)
    role_incidence_n = float(np.sum(role_count))
    possible_edges = host_n * max(host_n - 1, 0)
    degrees = np.asarray(host_role_degree, dtype=np.float64)
    topology = np.asarray([
        np.log1p(host_n),
        np.log1p(role_incidence_n),
        float(np.count_nonzero(role_count) / ROLE_COUNT),
        float(np.mean(degrees)) if len(degrees) else 0.0,
        float(np.mean(degrees >= 2.0)) if len(degrees) else 0.0,
        np.log1p(len(internal_edges)),
        np.log1p(len(boundary_edges)),
        safe_ratio(len(internal_edges), possible_edges),
        safe_ratio(reciprocal, len(internal_edges)),
        normalized_entropy(role_count),
    ], dtype=np.float64)
    result = np.concatenate([
        role_count / float(max(host_n, 1)),
        np.log1p(role_count),
        topology,
    ])
    if result.shape != (REGION_FEATURE_DIM,):
        raise RuntimeError("bad region feature shape: " + repr(result.shape))
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite region preview feature")
    return result


def quantile_fields(prefix, values):
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) == 0:
        return {
            prefix + "_min": 0.0,
            prefix + "_p25": 0.0,
            prefix + "_median": 0.0,
            prefix + "_p75": 0.0,
            prefix + "_p90": 0.0,
            prefix + "_max": 0.0,
            prefix + "_mean": 0.0,
        }
    return {
        prefix + "_min": float(np.min(values)),
        prefix + "_p25": float(np.quantile(values, 0.25)),
        prefix + "_median": float(np.quantile(values, 0.50)),
        prefix + "_p75": float(np.quantile(values, 0.75)),
        prefix + "_p90": float(np.quantile(values, 0.90)),
        prefix + "_max": float(np.max(values)),
        prefix + "_mean": float(np.mean(values)),
    }


def audit_one(row):
    asset = host_asset.reconstruct_host_roles(row["payload"])
    host_roles = asset["host_roles"]
    directed_edges = asset["host_host_edges"]
    hosts = sorted(host_roles)
    adjacency = undirected_adjacency(hosts, directed_edges)
    components = weak_components(adjacency)
    egos = unique_ego_regions(adjacency)
    host_regions = [frozenset([host]) for host in hosts]

    # Run the proposed local feature extractor on every available region scale.
    preview_vectors = []
    for regions in (host_regions, egos, components):
        preview_vectors.extend(
            region_vector(region, host_roles, directed_edges)
            for region in regions
        )
    preview_matrix = (
        np.asarray(preview_vectors, dtype=np.float64)
        if preview_vectors
        else np.zeros((0, REGION_FEATURE_DIM), dtype=np.float64)
    )
    if preview_matrix.shape[1] != REGION_FEATURE_DIM:
        raise RuntimeError("invalid preview matrix")

    host_role_degree = [len(host_roles[host]) for host in hosts]
    undirected_degree = [len(adjacency[host]) for host in hosts]
    ego_sizes = [len(region) for region in egos]
    component_sizes = [len(region) for region in components]
    incident_hosts = {
        int(value)
        for edge in directed_edges
        for value in edge
    }
    summary = row.get("summary") or {}
    output = {
        "sha256": host_asset.norm_sha(row.get("sha256")),
        "metadata_year": str(row.get("metadata_year", "")).strip(),
        "cache_line_number": int(row.get("_line_number", 0)),
        "node_count": int(len(asset["node_ids"])),
        "retained_graph_edge_count": int(len(asset["source"])),
        "selected_endpoint_count": int(len(asset["endpoint_nodes"])),
        "host_method_count": int(len(hosts)),
        "active_role_count": int(len({role for roles in host_roles.values() for role in roles})),
        "host_role_incidence_count": int(sum(host_role_degree)),
        "multi_role_host_count": int(sum(value >= 2 for value in host_role_degree)),
        "host_host_directed_edge_count": int(len(directed_edges)),
        "edge_incident_host_count": int(len(incident_hosts)),
        "edge_incident_host_ratio": safe_ratio(len(incident_hosts), len(hosts)),
        "isolated_host_count": int(sum(value == 0 for value in undirected_degree)),
        "isolated_host_ratio": safe_ratio(sum(value == 0 for value in undirected_degree), len(hosts)),
        "unique_ego_region_count": int(len(egos)),
        "nontrivial_ego_region_count": int(sum(value >= 2 for value in ego_sizes)),
        "has_nontrivial_ego": int(any(value >= 2 for value in ego_sizes)),
        "weak_component_count": int(len(components)),
        "nontrivial_component_count": int(sum(value >= 2 for value in component_sizes)),
        "component_size_ge3_count": int(sum(value >= 3 for value in component_sizes)),
        "has_component_size_ge3": int(any(value >= 3 for value in component_sizes)),
        "largest_component_size": int(max(component_sizes, default=0)),
        "largest_component_host_ratio": safe_ratio(max(component_sizes, default=0), len(hosts)),
        "region_preview_count": int(len(preview_matrix)),
        "region_preview_dimension": int(REGION_FEATURE_DIM),
        "region_preview_all_finite": int(np.isfinite(preview_matrix).all()),
        "endpoint_truncated": int(bool(summary.get("endpoint_truncated"))),
        "node_truncated": int(bool(summary.get("node_truncated"))),
        "edge_truncated": int(bool(summary.get("edge_truncated"))),
    }
    output.update(quantile_fields("host_role_degree", host_role_degree))
    output.update(quantile_fields("undirected_host_degree", undirected_degree))
    output.update(quantile_fields("ego_region_size", ego_sizes))
    output.update(quantile_fields("component_size", component_sizes))
    return output


def aggregate_rows(rows, cohort):
    n = len(rows)
    result = {
        "cohort": cohort,
        "audited_apk_n": int(n),
        "nonempty_host_apk_n": int(sum(row["host_method_count"] > 0 for row in rows)),
        "nonempty_host_apk_ratio": safe_ratio(sum(row["host_method_count"] > 0 for row in rows), n),
        "host_edge_apk_n": int(sum(row["host_host_directed_edge_count"] > 0 for row in rows)),
        "host_edge_apk_ratio": safe_ratio(sum(row["host_host_directed_edge_count"] > 0 for row in rows), n),
        "nontrivial_ego_apk_n": int(sum(row["has_nontrivial_ego"] for row in rows)),
        "nontrivial_ego_apk_ratio": safe_ratio(sum(row["has_nontrivial_ego"] for row in rows), n),
        "component_ge3_apk_n": int(sum(row["has_component_size_ge3"] for row in rows)),
        "component_ge3_apk_ratio": safe_ratio(sum(row["has_component_size_ge3"] for row in rows), n),
        "finite_preview_apk_n": int(sum(row["region_preview_all_finite"] for row in rows)),
        "finite_preview_apk_ratio": safe_ratio(sum(row["region_preview_all_finite"] for row in rows), n),
        "endpoint_truncated_apk_ratio": safe_ratio(sum(row["endpoint_truncated"] for row in rows), n),
        "node_truncated_apk_ratio": safe_ratio(sum(row["node_truncated"] for row in rows), n),
        "edge_truncated_apk_ratio": safe_ratio(sum(row["edge_truncated"] for row in rows), n),
    }
    for field in (
        "host_method_count",
        "active_role_count",
        "host_host_directed_edge_count",
        "edge_incident_host_ratio",
        "isolated_host_ratio",
        "unique_ego_region_count",
        "largest_component_size",
        "largest_component_host_ratio",
        "region_preview_count",
    ):
        result.update(quantile_fields(field, [row[field] for row in rows]))
    return result


def append_summary(lines, row):
    lines.extend([
        "",
        "Cohort: " + str(row["cohort"]),
        f"  audited APKs: {row['audited_apk_n']}",
        f"  nonempty caller-method bags: {row['nonempty_host_apk_ratio']:.2%}",
        f"  APKs with retained host-host edges: {row['host_edge_apk_ratio']:.2%}",
        f"  APKs with nontrivial 1-hop egos: {row['nontrivial_ego_apk_ratio']:.2%}",
        f"  APKs with a connected component of size >=3: {row['component_ge3_apk_ratio']:.2%}",
        f"  all candidate region previews finite: {row['finite_preview_apk_ratio']:.2%}",
        f"  host methods median / p90: {row['host_method_count_median']:.1f} / {row['host_method_count_p90']:.1f}",
        f"  unique ego regions median / p90: {row['unique_ego_region_count_median']:.1f} / {row['unique_ego_region_count_p90']:.1f}",
        f"  largest component size median / p90: {row['largest_component_size_median']:.1f} / {row['largest_component_size_p90']:.1f}",
        f"  isolated host ratio median / p90: {row['isolated_host_ratio_median']:.2%} / {row['isolated_host_ratio_p90']:.2%}",
        f"  endpoint/node/edge truncation ratios: {row['endpoint_truncated_apk_ratio']:.2%} / {row['node_truncated_apk_ratio']:.2%} / {row['edge_truncated_apk_ratio']:.2%}",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "artifacts/p25-endpoint-cache-full-cap256/"
            "endpoint_centered_tensor_cache.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--years", default="2017,2018,2019,2020,2021",
        help="Comma-separated metadata years; no labels are read.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means all matched rows")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/care-local-region-asset-audit-v1"),
    )
    args = parser.parse_args()

    if args.limit < 0:
        raise SystemExit("--limit must be nonnegative")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be positive")
    if not args.cache.exists():
        raise SystemExit("cache not found: " + str(args.cache))
    years = parse_years(args.years)
    all_rows, total_ok = host_asset.read_cache(args.cache, "all")
    rows = [
        row for row in all_rows
        if str(row.get("metadata_year", "")).strip() in set(years)
    ]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no cache rows matched requested years")
    args.output.mkdir(parents=True, exist_ok=True)

    audits = []
    errors = []
    for index, row in enumerate(rows, 1):
        try:
            audits.append(audit_one(row))
        except Exception as exc:
            errors.append({
                "sha256": host_asset.norm_sha(row.get("sha256")),
                "metadata_year": str(row.get("metadata_year", "")).strip(),
                "cache_line_number": int(row.get("_line_number", 0)),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
            })
        if index % args.progress_every == 0 or index == len(rows):
            print("audited:", index, "/", len(rows), "errors:", len(errors), flush=True)

    host_asset.write_csv(args.output / "apk_region_asset_audit.csv", audits)
    host_asset.write_csv(args.output / "audit_errors.csv", errors)
    by_year = []
    for year in years:
        selected = [row for row in audits if row["metadata_year"] == year]
        if selected:
            by_year.append(aggregate_rows(selected, year))
    overall = aggregate_rows(audits, "requested_years_pooled")
    summary_rows = [overall] + by_year
    host_asset.write_csv(args.output / "cohort_summary.csv", summary_rows)

    success_ratio = safe_ratio(len(audits), len(rows))
    criteria = {
        "reconstruction_success_ratio_at_least_0_99": success_ratio >= 0.99,
        "nonempty_host_bag_ratio_at_least_0_95": overall["nonempty_host_apk_ratio"] >= 0.95,
        "nontrivial_ego_ratio_at_least_0_70": overall["nontrivial_ego_apk_ratio"] >= 0.70,
        "component_size_ge3_ratio_at_least_0_50": overall["component_ge3_apk_ratio"] >= 0.50,
        "all_region_previews_finite": overall["finite_preview_apk_ratio"] == 1.0,
    }
    decisions = {
        "host_instance_branch_viable": bool(
            criteria["reconstruction_success_ratio_at_least_0_99"]
            and criteria["nonempty_host_bag_ratio_at_least_0_95"]
            and criteria["all_region_previews_finite"]
        ),
        "ego_branch_viable": bool(
            criteria["nontrivial_ego_ratio_at_least_0_70"]
            and criteria["all_region_previews_finite"]
        ),
        "topology_supports_later_community_probe": bool(
            criteria["component_size_ge3_ratio_at_least_0_50"]
        ),
    }
    metadata = {
        "experiment": "care-local-region-asset-audit-v1",
        "purpose": "label-free feasibility audit; no classifier is trained",
        "labels_read": False,
        "requested_years": years,
        "matched_cache_rows": int(len(rows)),
        "successful_audits": int(len(audits)),
        "errors": int(len(errors)),
        "success_ratio": float(success_ratio),
        "total_ok_cache_rows_all_years": int(total_ok),
        "host_semantics": "observed caller method with a retained edge to a selected sensitive endpoint",
        "ego_semantics": "host plus undirected one-hop neighbors in retained host-host edges",
        "component_semantics": "weak connected component; feasibility probe, not a claimed community",
        "region_preview_dimension": int(REGION_FEATURE_DIM),
        "region_feature_names": REGION_FEATURE_NAMES,
        "criteria": criteria,
        "decisions": decisions,
        "cache_limitations": (
            "P25 retains bounded callers/endpoints; topology and region counts are lower-bound observations"
        ),
    }
    (args.output / "audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "region_schema.json").write_text(
        json.dumps({
            "dimension": REGION_FEATURE_DIM,
            "feature_names": REGION_FEATURE_NAMES,
            "construction_uses_labels": False,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "CARE-Droid local-region asset audit v1",
        "",
        "This audit is label-free and trains no classifier.",
        "A host is an observed caller method, according to the existing cache construction.",
        "Connected components are only a topology probe, not claimed communities.",
        f"Requested rows / successful / errors: {len(rows)} / {len(audits)} / {len(errors)}",
        f"Reconstruction success ratio: {success_ratio:.2%}",
    ]
    for row in summary_rows:
        append_summary(lines, row)
    lines.extend([
        "",
        "Predeclared feasibility decisions",
        "---------------------------------",
        *[f"{key}: {value}" for key, value in decisions.items()],
        "",
        "Interpretation",
        "--------------",
        "If host_instance_branch_viable is true, host-level MIL can be implemented.",
        "If ego_branch_viable is true, the first CARE experiment should compare host-only with host+ego dual pooling.",
        "If topology_supports_later_community_probe is false, do not force a community module; retain host and ego regions.",
        "High truncation ratios do not invalidate the audit, but all topology claims must be stated as bounded-cache lower bounds.",
    ])
    (args.output / "REPORT.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()

