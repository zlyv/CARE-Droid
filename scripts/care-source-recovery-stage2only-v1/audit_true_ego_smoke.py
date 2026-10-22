#!/usr/bin/env python3
"""Run a true Host + directed 1-hop Ego audit over endpoint-cache payloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_host_ego_smoke import (
    BASE_DIM,
    REGION_DIM,
    read_cache,
    reconstruct,
    region_vector,
    role_names,
)


def role_labels(values, names):
    return [names[int(value)] for value in sorted(values)]


def neighbour_record(index, node_ids, host_roles, names):
    return {
        "host_id": str(node_ids[index]),
        "node_index": int(index),
        "roles": role_labels(host_roles.get(index, ()), names),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-apks", type=int, default=10)
    parser.add_argument("--target-apks", type=int, default=10)
    args = parser.parse_args()
    if not (10 <= args.min_apks <= args.target_apks <= 20):
        raise RuntimeError("smoke selection must contain 10-20 qualifying APKs")

    names = role_names(args.mapping)
    cache_rows = read_cache(args.cache)
    selected = []
    reconstructed = {}
    for row in cache_rows:
        sha = str(row.get("sha256", "")).strip().lower()
        node_ids, host_roles, edges = reconstruct(row["payload"])
        incoming = {host: set() for host in host_roles}
        outgoing = {host: set() for host in host_roles}
        for source, target in edges:
            if source in outgoing and target in incoming:
                outgoing[source].add(target)
                incoming[target].add(source)
        has_nontrivial = any(incoming[host] or outgoing[host] for host in host_roles)
        if not has_nontrivial:
            continue
        selected.append(row)
        reconstructed[sha] = (node_ids, host_roles, edges, incoming, outgoing)
        if len(selected) == args.target_apks:
            break

    args.output.mkdir(parents=True, exist_ok=True)
    examples_path = args.output / "ego_smoke_test_examples.csv"
    report_path = args.output / "ego_smoke_test_summary.txt"
    gate_path = args.output / "ego_smoke_gate.json"
    fields = [
        "apk_id", "host_id", "host_node_index", "center_role", "host_values",
        "ego_1hop_values", "neighbor_roles", "incoming_neighbors",
        "outgoing_neighbors", "ego_node_count", "ego_edge_count", "valid_ego",
        "host_ego_base_equal",
    ]
    output_rows = []
    human_examples = []
    totals = {
        "total_hosts": 0,
        "valid_egos": 0,
        "nontrivial_egos": 0,
        "single_node_egos": 0,
        "missing_egos": 0,
        "valid_ego_host_value_equal": 0,
    }
    for apk_row in selected:
        apk_id = str(apk_row.get("sha256", "")).strip().lower()
        node_ids, host_roles, edges, incoming, outgoing = reconstructed[apk_id]
        for center in sorted(host_roles):
            incoming_nodes = sorted(incoming[center])
            outgoing_nodes = sorted(outgoing[center])
            ego_nodes = frozenset({center, *incoming_nodes, *outgoing_nodes})
            ego_edges = sorted(
                (source, target) for source, target in edges
                if source in ego_nodes and target in ego_nodes
            )
            host_base = region_vector(frozenset([center]), host_roles, edges)
            ego_base = region_vector(ego_nodes, host_roles, edges)
            host_values = np.concatenate([host_base, [1.0, 0.0]])
            ego_values = np.concatenate([ego_base, [0.0, 1.0]])
            if host_values.shape != (REGION_DIM,) or ego_values.shape != (REGION_DIM,):
                raise RuntimeError("unexpected Host/Ego vector dimension")
            if not np.isfinite(host_values).all() or not np.isfinite(ego_values).all():
                raise RuntimeError("non-finite Host/Ego vector")
            valid = len(ego_nodes) > 1 or len(ego_edges) > 0
            base_equal = bool(np.allclose(host_values[:BASE_DIM], ego_values[:BASE_DIM]))
            neighbour_roles = sorted({
                role
                for node in ego_nodes if node != center
                for role in role_labels(host_roles.get(node, ()), names)
            })
            incoming_records = [neighbour_record(v, node_ids, host_roles, names) for v in incoming_nodes]
            outgoing_records = [neighbour_record(v, node_ids, host_roles, names) for v in outgoing_nodes]
            output_rows.append({
                "apk_id": apk_id,
                "host_id": str(node_ids[center]),
                "host_node_index": center,
                "center_role": json.dumps(role_labels(host_roles[center], names), ensure_ascii=False),
                "host_values": json.dumps(host_values.tolist(), separators=(",", ":")),
                "ego_1hop_values": json.dumps(ego_values.tolist(), separators=(",", ":")),
                "neighbor_roles": json.dumps(neighbour_roles, ensure_ascii=False),
                "incoming_neighbors": json.dumps(incoming_records, ensure_ascii=False),
                "outgoing_neighbors": json.dumps(outgoing_records, ensure_ascii=False),
                "ego_node_count": len(ego_nodes),
                "ego_edge_count": len(ego_edges),
                "valid_ego": int(valid),
                "host_ego_base_equal": int(base_equal),
            })
            totals["total_hosts"] += 1
            if valid:
                totals["valid_egos"] += 1
                totals["nontrivial_egos"] += 1
                totals["valid_ego_host_value_equal"] += int(base_equal)
            elif len(ego_nodes) == 1:
                totals["single_node_egos"] += 1
                totals["missing_egos"] += 1

            if valid and len(human_examples) < 5:
                lines = [
                    f"APK: {apk_id}",
                    f"Host: {', '.join(role_labels(host_roles[center], names))}",
                    f"host_id: {node_ids[center]}",
                    "1-hop nodes:",
                ]
                for node in sorted(ego_nodes):
                    marker = " (center)" if node == center else ""
                    lines.append(
                        f"  {', '.join(role_labels(host_roles[node], names))}"
                        f" [{node_ids[node]}]{marker}"
                    )
                lines.append("edges:")
                for source, target in ego_edges:
                    lines.append(
                        f"  {','.join(role_labels(host_roles[source], names))} -> "
                        f"{','.join(role_labels(host_roles[target], names))}"
                    )
                lines.extend([
                    f"ego_node_count: {len(ego_nodes)}",
                    f"ego_edge_count: {len(ego_edges)}",
                    "valid_ego: true",
                    f"host_ego_base_equal: {str(base_equal).lower()}",
                ])
                human_examples.append("\n".join(lines))

    with examples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    selected_apks = len(selected)
    passed = (
        selected_apks >= args.min_apks
        and totals["valid_egos"] > 0
        and totals["nontrivial_egos"] > 0
        and totals["valid_ego_host_value_equal"] == 0
        and len(human_examples) >= 5
    )
    gate = {
        "passed": passed,
        "probe_cache_apks": len(cache_rows),
        "selected_qualifying_apks": selected_apks,
        "selection_rule": "SHA-sorted, label-independent, first APKs with >=1 sensitive Host having an incoming or outgoing Host neighbor",
        **totals,
        "examples_printed": len(human_examples),
        "current_model_consumption": "HOST_ONLY",
        "warning": "This gate validates cached Ego construction; it does not claim that the current v4/MalBen model consumes Ego values.",
        "artifact_hashes": {str(args.cache.resolve()): sha256_file(args.cache)},
    }
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "CARE-Droid true Host + directed 1-hop Ego smoke test", "",
        f"gate_passed: {str(passed).lower()}",
        f"probe_cache_apks: {len(cache_rows)}",
        f"selected_qualifying_apks: {selected_apks}",
        f"selection_rule: {gate['selection_rule']}",
        f"total_hosts: {totals['total_hosts']}",
        f"valid_egos: {totals['valid_egos']}",
        f"nontrivial_egos: {totals['nontrivial_egos']}",
        f"single_node_egos: {totals['single_node_egos']}",
        f"missing_egos: {totals['missing_egos']}",
        f"valid_ego_host_value_equal: {totals['valid_ego_host_value_equal']}", "",
        "Interpretation boundary:",
        "  The smoke test runs real local-region reconstruction from APK-derived endpoint graphs.",
        "  It does not start OOF training, read 2018-2021, or extract all 799 Ego assets.",
        "  The established v4/MalBen model still consumes Host values only.", "",
        "Concrete examples:", "",
        "\n\n".join(human_examples),
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not passed:
        raise SystemExit("true Ego smoke gate failed; all downstream stages remain blocked")


if __name__ == "__main__":
    main()
