#!/usr/bin/env python3
"""Hard gate for the canonical 799-APK source local-asset cache."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=799)
    args = parser.parse_args()
    manifest_path = args.root / "local_asset_manifest.json"
    required = [
        manifest_path, args.root / "host_regions_2017.npz",
        args.root / "ego_1hop_regions_2017.npz",
        args.root / "regions_2017.npz", args.root / "global_role_graph_2017.npz",
        args.root / "endpoint_centered_tensor_cache_2017.jsonl.gz",
        args.root / "ego_neighbor_scope_audit.txt", args.root / "failures.csv",
    ]
    missing_files = [str(path) for path in required if not path.is_file()]
    if missing_files:
        raise SystemExit("missing source local assets: " + repr(missing_files))
    lineage = read_csv(args.lineage)
    lineage_ids = [row["sha256"].strip().lower() for row in lineage]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(args.root / "host_regions_2017.npz", allow_pickle=False) as host:
        cache_ids = host["sha256"].astype(str).tolist()
        offsets = host["host_offsets"].astype(np.int64)
        real_count = host["real_host_count"].astype(np.int64)
        host_shape = tuple(host["host_values"].shape)
    with np.load(args.root / "ego_1hop_regions_2017.npz", allow_pickle=False) as ego:
        ego_ids = ego["sha256"].astype(str).tolist()
        ego_offsets = ego["ego_offsets"].astype(np.int64)
        host_ids = ego["host_ids"].astype(str)
        host_instance_ids = ego["host_instance_ids"].astype(str)
        statuses = ego["ego_status"].astype(np.int64)
        ego_shape = tuple(ego["ego_values"].shape)
        node_count = ego["ego_node_count"].astype(np.int64)
        edge_count = ego["ego_edge_count"].astype(np.int64)
        role_offsets = ego["host_role_offsets"].astype(np.int64)
        in_offsets = ego["incoming_offsets"].astype(np.int64)
        out_offsets = ego["outgoing_offsets"].astype(np.int64)
        in_pos = ego["incoming_host_positions"].astype(np.int64)
        out_pos = ego["outgoing_host_positions"].astype(np.int64)
        in_ids = ego["incoming_neighbor_ids"].astype(str)
        out_ids = ego["outgoing_neighbor_ids"].astype(str)
    with np.load(args.root / "global_role_graph_2017.npz", allow_pickle=False) as graph:
        graph_ids = graph["sha256"].astype(str).tolist()
        node_shape = tuple(graph["role_node_counts"].shape)
        edge_shape = tuple(graph["role_edge_counts"].shape)
    with gzip.open(args.root / "endpoint_centered_tensor_cache_2017.jsonl.gz", "rt", encoding="utf-8") as handle:
        endpoint_ids = [json.loads(line)["sha256"].strip().lower() for line in handle if line.strip()]
    failures = read_csv(args.root / "failures.csv")
    checks = {
        "lineage_exact_unique_799": len(lineage_ids) == args.expected == len(set(lineage_ids)),
        "host_cache_exact_lineage_order": cache_ids == lineage_ids,
        "ego_cache_exact_lineage_order": ego_ids == lineage_ids,
        "graph_cache_exact_lineage_order": graph_ids == lineage_ids,
        "endpoint_cache_exact_unique_lineage": endpoint_ids == lineage_ids and len(set(endpoint_ids)) == args.expected,
        "no_true_extraction_failures": len(failures) == 0,
        "host_offsets_valid": len(offsets) == args.expected + 1 and offsets[0] == 0 and np.all(np.diff(offsets) >= 1) and offsets[-1] == host_shape[0],
        "host_feature_space_74d": len(host_shape) == 2 and host_shape[1] == 74,
        "ego_feature_space_74d": len(ego_shape) == 2 and ego_shape[1] == 74,
        "real_host_counts_match_ego_offsets": len(real_count) == args.expected and np.array_equal(np.diff(ego_offsets), real_count),
        "host_instance_ids_unique": len(host_instance_ids) == len(set(host_instance_ids.tolist())) == int(np.sum(real_count)),
        "host_metadata_lengths_match": len(host_ids) == len(statuses) == len(node_count) == len(edge_count) == int(np.sum(real_count)),
        "ragged_offsets_valid": len(role_offsets) == len(host_ids) + 1 and len(in_offsets) == len(host_ids) + 1 and len(out_offsets) == len(host_ids) + 1,
        "neighbor_positions_valid": (len(in_pos) == 0 or (np.min(in_pos) >= 0 and np.max(in_pos) < len(host_ids))) and (len(out_pos) == 0 or (np.min(out_pos) >= 0 and np.max(out_pos) < len(host_ids))),
        "neighbor_id_lengths_match": len(in_ids) == len(in_pos) and len(out_ids) == len(out_pos),
        "ego_status_semantics_valid": bool(np.all(np.isin(statuses, [0, 1]))) and bool(np.all((statuses == 1) == ((node_count > 1) | (edge_count > 0)))),
        "global_role_graph_shape_valid": node_shape == (args.expected, 31) and edge_shape == (args.expected, 31, 31),
        "role_order_contract_present": len(manifest.get("model_role_order", [])) == 31 and len(manifest.get("payload_to_model_role", [])) == 31,
        "manifest_future_free": manifest.get("source_only") is True and manifest.get("future_years_loaded") is False,
        "writer_lock_documented": manifest.get("writer_lock") == "fcntl exclusive nonblocking",
    }
    # Several NumPy comparisons above return np.bool_.  Normalise every gate
    # value before JSON serialisation so the audit result is portable across
    # Python/NumPy versions.
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    gate = {
        "passed": passed, "expected_apks": args.expected,
        "lineage_rows": len(lineage_ids), "real_hosts": int(np.sum(real_count)),
        "checks": checks,
    }
    (args.root / "source_asset_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["CARE-Droid source local-asset hard gate", "", f"passed: {str(passed).lower()}"]
    lines.extend(f"{name}: {str(value).lower()}" for name, value in checks.items())
    (args.root / "source_asset_gate.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("source local-asset gate failed; OOF is forbidden")
    print("SOURCE_LOCAL_ASSET_GATE_PASS")


if __name__ == "__main__":
    main()
