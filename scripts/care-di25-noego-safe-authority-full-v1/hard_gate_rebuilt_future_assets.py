#!/usr/bin/env python3
"""Hard-gate all rebuilt future assets before any CARE model code is imported."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np


EXPECTED_BY_YEAR = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
TAG = "2018_2021"


def norm(value) -> str:
    return str(value or "").strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    with args.lineage.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [norm(row.get("sha256") or row.get("sample_id") or row.get("apk_id")) for row in rows]
    years = [int(float(row.get("metadata_year") or row.get("year"))) for row in rows]
    files = {
        "host": args.root / f"host_regions_{TAG}.npz",
        "regions": args.root / f"regions_{TAG}.npz",
        "ego": args.root / f"ego_1hop_regions_{TAG}.npz",
        "graph": args.root / f"global_role_graph_{TAG}.npz",
        "endpoint": args.root / f"endpoint_centered_tensor_cache_{TAG}.jsonl.gz",
    }
    nonempty = {name: path.is_file() and path.stat().st_size > 100 for name, path in files.items()}
    if not all(nonempty.values()):
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty rebuilt assets " + repr({k: str(files[k]) for k, ok in nonempty.items() if not ok}))

    with np.load(files["host"], allow_pickle=False) as z:
        host_ids = [norm(x) for x in z["sha256"].astype(str)]
        host_values = np.asarray(z["host_values"])
        host_offsets = np.asarray(z["host_offsets"], dtype=np.int64)
        real_count = np.asarray(z["real_host_count"], dtype=np.int64)
        host_empty = np.asarray(z["host_was_empty"], dtype=np.int8)
    with np.load(files["regions"], allow_pickle=False) as z:
        region_ids = [norm(x) for x in z["sha256"].astype(str)]
        region_host_values = np.asarray(z["host_values"])
        region_host_offsets = np.asarray(z["host_offsets"], dtype=np.int64)
        ego_values_in_regions = np.asarray(z["ego_values"])
        ego_offsets_in_regions = np.asarray(z["ego_offsets"], dtype=np.int64)
    with np.load(files["ego"], allow_pickle=False) as z:
        ego_ids = [norm(x) for x in z["sha256"].astype(str)]
        ego_values = np.asarray(z["ego_values"])
        ego_offsets = np.asarray(z["ego_offsets"], dtype=np.int64)
        instance_ids = z["host_instance_ids"].astype(str).tolist()
        primary = np.asarray(z["host_primary_role"], dtype=np.int64)
        role_offsets = np.asarray(z["host_role_offsets"], dtype=np.int64)
        role_ids = np.asarray(z["host_role_ids"], dtype=np.int64)
        in_offsets = np.asarray(z["incoming_offsets"], dtype=np.int64)
        in_positions = np.asarray(z["incoming_host_positions"], dtype=np.int64)
        out_offsets = np.asarray(z["outgoing_offsets"], dtype=np.int64)
        out_positions = np.asarray(z["outgoing_host_positions"], dtype=np.int64)
        status = np.asarray(z["ego_status"], dtype=np.int64)
        node_count = np.asarray(z["ego_node_count"], dtype=np.int64)
        edge_count = np.asarray(z["ego_edge_count"], dtype=np.int64)
    with np.load(files["graph"], allow_pickle=False) as z:
        graph_ids = [norm(x) for x in z["sha256"].astype(str)]
        graph_nodes = np.asarray(z["role_node_counts"])
        graph_edges = np.asarray(z["role_edge_counts"])
    endpoint_ids = []
    with gzip.open(files["endpoint"], "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): endpoint_ids.append(norm(json.loads(line)["sha256"]))

    real_hosts = int(real_count.sum())
    model_hosts = int(np.maximum(real_count, 1).sum())
    checks = {
        "lineage_exact_unique_1989": len(ids) == len(set(ids)) == 1989 and "" not in ids,
        "year_counts_exact": dict(Counter(years)) == EXPECTED_BY_YEAR,
        "all_asset_orders_equal_lineage": host_ids == region_ids == ego_ids == graph_ids == endpoint_ids == ids,
        "host_shapes_valid": host_values.shape == (model_hosts, 74) and host_offsets.shape == (1990,) and host_offsets[0] == 0 and host_offsets[-1] == model_hosts,
        "host_offsets_exact": np.array_equal(np.diff(host_offsets), np.maximum(real_count, 1)),
        "empty_host_semantics": np.array_equal(host_empty, (real_count == 0).astype(np.int8)),
        "regions_duplicate_host_exact": np.array_equal(region_host_values, host_values) and np.array_equal(region_host_offsets, host_offsets),
        "ego_shapes_valid": ego_values.shape == ego_values_in_regions.shape == (real_hosts, 74) and ego_offsets.shape == ego_offsets_in_regions.shape == (1990,),
        "ego_offsets_exact": np.array_equal(ego_offsets, ego_offsets_in_regions) and np.array_equal(np.diff(ego_offsets), real_count),
        "host_instance_ids_unique": len(instance_ids) == len(set(instance_ids)) == real_hosts,
        "host_metadata_lengths": len(primary) == len(status) == len(node_count) == len(edge_count) == real_hosts,
        "ragged_offsets_valid": len(role_offsets) == len(in_offsets) == len(out_offsets) == real_hosts + 1 and role_offsets[-1] == len(role_ids) and in_offsets[-1] == len(in_positions) and out_offsets[-1] == len(out_positions),
        "neighbor_positions_valid": (len(in_positions) == 0 or (in_positions.min() >= 0 and in_positions.max() < real_hosts)) and (len(out_positions) == 0 or (out_positions.min() >= 0 and out_positions.max() < real_hosts)),
        "ego_status_semantics": np.all(np.isin(status, [0, 1])) and np.all((status == 1) == ((node_count > 1) | (edge_count > 0))),
        "global_role_graph_shape": graph_nodes.shape == (1989, 31) and graph_edges.shape == (1989, 31, 31),
        "endpoint_exact_unique_1989": len(endpoint_ids) == len(set(endpoint_ids)) == 1989,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    passed = all(checks.values())
    result = {
        "passed": passed, "checks": checks, "year_counts": dict(Counter(years)),
        "real_host_rows": real_hosts, "model_host_rows_with_placeholders": model_hosts,
        "empty_host_apks": int((real_count == 0).sum()),
        "file_bytes": {name: path.stat().st_size for name, path in files.items()},
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError("BLOCKED_BY_CONTRACT: rebuilt future asset hard gate failed")
    print("FUTURE_REBUILT_ASSET_HARD_GATE_PASS")


if __name__ == "__main__":
    main()
