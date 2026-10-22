#!/usr/bin/env python3
"""Hard gate for the frozen 1989-APK 2018--2021 local-asset cache."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np


TAG = "2018_2021"
EXPECTED_BY_YEAR = {2018: 793, 2019: 399, 2020: 397, 2021: 400}


def rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=1989)
    args = parser.parse_args()
    lineage = rows(args.lineage)
    lineage_ids = [row["sha256"].strip().lower() for row in lineage]
    years = [int(row["metadata_year"]) for row in lineage]
    required = [
        args.root / "local_asset_manifest.json",
        args.root / f"host_regions_{TAG}.npz",
        args.root / f"regions_{TAG}.npz",
        args.root / f"ego_1hop_regions_{TAG}.npz",
        args.root / f"global_role_graph_{TAG}.npz",
        args.root / f"endpoint_centered_tensor_cache_{TAG}.jsonl.gz",
        args.root / "ego_neighbor_scope_audit.txt",
        args.root / "failures.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing future local assets: " + repr(missing))
    manifest = json.loads(required[0].read_text(encoding="utf-8"))
    with np.load(required[1], allow_pickle=False) as host:
        host_ids = host["sha256"].astype(str).tolist()
        host_offsets = host["host_offsets"].astype(np.int64)
        host_values = host["host_values"]
        real_count = host["real_host_count"].astype(np.int64)
    with np.load(required[3], allow_pickle=False) as ego:
        ego_ids = ego["sha256"].astype(str).tolist()
        ego_values = ego["ego_values"]
        ego_offsets = ego["ego_offsets"].astype(np.int64)
        statuses = ego["ego_status"].astype(np.int64)
        node_count = ego["ego_node_count"].astype(np.int64)
        edge_count = ego["ego_edge_count"].astype(np.int64)
        instance_ids = ego["host_instance_ids"].astype(str).tolist()
    with np.load(required[4], allow_pickle=False) as graph:
        graph_ids = graph["sha256"].astype(str).tolist()
        node_shape = tuple(graph["role_node_counts"].shape)
        edge_shape = tuple(graph["role_edge_counts"].shape)
    with gzip.open(required[5], "rt", encoding="utf-8") as handle:
        endpoint_ids = [json.loads(line)["sha256"].strip().lower() for line in handle if line.strip()]
    failures = rows(required[7])
    checks = {
        "lineage_exact_unique_1989": len(lineage_ids) == args.expected == len(set(lineage_ids)),
        "year_counts_exact": dict(Counter(years)) == EXPECTED_BY_YEAR,
        "host_cache_exact_lineage_order": host_ids == lineage_ids,
        "ego_cache_exact_lineage_order": ego_ids == lineage_ids,
        "graph_cache_exact_lineage_order": graph_ids == lineage_ids,
        "endpoint_cache_exact_unique_lineage": endpoint_ids == lineage_ids and len(set(endpoint_ids)) == args.expected,
        "no_true_extraction_failures": len(failures) == 0,
        "host_offsets_valid": len(host_offsets) == args.expected + 1 and host_offsets[0] == 0 and np.all(np.diff(host_offsets) >= 1) and host_offsets[-1] == len(host_values),
        "host_feature_space_74d": tuple(host_values.shape[1:]) == (74,),
        "ego_feature_space_74d": tuple(ego_values.shape[1:]) == (74,),
        "real_host_counts_match_ego_offsets": len(real_count) == args.expected and np.array_equal(np.diff(ego_offsets), real_count),
        "host_instance_ids_unique": len(instance_ids) == len(set(instance_ids)) == int(np.sum(real_count)),
        "ego_status_semantics_valid": bool(np.all(np.isin(statuses, [0, 1]))) and bool(np.all((statuses == 1) == ((node_count > 1) | (edge_count > 0)))),
        "global_role_graph_shape_valid": node_shape == (args.expected, 31) and edge_shape == (args.expected, 31, 31),
        "manifest_future_scope": manifest.get("cohort_kind") == "future" and manifest.get("future_years_loaded") is True and manifest.get("cohort_years") == [2018, 2019, 2020, 2021],
        "role_order_contract_present": len(manifest.get("model_role_order", [])) == 31 and len(manifest.get("payload_to_model_role", [])) == 31,
        "writer_lock_documented": manifest.get("writer_lock") == "fcntl exclusive nonblocking",
    }
    # NumPy predicates yield np.bool_, which the standard JSON encoder does
    # not serialise.  Gate files contain only native Python booleans.
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    gate = {"passed": passed, "expected_apks": args.expected, "checks": checks}
    (args.root / "future_asset_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = ["CARE-Droid future local-asset hard gate", "", f"passed: {str(passed).lower()}"]
    text.extend(f"{key}: {str(value).lower()}" for key, value in checks.items())
    (args.root / "future_asset_gate.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("future local-asset gate failed")
    print("FUTURE_LOCAL_ASSET_GATE_PASS")


if __name__ == "__main__":
    main()
