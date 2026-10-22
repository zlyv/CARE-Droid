#!/usr/bin/env python3
"""Deterministically rebuild the frozen 2018--2021 merged assets from shards.

This program never parses an APK.  The 1,989 already extracted per-APK NPZ
shards are the sole feature authority; ``lineage`` supplies only their frozen
order.  Every output is written to a temporary file and atomically replaced
only after all shards have passed their structural checks.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


REGION_DIM = 74
REQUIRED_SHARD_KEYS = {
    "sha256", "endpoint_record_json", "host_ids", "host_values", "ego_values",
    "host_role_offsets", "host_role_ids", "host_primary_role",
    "incoming_offsets", "incoming_host_positions",
    "outgoing_offsets", "outgoing_host_positions",
    "ego_node_count", "ego_edge_count", "ego_status", "ego_direction",
    "role_node_counts", "role_edge_counts", "same_role_edge_count",
    "cross_role_edge_count",
}


def norm(value) -> str:
    return str(value or "").strip().lower()


def read_lineage(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [norm(row.get("sha256") or row.get("sample_id") or row.get("apk_id")) for row in rows]
    if len(ids) != 1989 or len(set(ids)) != 1989 or "" in ids:
        raise RuntimeError("BLOCKED_BY_CONTRACT: lineage is not 1,989 unique SHA values")
    return rows


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try: os.unlink(tmp_name)
        except FileNotFoundError: pass
        raise


def atomic_gzip_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with gzip.open(tmp_name, "wt", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        with open(tmp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try: os.unlink(tmp_name)
        except FileNotFoundError: pass
        raise


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def scalar_text(array) -> str:
    value = np.asarray(array)
    if value.shape != ():
        raise RuntimeError("expected scalar string in shard")
    return str(value.item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    rows = read_lineage(args.lineage)
    lineage_ids = [norm(row.get("sha256") or row.get("sample_id") or row.get("apk_id")) for row in rows]
    shard_paths = sorted(args.shards.glob("*.npz"))
    shard_ids = [path.stem.lower() for path in shard_paths]
    if len(shard_paths) != 1989 or len(set(shard_ids)) != 1989:
        raise RuntimeError("BLOCKED_BY_CONTRACT: expected exactly 1,989 unique shard files")
    missing = sorted(set(lineage_ids) - set(shard_ids))
    outside = sorted(set(shard_ids) - set(lineage_ids))
    if missing or outside:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: shard/lineage mismatch missing={len(missing)} outside={len(outside)}")
    empty = [str(path) for path in shard_paths if path.stat().st_size == 0]
    if empty:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: {len(empty)} empty shards")

    sha_values: list[str] = []
    host_values: list[np.ndarray] = []
    host_offsets = [0]
    real_host_count: list[int] = []
    host_was_empty: list[int] = []
    ego_values: list[np.ndarray] = []
    ego_offsets = [0]
    all_host_ids: list[str] = []
    all_host_instance_ids: list[str] = []
    all_primary: list[int] = []
    all_role_offsets = [0]
    all_role_ids: list[int] = []
    all_in_offsets = [0]
    all_in_positions: list[int] = []
    all_in_ids: list[str] = []
    all_in_primary_roles: list[int] = []
    all_out_offsets = [0]
    all_out_positions: list[int] = []
    all_out_ids: list[str] = []
    all_out_primary_roles: list[int] = []
    all_ego_node: list[int] = []
    all_ego_edge: list[int] = []
    all_status: list[int] = []
    all_direction: list[int] = []
    role_nodes: list[np.ndarray] = []
    role_edges: list[np.ndarray] = []
    endpoint_rows: list[dict] = []
    same_total = 0
    cross_total = 0

    for index, sha in enumerate(lineage_ids, start=1):
        path = args.shards / f"{sha}.npz"
        with np.load(path, allow_pickle=False) as saved:
            absent = REQUIRED_SHARD_KEYS - set(saved.files)
            if absent:
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: shard {sha} missing {sorted(absent)}")
            embedded_sha = norm(scalar_text(saved["sha256"]))
            if embedded_sha != sha:
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: shard identity mismatch {sha} != {embedded_sha}")
            actual_host = np.asarray(saved["host_values"], dtype=np.float32)
            actual_ego = np.asarray(saved["ego_values"], dtype=np.float32)
            if actual_host.ndim != 2 or actual_host.shape[1] != REGION_DIM:
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: invalid host_values shape in {sha}: {actual_host.shape}")
            if actual_ego.shape != actual_host.shape:
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: Host/Ego shape mismatch in {sha}")
            n_hosts = len(actual_host)
            model_host = actual_host if n_hosts else np.zeros((1, REGION_DIM), dtype=np.float32)
            sha_values.append(sha)
            host_values.append(model_host); host_offsets.append(host_offsets[-1] + len(model_host))
            real_host_count.append(n_hosts); host_was_empty.append(int(n_hosts == 0))
            ego_values.append(actual_ego); ego_offsets.append(ego_offsets[-1] + n_hosts)

            shard_host_ids = saved["host_ids"].astype(str).tolist()
            shard_primary = saved["host_primary_role"].astype(np.int16)
            if len(shard_host_ids) != n_hosts or len(shard_primary) != n_hosts:
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: Host metadata length mismatch in {sha}")
            host_base = len(all_host_ids)
            all_host_ids.extend(shard_host_ids)
            all_host_instance_ids.extend(f"{sha}:{value}" for value in shard_host_ids)
            all_primary.extend(shard_primary.tolist())

            shard_role_offsets = saved["host_role_offsets"].astype(np.int64)
            shard_role_ids = saved["host_role_ids"].astype(np.int16)
            shard_in_offsets = saved["incoming_offsets"].astype(np.int64)
            shard_in_positions = saved["incoming_host_positions"].astype(np.int64)
            shard_out_offsets = saved["outgoing_offsets"].astype(np.int64)
            shard_out_positions = saved["outgoing_host_positions"].astype(np.int64)
            for offsets, name in ((shard_role_offsets, "role"), (shard_in_offsets, "incoming"), (shard_out_offsets, "outgoing")):
                if len(offsets) != n_hosts + 1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
                    raise RuntimeError(f"BLOCKED_BY_CONTRACT: invalid {name} offsets in {sha}")
            if shard_role_offsets[-1] != len(shard_role_ids) or shard_in_offsets[-1] != len(shard_in_positions) or shard_out_offsets[-1] != len(shard_out_positions):
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: ragged terminal offset mismatch in {sha}")
            if (len(shard_in_positions) and (shard_in_positions.min() < 0 or shard_in_positions.max() >= n_hosts)) or (len(shard_out_positions) and (shard_out_positions.min() < 0 or shard_out_positions.max() >= n_hosts)):
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: neighbor position out of range in {sha}")

            for host_index in range(n_hosts):
                left, right = shard_role_offsets[host_index:host_index + 2]
                all_role_ids.extend(shard_role_ids[left:right].tolist()); all_role_offsets.append(len(all_role_ids))
                left, right = shard_in_offsets[host_index:host_index + 2]
                positions = shard_in_positions[left:right]
                all_in_positions.extend((positions + host_base).tolist())
                all_in_ids.extend(shard_host_ids[int(value)] for value in positions)
                all_in_primary_roles.extend(shard_primary[positions].tolist()); all_in_offsets.append(len(all_in_positions))
                left, right = shard_out_offsets[host_index:host_index + 2]
                positions = shard_out_positions[left:right]
                all_out_positions.extend((positions + host_base).tolist())
                all_out_ids.extend(shard_host_ids[int(value)] for value in positions)
                all_out_primary_roles.extend(shard_primary[positions].tolist()); all_out_offsets.append(len(all_out_positions))

            for key, dtype, target in (
                ("ego_node_count", np.int32, all_ego_node), ("ego_edge_count", np.int32, all_ego_edge),
                ("ego_status", np.int8, all_status), ("ego_direction", np.int8, all_direction),
            ):
                values = saved[key].astype(dtype).tolist()
                if len(values) != n_hosts:
                    raise RuntimeError(f"BLOCKED_BY_CONTRACT: {key} length mismatch in {sha}")
                target.extend(values)
            rn = np.asarray(saved["role_node_counts"], dtype=np.int64)
            re = np.asarray(saved["role_edge_counts"], dtype=np.int64)
            if rn.shape != (31,) or re.shape != (31, 31):
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: role graph shape mismatch in {sha}")
            role_nodes.append(rn); role_edges.append(re)
            same_total += int(np.asarray(saved["same_role_edge_count"]).item())
            cross_total += int(np.asarray(saved["cross_role_edge_count"]).item())
            endpoint_record = json.loads(scalar_text(saved["endpoint_record_json"]))
            if norm(endpoint_record.get("sha256")) != sha:
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: endpoint record identity mismatch in {sha}")
            endpoint_rows.append(endpoint_record)
        if index % 200 == 0 or index == 1989:
            print(f"validated/merged shards {index}/1989", flush=True)

    host_matrix = np.concatenate(host_values, axis=0)
    ego_matrix = np.concatenate(ego_values, axis=0) if any(len(value) for value in ego_values) else np.zeros((0, REGION_DIM), np.float32)
    common = dict(
        sha256=np.asarray(sha_values), host_values=host_matrix,
        host_offsets=np.asarray(host_offsets, dtype=np.int64),
        real_host_count=np.asarray(real_host_count, dtype=np.int32),
        host_was_empty=np.asarray(host_was_empty, dtype=np.int8),
    )
    atomic_npz(args.output / "host_regions_2018_2021.npz", **common)
    atomic_npz(args.output / "regions_2018_2021.npz", **common,
               ego_values=ego_matrix, ego_offsets=np.asarray(ego_offsets, dtype=np.int64))
    atomic_npz(
        args.output / "ego_1hop_regions_2018_2021.npz",
        sha256=np.asarray(sha_values), ego_values=ego_matrix,
        ego_offsets=np.asarray(ego_offsets, dtype=np.int64),
        host_ids=np.asarray(all_host_ids), host_primary_role=np.asarray(all_primary, dtype=np.int16),
        host_instance_ids=np.asarray(all_host_instance_ids),
        host_role_offsets=np.asarray(all_role_offsets, dtype=np.int64),
        host_role_ids=np.asarray(all_role_ids, dtype=np.int16),
        incoming_offsets=np.asarray(all_in_offsets, dtype=np.int64),
        incoming_host_positions=np.asarray(all_in_positions, dtype=np.int64),
        incoming_neighbor_ids=np.asarray(all_in_ids),
        incoming_neighbor_primary_roles=np.asarray(all_in_primary_roles, dtype=np.int16),
        outgoing_offsets=np.asarray(all_out_offsets, dtype=np.int64),
        outgoing_host_positions=np.asarray(all_out_positions, dtype=np.int64),
        outgoing_neighbor_ids=np.asarray(all_out_ids),
        outgoing_neighbor_primary_roles=np.asarray(all_out_primary_roles, dtype=np.int16),
        ego_node_count=np.asarray(all_ego_node, dtype=np.int32),
        ego_edge_count=np.asarray(all_ego_edge, dtype=np.int32),
        ego_status=np.asarray(all_status, dtype=np.int8),
        ego_direction=np.asarray(all_direction, dtype=np.int8),
    )
    atomic_npz(args.output / "global_role_graph_2018_2021.npz",
               sha256=np.asarray(sha_values), role_node_counts=np.stack(role_nodes), role_edge_counts=np.stack(role_edges))
    atomic_gzip_jsonl(args.output / "endpoint_centered_tensor_cache_2018_2021.jsonl.gz", endpoint_rows)

    outputs = {}
    for name in ("host_regions_2018_2021.npz", "regions_2018_2021.npz", "ego_1hop_regions_2018_2021.npz", "global_role_graph_2018_2021.npz", "endpoint_centered_tensor_cache_2018_2021.jsonl.gz"):
        path = args.output / name
        outputs[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    audit = {
        "passed": True, "apk_reanalysis": False, "lineage_rows": 1989,
        "shard_rows": 1989, "host_rows_with_placeholders": len(host_matrix),
        "real_host_rows": len(ego_matrix), "empty_host_apks": int(sum(host_was_empty)),
        "same_role_edges": same_total, "cross_role_edges": cross_total,
        "outputs": outputs,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("FUTURE_ASSETS_REBUILT_FROM_1989_FROZEN_SHARDS")


if __name__ == "__main__":
    main()
