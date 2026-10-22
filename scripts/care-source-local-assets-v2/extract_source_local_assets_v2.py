#!/usr/bin/env python3
"""One-pass, resumable 2017 Host/Ego/endpoint asset extraction.

Each APK is parsed once into an endpoint-centred bounded call-graph payload.
The payload and all Host-only / directed 1-hop Host-Ego raw assets are saved in
an atomic per-APK shard.  Final cohort files are merged atomically only after
all 799 canonical source APKs have successful shards.
"""

from __future__ import annotations

import argparse
import csv
try:
    import fcntl
except ImportError:  # permits platform-neutral reconstruction self-tests
    fcntl = None
import gzip
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


ROLE_COUNT = 31
BASE_DIM = 72
REGION_DIM = 74
NONTRIVIAL_EGO = 1
HOST_ONLY_NO_NEIGHBOR = 0


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp = Path(name)
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp = Path(name)
    try:
        with gzip.open(temp, "wt", encoding="utf-8", newline="") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def norm(value) -> str:
    return str(value).strip().lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entropy(counts: np.ndarray) -> float:
    value = np.asarray(counts, dtype=np.float64)
    if float(np.sum(value)) <= 0:
        return 0.0
    probability = value[value > 0] / np.sum(value)
    return float(-np.sum(probability * np.log(probability)) / np.log(ROLE_COUNT))


def region_vector(nodes, host_roles, host_edges) -> np.ndarray:
    nodes = frozenset(int(value) for value in nodes)
    counts = np.zeros(ROLE_COUNT, dtype=np.float64)
    role_incidence = []
    for host in nodes:
        roles = host_roles.get(host, ())
        role_incidence.append(len(roles))
        for role in roles:
            counts[int(role)] += 1
    internal = {(s, t) for s, t in host_edges if s in nodes and t in nodes and s != t}
    boundary = {(s, t) for s, t in host_edges if (s in nodes) != (t in nodes)}
    reciprocal = sum((t, s) in internal for s, t in internal)
    host_n = len(nodes)
    possible = host_n * max(host_n - 1, 0)
    degree = np.asarray(role_incidence, dtype=np.float64)
    topology = np.asarray([
        np.log1p(host_n), np.log1p(np.sum(counts)), np.count_nonzero(counts) / ROLE_COUNT,
        float(np.mean(degree)) if len(degree) else 0.0,
        float(np.mean(degree >= 2)) if len(degree) else 0.0,
        np.log1p(len(internal)), np.log1p(len(boundary)),
        len(internal) / possible if possible else 0.0,
        reciprocal / len(internal) if internal else 0.0,
        entropy(counts),
    ])
    result = np.concatenate([counts / max(host_n, 1), np.log1p(counts), topology])
    if result.shape != (BASE_DIM,) or not np.isfinite(result).all():
        raise RuntimeError("invalid local-region vector")
    return result.astype(np.float32)


def reconstruct(payload: dict, payload_to_model_role: np.ndarray) -> dict:
    node_ids = np.asarray(payload.get("node_ids") or [], dtype=str)
    role_rows = [
        tuple(sorted({int(payload_to_model_role[int(v)]) for v in (values or []) if 0 <= int(v) < ROLE_COUNT}))
        for values in (payload.get("node_role_ids") or [])
    ]
    base = np.asarray(payload.get("node_base_features") or [], dtype=np.float32)
    # A successfully parsed APK may have no retained sensitive endpoint and
    # therefore an empty endpoint-centred graph.  JSON represents its feature
    # matrix as [], which NumPy reads as shape (0,), rather than (0, 15).
    # This is a valid missing-region case, not a payload schema failure.
    if base.size == 0:
        base = np.empty((0, 15), dtype=np.float32)
    edge_index = payload.get("edge_index") or [[], []]
    if base.ndim != 2 or base.shape[1] != 15 or len(node_ids) != len(base) or len(role_rows) != len(base):
        raise RuntimeError(
            "endpoint payload node schema mismatch: "
            f"node_ids={len(node_ids)} role_rows={len(role_rows)} "
            f"base_shape={tuple(base.shape)}"
        )
    if len(edge_index) != 2:
        raise RuntimeError("endpoint payload edge_index schema mismatch")
    source = np.asarray(edge_index[0], dtype=np.int64)
    target = np.asarray(edge_index[1], dtype=np.int64)
    if len(source) != len(target):
        raise RuntimeError("endpoint edge rows disagree")

    endpoints = set(map(int, np.flatnonzero(base[:, 4] > 0.5)))
    caller_targets: dict[int, set[int]] = {}
    for src, dst in zip(source, target):
        src, dst = int(src), int(dst)
        if dst in endpoints:
            caller_targets.setdefault(src, set()).add(dst)
    host_roles = {}
    for host, targets in caller_targets.items():
        merged = tuple(sorted({role for endpoint in targets for role in role_rows[endpoint]}))
        if merged:
            host_roles[host] = merged
    hosts = sorted(host_roles)
    host_position = {node: index for index, node in enumerate(hosts)}
    host_edges = sorted({
        (int(src), int(dst)) for src, dst in zip(source, target)
        if int(src) in host_position and int(dst) in host_position and int(src) != int(dst)
    })
    incoming = {host: set() for host in hosts}
    outgoing = {host: set() for host in hosts}
    for src, dst in host_edges:
        outgoing[src].add(dst); incoming[dst].add(src)

    host_values = []
    ego_values = []
    status = []
    ego_nodes_n = []
    ego_edges_n = []
    role_offsets = [0]
    role_ids = []
    in_offsets = [0]
    in_position = []
    out_offsets = [0]
    out_position = []
    primary_role = []
    direction = []
    for host in hosts:
        in_nodes = sorted(incoming[host])
        out_nodes = sorted(outgoing[host])
        ego_nodes = frozenset({host, *in_nodes, *out_nodes})
        ego_edges = [(s, t) for s, t in host_edges if s in ego_nodes and t in ego_nodes]
        host_values.append(np.concatenate([region_vector([host], host_roles, host_edges), [1.0, 0.0]]))
        ego_values.append(np.concatenate([region_vector(ego_nodes, host_roles, host_edges), [0.0, 1.0]]))
        nontrivial = len(ego_nodes) > 1 or len(ego_edges) > 0
        status.append(NONTRIVIAL_EGO if nontrivial else HOST_ONLY_NO_NEIGHBOR)
        ego_nodes_n.append(len(ego_nodes)); ego_edges_n.append(len(ego_edges))
        role_ids.extend(host_roles[host]); role_offsets.append(len(role_ids))
        primary_role.append(min(host_roles[host]))
        in_position.extend(host_position[value] for value in in_nodes); in_offsets.append(len(in_position))
        out_position.extend(host_position[value] for value in out_nodes); out_offsets.append(len(out_position))
        direction.append(3 if in_nodes and out_nodes else (1 if in_nodes else (2 if out_nodes else 0)))

    role_node = np.zeros(ROLE_COUNT, dtype=np.int64)
    for roles in role_rows:
        for role in roles:
            role_node[role] += 1
    role_edge = np.zeros((ROLE_COUNT, ROLE_COUNT), dtype=np.int64)
    for src, dst in zip(source, target):
        for left in role_rows[int(src)]:
            for right in role_rows[int(dst)]:
                role_edge[left, right] += 1

    same_role_edges = 0
    cross_role_edges = 0
    for src, dst in host_edges:
        if min(host_roles[src]) == min(host_roles[dst]):
            same_role_edges += 1
        else:
            cross_role_edges += 1
    return {
        "node_ids": node_ids,
        "host_node_indices": np.asarray(hosts, dtype=np.int64),
        "host_ids": node_ids[hosts] if hosts else np.asarray([], dtype=str),
        "host_values": np.asarray(host_values, dtype=np.float32).reshape(-1, REGION_DIM),
        "ego_values": np.asarray(ego_values, dtype=np.float32).reshape(-1, REGION_DIM),
        "host_role_offsets": np.asarray(role_offsets, dtype=np.int64),
        "host_role_ids": np.asarray(role_ids, dtype=np.int16),
        "host_primary_role": np.asarray(primary_role, dtype=np.int16),
        "incoming_offsets": np.asarray(in_offsets, dtype=np.int64),
        "incoming_host_positions": np.asarray(in_position, dtype=np.int64),
        "outgoing_offsets": np.asarray(out_offsets, dtype=np.int64),
        "outgoing_host_positions": np.asarray(out_position, dtype=np.int64),
        "ego_node_count": np.asarray(ego_nodes_n, dtype=np.int32),
        "ego_edge_count": np.asarray(ego_edges_n, dtype=np.int32),
        "ego_status": np.asarray(status, dtype=np.int8),
        "ego_direction": np.asarray(direction, dtype=np.int8),
        "role_node_counts": role_node,
        "role_edge_counts": role_edge,
        "same_role_edge_count": np.asarray(same_role_edges, dtype=np.int64),
        "cross_role_edge_count": np.asarray(cross_role_edges, dtype=np.int64),
    }


def load_modules(extractor_root: Path):
    sys.path.insert(0, str(extractor_root / "scripts"))
    sys.path.insert(0, str(extractor_root / "src"))
    import run_endpoint_centered_tensor_cache_ag as endpoint
    return endpoint


def write_manifest(path: Path, lineage_ids: list[str], shard_dir: Path, failure_dir: Path) -> None:
    done = sorted(item.stem for item in shard_dir.glob("*.npz"))
    failed = sorted(item.stem for item in failure_dir.glob("*.json"))
    atomic_text(path, json.dumps({
        "schema_version": "2.0.0",
        "expected": len(lineage_ids),
        "done_count": len(done),
        "failed_count": len(failed),
        "done": done,
        "failed": failed,
    }, ensure_ascii=False, indent=2) + "\n")


def merge_assets(
    rows: list[dict], shard_dir: Path, output: Path, endpoint_rows: list[dict],
    asset_tag: str = "2017",
) -> dict:
    sha_values = []
    host_values = []
    host_offsets = [0]
    real_host_count = []
    host_was_empty = []
    ego_values = []
    ego_offsets = [0]
    all_host_ids = []
    all_host_instance_ids = []
    all_primary = []
    all_role_offsets = [0]
    all_role_ids = []
    all_in_offsets = [0]
    all_in_positions = []
    all_in_ids = []
    all_in_primary_roles = []
    all_out_offsets = [0]
    all_out_positions = []
    all_out_ids = []
    all_out_primary_roles = []
    all_ego_node = []
    all_ego_edge = []
    all_status = []
    all_direction = []
    role_nodes = []
    role_edges = []
    same_total = cross_total = 0
    for row in rows:
        sha = norm(row["sha256"])
        with np.load(shard_dir / f"{sha}.npz", allow_pickle=False) as saved:
            actual_host = np.asarray(saved["host_values"], dtype=np.float32)
            actual_ego = np.asarray(saved["ego_values"], dtype=np.float32)
            model_host = actual_host if len(actual_host) else np.zeros((1, REGION_DIM), dtype=np.float32)
            sha_values.append(sha)
            host_values.append(model_host); host_offsets.append(host_offsets[-1] + len(model_host))
            real_host_count.append(len(actual_host)); host_was_empty.append(int(len(actual_host) == 0))
            ego_values.append(actual_ego); ego_offsets.append(ego_offsets[-1] + len(actual_ego))
            shard_host_ids = saved["host_ids"].astype(str).tolist()
            shard_primary = saved["host_primary_role"].astype(np.int16)
            host_base = len(all_host_ids)
            all_host_ids.extend(shard_host_ids)
            all_host_instance_ids.extend(f"{sha}:{value}" for value in shard_host_ids)
            all_primary.extend(shard_primary.tolist())
            shard_role_offsets = saved["host_role_offsets"].astype(np.int64)
            shard_role_ids = saved["host_role_ids"].astype(np.int16)
            for host_index in range(len(shard_host_ids)):
                left, right = shard_role_offsets[host_index:host_index + 2]
                all_role_ids.extend(shard_role_ids[left:right].tolist())
                all_role_offsets.append(len(all_role_ids))
            shard_in_offsets = saved["incoming_offsets"].astype(np.int64)
            shard_in_positions = saved["incoming_host_positions"].astype(np.int64)
            shard_out_offsets = saved["outgoing_offsets"].astype(np.int64)
            shard_out_positions = saved["outgoing_host_positions"].astype(np.int64)
            for host_index in range(len(shard_host_ids)):
                left, right = shard_in_offsets[host_index:host_index + 2]
                positions = shard_in_positions[left:right]
                all_in_positions.extend((positions + host_base).tolist())
                all_in_ids.extend(shard_host_ids[int(value)] for value in positions)
                all_in_primary_roles.extend(shard_primary[positions].tolist())
                all_in_offsets.append(len(all_in_positions))
                left, right = shard_out_offsets[host_index:host_index + 2]
                positions = shard_out_positions[left:right]
                all_out_positions.extend((positions + host_base).tolist())
                all_out_ids.extend(shard_host_ids[int(value)] for value in positions)
                all_out_primary_roles.extend(shard_primary[positions].tolist())
                all_out_offsets.append(len(all_out_positions))
            all_ego_node.extend(saved["ego_node_count"].astype(np.int32).tolist())
            all_ego_edge.extend(saved["ego_edge_count"].astype(np.int32).tolist())
            all_status.extend(saved["ego_status"].astype(np.int8).tolist())
            all_direction.extend(saved["ego_direction"].astype(np.int8).tolist())
            role_nodes.append(np.asarray(saved["role_node_counts"], dtype=np.int64))
            role_edges.append(np.asarray(saved["role_edge_counts"], dtype=np.int64))
            same_total += int(saved["same_role_edge_count"])
            cross_total += int(saved["cross_role_edge_count"])
    host_matrix = np.concatenate(host_values) if host_values else np.zeros((0, REGION_DIM), np.float32)
    ego_matrix = np.concatenate(ego_values) if any(len(value) for value in ego_values) else np.zeros((0, REGION_DIM), np.float32)
    common = {
        "sha256": np.asarray(sha_values),
        "host_values": host_matrix,
        "host_offsets": np.asarray(host_offsets, dtype=np.int64),
        "real_host_count": np.asarray(real_host_count, dtype=np.int32),
        "host_was_empty": np.asarray(host_was_empty, dtype=np.int8),
    }
    atomic_npz(output / f"host_regions_{asset_tag}.npz", **common)
    atomic_npz(output / f"regions_{asset_tag}.npz", **common,
               ego_values=ego_matrix, ego_offsets=np.asarray(ego_offsets, dtype=np.int64))
    atomic_npz(
        output / f"ego_1hop_regions_{asset_tag}.npz",
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
    atomic_npz(
        output / f"global_role_graph_{asset_tag}.npz",
        sha256=np.asarray(sha_values),
        role_node_counts=np.stack(role_nodes), role_edge_counts=np.stack(role_edges),
    )
    atomic_gzip_jsonl(
        output / f"endpoint_centered_tensor_cache_{asset_tag}.jsonl.gz", endpoint_rows
    )

    nodes = np.asarray(all_ego_node, dtype=np.float64)
    edges = np.asarray(all_ego_edge, dtype=np.float64)
    status = np.asarray(all_status, dtype=np.int8)
    direction = np.asarray(all_direction, dtype=np.int8)
    total_hosts = len(status)
    nontrivial = int(np.sum(status == NONTRIVIAL_EGO))
    summary = {
        "total_apks": len(rows),
        "apks_with_host": int(np.sum(np.asarray(real_host_count) > 0)),
        "apks_without_host": int(np.sum(np.asarray(real_host_count) == 0)),
        "host_coverage": float(np.mean(np.asarray(real_host_count) > 0)),
        "total_hosts": total_hosts,
        "nontrivial_ego_hosts": nontrivial,
        "host_only_no_neighbor": int(np.sum(status == HOST_ONLY_NO_NEIGHBOR)),
        "true_extraction_failures": 0,
        "nontrivial_ego_rate": float(nontrivial / max(total_hosts, 1)),
        "ego_node_count_mean": float(np.mean(nodes)) if len(nodes) else 0.0,
        "ego_node_count_median": float(np.median(nodes)) if len(nodes) else 0.0,
        "ego_node_count_p90": float(np.percentile(nodes, 90)) if len(nodes) else 0.0,
        "ego_node_count_max": int(np.max(nodes)) if len(nodes) else 0,
        "ego_edge_count_mean": float(np.mean(edges)) if len(edges) else 0.0,
        "ego_edge_count_median": float(np.median(edges)) if len(edges) else 0.0,
        "ego_edge_count_p90": float(np.percentile(edges, 90)) if len(edges) else 0.0,
        "ego_edge_count_max": int(np.max(edges)) if len(edges) else 0,
        "same_role_ego_edge_count": same_total,
        "cross_role_ego_edge_count": cross_total,
        "incoming_only_ego_count": int(np.sum(direction == 1)),
        "outgoing_only_ego_count": int(np.sum(direction == 2)),
        "both_direction_ego_count": int(np.sum(direction == 3)),
        "no_neighbor_ego_count": int(np.sum(direction == 0)),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--priority", type=Path, required=True)
    parser.add_argument("--edge-semantics", type=Path, required=True)
    parser.add_argument("--extractor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=799)
    parser.add_argument("--asset-tag", default="2017")
    parser.add_argument("--cohort-kind", choices=("source", "future"), default="source")
    parser.add_argument("--years", default="2017")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()
    if fcntl is None:
        raise SystemExit("source extraction requires Linux fcntl writer locking")
    args.output.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output / "shards"
    failure_dir = args.output / "failure_shards"
    shard_dir.mkdir(exist_ok=True); failure_dir.mkdir(exist_ok=True)
    lock_handle = (args.output / ".exclusive_writer.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("another local-asset writer owns this output cache") from exc

    rows = read_csv(args.lineage)
    lineage_ids = [norm(row.get("sha256")) for row in rows]
    if len(rows) != args.expected or len(set(lineage_ids)) != args.expected or "" in lineage_ids:
        raise SystemExit("source lineage is not the exact unique expected cohort")
    endpoint = load_modules(args.extractor_root)
    endpoint.suppress_loguru(); endpoint.suppress_logs()
    mapping = endpoint.load_yaml(args.mapping)
    priority = endpoint.load_yaml(args.priority)
    edge_mapping = endpoint.load_yaml(args.edge_semantics)
    # Freeze provenance hashes before the long resumable extraction.  The
    # configuration is already loaded into memory at this point, so manifest
    # finalisation must not fail hours later merely because an external script
    # moved or replaced the package directory while extraction was running.
    mapping_sha256 = sha256_file(args.mapping)
    priority_sha256 = sha256_file(args.priority)
    edge_semantics_sha256 = sha256_file(args.edge_semantics)
    roles = list(mapping.get("roles", []))
    if len(roles) != ROLE_COUNT:
        raise SystemExit("semantic role vocabulary is not 31")
    # Endpoint payload IDs are alphabetical (see endpoint_centered_view.py),
    # while compact/S7 features follow the declared mapping order.  Remap once
    # here so the cached Host role columns and the 31-node global role graph
    # share an explicit, audited semantic coordinate system.
    payload_roles = sorted(roles)
    model_role_index = {role: index for index, role in enumerate(roles)}
    payload_to_model_role = np.asarray(
        [model_role_index[role] for role in payload_roles], dtype=np.int16
    )
    role_tier = endpoint.role_tiers(priority)
    entry_policy = endpoint.GraphPolicy(H=4, K_out=8, P_cap=8, E_cap=2000, max_states=20000)
    endpoint_policy = endpoint.EndpointViewPolicy(
        upstream_hops=2, downstream_hops=1, K_in=8, K_out=4,
        endpoint_cap=256, node_cap=2048, edge_cap=4096,
        reachability_max_states=100000,
    )
    pending = [row for row in rows if not (shard_dir / f"{norm(row['sha256'])}.npz").is_file()]
    print(f"local asset extraction: total={len(rows)} completed={len(rows)-len(pending)} pending={len(pending)}", flush=True)
    for index, row in enumerate(pending, 1):
        sha = norm(row["sha256"])
        shard_path = shard_dir / f"{sha}.npz"
        if shard_path.exists():
            continue
        result = endpoint.analyze_one(
            row, list(mapping.get("rules", [])), priority,
            list(edge_mapping.get("edge_rules", [])), roles, role_tier,
            entry_policy, endpoint_policy,
        )
        if result.get("status") != "OK":
            atomic_text(failure_dir / f"{sha}.json", json.dumps({
                "sha256": sha, "error_type": result.get("error_type", ""),
                "error_message": result.get("error_message", ""),
            }, ensure_ascii=False, indent=2) + "\n")
        else:
            local = reconstruct(result["payload"], payload_to_model_role)
            atomic_npz(
                shard_path,
                sha256=np.asarray(sha),
                endpoint_record_json=np.asarray(json.dumps(result, ensure_ascii=False, separators=(",", ":"))),
                **local,
            )
            failure = failure_dir / f"{sha}.json"
            if failure.exists():
                failure.unlink()
        if index % max(1, args.checkpoint_every) == 0 or index == len(pending):
            write_manifest(args.output / "done_manifest.json", lineage_ids, shard_dir, failure_dir)
        if index % max(1, args.progress_every) == 0 or index == len(pending):
            print(f"processed {index}/{len(pending)}", flush=True)

    write_manifest(args.output / "done_manifest.json", lineage_ids, shard_dir, failure_dir)
    shard_ids = {item.stem for item in shard_dir.glob("*.npz")}
    failure_ids = {item.stem for item in failure_dir.glob("*.json")}
    missing = sorted(set(lineage_ids) - shard_ids)
    outside = sorted(shard_ids - set(lineage_ids))
    failures = []
    for sha in sorted(failure_ids):
        failures.append(json.loads((failure_dir / f"{sha}.json").read_text(encoding="utf-8")))
    with (args.output / "failures.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sha256", "error_type", "error_message"])
        writer.writeheader(); writer.writerows(failures)
    hard = {
        "unique_apks_eq_expected": len(shard_ids) == args.expected,
        "duplicate_apks_eq_0": len(list(shard_dir.glob("*.npz"))) == len(shard_ids),
        "missing_apks_eq_0": not missing,
        "outside_lineage_apks_eq_0": not outside,
        "true_failures_eq_0": not failures,
    }
    if not all(hard.values()):
        atomic_text(args.output / "extraction_summary.txt", json.dumps({
            "gate_passed": False, "hard_assertions": hard,
            "missing": missing, "outside": outside, "failures": failures,
        }, ensure_ascii=False, indent=2) + "\n")
        raise SystemExit("local asset completeness gate failed; OOF remains blocked")

    endpoint_rows = []
    for sha in lineage_ids:
        with np.load(shard_dir / f"{sha}.npz", allow_pickle=False) as saved:
            endpoint_rows.append(json.loads(str(saved["endpoint_record_json"])))
    summary = merge_assets(rows, shard_dir, args.output, endpoint_rows, args.asset_tag)
    scope_lines = [
        "CARE-Droid Ego neighbor scope audit", "",
        "Current definition: sensitive-endpoint directed 1-hop Host Ego.", "",
        "1. Can ordinary non-sensitive methods enter an Ego?",
        "   Yes, but only when the method is retained in the bounded endpoint-centred graph and itself qualifies as a Host by calling at least one selected sensitive endpoint.",
        "2. Can only sensitive APIs/endpoints become neighbors?",
        "   No. In the current implementation sensitive endpoint nodes are not the ordinary neighbor population. Ego neighbors are other Host instances; a Host may be a non-sensitive application method.",
        "3. What edges are used?",
        "   Raw function-call edges retained by the bounded sensitive-endpoint-centred subgraph, followed by a Host-to-Host filter. This is not the complete raw call graph.",
        "4. What does same-role -> same-role mean?",
        "   Two distinct Host instance IDs whose primary semantic roles match. Self edges are removed; role labels are never used as node identity.", "",
        "Required manuscript name: sensitive-endpoint directed 1-hop Ego.",
        "Forbidden overclaim: complete function-call-graph one-hop context.",
    ]
    atomic_text(args.output / "ego_neighbor_scope_audit.txt", "\n".join(scope_lines) + "\n")
    cohort_years = [int(value.strip()) for value in args.years.split(",") if value.strip()]
    manifest = {
        "schema_version": "2.0.0",
        "cohort_kind": args.cohort_kind,
        "cohort_years": cohort_years,
        "source_only": args.cohort_kind == "source",
        "future_years_loaded": args.cohort_kind == "future",
        "future_diagnostic_only": args.cohort_kind == "future",
        "lineage_sha256": sha256_file(args.lineage),
        "mapping_sha256": mapping_sha256,
        "priority_sha256": priority_sha256,
        "edge_semantics_sha256": edge_semantics_sha256,
        "neighbor_scope": "sensitive-endpoint directed 1-hop Host Ego",
        "payload_role_order": payload_roles,
        "model_role_order": roles,
        "payload_to_model_role": payload_to_model_role.tolist(),
        "region_dim": REGION_DIM,
        "status_codes": {"0": "HOST_ONLY_NO_NEIGHBOR", "1": "NONTRIVIAL_EGO"},
        "hard_assertions": hard,
        "summary": summary,
        "writer_lock": "fcntl exclusive nonblocking",
        "checkpoint": "one atomic NPZ shard per APK; atomic done manifest every N APKs",
    }
    atomic_text(args.output / "local_asset_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    summary_lines = [
        f"CARE-Droid {args.cohort_kind} local assets v2", "",
        f"All {args.expected} APKs for years {cohort_years} parsed successfully.",
    ]
    summary_lines.extend(f"{key}: {value}" for key, value in summary.items())
    summary_lines.extend(["", "Hard assertions:"])
    summary_lines.extend(f"  {key}: {str(value).lower()}" for key, value in hard.items())
    atomic_text(args.output / "extraction_summary.txt", "\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines), flush=True)


if __name__ == "__main__":
    main()
