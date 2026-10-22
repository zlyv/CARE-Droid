#!/usr/bin/env python3
"""Directed Host-graph one-hop packing and lightweight embedding fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn

RAW_DIM = 74
META_DIM = 3


@dataclass
class ApkEgoMeta:
    sha256: str
    host_ids: list[str]
    primary_roles: list[int]
    incoming: list[list[int]]
    outgoing: list[list[int]]
    status: list[int]
    node_count: list[int]
    edge_count: list[int]
    real_host_count: int


def pack(host: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    host = np.asarray(host, dtype=np.float32)
    width = host.shape[1]
    result = np.zeros((len(host) + len(edges), width + META_DIM), dtype=np.float32)
    result[:len(host), :width] = host
    result[:len(host), width] = 0.0
    for offset, (source, target) in enumerate(edges, start=len(host)):
        result[offset, width] = 1.0
        result[offset, width + 1] = float(source)
        result[offset, width + 2] = float(target)
    return result


def unpack(values: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < RAW_DIM + META_DIM:
        raise ValueError("packed Host/Ego bag has invalid shape")
    width = values.shape[1] - META_DIM
    row_type = values[:, width]
    host = values[row_type < 0.5, :width]
    edge_rows = values[row_type >= 0.5]
    edges = [(int(round(row[width + 1])), int(round(row[width + 2]))) for row in edge_rows]
    if any(s < 0 or t < 0 or s >= len(host) or t >= len(host) or s == t for s, t in edges):
        raise ValueError("packed Host/Ego edge is outside its APK bag")
    return host, edges


def repack_augmented(augmented: list[np.ndarray], originals: list[np.ndarray]) -> list[np.ndarray]:
    result = []
    for values, original in zip(augmented, originals):
        _, edges = unpack(original)
        result.append(pack(np.asarray(values, dtype=np.float32), edges))
    return result


def make_fit_adapter(original_fit):
    """Keep scaler/prototypes Host-only, then restore frozen directed edges."""
    def fit(train_bags, test_bags, labels, args, seed):
        train_host = [unpack(value)[0] for value in train_bags]
        test_host = [unpack(value)[0] for value in test_bags]
        train_aug, test_aug, audit = original_fit(train_host, test_host, labels, args, seed)
        audit = dict(audit)
        audit.update({
            "ego_changes_scaler": False, "ego_changes_prototypes": False,
            "ego_edge_scope": "directed Host-graph one-hop",
        })
        return repack_augmented(train_aug, train_bags), repack_augmented(test_aug, test_bags), audit
    return fit


def load_source_packed_bags(rows, args, asset_root: Path | None = None):
    root = Path(asset_root or "artifacts/care-source-local-assets-v2")
    host_path = root / "host_regions_2017.npz"
    ego_path = root / "ego_1hop_regions_2017.npz"
    if not host_path.is_file() or not ego_path.is_file():
        raise RuntimeError("BLOCKED_BY_CONTRACT: source Host/Ego cache is missing")
    expected = [str(row["sha256"]).strip().lower() for row in rows]
    with np.load(host_path, allow_pickle=False) as h, np.load(ego_path, allow_pickle=False) as e:
        ids = h["sha256"].astype(str).tolist()
        eids = e["sha256"].astype(str).tolist()
        if ids != expected or eids != expected:
            raise RuntimeError("BLOCKED_BY_CONTRACT: Host/Ego cache lineage/order differs from source")
        host_values = np.asarray(h["host_values"], np.float32)
        host_offsets = np.asarray(h["host_offsets"], np.int64)
        real_count = np.asarray(h["real_host_count"], np.int64)
        ego_offsets = np.asarray(e["ego_offsets"], np.int64)
        host_ids = e["host_ids"].astype(str)
        roles = np.asarray(e["host_primary_role"], np.int64)
        status = np.asarray(e["ego_status"], np.int64)
        nodes = np.asarray(e["ego_node_count"], np.int64)
        edges_n = np.asarray(e["ego_edge_count"], np.int64)
        in_offsets = np.asarray(e["incoming_offsets"], np.int64)
        out_offsets = np.asarray(e["outgoing_offsets"], np.int64)
        in_pos = np.asarray(e["incoming_host_positions"], np.int64)
        out_pos = np.asarray(e["outgoing_host_positions"], np.int64)
        # Load and validate the cached aggregate even though the model uses the
        # stricter encoded-neighbour mean required by this experiment.
        ego_values = np.asarray(e["ego_values"], np.float32)
        if ego_values.shape != (int(ego_offsets[-1]), RAW_DIM):
            raise RuntimeError("BLOCKED_BY_CONTRACT: ego_values schema mismatch")
        bags, metadata, audits = [], [], []
        for i, sha in enumerate(expected):
            hb, he = int(host_offsets[i]), int(host_offsets[i + 1])
            eb, ee = int(ego_offsets[i]), int(ego_offsets[i + 1])
            host = host_values[hb:he]
            nreal = int(real_count[i])
            if ee - eb != nreal or (nreal and len(host) != nreal) or (not nreal and len(host) != 1):
                raise RuntimeError("BLOCKED_BY_CONTRACT: missing-region Host/Ego alignment failed")
            incoming, outgoing, edge_set = [], [], set()
            for g in range(eb, ee):
                inc = (in_pos[in_offsets[g]:in_offsets[g+1]] - eb).astype(int).tolist()
                out = (out_pos[out_offsets[g]:out_offsets[g+1]] - eb).astype(int).tolist()
                if any(v < 0 or v >= nreal for v in inc + out):
                    raise RuntimeError("BLOCKED_BY_CONTRACT: cross-APK Host neighbor detected")
                incoming.append(inc); outgoing.append(out)
                edge_set.update((g-eb, v) for v in out)
            bags.append(pack(host, sorted(edge_set)))
            meta = ApkEgoMeta(
                sha, host_ids[eb:ee].tolist() if nreal else ["__missing_host__"],
                roles[eb:ee].astype(int).tolist() if nreal else [-1],
                incoming if nreal else [[]], outgoing if nreal else [[]],
                status[eb:ee].astype(int).tolist() if nreal else [0],
                nodes[eb:ee].astype(int).tolist() if nreal else [1],
                edges_n[eb:ee].astype(int).tolist() if nreal else [0], nreal,
            )
            metadata.append(meta)
            audits.append({
                "sha256": sha, "host_region_n": len(host), "real_host_region_n": nreal,
                "nontrivial_ego_n": int(sum(meta.status)), "host_was_empty": int(nreal == 0),
            })
    return bags, audits, True, metadata


class DirectedHostEgoFusion(nn.Module):
    """One small projection and one scalar gate; no new attention or GNN."""
    def __init__(self, local_dim: int):
        super().__init__()
        self.ego_projection = nn.Sequential(nn.Linear(2 * local_dim, local_dim), nn.ReLU())
        self.ego_gate = nn.Linear(2 * local_dim, 1)

    def forward(self, host_local: torch.Tensor, edges: torch.Tensor):
        n, dim = host_local.shape
        incoming = torch.zeros_like(host_local); outgoing = torch.zeros_like(host_local)
        in_count = torch.zeros(n, device=host_local.device); out_count = torch.zeros(n, device=host_local.device)
        if edges.numel():
            source, target = edges[:, 0].long(), edges[:, 1].long()
            incoming.index_add_(0, target, host_local[source]); in_count.index_add_(0, target, torch.ones_like(target, dtype=host_local.dtype))
            outgoing.index_add_(0, source, host_local[target]); out_count.index_add_(0, source, torch.ones_like(source, dtype=host_local.dtype))
        incoming = incoming / in_count.clamp_min(1.0)[:, None]
        outgoing = outgoing / out_count.clamp_min(1.0)[:, None]
        ego = self.ego_projection(torch.cat([incoming, outgoing], dim=1))
        gamma = torch.sigmoid(self.ego_gate(torch.cat([host_local, ego], dim=1))).reshape(-1)
        used = (in_count + out_count) > 0
        local = torch.where(used[:, None], host_local + gamma[:, None] * ego, host_local)
        return local, ego, gamma, used, in_count, out_count


def split_packed_tensor(regions: torch.Tensor):
    width = regions.shape[1] - META_DIM
    host_mask = regions[:, width] < 0.5
    host = regions[host_mask, :width]
    edge_rows = regions[~host_mask]
    edges = edge_rows[:, width + 1:width + 3].long() if len(edge_rows) else torch.empty((0, 2), dtype=torch.long, device=regions.device)
    return host, edges

