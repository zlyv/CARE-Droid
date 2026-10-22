#!/usr/bin/env python3
"""Controlled GraphGlobal-path models for strict-source CARE-Droid.

The 31-role Node/Edge encoder is identical in every arm.  GraphGlobal is
encoded separately and can affect (Q) Host attention, (D) the bounded-delta
head, both, or neither.  The model width and instantiated parameter set are
identical across arms; the two path switches are fixed non-trainable buffers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


ARMS = {
    "N0_NO_GRAPHGLOBAL": (0.0, 0.0),
    "D_DELTA_ONLY": (0.0, 1.0),
    "Q_QUERY_ONLY": (1.0, 0.0),
    "QD_QUERY_AND_DELTA": (1.0, 1.0),
}


@dataclass
class LocalEvidenceCache:
    """Values after Host selection, suitable for the direct-bypass audit."""

    role_token: torch.Tensor
    critical: torch.Tensor
    attended: torch.Tensor
    scalar_summary: torch.Tensor
    attention: torch.Tensor
    score: torch.Tensor
    graph_entropy: torch.Tensor


def make_graphglobal_path_class(v4, arm: str):
    if arm not in ARMS:
        raise KeyError(arm)
    query_switch, delta_switch = ARMS[arm]

    class GraphGlobalPathHostAnomaly(nn.Module):
        graphglobal_path_arm = arm

        def __init__(self, global_dim: int, variant: str, args):
            super().__init__()
            if str(variant) != "RGCondAttn":
                raise ValueError("GraphGlobal path audit supports RGCondAttn only")
            if int(global_dim) != 24:
                raise RuntimeError(f"Expected Global24 width 24, got {global_dim}")
            self.max_delta = float(args.max_delta)
            graph_dim = int(args.graph_dim)
            local_dim = int(args.embedding_dim)
            hidden_dim = int(args.hidden_dim)

            # Node/Edge-only semantic encoder. GraphGlobal is deliberately absent.
            self.role_embedding = nn.Embedding(v4.ROLE_COUNT, graph_dim)
            self.node_input = nn.Linear(v4.ROLE_NODE_WIDTH, graph_dim)
            self.blocks = nn.ModuleList([
                v4.EdgeBiasedGraphBlock(
                    graph_dim, int(args.graph_heads), float(args.dropout)
                )
                for _ in range(int(args.graph_layers))
            ])
            self.pool_score = nn.Linear(graph_dim, 1)

            # Separate GraphGlobal representation, shared by the Q/D switches.
            self.graphglobal_encoder = nn.Sequential(
                nn.Linear(v4.GRAPH_GLOBAL_DIM, graph_dim),
                nn.LayerNorm(graph_dim),
                nn.GELU(),
            )
            self.global_to_query = nn.Linear(graph_dim, local_dim, bias=False)
            self.global_to_delta = nn.Linear(graph_dim, graph_dim, bias=False)

            self.region_encoder = nn.Sequential(
                nn.Linear(v4.AUGMENTED_REGION_DIM, hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(args.dropout)),
                nn.Linear(hidden_dim, local_dim),
                nn.ReLU(),
            )
            self.local_risk = nn.Linear(local_dim, 1)
            self.role_query = nn.Linear(graph_dim, local_dim)
            self.local_key = nn.Linear(local_dim, local_dim)
            self.residual_beta_raw = nn.Parameter(torch.tensor(0.0))

            # Fixed 113-D correction interface:
            # Global24(24)+base logit(1)+role slot(32)+critical(24)
            # +attended(24)+summary(8).
            fused_dim = int(global_dim) + 1 + graph_dim + 2 * local_dim + 8
            if fused_dim != 113:
                raise RuntimeError(f"Correction width {fused_dim} != 113")
            self.delta_head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(args.dropout)),
                nn.Linear(hidden_dim, 1),
            )
            self.register_buffer(
                "graphglobal_query_switch",
                torch.tensor(float(query_switch)),
                persistent=True,
            )
            self.register_buffer(
                "graphglobal_delta_switch",
                torch.tensor(float(delta_switch)),
                persistent=True,
            )

        def encode_role_graph(self, node, edge):
            role_index = torch.arange(v4.ROLE_COUNT, device=node.device)
            hidden = self.node_input(node) + self.role_embedding(role_index)
            for block in self.blocks:
                hidden = block(hidden, edge)
            pool_weight = torch.softmax(self.pool_score(hidden).reshape(-1), dim=0)
            role_token = torch.sum(pool_weight[:, None] * hidden, dim=0)
            entropy = -torch.sum(
                pool_weight * torch.log(pool_weight.clamp_min(1e-8))
            ) / math.log(v4.ROLE_COUNT)
            return role_token, entropy

        def select_local_evidence(self, graph, regions):
            role_token, graph_entropy = self.encode_role_graph(
                graph["node"], graph["edge"]
            )
            graphglobal = self.graphglobal_encoder(graph["global"].reshape(-1))
            local = self.region_encoder(regions)
            residual = torch.log1p(torch.clamp(regions[:, -1], min=0.0))
            score = self.local_risk(local).reshape(-1)
            score = score + nn.functional.softplus(self.residual_beta_raw) * residual

            # Role structure always provides the base query. GraphGlobal can add
            # only the explicitly switched query term.
            query = self.role_query(role_token)
            query = query + self.graphglobal_query_switch * self.global_to_query(
                graphglobal
            )
            key = self.local_key(local)
            score = score + torch.mv(key, query) / math.sqrt(local.shape[1])
            attention = torch.softmax(score, dim=0)
            attended = torch.sum(attention[:, None] * local, dim=0)
            critical_index = torch.argmax(score)
            critical = local[critical_index]
            attention_entropy = -torch.sum(
                attention * torch.log(attention.clamp_min(1e-8))
            ) / math.log(max(int(len(attention)), 2))
            prototype_summary = torch.stack([
                torch.mean(regions[:, -4]),
                torch.mean(regions[:, -3]),
                torch.mean(regions[:, -2]),
                torch.max(regions[:, -1]),
            ])
            scalar_summary = torch.cat([
                score[critical_index].reshape(1),
                torch.mean(score).reshape(1),
                attention_entropy.reshape(1),
                torch.max(attention).reshape(1),
                prototype_summary,
            ])
            cache = LocalEvidenceCache(
                role_token=role_token,
                critical=critical,
                attended=attended,
                scalar_summary=scalar_summary,
                attention=attention,
                score=score,
                graph_entropy=graph_entropy,
            )
            return cache, graphglobal

        def delta_from_selected_evidence(
            self, global_features, base_logit, cache, graphglobal
        ):
            # The D path occupies the existing 32-D role slot by a residual
            # projection. Q-only has switch=0, so no direct GraphGlobal value is
            # present in the correction vector after Host selection is frozen.
            role_slot = cache.role_token + (
                self.graphglobal_delta_switch * self.global_to_delta(graphglobal)
            )
            fused = torch.cat([
                global_features.reshape(-1),
                base_logit.reshape(1),
                role_slot,
                cache.critical,
                cache.attended,
                cache.scalar_summary,
            ])
            if fused.numel() != 113:
                raise RuntimeError(f"Correction input width {fused.numel()} != 113")
            raw_delta = self.delta_head(fused).reshape(())
            bounded_delta = self.max_delta * torch.tanh(raw_delta)
            return raw_delta, bounded_delta, fused

        def forward_with_trace(self, global_features, base_logit, graph, regions):
            cache, graphglobal = self.select_local_evidence(graph, regions)
            raw_delta, bounded_delta, fused = self.delta_from_selected_evidence(
                global_features, base_logit, cache, graphglobal
            )
            final_logit = base_logit.reshape(()) + bounded_delta
            diagnostic = torch.cat([
                bounded_delta.reshape(1),
                cache.score[torch.argmax(cache.score)].reshape(1),
                torch.mean(cache.score).reshape(1),
                cache.scalar_summary[2].reshape(1),
                torch.max(cache.attention).reshape(1),
                cache.scalar_summary[2].reshape(1),
                cache.scalar_summary[-4:],
                cache.graph_entropy.reshape(1),
            ])
            trace = {
                "role_token": cache.role_token,
                "graphglobal_embedding": graphglobal,
                "attention": cache.attention,
                "critical": cache.critical,
                "attended": cache.attended,
                "correction_input": fused,
                "raw_delta": raw_delta,
                "bounded_delta": bounded_delta,
            }
            return final_logit, diagnostic, trace

        def forward(self, global_features, base_logit, graph, regions):
            final_logit, diagnostic, _ = self.forward_with_trace(
                global_features, base_logit, graph, regions
            )
            return final_logit, diagnostic

    GraphGlobalPathHostAnomaly.__name__ = "GraphGlobalPath_" + arm
    return GraphGlobalPathHostAnomaly

