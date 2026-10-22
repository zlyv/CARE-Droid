#!/usr/bin/env python3
"""Frozen N0/Q GraphGlobal-path models for the future diagnostic."""
from __future__ import annotations

import math
from dataclasses import dataclass
import torch
from torch import nn

ARMS = {
    "N0_NO_GRAPHGLOBAL": (0.0, 0.0),
    "Q_QUERY_ONLY": (1.0, 0.0),
}

@dataclass
class LocalEvidenceCache:
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

    class FrozenGraphGlobalPath(nn.Module):
        graphglobal_path_arm = arm

        def __init__(self, global_dim: int, variant: str, args):
            super().__init__()
            if str(variant) != "RGCondAttn" or int(global_dim) != 24:
                raise RuntimeError("frozen Q/N0 model contract changed")
            self.max_delta = float(args.max_delta)
            graph_dim, local_dim, hidden_dim = int(args.graph_dim), int(args.embedding_dim), int(args.hidden_dim)
            self.role_embedding = nn.Embedding(v4.ROLE_COUNT, graph_dim)
            self.node_input = nn.Linear(v4.ROLE_NODE_WIDTH, graph_dim)
            self.blocks = nn.ModuleList([
                v4.EdgeBiasedGraphBlock(graph_dim, int(args.graph_heads), float(args.dropout))
                for _ in range(int(args.graph_layers))
            ])
            self.pool_score = nn.Linear(graph_dim, 1)
            self.graphglobal_encoder = nn.Sequential(
                nn.Linear(v4.GRAPH_GLOBAL_DIM, graph_dim), nn.LayerNorm(graph_dim), nn.GELU()
            )
            self.global_to_query = nn.Linear(graph_dim, local_dim, bias=False)
            self.global_to_delta = nn.Linear(graph_dim, graph_dim, bias=False)
            self.region_encoder = nn.Sequential(
                nn.Linear(v4.AUGMENTED_REGION_DIM, hidden_dim), nn.ReLU(),
                nn.Dropout(float(args.dropout)), nn.Linear(hidden_dim, local_dim), nn.ReLU(),
            )
            self.local_risk = nn.Linear(local_dim, 1)
            self.role_query = nn.Linear(graph_dim, local_dim)
            self.local_key = nn.Linear(local_dim, local_dim)
            self.residual_beta_raw = nn.Parameter(torch.tensor(0.0))
            fused_dim = int(global_dim) + 1 + graph_dim + 2 * local_dim + 8
            if fused_dim != 113:
                raise RuntimeError(f"correction width changed: {fused_dim}")
            self.delta_head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim), nn.ReLU(), nn.Dropout(float(args.dropout)), nn.Linear(hidden_dim, 1)
            )
            self.register_buffer("graphglobal_query_switch", torch.tensor(query_switch), persistent=True)
            self.register_buffer("graphglobal_delta_switch", torch.tensor(delta_switch), persistent=True)

        def encode_role_graph(self, node, edge):
            index = torch.arange(v4.ROLE_COUNT, device=node.device)
            hidden = self.node_input(node) + self.role_embedding(index)
            for block in self.blocks:
                hidden = block(hidden, edge)
            weight = torch.softmax(self.pool_score(hidden).reshape(-1), dim=0)
            token = torch.sum(weight[:, None] * hidden, dim=0)
            entropy = -torch.sum(weight * torch.log(weight.clamp_min(1e-8))) / math.log(v4.ROLE_COUNT)
            return token, entropy

        def select_local_evidence(self, graph, regions):
            role_token, graph_entropy = self.encode_role_graph(graph["node"], graph["edge"])
            graphglobal = self.graphglobal_encoder(graph["global"].reshape(-1))
            local = self.region_encoder(regions)
            residual = torch.log1p(torch.clamp(regions[:, -1], min=0.0))
            score = self.local_risk(local).reshape(-1)
            score = score + nn.functional.softplus(self.residual_beta_raw) * residual
            query = self.role_query(role_token)
            query = query + self.graphglobal_query_switch * self.global_to_query(graphglobal)
            score = score + torch.mv(self.local_key(local), query) / math.sqrt(local.shape[1])
            attention = torch.softmax(score, dim=0)
            attended = torch.sum(attention[:, None] * local, dim=0)
            critical_index = torch.argmax(score)
            critical = local[critical_index]
            attention_entropy = -torch.sum(attention * torch.log(attention.clamp_min(1e-8))) / math.log(max(len(attention), 2))
            proto = torch.stack([
                torch.mean(regions[:, -4]), torch.mean(regions[:, -3]),
                torch.mean(regions[:, -2]), torch.max(regions[:, -1]),
            ])
            summary = torch.cat([
                score[critical_index].reshape(1), torch.mean(score).reshape(1),
                attention_entropy.reshape(1), torch.max(attention).reshape(1), proto,
            ])
            return LocalEvidenceCache(role_token, critical, attended, summary, attention, score, graph_entropy), graphglobal

        def forward(self, global_features, base_logit, graph, regions):
            cache, graphglobal = self.select_local_evidence(graph, regions)
            role_slot = cache.role_token + self.graphglobal_delta_switch * self.global_to_delta(graphglobal)
            fused = torch.cat([
                global_features.reshape(-1), base_logit.reshape(1), role_slot,
                cache.critical, cache.attended, cache.scalar_summary,
            ])
            if fused.numel() != 113:
                raise RuntimeError("correction input width is not 113")
            bounded_delta = self.max_delta * torch.tanh(self.delta_head(fused).reshape(()))
            final_logit = base_logit.reshape(()) + bounded_delta
            diagnostic = torch.cat([
                bounded_delta.reshape(1), cache.score[torch.argmax(cache.score)].reshape(1),
                torch.mean(cache.score).reshape(1), cache.scalar_summary[2].reshape(1),
                torch.max(cache.attention).reshape(1), cache.scalar_summary[2].reshape(1),
                cache.scalar_summary[-4:], cache.graph_entropy.reshape(1),
            ])
            return final_logit, diagnostic

    FrozenGraphGlobalPath.__name__ = "FrozenGraphGlobalPath_" + arm
    return FrozenGraphGlobalPath
