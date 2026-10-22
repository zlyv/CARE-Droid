#!/usr/bin/env python3
"""Bit-for-bit arithmetic copy of frozen M1 with read-only capture hooks."""
from __future__ import annotations

import math
import torch
from torch import nn

from host_ego_encoder import DirectedHostEgoFusion, split_packed_tensor


def instrumented_class(baseline_module, v4):
    Base = baseline_module.DedupBoundedRGCondAttn

    class InstrumentedHostEgoAnomaly(Base):
        def __init__(self, global_dim, variant, args):
            super().__init__(global_dim, variant, args)
            self.host_ego_fusion = DirectedHostEgoFusion(int(args.embedding_dim))
            self.last_internal_capture = None

        def forward(self, global_features, base_logit, graph, regions):
            # Keep this arithmetic in the exact order of frozen HostEgoAnomaly.
            host_regions, edges = split_packed_tensor(regions)
            graph_token, graph_entropy = self.graph_encoder(
                graph["node"], graph["edge"], graph["global"])
            host_local = self.region_encoder(host_regions)
            local, ego, gamma, used, in_count, out_count = self.host_ego_fusion(
                host_local, edges)
            local_score = self.local_risk(local).reshape(-1)
            residual = torch.log1p(torch.clamp(host_regions[:, -1], min=0.0))
            score = local_score + nn.functional.softplus(self.residual_beta_raw) * residual
            query = self.graph_query(graph_token)
            key = self.local_key(local)
            score = score + torch.mv(key, query) / math.sqrt(local.shape[1])
            attention = torch.softmax(score, dim=0)
            attended = torch.sum(attention[:, None] * local, dim=0)
            critical_index = torch.argmax(score)
            critical = local[critical_index]
            attention_entropy = -torch.sum(
                attention * torch.log(attention.clamp_min(1e-8))) / math.log(max(int(len(attention)), 2))
            prototype_summary = torch.stack([
                torch.mean(host_regions[:, -4]), torch.mean(host_regions[:, -3]),
                torch.mean(host_regions[:, -2]), torch.max(host_regions[:, -1]),
            ])
            scalar_summary = torch.cat([
                score[critical_index].reshape(1), torch.mean(score).reshape(1),
                attention_entropy.reshape(1), torch.max(attention).reshape(1), prototype_summary,
            ])
            fused = torch.cat([
                global_features.reshape(-1), base_logit.reshape(1), graph_token,
                critical, attended, scalar_summary,
            ])
            raw_delta = self.delta_head(fused).reshape(())
            bounded_delta = self.max_delta * torch.tanh(raw_delta)
            final_logit = base_logit.reshape(()) + bounded_delta
            diagnostic = torch.cat([
                bounded_delta.reshape(1), score[critical_index].reshape(1),
                torch.mean(score).reshape(1), attention_entropy.reshape(1),
                torch.max(attention).reshape(1), attention_entropy.reshape(1),
                prototype_summary, graph_entropy.reshape(1),
                torch.sigmoid(raw_delta).reshape(1),
            ])

            # Detach only after the frozen forward result has been constructed.
            def array(value):
                return value.detach().cpu().numpy()

            self.last_internal_capture = {
                "raw_delta": float(raw_delta.detach().cpu()),
                "bounded_delta": float(bounded_delta.detach().cpu()),
                "attention_logit": array(score),
                "attention_weight": array(attention),
                "critical_index": int(critical_index.detach().cpu()),
                "external_background_value": array(host_regions[:, -4]),
                "internal_background_value": array(host_regions[:, -3]),
                "shared_commonness_value": array(host_regions[:, -2]),
                "anomaly_score": array(host_regions[:, -1]),
                "host_embedding_norm": array(torch.linalg.vector_norm(host_local, dim=1)),
                "ego_embedding_norm": array(torch.linalg.vector_norm(ego, dim=1)),
                "gamma": array(gamma),
                "gamma_ego_norm": array(gamma * torch.linalg.vector_norm(ego, dim=1)),
                "local_embedding_norm": array(torch.linalg.vector_norm(local, dim=1)),
                "ego_used": array(used).astype(int),
                "incoming_count": array(in_count),
                "outgoing_count": array(out_count),
                "attention_entropy": float(attention_entropy.detach().cpu()),
                "max_attention": float(torch.max(attention).detach().cpu()),
                "graph_entropy": float(graph_entropy.detach().cpu()),
            }
            return final_logit, diagnostic

    return InstrumentedHostEgoAnomaly
