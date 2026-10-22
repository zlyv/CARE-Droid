#!/usr/bin/env python3
"""The only permitted architectural change: replace M1's scalar correction head."""
from __future__ import annotations

import math
import torch
from torch import nn


def make_dmd_class(base_class, split_packed_tensor, directed_fusion, action_classes: int):
    if action_classes not in (2, 3):
        raise ValueError("action_classes must be 2 (NoKeep) or 3 (Full)")

    class DirectionMagnitudeDecoupledBoundedCorrection(base_class):
        ACTION_CLASSES = action_classes

        def __init__(self, global_dim, variant, args):
            super().__init__(global_dim, variant, args)
            old = self.delta_head
            first = next(layer for layer in old if isinstance(layer, nn.Linear))
            self.dmd_input_dim = int(first.in_features)
            del self.delta_head
            self.host_ego_fusion = directed_fusion(int(args.embedding_dim))
            self.action_head = nn.Linear(self.dmd_input_dim, action_classes)
            self.magnitude_head = nn.Linear(self.dmd_input_dim, 1)
            self.last_dmd = None
            self.last_internal_capture = None

        def forward(self, global_features, base_logit, graph, regions):
            # This block is intentionally arithmetic-identical to frozen M1 up
            # to the construction of `fused`. Only the old scalar head changes.
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
                attention_entropy.reshape(1), torch.max(attention).reshape(1),
                prototype_summary,
            ])
            # z excludes Stage-I; u appends exactly one fold-safe Stage-I logit.
            z = torch.cat([
                global_features.reshape(-1), graph_token, critical, attended,
                scalar_summary,
            ])
            u = torch.cat([z, base_logit.reshape(1)])
            if int(u.numel()) != self.dmd_input_dim:
                raise RuntimeError(f"DMD input changed: {u.numel()} != {self.dmd_input_dim}")
            action_logits = self.action_head(u)
            action_probability = torch.softmax(action_logits, dim=0)
            if self.ACTION_CLASSES == 3:
                q_neg, q_keep, q_pos = action_probability
            else:
                q_neg, q_pos = action_probability
                q_keep = torch.full_like(q_neg, float("nan"))
            direction = q_pos - q_neg
            magnitude = self.max_delta * torch.sigmoid(self.magnitude_head(u).reshape(()))
            delta = direction * magnitude
            final_logit = base_logit.reshape(()) + delta
            diagnostic = torch.cat([
                delta.reshape(1), score[critical_index].reshape(1), torch.mean(score).reshape(1),
                attention_entropy.reshape(1), torch.max(attention).reshape(1),
                attention_entropy.reshape(1), prototype_summary, graph_entropy.reshape(1),
                direction.reshape(1), magnitude.reshape(1), action_probability.reshape(-1),
            ])
            self.last_dmd = {
                "action_logits": action_logits,
                "action_probability": action_probability,
                "q_neg": q_neg, "q_keep": q_keep, "q_pos": q_pos,
                "direction": direction, "magnitude": magnitude, "delta": delta,
            }
            # Detached logging must not affect the forward graph.
            self.last_internal_capture = {
                "gamma": gamma.detach().cpu().numpy(),
                "anomaly": residual.detach().cpu().numpy(),
                "attention": attention.detach().cpu().numpy(),
                "critical_host_index": int(critical_index.detach().cpu()),
                "used_ego": used.detach().cpu().numpy(),
                "incoming": in_count.detach().cpu().numpy(),
                "outgoing": out_count.detach().cpu().numpy(),
            }
            return final_logit, diagnostic

    DirectionMagnitudeDecoupledBoundedCorrection.__name__ = (
        "DMDNoKeep" if action_classes == 2 else "DMDFull")
    return DirectionMagnitudeDecoupledBoundedCorrection

