#!/usr/bin/env python3
"""Capacity-matched correction-input ablations for Host-only CARE-Droid."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import nn


BLOCKS = OrderedDict([
("global24", (0, 24)),
("stage1_logit", (24, 25)),
("role_graph_token", (25, 57)),
("global_context", (57, 89)),
("critical_host", (89, 113)),
("attended_host", (113, 137)),
("scalar_summary", (137, 145)),
])

ARMS = OrderedDict([
    ("FULL", ()),
    ("DROP_GLOBAL24", ("global24",)),
    ("DROP_STAGE1_LOGIT", ("stage1_logit",)),
    ("DROP_ROLE_GRAPH", ("role_graph_token",)),
    ("DROP_CRITICAL_HOST", ("critical_host",)),
    ("DROP_ATTENDED_HOST", ("attended_host",)),
    ("DROP_SCALAR_SUMMARY", ("scalar_summary",)),
    # Sufficiency controls. They are secondary and must not replace LOO results.
    ("STAGE1_CONTEXT_ONLY", (
        "role_graph_token", "global_context", "critical_host", "attended_host", "scalar_summary",
    )),
    ("STRUCTURE_ONLY", ("global24", "stage1_logit")),
])


def correction_input_mask(arm: str) -> torch.Tensor:
    if arm not in ARMS:
        raise KeyError(arm)
    mask = torch.ones(145, dtype=torch.float32)
    for block in ARMS[arm]:
        left, right = BLOCKS[block]
        mask[left:right] = 0.0
    return mask


def make_masked_host_class(baseline_module, arm: str):
    """Return the exact Host-only model with a fixed, parameter-free input mask."""

    Base = baseline_module.DedupBoundedRGCondAttn
    fixed_mask = correction_input_mask(arm)

    class MaskedHostAnomaly(Base):
        correction_ablation_arm = arm

        def __init__(self, global_dim, variant, args):
            super().__init__(global_dim, variant, args)
            if int(global_dim) != 24:
                raise RuntimeError(f"Expected Global24, received width={global_dim}")
            self.register_buffer("correction_mask", fixed_mask.clone(), persistent=True)

        def forward(self, global_features, base_logit, graph, regions):
            graph_token, graph_entropy, global_context = self.graph_encoder(
                graph["node"], graph["edge"], graph["global"]
            )
            local = self.region_encoder(regions)
            local_score = self.local_risk(local).reshape(-1)
            residual = torch.log1p(torch.clamp(regions[:, -1], min=0.0))
            score = local_score + nn.functional.softplus(self.residual_beta_raw) * residual
            query = self.graph_query(graph_token)
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
            fused = torch.cat([
                global_features.reshape(-1),
                base_logit.reshape(1),
                graph_token,
                global_context,
                critical,
                attended,
                scalar_summary,
            ])
            if fused.numel() != 113 + global_context.numel():
                raise RuntimeError(f"Correction input width {fused.numel()} != 113")
            masked = fused * self.correction_mask.to(dtype=fused.dtype)
            raw_delta = self.delta_head(masked).reshape(())
            bounded_delta = self.max_delta * torch.tanh(raw_delta)
            final_logit = base_logit.reshape(()) + bounded_delta
            diagnostic = torch.cat([
                bounded_delta.reshape(1),
                score[critical_index].reshape(1),
                torch.mean(score).reshape(1),
                attention_entropy.reshape(1),
                torch.max(attention).reshape(1),
                attention_entropy.reshape(1),
                prototype_summary,
                graph_entropy.reshape(1),
            ])
            return final_logit, diagnostic

    MaskedHostAnomaly.__name__ = f"HostAnomaly_{arm}"
    return MaskedHostAnomaly

