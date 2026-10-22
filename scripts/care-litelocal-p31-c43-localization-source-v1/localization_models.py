#!/usr/bin/env python3
"""Capacity-matched C43 localization variants for strict-2017 P31-DI25.

The correction vector, C43 Host core, Relation31, backgrounds, bounded head,
and parameterization stay unchanged.  Two fixed, parameter-free factors are
varied: hard argmax versus top-3 critical pooling, and raw Relation31 query
scores versus mean-preserving local-evidence alignment.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import nn

from graphglobal_path_models import LocalEvidenceCache, make_graphglobal_path_class


CORE_WIDTH = 47
RELATION_WIDTH = 31

ARM_SPECS = OrderedDict([
    ("L1_TOPK3_RAW", {"topk": True, "aligned": False}),
    ("L2_ARGMAX_ALIGNED", {"topk": False, "aligned": True}),
    ("L3_TOPK3_ALIGNED", {"topk": True, "aligned": True}),
])


def align_relation_score(base_score, relation_score, enabled: bool):
    """Gate Relation31 by within-APK non-relation evidence without new weights.

    The factor is 2*sigmoid(z), so a one-Host or constant-score APK has factor
    exactly one.  Above-average locally supported Hosts can retain/amplify the
    relation term; below-average Hosts are downweighted.  This prevents a
    sparse structural relation alone from dominating localization.
    """
    if not enabled:
        return relation_score
    centered = base_score - torch.mean(base_score)
    scale = torch.sqrt(torch.mean(centered * centered) + 1e-6)
    support = 2.0 * torch.sigmoid(centered / scale)
    return relation_score * support


def critical_pool(local, score, attention, use_topk: bool):
    if not use_topk:
        return local[torch.argmax(score)]
    k = min(3, int(local.shape[0]))
    index = torch.topk(score, k=k, largest=True, sorted=False).indices
    weight = attention[index]
    weight = weight / weight.sum().clamp_min(1e-12)
    return torch.sum(weight[:, None] * local[index], dim=0)


def make_localization_class(v4, arm: str):
    if arm not in ARM_SPECS:
        raise KeyError(arm)
    spec = ARM_SPECS[arm]
    Base = make_graphglobal_path_class(v4, "Q_QUERY_ONLY")

    class C43Localization(Base):
        localization_arm = arm

        def __init__(self, global_dim, variant, args):
            super().__init__(global_dim, variant, args)
            hidden = int(args.hidden_dim)
            local = int(args.embedding_dim)
            self.region_encoder = nn.Sequential(
                nn.Linear(CORE_WIDTH, hidden), nn.ReLU(),
                nn.Dropout(float(args.dropout)),
                nn.Linear(hidden, local), nn.ReLU(),
            )
            self.relation_key = nn.Linear(RELATION_WIDTH, local, bias=False)
            self.register_buffer(
                "critical_topk_switch",
                torch.tensor(float(spec["topk"])), persistent=True,
            )
            self.register_buffer(
                "relation_alignment_switch",
                torch.tensor(float(spec["aligned"])), persistent=True,
            )

        def select_local_evidence(self, graph, regions):
            if regions.ndim != 2 or regions.shape[1] != CORE_WIDTH + RELATION_WIDTH:
                raise RuntimeError(
                    f"Localization model expects N x 78, got {tuple(regions.shape)}"
                )
            core = regions[:, :CORE_WIDTH]
            relation = regions[:, CORE_WIDTH:]
            role_token, graph_entropy = self.encode_role_graph(
                graph["node"], graph["edge"]
            )
            graphglobal = self.graphglobal_encoder(graph["global"].reshape(-1))
            local = self.region_encoder(core)
            residual = torch.log1p(torch.clamp(core[:, -1], min=0.0))
            base_score = self.local_risk(local).reshape(-1)
            base_score = base_score + nn.functional.softplus(
                self.residual_beta_raw
            ) * residual
            query = self.role_query(role_token) + self.global_to_query(graphglobal)
            base_score = base_score + torch.mv(
                self.local_key(local), query
            ) / math.sqrt(local.shape[1])
            relation_score = torch.mv(
                self.relation_key(relation), query
            ) / math.sqrt(local.shape[1])
            effective_relation = align_relation_score(
                base_score, relation_score, bool(spec["aligned"])
            )
            score = base_score + effective_relation
            attention = torch.softmax(score, dim=0)
            attended = torch.sum(attention[:, None] * local, dim=0)
            critical = critical_pool(local, score, attention, bool(spec["topk"]))
            attention_entropy = -torch.sum(
                attention * torch.log(attention.clamp_min(1e-8))
            ) / math.log(max(int(len(attention)), 2))
            prototype_summary = torch.stack([
                torch.mean(core[:, -4]), torch.mean(core[:, -3]),
                torch.mean(core[:, -2]), torch.max(core[:, -1]),
            ])
            # Preserve the exact eight-dimensional correction summary.
            scalar_summary = torch.cat([
                torch.max(score).reshape(1), torch.mean(score).reshape(1),
                attention_entropy.reshape(1), torch.max(attention).reshape(1),
                prototype_summary,
            ])
            cache = LocalEvidenceCache(
                role_token=role_token, critical=critical, attended=attended,
                scalar_summary=scalar_summary, attention=attention, score=score,
                graph_entropy=graph_entropy,
            )
            return cache, graphglobal

    C43Localization.__name__ = "C43Localization_" + arm
    return C43Localization
