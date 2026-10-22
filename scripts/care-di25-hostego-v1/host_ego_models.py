#!/usr/bin/env python3
"""Frozen CARE heads with only the Host representation replaced by Host+Ego."""

from __future__ import annotations

import math
import torch
from torch import nn

from host_ego_encoder import DirectedHostEgoFusion, split_packed_tensor


def direct_probability_from_delta(delta: torch.Tensor, max_delta: float) -> torch.Tensor:
    ratio = torch.clamp(delta / float(max_delta), -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.sigmoid(torch.atanh(ratio))


def anomaly_classes(baseline_module, v4):
    Base = baseline_module.DedupBoundedRGCondAttn

    class HostAnomalyWithDirect(Base):
        def forward(self, global_features, base_logit, graph, regions):
            final, diagnostic = super().forward(global_features, base_logit, graph, regions)
            direct = direct_probability_from_delta(diagnostic[0], self.max_delta)
            return final, torch.cat([diagnostic, direct.reshape(1)])

    class HostEgoAnomaly(Base):
        def __init__(self, global_dim, variant, args):
            super().__init__(global_dim, variant, args)
            self.host_ego_fusion = DirectedHostEgoFusion(int(args.embedding_dim))
            self.last_ego_capture = None

        def forward(self, global_features, base_logit, graph, regions):
            host_regions, edges = split_packed_tensor(regions)
            graph_token, graph_entropy = self.graph_encoder(graph["node"], graph["edge"], graph["global"])
            host_local = self.region_encoder(host_regions)
            local, ego, gamma, used, in_count, out_count = self.host_ego_fusion(host_local, edges)
            local_score = self.local_risk(local).reshape(-1)
            residual = torch.log1p(torch.clamp(host_regions[:, -1], min=0.0))
            score = local_score + nn.functional.softplus(self.residual_beta_raw) * residual
            query = self.graph_query(graph_token); key = self.local_key(local)
            score = score + torch.mv(key, query) / math.sqrt(local.shape[1])
            attention = torch.softmax(score, dim=0)
            attended = torch.sum(attention[:, None] * local, dim=0)
            critical_index = torch.argmax(score); critical = local[critical_index]
            attention_entropy = -torch.sum(attention * torch.log(attention.clamp_min(1e-8))) / math.log(max(int(len(attention)), 2))
            prototype_summary = torch.stack([
                torch.mean(host_regions[:, -4]), torch.mean(host_regions[:, -3]),
                torch.mean(host_regions[:, -2]), torch.max(host_regions[:, -1]),
            ])
            scalar_summary = torch.cat([
                score[critical_index].reshape(1), torch.mean(score).reshape(1),
                attention_entropy.reshape(1), torch.max(attention).reshape(1), prototype_summary,
            ])
            fused = torch.cat([global_features.reshape(-1), base_logit.reshape(1), graph_token, critical, attended, scalar_summary])
            raw_delta = self.delta_head(fused).reshape(())
            bounded_delta = self.max_delta * torch.tanh(raw_delta)
            final_logit = base_logit.reshape(()) + bounded_delta
            diagnostic = torch.cat([
                bounded_delta.reshape(1), score[critical_index].reshape(1), torch.mean(score).reshape(1),
                attention_entropy.reshape(1), torch.max(attention).reshape(1), attention_entropy.reshape(1),
                prototype_summary, graph_entropy.reshape(1), torch.sigmoid(raw_delta).reshape(1),
            ])
            self.last_ego_capture = {
                "gamma": gamma.detach().cpu().numpy(), "used": used.detach().cpu().numpy(),
                "incoming": in_count.detach().cpu().numpy(), "outgoing": out_count.detach().cpu().numpy(),
                "host_norm": torch.linalg.vector_norm(host_local, dim=1).detach().cpu().numpy(),
                "ego_norm": torch.linalg.vector_norm(ego, dim=1).detach().cpu().numpy(),
                "local_norm": torch.linalg.vector_norm(local, dim=1).detach().cpu().numpy(),
                "max_fallback_diff": torch.max(torch.abs(local[~used] - host_local[~used])).detach().cpu().item() if bool(torch.any(~used)) else 0.0,
            }
            return final_logit, diagnostic

    return HostAnomalyWithDirect, HostEgoAnomaly


def malben_classes(malben_module):
    Base = malben_module.RoleConditionedEvidenceModel

    class HostMalBenWithDirect(Base):
        def forward(self, global_features, base_logit, graph, regions):
            final, diagnostic = super().forward(global_features, base_logit, graph, regions)
            direct = direct_probability_from_delta(diagnostic[0], self.max_delta)
            return final, torch.cat([diagnostic, direct.reshape(1)])

    class HostEgoMalBen(Base):
        def __init__(self, global_dim, variant, args):
            super().__init__(global_dim, variant, args)
            self.host_ego_fusion = DirectedHostEgoFusion(int(args.embedding_dim))
            self.last_ego_capture = None

        def forward(self, global_features, base_logit, graph, regions):
            host_regions, edges = split_packed_tensor(regions)
            graph_token, graph_entropy = self.graph_encoder(graph["node"], graph["edge"], graph["global"])
            host_local = self.region_encoder(host_regions[:, :malben_module.OLD_AUGMENTED_DIM])
            local, ego, gamma, used, in_count, out_count = self.host_ego_fusion(host_local, edges)
            residual = host_regions[:, malben_module.BASE_REGION_DIM + 3]
            score = self.local_risk(local).reshape(-1)
            score = score + nn.functional.softplus(self.residual_beta_raw) * torch.log1p(torch.clamp(residual, min=0.0))
            query = self.graph_query(graph_token); key = self.local_key(local)
            score = score + torch.mv(key, query) / math.sqrt(local.shape[1])
            attention = torch.softmax(score, dim=0); critical_index = torch.argmax(score); critical = local[critical_index]
            role_columns = host_regions[:, malben_module.OLD_AUGMENTED_DIM:]
            s_ben = role_columns[:, 0].clamp(0.0, 1.0); s_mal = role_columns[:, 1].clamp(0.0, 1.0)
            margin = role_columns[:, 2].clamp(-1.0, 1.0); strength = role_columns[:, 3].clamp(0.0, 1.0)
            e_ben_region = torch.relu(-margin); e_mal_region = torch.relu(margin)
            z_mal = torch.sum((attention * e_mal_region)[:, None] * local, dim=0)
            z_ben = torch.sum((attention * e_ben_region)[:, None] * local, dim=0)
            z_difference = z_mal - z_ben
            e_mal = torch.sum(attention * e_mal_region); e_ben = torch.sum(attention * e_ben_region)
            evidence_difference = e_mal - e_ben
            attention_entropy = -torch.sum(attention * torch.log(attention.clamp_min(1e-8))) / math.log(max(int(len(attention)), 2))
            scalar_summary = torch.stack([
                score[critical_index], torch.mean(score), attention_entropy, torch.max(attention),
                e_mal, e_ben, torch.mean(strength), torch.max(torch.abs(margin)),
            ])
            fused = torch.cat([global_features.reshape(-1), base_logit.reshape(1), graph_token, critical, z_difference, scalar_summary])
            raw_delta = self.delta_head(fused).reshape(())
            bounded_delta = self.max_delta * torch.tanh(raw_delta)
            final_logit = base_logit.reshape(()) + bounded_delta
            diagnostic = torch.stack([
                bounded_delta, score[critical_index], torch.mean(score), attention_entropy, torch.max(attention),
                e_mal, e_ben, torch.mean(strength), torch.max(torch.abs(margin)), evidence_difference,
                graph_entropy, torch.sigmoid(raw_delta),
            ])
            self.last_region_capture = torch.stack([
                attention, s_ben, s_mal, margin, e_ben_region, e_mal_region, strength,
                role_columns[:, 4], role_columns[:, 5], role_columns[:, 6], role_columns[:, 7], role_columns[:, 8],
            ], dim=1).detach()
            self.last_apk_capture = {
                "e_mal": float(e_mal.detach().cpu()), "e_ben": float(e_ben.detach().cpu()),
                "z_mal": z_mal.detach().cpu().numpy(), "z_ben": z_ben.detach().cpu().numpy(),
                "stage2_feature": torch.cat([graph_token, z_difference, critical, evidence_difference.reshape(1)]).detach().cpu().numpy(),
            }
            self.last_ego_capture = {
                "gamma": gamma.detach().cpu().numpy(), "used": used.detach().cpu().numpy(),
                "incoming": in_count.detach().cpu().numpy(), "outgoing": out_count.detach().cpu().numpy(),
                "host_norm": torch.linalg.vector_norm(host_local, dim=1).detach().cpu().numpy(),
                "ego_norm": torch.linalg.vector_norm(ego, dim=1).detach().cpu().numpy(),
                "local_norm": torch.linalg.vector_norm(local, dim=1).detach().cpu().numpy(),
                "max_fallback_diff": torch.max(torch.abs(local[~used] - host_local[~used])).detach().cpu().item() if bool(torch.any(~used)) else 0.0,
            }
            return final_logit, diagnostic

    return HostMalBenWithDirect, HostEgoMalBen

