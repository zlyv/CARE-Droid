#!/usr/bin/env python3
from __future__ import annotations


def install_variant(v4, arm):
    import torch

    Base = v4.RoleGraphConditionedEvidence
    if arm == "U1_UNBOUNDED_RESIDUAL":
        class UnboundedResidual(Base):
            correction_semantics = "base_logit_plus_raw_delta_without_tanh_or_clip"

            def delta_from_selected_evidence(self, global_features, base_logit, cache, graphglobal):
                raw_delta, _, fused = super().delta_from_selected_evidence(
                    global_features, base_logit, cache, graphglobal
                )
                return raw_delta, raw_delta, fused
        v4.RoleGraphConditionedEvidence = UnboundedResidual
        return

    if arm == "U2_DIRECT_STAGE2":
        class DirectStage2(Base):
            correction_semantics = "direct_stage2_logit_without_stage1_logit_input"

            def forward_with_trace(self, global_features, base_logit, graph, regions):
                cache, graphglobal = self.select_local_evidence(graph, regions)
                zero_base = base_logit.reshape(()) * 0.0
                raw_logit, _, fused = super().delta_from_selected_evidence(
                    global_features, zero_base, cache, graphglobal
                )
                diagnostic = torch.cat([
                    raw_logit.reshape(1),
                    cache.score[torch.argmax(cache.score)].reshape(1),
                    torch.mean(cache.score).reshape(1),
                    cache.scalar_summary[2].reshape(1),
                    torch.max(cache.attention).reshape(1),
                    cache.scalar_summary[2].reshape(1),
                    cache.scalar_summary[-4:],
                    cache.graph_entropy.reshape(1),
                ])
                trace = {"attention": cache.attention, "critical": cache.critical,
                         "attended": cache.attended, "correction_input": fused,
                         "raw_delta": raw_logit, "bounded_delta": raw_logit}
                return raw_logit, diagnostic, trace
        v4.RoleGraphConditionedEvidence = DirectStage2
        return

    raise KeyError(arm)
