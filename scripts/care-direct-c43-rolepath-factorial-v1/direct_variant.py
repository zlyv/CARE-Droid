#!/usr/bin/env python3
from __future__ import annotations

def install_direct(v4):
    import torch
    Base=v4.RoleGraphConditionedEvidence
    class DirectStage2(Base):
        correction_semantics="direct_stage2_logit_without_stage1_logit_input"
        def forward_with_trace(self,global_features,base_logit,graph,regions):
            cache,graphglobal=self.select_local_evidence(graph,regions)
            zero_base=base_logit.reshape(())*0.0
            raw_logit,_,fused=super().delta_from_selected_evidence(global_features,zero_base,cache,graphglobal)
            diagnostic=torch.cat([raw_logit.reshape(1),cache.score[torch.argmax(cache.score)].reshape(1),
                torch.mean(cache.score).reshape(1),cache.scalar_summary[2].reshape(1),torch.max(cache.attention).reshape(1),
                cache.scalar_summary[2].reshape(1),cache.scalar_summary[-4:],cache.graph_entropy.reshape(1)])
            trace={"attention":cache.attention,"critical":cache.critical,"attended":cache.attended,
                "correction_input":fused,"raw_delta":raw_logit,"bounded_delta":raw_logit}
            return raw_logit,diagnostic,trace
    v4.RoleGraphConditionedEvidence=DirectStage2
