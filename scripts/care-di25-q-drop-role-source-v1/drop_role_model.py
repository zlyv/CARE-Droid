#!/usr/bin/env python3
"""Q-Query with RoleToken retained for localization but removed from delta."""
from __future__ import annotations
import torch
from graphglobal_path_models import make_graphglobal_path_class

ARM="Q_DROP_ROLE_DIRECT"

def make_model(v4):
    Base=make_graphglobal_path_class(v4,"Q_QUERY_ONLY")
    mask=torch.ones(113,dtype=torch.float32);mask[25:57]=0.0
    class QDropRoleDirect(Base):
        def __init__(self,global_dim,variant,args):
            super().__init__(global_dim,variant,args)
            self.register_buffer("correction_mask",mask.clone(),persistent=True)
        def delta_from_selected_evidence(self,global_features,base_logit,cache,graphglobal):
            # RoleToken has already contributed to Host attention in
            # select_local_evidence. It is masked only at the delta interface.
            fused=torch.cat([global_features.reshape(-1),base_logit.reshape(1),cache.role_token,cache.critical,cache.attended,cache.scalar_summary])
            if fused.numel()!=113:raise RuntimeError(f"Correction input width {fused.numel()} != 113")
            masked=fused*self.correction_mask.to(device=fused.device,dtype=fused.dtype)
            raw=self.delta_head(masked).reshape(());bounded=self.max_delta*torch.tanh(raw)
            return raw,bounded,masked
    QDropRoleDirect.__name__=ARM
    return QDropRoleDirect
