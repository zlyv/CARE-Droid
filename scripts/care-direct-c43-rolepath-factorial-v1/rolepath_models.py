#!/usr/bin/env python3
from __future__ import annotations
import math

ARMS={
    "R11_ROLE_QUERY_READOUT":(1.0,1.0),
    "R10_ROLE_QUERY_ONLY":(1.0,0.0),
    "R01_ROLE_READOUT_ONLY":(0.0,1.0),
    "R00_ROLE_NEITHER":(0.0,0.0),
}
CORE=111; REL=31

def install_rolepath(v4,arm):
    if arm not in ARMS: raise KeyError(arm)
    import torch
    from torch import nn
    query_switch,readout_switch=ARMS[arm]; Base=v4.RoleGraphConditionedEvidence
    class RolePathFactorial(Base):
        rolepath_arm=arm
        def __init__(self,global_dim,variant,args):
            super().__init__(global_dim,variant,args)
            self.register_buffer("role_query_path_switch",torch.tensor(query_switch),persistent=True)
            self.register_buffer("role_readout_path_switch",torch.tensor(readout_switch),persistent=True)
        def select_local_evidence(self,graph,regions):
            if regions.ndim!=2 or regions.shape[1]!=CORE+REL:
                raise RuntimeError(f"Role-path C43 expects N x {CORE+REL}, got {tuple(regions.shape)}")
            core=regions[:,:CORE]; relation=regions[:,CORE:]
            role,ge=self.encode_role_graph(graph["node"],graph["edge"])
            gg=self.graphglobal_encoder(graph["global"].reshape(-1)); local=self.region_encoder(core)
            residual=torch.log1p(torch.clamp(core[:,-1],min=0)); score=self.local_risk(local).reshape(-1)
            score=score+nn.functional.softplus(self.residual_beta_raw)*residual
            query=self.role_query(role)*self.role_query_path_switch+self.global_to_query(gg)
            score=score+torch.mv(self.local_key(local),query)/math.sqrt(local.shape[1])
            score=score+torch.mv(self.relation_key(relation),query)/math.sqrt(local.shape[1])
            attention=torch.softmax(score,0); critical_index=torch.argmax(score)
            entropy=-torch.sum(attention*torch.log(attention.clamp_min(1e-8)))/math.log(max(len(attention),2))
            prototype=torch.stack([torch.mean(core[:,-4]),torch.mean(core[:,-3]),torch.mean(core[:,-2]),torch.max(core[:,-1])])
            summary=torch.cat([score[critical_index].reshape(1),torch.mean(score).reshape(1),entropy.reshape(1),torch.max(attention).reshape(1),prototype])
            Cache=__import__("graphglobal_path_models").LocalEvidenceCache
            return Cache(role_token=role,critical=local[critical_index],attended=torch.sum(attention[:,None]*local,0),
                scalar_summary=summary,attention=attention,score=score,graph_entropy=ge),gg
        def delta_from_selected_evidence(self,global_features,base_logit,cache,graphglobal):
            Cache=type(cache)
            selected=Cache(role_token=cache.role_token*self.role_readout_path_switch,critical=cache.critical,
                attended=cache.attended,scalar_summary=cache.scalar_summary,attention=cache.attention,
                score=cache.score,graph_entropy=cache.graph_entropy)
            return super().delta_from_selected_evidence(global_features,base_logit,selected,graphglobal)
    RolePathFactorial.__name__="RolePathFactorial_"+arm
    v4.RoleGraphConditionedEvidence=RolePathFactorial
