#!/usr/bin/env python3
"""Frozen W0 plus an equal-capacity, zero-initialized Host semantic Adapter."""
from __future__ import annotations
import math
from semantic_common import BASE_WIDTH,FEATURE_WIDTH,MODEL_WIDTH

def make_model(v4,args,base_state):
    torch=v4.torch; nn=torch.nn; Base=v4.RoleGraphConditionedEvidence
    from graphglobal_path_models import LocalEvidenceCache
    class Adapter(Base):
        def __init__(self):
            super().__init__(24,'RGCondAttn',args)
            self.semantic_adapter=nn.Sequential(nn.Linear(FEATURE_WIDTH,16),nn.ReLU(),nn.Dropout(float(args.dropout)))
            self.semantic_delta=nn.Linear(16,24,bias=False); nn.init.zeros_(self.semantic_delta.weight)
        def select_local_evidence(self,graph,regions):
            if regions.ndim!=2 or regions.shape[1]!=MODEL_WIDTH: raise RuntimeError(f'expected Nx{MODEL_WIDTH}')
            base=regions[:,:BASE_WIDTH]; semantic=regions[:,BASE_WIDTH:BASE_WIDTH+FEATURE_WIDTH]; mask=regions[:,-1:]
            core=base[:,:111]; relation=base[:,111:]
            role_token,graph_entropy=self.encode_role_graph(graph['node'],graph['edge']); graphglobal=self.graphglobal_encoder(graph['global'].reshape(-1))
            local=self.region_encoder(core)+mask*self.semantic_delta(self.semantic_adapter(semantic))
            residual=torch.log1p(torch.clamp(core[:,-1],min=0)); score=self.local_risk(local).reshape(-1)+nn.functional.softplus(self.residual_beta_raw)*residual
            query=self.role_query(role_token)+self.graphglobal_query_switch*self.global_to_query(graphglobal)
            score=score+torch.mv(self.local_key(local),query)/math.sqrt(local.shape[1])+torch.mv(self.relation_key(relation),query)/math.sqrt(local.shape[1])
            attention=torch.softmax(score,0); critical=torch.argmax(score); attended=torch.sum(attention[:,None]*local,0)
            entropy=-torch.sum(attention*torch.log(attention.clamp_min(1e-8)))/math.log(max(len(attention),2))
            summary=torch.cat([score[critical].reshape(1),torch.mean(score).reshape(1),entropy.reshape(1),torch.max(attention).reshape(1),torch.stack([torch.mean(core[:,-4]),torch.mean(core[:,-3]),torch.mean(core[:,-2]),torch.max(core[:,-1])])])
            return LocalEvidenceCache(role_token=role_token,critical=local[critical],attended=attended,scalar_summary=summary,attention=attention,score=score,graph_entropy=graph_entropy),graphglobal
    model=Adapter(); result=model.load_state_dict(base_state,strict=False)
    expected={k for k in model.state_dict() if k.startswith('semantic_')}
    if set(result.missing_keys)!=expected or result.unexpected_keys: raise RuntimeError(f'base load mismatch {result}')
    for n,p in model.named_parameters(): p.requires_grad_(n.startswith('semantic_'))
    return model

def set_train_mode(model):
    model.eval(); model.semantic_adapter.train(); model.semantic_delta.train()

def trainable_parameters(model): return [p for p in model.parameters() if p.requires_grad]
