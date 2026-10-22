#!/usr/bin/env python3
"""Frozen-W0 Host--Prototype Relation Network with a zero-initialized Host residual."""
from __future__ import annotations

import math

from hprn_common import (BASE_WIDTH, MODEL_WIDTH, PROTO_WIDTH,
                         RELATION_ARMS, SIDECAR_PER_BACKGROUND)


def make_model(v4, args, arm, base_state):
    if arm not in RELATION_ARMS: raise ValueError(arm)
    torch=v4.torch; nn=torch.nn; Base=v4.RoleGraphConditionedEvidence
    from graphglobal_path_models import LocalEvidenceCache

    class HPRN(Base):
        def __init__(self):
            super().__init__(24,'RGCondAttn',args)
            self.hprn_arm=arm
            self.hprn_host_proj=nn.ModuleList([nn.Linear(PROTO_WIDTH,16) for _ in range(3)])
            self.hprn_proto_proj=nn.ModuleList([nn.Linear(PROTO_WIDTH,16) for _ in range(3)])
            self.hprn_relation=nn.ModuleList([nn.Sequential(
                nn.Linear(65,16),nn.ReLU(),nn.Dropout(float(args.dropout)),nn.Linear(16,8)
            ) for _ in range(3)])
            self.hprn_fuse=nn.Sequential(nn.Linear(24,16),nn.ReLU(),nn.Dropout(float(args.dropout)))
            self.hprn_delta=nn.Linear(16,24,bias=False)
            nn.init.zeros_(self.hprn_delta.weight)

        def relation_delta(self, core, sidecar):
            n=len(core); pieces=[]; cursor=0
            host=core[:,:PROTO_WIDTH] if self.hprn_arm in ('R2_TRUE_RELATION','R3_BINDING_SHUFFLE') else torch.zeros_like(core[:,:PROTO_WIDTH])
            for j in range(3):
                proto=sidecar[:,cursor:cursor+PROTO_WIDTH]; cursor+=PROTO_WIDTH
                distance=sidecar[:,cursor:cursor+1]; cursor+=1
                zh=self.hprn_host_proj[j](host); zp=self.hprn_proto_proj[j](proto)
                relation=torch.cat([zh,zp,torch.abs(zh-zp),zh*zp,distance],dim=1)
                pieces.append(self.hprn_relation[j](relation))
            if cursor!=sidecar.shape[1]: raise RuntimeError('HPRN sidecar parse mismatch')
            return self.hprn_delta(self.hprn_fuse(torch.cat(pieces,dim=1)))

        def select_local_evidence(self, graph, regions):
            if regions.ndim!=2 or regions.shape[1]!=MODEL_WIDTH:
                raise RuntimeError(f'HPRN expects N x {MODEL_WIDTH}, got {tuple(regions.shape)}')
            base=regions[:,:BASE_WIDTH]; sidecar=regions[:,BASE_WIDTH:]
            core=base[:,:111]; relation=base[:,111:]
            role_token,graph_entropy=self.encode_role_graph(graph['node'],graph['edge'])
            graphglobal=self.graphglobal_encoder(graph['global'].reshape(-1))
            local=self.region_encoder(core)+self.relation_delta(core,sidecar)
            residual=torch.log1p(torch.clamp(core[:,-1],min=0))
            score=self.local_risk(local).reshape(-1)+nn.functional.softplus(self.residual_beta_raw)*residual
            query=self.role_query(role_token)+self.graphglobal_query_switch*self.global_to_query(graphglobal)
            score=score+torch.mv(self.local_key(local),query)/math.sqrt(local.shape[1])
            score=score+torch.mv(self.relation_key(relation),query)/math.sqrt(local.shape[1])
            attention=torch.softmax(score,0); critical=torch.argmax(score)
            attended=torch.sum(attention[:,None]*local,0)
            entropy=-torch.sum(attention*torch.log(attention.clamp_min(1e-8)))/math.log(max(len(attention),2))
            summary=torch.cat([score[critical].reshape(1),torch.mean(score).reshape(1),entropy.reshape(1),
                               torch.max(attention).reshape(1),torch.stack([torch.mean(core[:,-4]),
                               torch.mean(core[:,-3]),torch.mean(core[:,-2]),torch.max(core[:,-1])])])
            return LocalEvidenceCache(role_token=role_token,critical=local[critical],attended=attended,
                scalar_summary=summary,attention=attention,score=score,graph_entropy=graph_entropy),graphglobal

    model=HPRN()
    result=model.load_state_dict(base_state,strict=False)
    expected={name for name in model.state_dict() if name.startswith('hprn_')}
    if set(result.missing_keys)!=expected or result.unexpected_keys:
        raise RuntimeError(f'W0 state load mismatch missing={result.missing_keys} unexpected={result.unexpected_keys}')
    for name,param in model.named_parameters(): param.requires_grad_(name.startswith('hprn_'))
    return model


def set_train_mode(model):
    model.eval()
    model.hprn_host_proj.train(); model.hprn_proto_proj.train(); model.hprn_relation.train()
    model.hprn_fuse.train(); model.hprn_delta.train()


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
