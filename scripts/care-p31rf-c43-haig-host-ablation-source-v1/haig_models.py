#!/usr/bin/env python3
from __future__ import annotations
import math,torch
from torch import nn
from graphglobal_path_models import LocalEvidenceCache,make_graphglobal_path_class
CORE=111;REL=31
def make_class(v4):
    Base=make_graphglobal_path_class(v4,"Q_QUERY_ONLY")
    class Model(Base):
        def __init__(self,global_dim,variant,args):
            super().__init__(global_dim,variant,args);h=int(args.hidden_dim);d=int(args.embedding_dim);self.region_encoder=nn.Sequential(nn.Linear(CORE,h),nn.ReLU(),nn.Dropout(float(args.dropout)),nn.Linear(h,d),nn.ReLU());self.relation_key=nn.Linear(REL,d,bias=False)
        def select_local_evidence(self,graph,regions):
            if regions.ndim!=2 or regions.shape[1]!=CORE+REL:raise RuntimeError(f"HAIG C43 expects N x {CORE+REL}, got {tuple(regions.shape)}")
            core=regions[:,:CORE];relation=regions[:,CORE:];role,ge=self.encode_role_graph(graph["node"],graph["edge"]);gg=self.graphglobal_encoder(graph["global"].reshape(-1));local=self.region_encoder(core);res=torch.log1p(torch.clamp(core[:,-1],min=0));score=self.local_risk(local).reshape(-1)+nn.functional.softplus(self.residual_beta_raw)*res;query=self.role_query(role)+self.global_to_query(gg);score=score+torch.mv(self.local_key(local),query)/math.sqrt(local.shape[1])+torch.mv(self.relation_key(relation),query)/math.sqrt(local.shape[1]);att=torch.softmax(score,0);critical=torch.argmax(score);summary=torch.cat([score[critical].reshape(1),torch.mean(score).reshape(1),(-torch.sum(att*torch.log(att.clamp_min(1e-8)))/math.log(max(len(att),2))).reshape(1),torch.max(att).reshape(1),torch.stack([torch.mean(core[:,-4]),torch.mean(core[:,-3]),torch.mean(core[:,-2]),torch.max(core[:,-1])])]);return LocalEvidenceCache(role_token=role,critical=local[critical],attended=torch.sum(att[:,None]*local,0),scalar_summary=summary,attention=att,score=score,graph_entropy=ge),gg
    return Model

