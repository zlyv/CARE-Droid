#!/usr/bin/env python3
from __future__ import annotations
import math,torch
from torch import nn
from graphglobal_path_models import LocalEvidenceCache,make_graphglobal_path_class

CORE_WIDTH=53 # weighted Compact43+Prov6 (49) + four background columns
RELATION_WIDTH=31
def make_class(v4):
    Base=make_graphglobal_path_class(v4,"Q_QUERY_ONLY")
    class C43Prov6(Base):
        def __init__(self,global_dim,variant,args):
            super().__init__(global_dim,variant,args);hidden=int(args.hidden_dim);local=int(args.embedding_dim)
            self.region_encoder=nn.Sequential(nn.Linear(CORE_WIDTH,hidden),nn.ReLU(),nn.Dropout(float(args.dropout)),nn.Linear(hidden,local),nn.ReLU())
            self.relation_key=nn.Linear(RELATION_WIDTH,local,bias=False)
        def select_local_evidence(self,graph,regions):
            if regions.ndim!=2 or regions.shape[1]!=84:raise RuntimeError(f"Prov6 model expects N x 84, got {tuple(regions.shape)}")
            core=regions[:,:CORE_WIDTH];relation=regions[:,CORE_WIDTH:]
            role_token,graph_entropy=self.encode_role_graph(graph["node"],graph["edge"]);gg=self.graphglobal_encoder(graph["global"].reshape(-1));local=self.region_encoder(core)
            residual=torch.log1p(torch.clamp(core[:,-1],min=0));score=self.local_risk(local).reshape(-1)+nn.functional.softplus(self.residual_beta_raw)*residual
            query=self.role_query(role_token)+self.global_to_query(gg);score=score+torch.mv(self.local_key(local),query)/math.sqrt(local.shape[1]);score=score+torch.mv(self.relation_key(relation),query)/math.sqrt(local.shape[1])
            attention=torch.softmax(score,0);attended=torch.sum(attention[:,None]*local,0);critical_index=torch.argmax(score);critical=local[critical_index]
            entropy=-torch.sum(attention*torch.log(attention.clamp_min(1e-8)))/math.log(max(int(len(attention)),2));proto=torch.stack([torch.mean(core[:,-4]),torch.mean(core[:,-3]),torch.mean(core[:,-2]),torch.max(core[:,-1])]);summary=torch.cat([score[critical_index].reshape(1),torch.mean(score).reshape(1),entropy.reshape(1),torch.max(attention).reshape(1),proto])
            return LocalEvidenceCache(role_token=role_token,critical=critical,attended=attended,scalar_summary=summary,attention=attention,score=score,graph_entropy=graph_entropy),gg
    C43Prov6.__name__="C43RelationQueryProv6"
    return C43Prov6
