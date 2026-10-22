#!/usr/bin/env python3
"""Compact43 correction evidence with relation semantics restricted to attention."""
from __future__ import annotations
import math
import torch
from torch import nn
from graphglobal_path_models import LocalEvidenceCache, make_graphglobal_path_class

ARMS={"C43_COMPACT":0.0,"C43_RELATION_QUERY":1.0}
CORE_WIDTH=47  # weighted Compact43 + four anomaly-background columns
RELATION_WIDTH=31


def make_compact_class(v4,arm):
    if arm not in ARMS: raise KeyError(arm)
    switch=float(ARMS[arm]); Base=make_graphglobal_path_class(v4,"Q_QUERY_ONLY")

    class CompactRelationQuery(Base):
        compact_arm=arm
        def __init__(self,global_dim,variant,args):
            super().__init__(global_dim,variant,args)
            hidden=int(args.hidden_dim); local=int(args.embedding_dim)
            self.region_encoder=nn.Sequential(nn.Linear(CORE_WIDTH,hidden),nn.ReLU(),nn.Dropout(float(args.dropout)),nn.Linear(hidden,local),nn.ReLU())
            # Instantiated in both arms. The fixed switch isolates relation use.
            self.relation_key=nn.Linear(RELATION_WIDTH,local,bias=False)
            self.register_buffer("relation_query_switch",torch.tensor(switch),persistent=True)

        def select_local_evidence(self,graph,regions):
            if regions.ndim!=2 or regions.shape[1]!=(CORE_WIDTH+RELATION_WIDTH): raise RuntimeError(f"Compact model expects N x 78, got {tuple(regions.shape)}")
            core=regions[:,:CORE_WIDTH]; relation=regions[:,CORE_WIDTH:]
            role_token,graph_entropy=self.encode_role_graph(graph["node"],graph["edge"]); graphglobal=self.graphglobal_encoder(graph["global"].reshape(-1)); local=self.region_encoder(core)
            residual=torch.log1p(torch.clamp(core[:,-1],min=0.0)); score=self.local_risk(local).reshape(-1)+nn.functional.softplus(self.residual_beta_raw)*residual
            query=self.role_query(role_token)+self.global_to_query(graphglobal); key=self.local_key(local)
            score=score+torch.mv(key,query)/math.sqrt(local.shape[1])
            relation_score=torch.mv(self.relation_key(relation),query)/math.sqrt(local.shape[1])
            score=score+self.relation_query_switch*relation_score
            attention=torch.softmax(score,dim=0); attended=torch.sum(attention[:,None]*local,dim=0); critical_index=torch.argmax(score); critical=local[critical_index]
            attention_entropy=-torch.sum(attention*torch.log(attention.clamp_min(1e-8)))/math.log(max(int(len(attention)),2))
            prototype_summary=torch.stack([torch.mean(core[:,-4]),torch.mean(core[:,-3]),torch.mean(core[:,-2]),torch.max(core[:,-1])])
            scalar_summary=torch.cat([score[critical_index].reshape(1),torch.mean(score).reshape(1),attention_entropy.reshape(1),torch.max(attention).reshape(1),prototype_summary])
            cache=LocalEvidenceCache(role_token=role_token,critical=critical,attended=attended,scalar_summary=scalar_summary,attention=attention,score=score,graph_entropy=graph_entropy)
            return cache,graphglobal

    CompactRelationQuery.__name__="Compact43_"+arm
    return CompactRelationQuery
