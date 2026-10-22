#!/usr/bin/env python3
"""Exact static Host-own-Role31 bipartite residual fusion; reads tensors only and writes nothing."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EvidenceCache:
    role_token: object
    critical: object
    attended: object
    scalar_summary: object
    attention: object
    score: object
    graph_entropy: object


def make_model(v4,args):
    torch=v4.torch; nn=torch.nn; Base=v4.RoleGraphConditionedEvidence

    class RoleHostAlignedFusion(Base):
        fusion_semantics='exact Host-own-Role31, node-level role residual plus reverse Host residual'
        def __init__(self):
            super().__init__(24,'RGCondAttn',args)
            graph_dim=int(args.graph_dim); local_dim=int(args.embedding_dim)
            self.host_to_role=nn.Linear(local_dim,graph_dim,bias=False)
            self.role_fuse_gate=nn.Linear(2*graph_dim,graph_dim)
            self.role_to_host=nn.Linear(graph_dim,local_dim,bias=False)
            self.host_fuse_gate=nn.Linear(2*local_dim,local_dim)
            self.role_residual_scale=nn.Parameter(torch.tensor(0.0))
            self.host_residual_scale=nn.Parameter(torch.tensor(0.0))

        def select_local_evidence(self,graph,regions):
            if regions.ndim!=2 or regions.shape[1]!=173:
                raise RuntimeError(f'aligned fusion expects N x 173, got {tuple(regions.shape)}')
            base=regions[:,:142]; binding=regions[:,142:]
            core=base[:,:111]; relation=base[:,111:]
            role_index=torch.arange(v4.ROLE_COUNT,device=regions.device)
            role_nodes=self.node_input(graph['node'])+self.role_embedding(role_index)
            for block in self.blocks: role_nodes=block(role_nodes,graph['edge'])
            local=self.region_encoder(core)

            projected=self.host_to_role(local)
            mass=torch.sum(binding,dim=0)
            host_role=torch.matmul(binding.transpose(0,1),projected)/mass.clamp_min(1e-6)[:,None]
            host_role=torch.where((mass>0)[:,None],host_role,torch.zeros_like(host_role))
            role_gate=torch.sigmoid(self.role_fuse_gate(torch.cat([role_nodes,host_role],dim=1)))
            fused_role=role_nodes+torch.tanh(self.role_residual_scale)*role_gate*host_role
            pool_weight=torch.softmax(self.pool_score(fused_role).reshape(-1),dim=0)
            role_token=torch.sum(pool_weight[:,None]*fused_role,dim=0)
            graph_entropy=-torch.sum(pool_weight*torch.log(pool_weight.clamp_min(1e-8)))/math.log(v4.ROLE_COUNT)

            role_values=self.role_to_host(fused_role)
            host_context=torch.matmul(binding,role_values)
            host_gate=torch.sigmoid(self.host_fuse_gate(torch.cat([local,host_context],dim=1)))
            enriched=local+torch.tanh(self.host_residual_scale)*host_gate*host_context

            graphglobal=self.graphglobal_encoder(graph['global'].reshape(-1))
            residual=torch.log1p(torch.clamp(core[:,-1],min=0.0))
            score=self.local_risk(enriched).reshape(-1)+nn.functional.softplus(self.residual_beta_raw)*residual
            query=self.role_query(role_token)+self.global_to_query(graphglobal)
            score=score+torch.mv(self.local_key(enriched),query)/math.sqrt(enriched.shape[1])
            score=score+torch.mv(self.relation_key(relation),query)/math.sqrt(enriched.shape[1])
            attention=torch.softmax(score,dim=0); critical_index=torch.argmax(score)
            attended=torch.sum(attention[:,None]*enriched,dim=0); critical=enriched[critical_index]
            entropy=-torch.sum(attention*torch.log(attention.clamp_min(1e-8)))/math.log(max(int(len(attention)),2))
            prototype=torch.stack([torch.mean(core[:,-4]),torch.mean(core[:,-3]),torch.mean(core[:,-2]),torch.max(core[:,-1])])
            summary=torch.cat([score[critical_index].reshape(1),torch.mean(score).reshape(1),entropy.reshape(1),torch.max(attention).reshape(1),prototype])
            return EvidenceCache(role_token,critical,attended,summary,attention,score,graph_entropy),graphglobal

    return RoleHostAlignedFusion()
