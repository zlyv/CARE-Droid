#!/usr/bin/env python3
import math
import torch
from torch import nn
from host_composition_encoder import HostSemanticCompositionEncoder
from host_position_encoder import HostStructuralPositionEncoder,POSITION_DIM

def make_model_class(Base,DirectedHostEgoFusion,role_count:int,augmented_dim:int,arm:str):
    if arm not in {"A","B","AB"}:raise ValueError(arm)
    class HostComplementModel(Base):
        def __init__(self,global_dim,variant,args):
            super().__init__(global_dim,variant,args);dim=int(args.embedding_dim)
            self.host_ego_fusion=DirectedHostEgoFusion(dim);self.arm=arm;self.role_count=role_count;self.augmented_dim=augmented_dim
            if arm in {"A","AB"}:
                self.composition_encoder=HostSemanticCompositionEncoder(role_count,dim);self.composition_gate=nn.Linear(2*dim,1)
            if arm in {"B","AB"}:
                self.position_encoder=HostStructuralPositionEncoder(dim);self.position_gate=nn.Linear(2*dim,1)
            self.last_complement_capture=None

        def forward(self,global_features,base_logit,graph,regions):
            width=regions.shape[1]-3;host_mask=regions[:,width]<.5;host_rows=regions[host_mask,:width];edge_rows=regions[~host_mask]
            edges=edge_rows[:,width+1:width+3].long() if len(edge_rows) else torch.empty((0,2),dtype=torch.long,device=regions.device)
            base=host_rows[:,:self.augmented_dim];membership=host_rows[:,self.augmented_dim:self.augmented_dim+self.role_count]
            position=host_rows[:,self.augmented_dim+self.role_count:self.augmented_dim+self.role_count+POSITION_DIM]
            graph_token,graph_entropy=self.graph_encoder(graph["node"],graph["edge"],graph["global"])
            host_local=self.region_encoder(base);local,ego,gamma,used,in_count,out_count=self.host_ego_fusion(host_local,edges)
            augmented=local;alpha=torch.zeros(len(local),device=local.device);beta=torch.zeros(len(local),device=local.device);composition=torch.zeros_like(local);structural=torch.zeros_like(local)
            if self.arm in {"A","AB"}:
                composition=self.composition_encoder(membership);alpha=torch.sigmoid(self.composition_gate(torch.cat([local,composition],1))).reshape(-1);augmented=augmented+alpha[:,None]*composition
            if self.arm in {"B","AB"}:
                structural=self.position_encoder(position);beta=torch.sigmoid(self.position_gate(torch.cat([local,structural],1))).reshape(-1);augmented=augmented+beta[:,None]*structural
            local_score=self.local_risk(augmented).reshape(-1);residual=torch.log1p(torch.clamp(base[:,-1],min=0.0));score=local_score+nn.functional.softplus(self.residual_beta_raw)*residual
            score=score+torch.mv(self.local_key(augmented),self.graph_query(graph_token))/math.sqrt(augmented.shape[1]);attention=torch.softmax(score,0)
            attended=torch.sum(attention[:,None]*augmented,0);critical_index=torch.argmax(score);critical=augmented[critical_index]
            entropy=-torch.sum(attention*torch.log(attention.clamp_min(1e-8)))/math.log(max(int(len(attention)),2))
            proto=torch.stack([torch.mean(base[:,-4]),torch.mean(base[:,-3]),torch.mean(base[:,-2]),torch.max(base[:,-1])])
            summary=torch.cat([score[critical_index].reshape(1),score.mean().reshape(1),entropy.reshape(1),attention.max().reshape(1),proto])
            fused=torch.cat([global_features.reshape(-1),base_logit.reshape(1),graph_token,critical,attended,summary]);raw=self.delta_head(fused).reshape(());delta=self.max_delta*torch.tanh(raw)
            self.last_complement_capture={"alpha":alpha.detach(),"beta":beta.detach(),"gamma":gamma.detach(),"ego_used":used.detach(),"membership":membership.detach(),"position":position.detach(),"anomaly":base[:,-4:].detach()}
            noego=~used;fallback=torch.max(torch.abs(local[noego]-host_local[noego])) if bool(torch.any(noego)) else torch.zeros((),device=local.device)
            multi=torch.mean((membership.sum(1)>=2).to(local.dtype))
            diagnostic=torch.cat([delta.reshape(1),score[critical_index].reshape(1),score.mean().reshape(1),entropy.reshape(1),attention.max().reshape(1),entropy.reshape(1),proto,graph_entropy.reshape(1),torch.sigmoid(raw).reshape(1),
                                  alpha.mean().reshape(1),alpha.median().reshape(1),alpha.max().reshape(1),beta.mean().reshape(1),beta.median().reshape(1),beta.max().reshape(1),multi.reshape(1),fallback.reshape(1)])
            return base_logit.reshape(())+delta,diagnostic
    return HostComplementModel
