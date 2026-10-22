#!/usr/bin/env python3
from __future__ import annotations
import math,numpy as np,torch
from torch import nn
from graphglobal_path_models import LocalEvidenceCache,make_graphglobal_path_class
CORE=47; REL=31; API_DIM=16; ROLE_N=31; TOTAL=CORE+REL+ROLE_N*API_DIM
ARMS=("R1_UNCONDITIONAL_API","R2_FIXED_ROLE_API","R2S_FIXED_SHUFFLED","R3_LEARNED_ROLE_API","R4_SHUFFLED_WITHIN_ROLE")
GATE_SNAPSHOTS=[]
def make_class(v4,arm):
    if arm not in ARMS: raise KeyError(arm)
    Base=make_graphglobal_path_class(v4,"Q_QUERY_ONLY")
    class Model(Base):
        role_api_arm=arm
        def __init__(self,global_dim,variant,args):
            super().__init__(global_dim,variant,args); h=int(args.hidden_dim); d=int(args.embedding_dim)
            self.region_encoder=nn.Sequential(nn.Linear(CORE,h),nn.ReLU(),nn.Dropout(float(args.dropout)),nn.Linear(h,d),nn.ReLU())
            self.relation_key=nn.Linear(REL,d,bias=False)
            self.api_unconditional=nn.Linear(API_DIM,ROLE_N,bias=False)
            self.api_role_residual=nn.Sequential(nn.Linear(API_DIM,API_DIM),nn.ReLU(),nn.Linear(API_DIM,1,bias=False))
            self.api_gate_logits=nn.Parameter(torch.zeros(ROLE_N))
        def alpha(self):
            if arm=="R1_UNCONDITIONAL_API": return torch.full((ROLE_N,),.25,device=self.api_gate_logits.device)
            if arm in ("R2_FIXED_ROLE_API","R2S_FIXED_SHUFFLED"): return torch.full((ROLE_N,),.25,device=self.api_gate_logits.device)
            return .30*torch.sigmoid(self.api_gate_logits)
        def refine(self,core,api):
            role=core[:,:ROLE_N]
            if arm=="R1_UNCONDITIONAL_API":
                mask=(torch.linalg.vector_norm(api,dim=2)>0).float(); pooled=torch.sum(api*mask[:,:,None],dim=1)/torch.clamp(mask.sum(1,keepdim=True),min=1.0)
                delta=torch.tanh(self.api_unconditional(pooled))
            else: delta=torch.tanh(self.api_role_residual(api).squeeze(-1))
            refined=role+self.alpha()[None,:]*delta
            return torch.cat([refined,core[:,ROLE_N:]],1)
        def select_local_evidence(self,graph,regions):
            if regions.ndim!=2 or regions.shape[1]!=TOTAL: raise RuntimeError(f"Role API model expects N x {TOTAL}, got {tuple(regions.shape)}")
            core=regions[:,:CORE]; relation=regions[:,CORE:CORE+REL]; api=regions[:,CORE+REL:].reshape(-1,ROLE_N,API_DIM); core=self.refine(core,api)
            role,ge=self.encode_role_graph(graph["node"],graph["edge"]); gg=self.graphglobal_encoder(graph["global"].reshape(-1)); local=self.region_encoder(core)
            res=torch.log1p(torch.clamp(core[:,-1],min=0)); score=self.local_risk(local).reshape(-1)+nn.functional.softplus(self.residual_beta_raw)*res
            query=self.role_query(role)+self.global_to_query(gg); score=score+torch.mv(self.local_key(local),query)/math.sqrt(local.shape[1])+torch.mv(self.relation_key(relation),query)/math.sqrt(local.shape[1])
            att=torch.softmax(score,0); critical=torch.argmax(score); ent=-torch.sum(att*torch.log(att.clamp_min(1e-8)))/math.log(max(len(att),2))
            summary=torch.cat([score[critical].reshape(1),torch.mean(score).reshape(1),ent.reshape(1),torch.max(att).reshape(1),torch.stack([torch.mean(core[:,-4]),torch.mean(core[:,-3]),torch.mean(core[:,-2]),torch.max(core[:,-1])])])
            return LocalEvidenceCache(role_token=role,critical=local[critical],attended=torch.sum(att[:,None]*local,0),scalar_summary=summary,attention=att,score=score,graph_entropy=ge),gg
    return Model
def install_gate_capture(v4,arm):
    original=v4.predict_model
    def wrapped(model,*args,**kwargs):
        GATE_SNAPSHOTS.append({"arm":arm,"alpha":model.alpha().detach().cpu().numpy().astype(float).tolist()})
        return original(model,*args,**kwargs)
    v4.predict_model=wrapped
