#!/usr/bin/env python3
import torch
from torch import nn

POSITION_DIM=13

def structural_vector(host_pred,host_succ,ordinary_pred,ordinary_succ,all_pred=None,all_succ=None):
    hp,hs,op,os=[float(x) for x in (host_pred,host_succ,ordinary_pred,ordinary_succ)]
    ai=float(hp+op if all_pred is None else all_pred);ao=float(hs+os if all_succ is None else all_succ); eps=1e-8
    source=float(ai==0 and ao>0);sink=float(ai>0 and ao==0);bridge=float(ai>0 and ao>0);isolated=float(ai==0 and ao==0)
    return [hp/(ai+eps),hs/(ao+eps),op/(ai+eps),os/(ao+eps),(ao-ai)/(ao+ai+eps),(hs-hp)/(hs+hp+eps),(os-op)/(os+op+eps),source,sink,bridge,isolated,__import__('math').log1p(ai),__import__('math').log1p(ao)]

class HostStructuralPositionEncoder(nn.Module):
    def __init__(self,output_dim:int=24):
        super().__init__();self.output_dim=int(output_dim);self.net=nn.Sequential(nn.Linear(POSITION_DIM,output_dim),nn.ReLU(),nn.Linear(output_dim,output_dim))
    def forward(self,values:torch.Tensor):return self.net(values)
