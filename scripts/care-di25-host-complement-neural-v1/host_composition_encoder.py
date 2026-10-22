#!/usr/bin/env python3
import torch
from torch import nn

class HostSemanticCompositionEncoder(nn.Module):
    """Pair-only semantic composition; single-role Hosts map exactly to zero."""
    def __init__(self, role_count:int, output_dim:int=24):
        super().__init__(); self.role_count=int(role_count); self.output_dim=int(output_dim)
        embed_dim=max(4,output_dim//3)
        self.role_embedding=nn.Embedding(self.role_count,embed_dim)
        self.pair_mlp=nn.Sequential(nn.Linear(3*embed_dim,output_dim),nn.ReLU(),nn.Linear(output_dim,output_dim))
        self.pool=nn.Linear(2*output_dim,output_dim)

    @staticmethod
    def unique_membership(role_ids, role_count:int):
        result=torch.zeros(int(role_count),dtype=torch.int64)
        for value in set(int(x) for x in role_ids):
            if 0 <= value < role_count: result[value]=1
        return result

    @staticmethod
    def pair_count(membership:torch.Tensor)->int:
        n=int(torch.sum(membership>0.5).item())
        return n*(n-1)//2

    def forward(self,membership:torch.Tensor):
        rows=[]
        for mask in membership>0.5:
            ids=torch.nonzero(mask,as_tuple=False).reshape(-1)
            if len(ids)<2:
                rows.append(torch.zeros(self.output_dim,device=membership.device,dtype=membership.dtype));continue
            pairs=[]
            for i in range(len(ids)):
                for j in range(i+1,len(ids)):
                    left,right=self.role_embedding(ids[i]),self.role_embedding(ids[j])
                    pairs.append(self.pair_mlp(torch.cat([left,right,left*right])))
            values=torch.stack(pairs);rows.append(self.pool(torch.cat([values.mean(0),values.max(0).values])))
        return torch.stack(rows)
