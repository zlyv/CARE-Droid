#!/usr/bin/env python3
"""HDBC-V1 fixed-capacity fusion head. Reads arrays only, not APKs; caller writes new HDBC outputs."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedShuffleSplit

ARMS=('H0_ORIGINAL','H1_CAPACITY_CONTROL','H2_ROLE_PAIR_BASELINE',
      'H3_UNDIRECTED_HOST','H4_HDBC_DIRECTED','H5_DIRECTION_SHUFFLE')

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(128,16),nn.GELU())
        self.readout=nn.Linear(113+16,1)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)
    def forward(self,z,x,baseline):
        return baseline.reshape(-1)+self.readout(torch.cat([z,self.project(x)],1)).reshape(-1)

def scaler(z,x):
    allv=np.concatenate([z,x],axis=1).astype(np.float64)
    mean=allv.mean(0);std=allv.std(0)
    std=np.where(std>1e-6,std,1.)
    return mean.astype(np.float32),std.astype(np.float32)

def transform(z,x,mean,std):
    allv=np.concatenate([z,x],axis=1)
    if allv.shape[1]!=241:raise RuntimeError('fusion width contract')
    val=np.clip((allv-mean)/std,-10,10).astype(np.float32)
    if not np.isfinite(val).all():raise RuntimeError('nonfinite fusion input')
    return val[:,:113],val[:,113:]

def fit(z,x,baseline,y,seed,lr,weight_decay,max_epochs=30,patience=5):
    y=np.asarray(y,int);baseline=np.asarray(baseline,np.float32)
    if len(y)!=len(z) or x.shape!=(len(y),128):raise RuntimeError('fusion sample contract')
    split=StratifiedShuffleSplit(n_splits=1,test_size=.15,random_state=seed+817)
    it,iv=next(split.split(np.zeros(len(y)),y))
    m,s=scaler(z[it],x[it]);zt,xt=transform(z[it],x[it],m,s);zv,xv=transform(z[iv],x[iv],m,s)
    torch.manual_seed(seed);model=Fusion();opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    zt=torch.from_numpy(zt);xt=torch.from_numpy(xt);bt=torch.from_numpy(baseline[it]);yt=torch.from_numpy(y[it].astype(np.float32))
    zv=torch.from_numpy(zv);xv=torch.from_numpy(xv);bv=torch.from_numpy(baseline[iv]);yv=torch.from_numpy(y[iv].astype(np.float32))
    best=float('inf');epoch_best=0;wait=0
    for epoch in range(1,max_epochs+1):
        model.train();opt.zero_grad(set_to_none=True)
        loss=nn.functional.binary_cross_entropy_with_logits(model(zt,xt,bt),yt)
        loss.backward();opt.step()
        model.eval()
        with torch.no_grad():val=float(nn.functional.binary_cross_entropy_with_logits(model(zv,xv,bv),yv))
        if val<best-1e-6:best=val;epoch_best=epoch;wait=0
        else:wait+=1
        if wait>=patience:break
    if epoch_best<1:raise RuntimeError('early stop invalid')
    m,s=scaler(z,x);za,xa=transform(z,x,m,s)
    torch.manual_seed(seed);model=Fusion();opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    za=torch.from_numpy(za);xa=torch.from_numpy(xa);ba=torch.from_numpy(baseline);ya=torch.from_numpy(y.astype(np.float32))
    for _ in range(epoch_best):
        model.train();opt.zero_grad(set_to_none=True)
        loss=nn.functional.binary_cross_entropy_with_logits(model(za,xa,ba),ya)
        loss.backward();opt.step()
    return dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},mean=m,std=s,
                best_epoch=epoch_best,validation_loss=best,train_loss=float(loss),
                parameter_n=sum(p.numel() for p in model.parameters()))

def predict(fit_result,z,x,baseline):
    model=Fusion();model.load_state_dict(fit_result['state']);model.eval()
    zz,xx=transform(z,x,fit_result['mean'],fit_result['std'])
    with torch.no_grad():v=model(torch.from_numpy(zz),torch.from_numpy(xx),torch.from_numpy(np.asarray(baseline,np.float32))).numpy()
    out=1/(1+np.exp(-np.clip(v,-50,50)))
    if not np.isfinite(out).all():raise RuntimeError('nonfinite fusion probability')
    return out
