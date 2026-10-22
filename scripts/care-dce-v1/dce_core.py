#!/usr/bin/env python3
"""DCE-V1 frozen fused-vector extraction and head models; reads cached tensors, writes nowhere; no APK access."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

WIDTH=113
ARMS=('D0_ORIGINAL','D1_SINGLE_CAPACITY','D2_DCE','D3_SHUFFLED_DIRECTION','D4_DIRECTION_FEATURE')

class Head(nn.Module):
    def __init__(self,arm):
        super().__init__()
        self.arm=arm
        if arm in (ARMS[1],ARMS[2],ARMS[3]):
            self.left=nn.Linear(WIDTH,1)
            self.right=nn.Linear(WIDTH,1)
            nn.init.zeros_(self.left.weight);nn.init.zeros_(self.left.bias)
            nn.init.zeros_(self.right.weight);nn.init.zeros_(self.right.bias)
        elif arm==ARMS[4]:
            self.single=nn.Linear(WIDTH+1,2)
            nn.init.zeros_(self.single.weight);nn.init.zeros_(self.single.bias)
        else:raise ValueError(arm)

    def forward(self,z,base,direction):
        if self.arm==ARMS[1]:
            delta=self.left(z).reshape(-1)+self.right(z).reshape(-1)
        elif self.arm in (ARMS[2],ARMS[3]):
            a=self.left(z).reshape(-1);b=self.right(z).reshape(-1)
            delta=torch.where(direction.reshape(-1)>0,b,a)
        else:
            x=torch.cat([z,direction.reshape(-1,1).to(z.dtype)],dim=1)
            delta=self.single(x).sum(dim=1)
        return base.reshape(-1)+delta

def fit_scaler(z):
    a=np.asarray(z,np.float64)
    mean=a.mean(0);std=a.std(0)
    std=np.where(std>1e-6,std,1.)
    return mean.astype(np.float32),std.astype(np.float32)

def scale(z,mean,std):
    a=np.asarray((z-mean)/std,np.float32)
    if a.ndim!=2 or a.shape[1]!=WIDTH or not np.isfinite(a).all():
        raise RuntimeError('invalid frozen head coordinate')
    return a

def extract(v4,model,global_values,graphs,bags,device):
    """Exact Direct-C43 forward_with_trace pre-head vector, replayed against frozen probability."""
    gt=torch.as_tensor(global_values,dtype=torch.float32,device=device)
    graph=v4.graph_tensors(graphs,device)
    vectors=[];logits=[]
    model.eval()
    with torch.no_grad():
        for i,bag in enumerate(bags):
            one={k:v[i] for k,v in graph.items()}
            tensor=torch.as_tensor(bag,dtype=torch.float32,device=device)
            if tensor.ndim!=2 or tensor.shape[1]!=142:raise RuntimeError('Host 142D contract')
            logit,_,trace=model.forward_with_trace(gt[i],torch.zeros((),device=device),one,tensor)
            fused=trace['correction_input'].reshape(-1)
            if fused.numel()!=WIDTH or abs(float(fused[24]))>1e-7:
                raise RuntimeError('Direct-C43 pre-head tensor/base slot mismatch')
            vectors.append(fused.cpu().numpy());logits.append(float(logit.cpu()))
    z=np.asarray(vectors,np.float32);p=np.asarray(logits,np.float32)
    if not (np.isfinite(z).all() and np.isfinite(p).all()):raise RuntimeError('nonfinite frozen encoding')
    return z,p

def sigmoid(x):
    x=np.asarray(x,np.float64)
    return 1/(1+np.exp(-np.clip(x,-50,50)))

def fit_heads(z,base,y,direction,seed,lr=1e-3,weight_decay=1e-4,max_epochs=30,
              patience=5,shuffle_seed=None):
    """Same BCE/Adam/holdout policy for D1-D4; trains only small heads, no APK access."""
    from sklearn.model_selection import StratifiedShuffleSplit
    n=len(y);y=np.asarray(y,int);direction=np.asarray(direction,int)
    if z.shape!=(n,WIDTH) or base.shape!=(n,) or direction.shape!=(n,):raise RuntimeError('head inputs')
    if set(np.unique(direction))!={0,1}:raise RuntimeError('missing Stage-I direction')
    split=StratifiedShuffleSplit(n_splits=1,test_size=.15,random_state=seed+681)
    fit,hold=next(split.split(np.zeros(n),y))
    perm=np.random.RandomState(seed+1123 if shuffle_seed is None else shuffle_seed).permutation(n)
    shuffled=direction[perm]
    if sorted(shuffled.tolist())!=sorted(direction.tolist()):raise RuntimeError('shuffle counts changed')
    audit=dict(seed=int(seed),fit_n=len(fit),hold_n=len(hold),
        train_direction_benign=int((direction==0).sum()),train_direction_malicious=int((direction==1).sum()),
        benign_head_benign=int(((direction==0)&(y==0)).sum()),
        benign_head_malware=int(((direction==0)&(y==1)).sum()),
        malware_head_benign=int(((direction==1)&(y==0)).sum()),
        malware_head_malware=int(((direction==1)&(y==1)).sum()),
        shuffled_changed_fraction=float(np.mean(shuffled!=direction)),
        oof_direction_not_true_label=True)
    results={}
    for arm in ARMS[1:]:
        torch.manual_seed(seed)
        head=Head(arm)
        opt=torch.optim.Adam(head.parameters(),lr=lr,weight_decay=weight_decay)
        mean,std=fit_scaler(z[fit]);xf=scale(z[fit],mean,std);xh=scale(z[hold],mean,std)
        zz=torch.from_numpy(xf);vh=torch.from_numpy(xh)
        bb=torch.from_numpy(np.asarray(base[fit],np.float32));bh=torch.from_numpy(np.asarray(base[hold],np.float32))
        yy=torch.from_numpy(np.asarray(y[fit],np.float32));yh=torch.from_numpy(np.asarray(y[hold],np.float32))
        dd=torch.from_numpy(np.asarray(shuffled[fit] if arm==ARMS[3] else direction[fit],np.float32))
        dh=torch.from_numpy(np.asarray(direction[hold],np.float32))
        best=float('inf');best_epoch=0;best_train=float('nan');wait=0
        for epoch in range(1,max_epochs+1):
            head.train();opt.zero_grad(set_to_none=True)
            logit=head(zz,bb,dd)
            loss=nn.functional.binary_cross_entropy_with_logits(logit,yy)
            loss.backward();opt.step()
            head.eval()
            with torch.no_grad():
                val=nn.functional.binary_cross_entropy_with_logits(head(vh,bh,dh),yh).item()
            if val<best-1e-6:
                best=val;best_epoch=epoch;best_train=float(loss);wait=0
            else:wait+=1
            if wait>=patience:break
        if best_epoch<1:raise RuntimeError('invalid head early stopping')
        # Refit on all outer-train APKs for the source-selected epoch count.
        mean,std=fit_scaler(z);allx=torch.from_numpy(scale(z,mean,std));allb=torch.from_numpy(np.asarray(base,np.float32))
        ally=torch.from_numpy(np.asarray(y,np.float32));alld=torch.from_numpy(np.asarray(shuffled if arm==ARMS[3] else direction,np.float32))
        torch.manual_seed(seed);head=Head(arm)
        opt=torch.optim.Adam(head.parameters(),lr=lr,weight_decay=weight_decay)
        for _ in range(best_epoch):
            head.train();opt.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(head(allx,allb,alld),ally)
            loss.backward();opt.step()
        state={k:v.detach().cpu() for k,v in head.state_dict().items()}
        results[arm]=dict(state_dict=state,mean=mean,std=std,best_epoch=best_epoch,
            train_loss=float(loss),validation_loss=best,inner_train_loss=best_train,
            parameter_n=sum(v.numel() for v in head.parameters()),
            shuffle_seed=(seed+1123 if shuffle_seed is None else shuffle_seed) if arm==ARMS[3] else None,
            train_direction=shuffled if arm==ARMS[3] else direction)
    return results,audit

def predict_head(arm,obj,z,base,direction):
    head=Head(arm);head.load_state_dict(obj['state_dict']);head.eval()
    x=torch.from_numpy(scale(z,obj['mean'],obj['std']))
    b=torch.from_numpy(np.asarray(base,np.float32));d=torch.from_numpy(np.asarray(direction,np.float32))
    with torch.no_grad():logit=head(x,b,d).numpy()
    p=sigmoid(logit)
    if not np.isfinite(p).all():raise RuntimeError('invalid head output')
    return p
