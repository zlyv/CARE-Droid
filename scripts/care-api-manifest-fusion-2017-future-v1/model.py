#!/usr/bin/env python3
"""Equal-capacity residual fusion head for frozen W0 code, API and Manifest views.

Inputs are frozen W0 113D readout tensors plus fixed-width API512 and Manifest135
blocks. Outputs are independent Stage-II malware probabilities and isolated checkpoints.
"""
from __future__ import annotations

import copy, time
import numpy as np


def make_model(torch,dropout=.10):
    nn=torch.nn
    class FusionHead(nn.Module):
        def __init__(self):
            super().__init__();self.code=nn.Sequential(nn.Linear(113,32),nn.ReLU());self.api=nn.Sequential(nn.Linear(512,16,bias=False),nn.ReLU());self.manifest=nn.Sequential(nn.Linear(135,16,bias=False),nn.ReLU());self.fuse=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,16),nn.ReLU());self.delta=nn.Linear(16,1,bias=False);nn.init.zeros_(self.delta.weight)
        def forward(self,base_logit,code,api,manifest):
            z=torch.cat([self.code(code),self.api(api),self.manifest(manifest)],1);return base_logit.reshape(-1)+self.delta(self.fuse(z)).reshape(-1)
    return FusionHead()


def train_model(torch,code_train,api_train,manifest_train,base_train,y_train,code_target,api_target,manifest_target,base_target,seed,device,epochs=40):
    torch.manual_seed(int(seed));np.random.seed(int(seed)&0xffffffff)
    if device.type=='cuda':torch.cuda.manual_seed_all(int(seed))
    model=make_model(torch).to(device);model.eval()
    def tensor(x):return torch.as_tensor(np.asarray(x,np.float32),dtype=torch.float32,device=device)
    ct,at,mt,bt,yt=map(tensor,[code_train,api_train,manifest_train,base_train,y_train]);cv,av,mv,bv=map(tensor,[code_target,api_target,manifest_target,base_target])
    with torch.no_grad():zero=model(bt,ct,at*0,mt*0);zero_diff=float(torch.max(torch.abs(zero-bt)).cpu())
    if zero_diff!=0.:raise RuntimeError('zero residual does not exactly replay W0')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);lossfn=torch.nn.BCEWithLogitsLoss();started=time.perf_counter()
    for _ in range(int(epochs)):
        model.train();opt.zero_grad(set_to_none=True);loss=lossfn(model(bt,ct,at,mt),yt);loss.backward();opt.step()
    model.eval()
    with torch.no_grad():pred=torch.sigmoid(model(bv,cv,av,mv)).cpu().numpy();train_pred=torch.sigmoid(model(bt,ct,at,mt)).cpu().numpy()
    state={k:v.detach().cpu() for k,v in model.state_dict().items()};params=sum(p.numel() for p in model.parameters())
    return pred,train_pred,state,{'epochs':epochs,'parameters':params,'seconds':time.perf_counter()-started,'zero_residual_max_logit_diff':zero_diff,'final_train_loss':float(loss.detach().cpu())}


def predict_model(torch,state,code,api,manifest,base,device):
    model=make_model(torch).to(device);model.load_state_dict(state);model.eval()
    def t(x):return torch.as_tensor(np.asarray(x,np.float32),dtype=torch.float32,device=device)
    with torch.no_grad():return torch.sigmoid(model(t(base),t(code),t(api),t(manifest))).cpu().numpy()
