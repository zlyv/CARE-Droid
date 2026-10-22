#!/usr/bin/env python3
"""Train equal-capacity S1/S3 Adapters on frozen 2017 W0 folds or full 2017.

Inputs are formal_split W0 assets plus v2 semantic sidecars. Outputs are
fold/seed or T1 checkpoints and held-out probabilities. Only semantic_* model
parameters receive gradients; supervision remains APK-level BCE.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from semantic_common import *
from semantic_model import make_model,set_train_mode,trainable_parameters

def predict(v4,model,g,graph,bags,dev): return v4.predict_model(model,g,np.full(len(bags),.5),graph,bags,dev)[0]

def frozen_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('semantic_')}
def maxdiff(a,b): return max([float(np.max(np.abs(a[k].numpy()-b[k].numpy()))) for k in a] or [0.])

def train_one(pack,arm,seed,epochs,trbags,vabags,labels):
    v4,args,dev=pack['v4'],pack['args'],pack['dev']; torch=v4.torch
    v4.seed_everything(seed+1000*(ADAPTER_ARMS.index(arm)+1)); model=make_model(v4,args,pack['old']['state_dict']).to(dev)
    frozen=frozen_state(model); params=trainable_parameters(model); opt=torch.optim.Adam(params,lr=float(args.learning_rate),weight_decay=float(args.weight_decay))
    gt=torch.as_tensor(pack['global_train'],dtype=torch.float32,device=dev); graph=v4.graph_tensors(pack['graph_train'],dev)
    bags=[torch.as_tensor(x,dtype=torch.float32,device=dev) for x in trbags]; y=torch.as_tensor(labels,dtype=torch.float32,device=dev)
    rng=np.random.RandomState(seed+17031); bs=int(args.batch_size); start=time.perf_counter(); loss_value=float('nan')
    if dev.type=='cuda': torch.cuda.reset_peak_memory_stats(dev)
    for _ in range(int(epochs)):
        set_train_mode(model); order=rng.permutation(len(labels)); total=0.
        for lo in range(0,len(order),bs):
            ix=order[lo:lo+bs]; opt.zero_grad(set_to_none=True)
            logits=v4.model_forward_batch(model,gt,torch.zeros(len(labels),dtype=torch.float32,device=dev),graph,bags,ix)
            target=y[torch.as_tensor(ix,dtype=torch.long,device=dev)]; loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,float(args.grad_clip)); opt.step(); total+=float(loss.detach().cpu())*len(ix)
        loss_value=total/len(labels)
    train_s=time.perf_counter()-start; model.eval(); start=time.perf_counter()
    if vabags is None:
        p=np.asarray([],np.float32); infer_s=0.0
    else:
        p=predict(v4,model,pack['global_val'],v4.graph_tensors(pack['graph_val'],dev),vabags,dev); infer_s=time.perf_counter()-start
    fd=maxdiff(frozen,frozen_state(model))
    if fd!=0: raise RuntimeError(f'frozen W0 changed {fd}')
    cost={'training_seconds':train_s,'inference_seconds':infer_s,'trainable_parameters':sum(x.numel() for x in params),'final_BCE':loss_value,'gpu_peak_mib':float(torch.cuda.max_memory_allocated(dev))/1024**2 if dev.type=='cuda' else 0.}
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}; return np.asarray(p),state,cost

def zero_replay(pack,arm,bags):
    v4,dev=pack['v4'],pack['dev']; model=make_model(v4,pack['args'],pack['old']['state_dict']).to(dev); model.eval()
    p=predict(v4,model,pack['global_val'],v4.graph_tensors(pack['graph_val'],dev),bags,dev)
    return float(np.max(np.abs(np.asarray(p)-np.asarray(pack['old']['probability']))))

def smoke(device):
    from bfa_core import formal_split,load_data_contract
    data,_=load_data_contract(); pack=formal_split(data,'S1',SEEDS[0],device); counts=[]; rows={}
    for arm in ADAPTER_ARMS:
        bags,audit=build_semantic_bags(pack['bags_val'],pack['raw_val'],data['ids'][pack['val']],arm,SEEDS[0]); d=zero_replay(pack,arm,bags)
        m=make_model(pack['v4'],pack['args'],pack['old']['state_dict']); n=sum(p.numel() for p in m.parameters() if p.requires_grad); counts.append(n)
        if d>1e-7: raise RuntimeError(f'zero replay failed {arm}: {d}')
        rows[arm]={'zero_replay_max_abs':d,'trainable_parameters':n,'audit':audit}
    if len(set(counts))!=1: raise RuntimeError('capacity mismatch')
    # One real APK-level optimization step verifies gradients are confined to the Adapter.
    trbags,_=build_semantic_bags(pack['bags_train'],pack['raw_train'],data['ids'][pack['train']],'C1_S1',SEEDS[0]); v4,dev=pack['v4'],pack['dev']; torch=v4.torch
    m=make_model(v4,pack['args'],pack['old']['state_dict']).to(dev); before=frozen_state(m); params=trainable_parameters(m); opt=torch.optim.Adam(params,lr=float(pack['args'].learning_rate)); set_train_mode(m)
    gt=torch.as_tensor(pack['global_train'],dtype=torch.float32,device=dev); graph=v4.graph_tensors(pack['graph_train'],dev); bt=[torch.as_tensor(x,dtype=torch.float32,device=dev) for x in trbags]; ix=np.asarray([0,1],int)
    logits=v4.model_forward_batch(m,gt,torch.zeros(len(trbags),dtype=torch.float32,device=dev),graph,bt,ix); target=torch.as_tensor(np.asarray(data['y'][pack['train']])[ix],dtype=torch.float32,device=dev); loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target); opt.zero_grad(); loss.backward(); opt.step(); frozen_change=maxdiff(before,frozen_state(m))
    if frozen_change!=0: raise RuntimeError('smoke optimization changed W0')
    atomic_json(OUT/'audit/unit_test_results.json',{'status':'PASS','equal_capacity':True,'trainable_parameters':counts[0],'one_training_step_BCE':float(loss.detach().cpu()),'one_step_frozen_W0_max_change':frozen_change,'arms':rows})
    print('SEMANTIC_ADAPTER_SMOKE_PASS',flush=True)

def run(mode,device,resume):
    from bfa_core import formal_split,load_data_contract
    data,_=load_data_contract(); splits=SOURCE_SPLITS if mode=='source' else ('T1',)
    for split in splits:
      for seed in SEEDS:
        pack=formal_split(data,split,seed,device); labels=np.asarray(data['y'][pack['train']],int); epoch=int(pack['old']['selected_epochs'])
        for arm in ADAPTER_ARMS:
            path=OUT/'models'/split/str(seed)/(arm+'.pkl')
            if path.is_file() and resume: print('RESUME',split,seed,arm,flush=True); continue
            if path.exists(): raise RuntimeError('checkpoint exists without resume')
            tr,ta=build_semantic_bags(pack['bags_train'],pack['raw_train'],data['ids'][pack['train']],arm,seed)
            if split=='T1':
                va=None; vaudit={'status':'DEFERRED_UNTIL_SOURCE_FREEZE','future_semantics_or_labels_accessed':False}
            else:
                va,vaudit=build_semantic_bags(pack['bags_val'],pack['raw_val'],data['ids'][pack['val']],arm,seed)
            p,state,cost=train_one(pack,arm,seed,epoch,tr,va,labels)
            heldout_ids=[] if split=='T1' else list(map(str,data['ids'][pack['val']]))
            write_pickle(path,{'arm':arm,'split':split,'seed':seed,'ids':heldout_ids,'probability':p,'state_dict':state,'selected_epochs':epoch,'cost':cost,'train_audit':ta,'val_audit':vaudit,'training_years':[2017],'future_labels_used':False})
            print('ADAPTER_DONE',split,seed,arm,'epochs='+str(epoch),flush=True)

def main():
    ap=argparse.ArgumentParser(); g=ap.add_mutually_exclusive_group(required=True); g.add_argument('--smoke',action='store_true'); g.add_argument('--source',action='store_true'); g.add_argument('--full',action='store_true')
    ap.add_argument('--device',default='cuda',choices=('cpu','cuda')); ap.add_argument('--resume',action='store_true'); a=ap.parse_args(); os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if a.smoke: smoke(a.device)
    else: run('source' if a.source else 'full',a.device,a.resume)
if __name__=='__main__': main()

