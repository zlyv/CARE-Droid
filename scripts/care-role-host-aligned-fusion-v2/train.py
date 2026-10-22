#!/usr/bin/env python3
"""Train three equal-capacity static Role-Host fusion arms; reads no APKs and writes only isolated v2 checkpoints."""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from rhaf_common import (ARMS,OUT,SEEDS,SPLITS,atomic_json,augment_bags,
                         preflight,verify_assets,write_pickle)
from model import make_model


def train_one(pack,seed,epochs,bags_train,bags_val,labels):
    v4,args,dev=pack['v4'],pack['args'],pack['dev']; torch=v4.torch
    neutral_train=np.full(len(labels),.5); neutral_val=np.full(len(bags_val),.5)
    weights,correct=v4.fixkeep_weights(labels,neutral_train,args)
    if not np.array_equal(weights,np.ones(len(labels),np.float32)) or args.keep_consistency!=0:
        raise RuntimeError('equal BCE contract changed')
    v4.seed_everything(seed); model=make_model(v4,args).to(dev)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    global_tensor=torch.as_tensor(pack['global_train'],dtype=torch.float32,device=dev)
    graph=v4.graph_tensors(pack['graph_train'],dev)
    bags=[torch.as_tensor(x,dtype=torch.float32,device=dev) for x in bags_train]
    if dev.type=='cuda': torch.cuda.reset_peak_memory_stats(dev)
    started=time.perf_counter()
    loss=v4.train_epochs(model,optimizer,global_tensor,neutral_train,graph,bags,labels,weights,correct,epochs,args,seed+5901)
    training_seconds=time.perf_counter()-started; started=time.perf_counter()
    probability,_=v4.predict_model(model,pack['global_val'],neutral_val,pack['graph_val'],bags_val,dev)
    inference_seconds=time.perf_counter()-started
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    cost={'training_seconds':training_seconds,'inference_seconds':inference_seconds,
          'parameters':sum(v.numel() for v in model.parameters()),'final_BCE':float(loss),
          'gpu_peak_mib':float(torch.cuda.max_memory_allocated(dev))/1024**2 if dev.type=='cuda' else 0.}
    scales={'role_residual_scale':float(torch.tanh(model.role_residual_scale).detach().cpu()),
            'host_residual_scale':float(torch.tanh(model.host_residual_scale).detach().cpu())}
    del model,optimizer,global_tensor,graph,bags
    if dev.type=='cuda': torch.cuda.empty_cache()
    if not np.isfinite(probability).all(): raise RuntimeError('nonfinite probability')
    return np.asarray(probability),state,cost,scales


def run(which,device,resume):
    from bfa_core import formal_split,load_data_contract
    if verify_assets()['status']!='PASS': raise RuntimeError('assets changed before training')
    data,_=load_data_contract(); splits=SPLITS[:5] if which=='source' else SPLITS[5:]
    for split in splits:
        for seed in SEEDS:
            pack=formal_split(data,split,seed,device); labels=np.asarray(data['y'][pack['train']],int)
            epochs=int(pack['old']['selected_epochs'])
            for arm in ARMS[1:]:
                path=OUT/'models'/split/str(seed)/(arm+'.pkl')
                if path.is_file():
                    if not resume: raise RuntimeError('checkpoint exists without --resume')
                    print('RHAF_RESUME',split,seed,arm,flush=True); continue
                train_bags,audit_train=augment_bags(pack['bags_train'],pack['raw_train'],data['ids'][pack['train']],arm,seed)
                val_bags,audit_val=augment_bags(pack['bags_val'],pack['raw_val'],data['ids'][pack['val']],arm,seed)
                probability,state,cost,scales=train_one(pack,seed,epochs,train_bags,val_bags,labels)
                write_pickle(path,{'arm':arm,'split':split,'seed':seed,'ids':list(map(str,data['ids'][pack['val']])),
                    'probability':probability,'state_dict':state,'selected_epochs':epochs,'cost':cost,
                    'fusion_scales':scales,'train_binding_audit':audit_train,'val_binding_audit':audit_val,
                    'Stage1_changed':False,'Router_changed':False,'backgrounds_changed':False,'dynamic_analysis':False})
                print('RHAF_ARM_DONE',split,seed,arm,'epochs='+str(epochs),flush=True)
            print('RHAF_SPLIT_SEED_DONE',split,seed,flush=True)


def smoke(device):
    from bfa_core import formal_split,load_data_contract
    data,_=load_data_contract(); pack=formal_split(data,'S1',SEEDS[0],device)
    audits={}; predictions={}
    for arm in ARMS[1:]:
        bags,audit=augment_bags(pack['bags_val'],pack['raw_val'],data['ids'][pack['val']],arm,SEEDS[0]); audits[arm]=audit
        model=make_model(pack['v4'],pack['args']).to(pack['dev'])
        probability,_=pack['v4'].predict_model(model,pack['global_val'],np.full(len(bags),.5),pack['graph_val'],bags,pack['dev'])
        if len(probability)!=len(bags) or not np.isfinite(probability).all(): raise RuntimeError('smoke prediction failed')
        predictions[arm]={'n':len(probability),'parameters':sum(v.numel() for v in model.parameters())}
        del model
    if len({v['parameters'] for v in predictions.values()})!=1: raise RuntimeError('control capacities differ')
    atomic_json(OUT/'smoke_test.json',{'status':'PASS','split':'S1','baseline_replay_max_difference':pack['replay_diff'],
                'audits':audits,'predictions':predictions})
    print('RHAF_SMOKE_PASS',flush=True)


def main():
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight',action='store_true'); group.add_argument('--smoke-test',action='store_true')
    group.add_argument('--train-source',action='store_true'); group.add_argument('--train-rolling',action='store_true')
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args(); os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if args.preflight: preflight(args.resume); print('RHAF_PREFLIGHT_PASS',flush=True); return
    if verify_assets()['status']!='PASS': raise RuntimeError('run preflight first')
    if args.device=='cuda':
        import torch; probe=torch.zeros(1,device='cuda'); del probe
    if args.smoke_test: smoke(args.device)
    elif args.train_source: run('source',args.device,args.resume)
    else: run('rolling',args.device,args.resume)


if __name__=='__main__': main()
