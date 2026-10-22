#!/usr/bin/env python3
"""Train only the equal-capacity HPRN branches on 2017; the complete W0 backbone stays frozen."""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from hprn_common import (ARMS, OUT, RELATION_ARMS, SEEDS, SOURCE_SPLITS,
                         TRAIN_SPLITS, atomic_json, branch_state,
                         build_relation_bags, frozen_state, preflight,
                         state_max_difference, verify_assets, write_pickle)
from model import make_model, set_train_mode, trainable_parameters


def predict(v4,model,global_values,graph_values,bags,device):
    return v4.predict_model(model,global_values,np.full(len(bags),.5),graph_values,bags,device)[0]


def train_one(pack,arm,seed,epochs,bags_train,bags_val,labels):
    v4,args,dev=pack['v4'],pack['args'],pack['dev']; torch=v4.torch
    v4.seed_everything(seed+1000*(RELATION_ARMS.index(arm)+1))
    model=make_model(v4,args,arm,pack['old']['state_dict']).to(dev)
    original=frozen_state(model); params=trainable_parameters(model)
    optimizer=torch.optim.Adam(params,lr=float(args.learning_rate),weight_decay=float(args.weight_decay))
    global_tensor=torch.as_tensor(pack['global_train'],dtype=torch.float32,device=dev)
    graph=v4.graph_tensors(pack['graph_train'],dev)
    bag_tensors=[torch.as_tensor(x,dtype=torch.float32,device=dev) for x in bags_train]
    labels_t=torch.as_tensor(labels,dtype=torch.float32,device=dev)
    rng=np.random.RandomState(seed+7301); batch_size=int(args.batch_size); final_loss=float('nan')
    if dev.type=='cuda': torch.cuda.reset_peak_memory_stats(dev)
    started=time.perf_counter()
    for epoch in range(int(epochs)):
        set_train_mode(model); order=rng.permutation(len(labels)); total=0.0
        for start in range(0,len(order),batch_size):
            batch=order[start:start+batch_size]; optimizer.zero_grad(set_to_none=True)
            logits=v4.model_forward_batch(model,global_tensor,
                torch.zeros(len(labels),dtype=torch.float32,device=dev),graph,bag_tensors,batch)
            target=labels_t[torch.as_tensor(batch,dtype=torch.long,device=dev)]
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,float(args.grad_clip)); optimizer.step()
            total+=float(loss.detach().cpu())*len(batch)
        final_loss=total/len(labels)
    training_seconds=time.perf_counter()-started; model.eval(); started=time.perf_counter()
    probability=predict(v4,model,pack['global_val'],pack['graph_val'],bags_val,dev)
    inference_seconds=time.perf_counter()-started
    frozen_diff=state_max_difference(original,frozen_state(model))
    if frozen_diff!=0: raise RuntimeError(f'frozen W0 parameters changed: {frozen_diff}')
    cost={'training_seconds':training_seconds,'inference_seconds':inference_seconds,
          'total_parameters':sum(p.numel() for p in model.parameters()),
          'trainable_parameters':sum(p.numel() for p in params),'final_BCE':final_loss,
          'gpu_peak_mib':float(torch.cuda.max_memory_allocated(dev))/1024**2 if dev.type=='cuda' else 0.0}
    delta_norm=float(model.hprn_delta.weight.detach().norm().cpu())
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    del model,optimizer,global_tensor,graph,bag_tensors
    if dev.type=='cuda': torch.cuda.empty_cache()
    if not np.isfinite(probability).all(): raise RuntimeError('nonfinite HPRN probability')
    return np.asarray(probability),state,cost,delta_norm,frozen_diff


def zero_replay(pack,arm,bags):
    v4,dev=pack['v4'],pack['dev']; model=make_model(v4,pack['args'],arm,pack['old']['state_dict']).to(dev); model.eval()
    probability=predict(v4,model,pack['global_val'],pack['graph_val'],bags,dev)
    diff=float(np.max(np.abs(np.asarray(probability)-np.asarray(pack['old']['probability']))))
    base=pack['model']; base.eval(); max_attention=0.; critical_mismatch=0
    graph=v4.graph_tensors(pack['graph_val'],dev)
    with v4.torch.no_grad():
        for i in range(len(bags)):
            one={k:v[i] for k,v in graph.items()}; b=v4.torch.as_tensor(pack['bags_val'][i],dtype=v4.torch.float32,device=dev)
            a=v4.torch.as_tensor(bags[i],dtype=v4.torch.float32,device=dev)
            cb,_=base.select_local_evidence(one,b); ch,_=model.select_local_evidence(one,a)
            max_attention=max(max_attention,float(v4.torch.max(v4.torch.abs(cb.attention-ch.attention)).cpu()))
            critical_mismatch+=int(int(v4.torch.argmax(cb.score))!=int(v4.torch.argmax(ch.score)))
    del model,graph
    if dev.type=='cuda': v4.torch.cuda.empty_cache()
    return {'probability_max_abs_difference':diff,'attention_max_abs_difference':max_attention,
            'critical_host_mismatches':critical_mismatch,'n':len(bags)}


def smoke(device):
    from bfa_core import formal_split,load_data_contract
    data,_=load_data_contract(); pack=formal_split(data,'S1',SEEDS[0],device)
    results={}; parameters=[]
    for arm in RELATION_ARMS:
        bags,audit=build_relation_bags(pack['bags_val'],pack['raw_val'],data['ids'][pack['val']],pack['old']['background'],arm,SEEDS[0])
        replay=zero_replay(pack,arm,bags); model=make_model(pack['v4'],pack['args'],arm,pack['old']['state_dict'])
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); parameters.append(trainable)
        results[arm]={'audit':audit,'zero_residual_replay':replay,'trainable_parameters':trainable}
        if replay['probability_max_abs_difference']>1e-7 or replay['attention_max_abs_difference']>1e-7 or replay['critical_host_mismatches']:
            raise RuntimeError(f'zero residual replay failed {arm}: {replay}')
    if len(set(parameters))!=1: raise RuntimeError('R1-R4 trainable capacity differs')
    atomic_json(OUT/'smoke_test.json',{'status':'PASS','W0_replay_max_difference':pack['replay_diff'],
        'equal_trainable_capacity':True,'trainable_parameters':parameters[0],'arms':results})
    print('HPRN_SMOKE_PASS',flush=True)


def run(which,device,resume):
    from bfa_core import formal_split,load_data_contract
    if verify_assets()['status']!='PASS': raise RuntimeError('assets changed before training')
    data,_=load_data_contract(); splits=SOURCE_SPLITS if which=='source' else ('T1',)
    for split in splits:
        for seed in SEEDS:
            pack=formal_split(data,split,seed,device); labels=np.asarray(data['y'][pack['train']],int)
            epochs=int(pack['old']['selected_epochs']); counts=[]
            for arm in RELATION_ARMS:
                path=OUT/'models'/split/str(seed)/(arm+'.pkl')
                if path.is_file():
                    if not resume: raise RuntimeError('checkpoint exists without --resume')
                    print('HPRN_RESUME',split,seed,arm,flush=True); continue
                train_bags,audit_train=build_relation_bags(pack['bags_train'],pack['raw_train'],data['ids'][pack['train']],pack['old']['background'],arm,seed)
                val_bags,audit_val=build_relation_bags(pack['bags_val'],pack['raw_val'],data['ids'][pack['val']],pack['old']['background'],arm,seed)
                probability,state,cost,delta_norm,frozen_diff=train_one(pack,arm,seed,epochs,train_bags,val_bags,labels)
                counts.append(cost['trainable_parameters'])
                write_pickle(path,{'arm':arm,'split':split,'seed':seed,'ids':list(map(str,data['ids'][pack['val']])),
                    'probability':probability,'state_dict':state,'selected_epochs':epochs,'cost':cost,
                    'delta_weight_norm':delta_norm,'frozen_parameter_max_change':frozen_diff,
                    'train_sidecar_audit':audit_train,'val_sidecar_audit':audit_val,
                    'baseline':'W0','training_years':[2017],'future_used_for_training':False})
                print('HPRN_ARM_DONE',split,seed,arm,'epochs='+str(epochs),flush=True)
            if counts and len(set(counts))!=1: raise RuntimeError('capacity mismatch')
            print('HPRN_SPLIT_SEED_DONE',split,seed,flush=True)


def main():
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight',action='store_true'); group.add_argument('--smoke-test',action='store_true')
    group.add_argument('--train-source',action='store_true'); group.add_argument('--train-full2017',action='store_true')
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args(); os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if args.preflight: preflight(args.resume); print('HPRN_PREFLIGHT_PASS',flush=True); return
    if verify_assets()['status']!='PASS': raise RuntimeError('run preflight first')
    if args.device=='cuda':
        import torch; probe=torch.zeros(1,device='cuda'); del probe
    if args.smoke_test: smoke(args.device)
    elif args.train_source: run('source',args.device,args.resume)
    else: run('full2017',args.device,args.resume)


if __name__=='__main__': main()
