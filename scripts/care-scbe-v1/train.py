#!/usr/bin/env python3
"""Train fixed-route Direct-C43 M0–M5 in CARE-Droid; writes only outputs/care_scbe_v1, never opens APKs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
HERE=ROOT/'scripts/care-scbe-v1'
OUT=ROOT/'outputs/care_scbe_v1'
OLD=ROOT/'outputs/care_multiyear_development_v1'
MYD=ROOT/'scripts/care-multiyear-development-v1'
sys.path[:0]=[str(HERE),str(MYD)]
from preflight import verify, atomic_new
from scbe_features import make_arms

ARMS=('M0_REPRODUCED','M1_CLASS_SUPPORT','M2_CONDITION_ONLY','M3_SCBE','M4_ROLE_SHUFFLE','M5_CAPACITY_PLACEBO')
SEEDS=(900000,900001,900002)

def write_pickle(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL)

def load_source_contract(data):
    path=ROOT/'outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv'
    rows={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['arm']=='SVM':rows[r['sha256']]=r
    if len(rows)!=799:raise RuntimeError('source SVM rows are not 799')
    source=np.flatnonzero(data['years']==2017)
    if {str(data['ids'][i]) for i in source}!=set(rows):raise RuntimeError('source SHA set differs')
    return rows

def split_contract(data,name,source_rows):
    from protocol import split_window
    if name.startswith('S'):
        fold=int(name[1:]);val=np.asarray([i for i,y in enumerate(data['years'])
                   if int(y)==2017 and int(source_rows[str(data['ids'][i])]['fold'])==fold],int)
        train=np.asarray([i for i,y in enumerate(data['years'])
                   if int(y)==2017 and int(source_rows[str(data['ids'][i])]['fold'])!=fold],int)
        base=np.asarray([float(source_rows[str(data['ids'][i])]['stage1_prob']) for i in val])
        route=np.asarray([int(source_rows[str(data['ids'][i])]['routed'])==1 for i in val],bool)
        router_score=np.full(len(val),np.nan)
    else:
        records=[]
        with (OLD/'sample_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
            records=[dict(r,year=int(r['year']),label=int(r['label'])) for r in csv.DictReader(f)]
        train,val=split_window(records,name)
        with np.load(OLD/'predictions'/name/'router.npz',allow_pickle=False) as a:
            if not np.array_equal(a['ids'],data['ids'][val]) or not np.array_equal(a['val_index'],val):
                raise RuntimeError('frozen router SHA/index mismatch '+name)
            base=a['s1'].copy();route=a['new_route'].copy();router_score=a['new_score'].copy()
    expected=int(round(.25*len(val)))
    if int(route.sum())!=expected:raise RuntimeError('Top25% budget changed '+name)
    if set(map(str,data['ids'][train]))&set(map(str,data['ids'][val])):raise RuntimeError('train/test SHA leakage')
    return train,val,base,route,router_score

def global24(v4,data,train,val):
    f=data['feature_rows'];rows=f['rows']
    gx,gz,names=v4.audit.compact_matrix_variant(v4.VARIANT,[rows[i] for i in train],
        [rows[i] for i in val],f['compact_by_sha'],f['schema'],f['canonical_names'])
    ix,_=v4.evidence.split_view_indices(names)
    a=np.asarray(gx[:,ix],np.float32);b=np.asarray(gz[:,ix],np.float32)
    if a.shape!=(len(train),24) or b.shape!=(len(val),24):raise RuntimeError('Global24 mismatch')
    return a,b

def original_reference(data,name,seed,val):
    if name.startswith('S'):return None
    tag='C0R_2017' if name=='T1' else name+'_C1'
    with (OLD/'models/stage2'/tag/(str(seed)+'.pkl')).open('rb') as f:obj=pickle.load(f)
    ids=[str(x) for x in obj['metadata']['validation_sha']]
    loc={s:i for i,s in enumerate(ids)}
    if any(str(data['ids'][i]) not in loc for i in val):raise RuntimeError('old checkpoint validation SHA missing')
    return np.asarray(obj['probability'])[[loc[str(data['ids'][i])] for i in val]]

def train_one(v4,args,dev,seed,epochs,gtrain,gtest,graph_train,graph_test,train_bags,test_bags,y):
    torch=v4.torch
    neutral_train=np.full(len(gtrain),.5);neutral_test=np.full(len(gtest),.5)
    weights,correct=v4.fixkeep_weights(y,neutral_train,args)
    if not np.array_equal(weights,np.ones(len(y),np.float32)) or args.keep_consistency!=0:
        raise RuntimeError('original equal BCE contract changed')
    v4.seed_everything(seed)
    model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args).to(dev)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    gt=torch.as_tensor(gtrain,dtype=torch.float32,device=dev)
    graph=v4.graph_tensors(graph_train,dev)
    bt=[torch.as_tensor(x,dtype=torch.float32,device=dev) for x in train_bags]
    started=time.perf_counter()
    loss=v4.train_epochs(model,optimizer,gt,neutral_train,graph,bt,y,weights,correct,epochs,args,seed+3301)
    train_sec=time.perf_counter()-started
    started=time.perf_counter()
    p,_=v4.predict_model(model,gtest,neutral_test,graph_test,test_bags,dev)
    infer_sec=time.perf_counter()-started
    state={k:value.detach().cpu() for k,value in model.state_dict().items()}
    param=sum(t.numel() for t in model.parameters())
    gpu_peak=float(torch.cuda.max_memory_allocated(dev))/1024**2 if dev.type=='cuda' else 0.
    del model,optimizer,gt,graph,bt
    if dev.type=='cuda':torch.cuda.empty_cache()
    if not np.isfinite(p).all():raise RuntimeError('nonfinite model probability')
    return np.asarray(p),state,{'training_seconds':train_sec,'inference_seconds':infer_sec,
        'parameters':param,'gpu_peak_mib':gpu_peak,'final_BCE':loss}

def run_split_seed(data,name,seed,device,smoke=False):
    from audit_data import load_data
    from myd_common import runtime
    from train_stage2 import raw_bags,capture_background
    source_rows=load_source_contract(data)
    train,val,base,route,router_score=split_contract(data,name,source_rows)
    if smoke:
        train=np.concatenate([train[data['y'][train]==label][:40] for label in (0,1)])
        val=val[:min(12,len(val))]
        base=base[:len(val)];route=route[:len(val)];router_score=router_score[:len(val)]
        # Smoke deliberately tests the tensor interface, not Top25% performance.
    v4,_,args=runtime();args.device=device;args.keep_weight=args.fix_weight=1.;args.keep_consistency=0.
    dev=v4.resolve_device(device)
    if dev.type=='cuda' and v4.torch.cuda.mem_get_info()[0]<1024**3:
        raise RuntimeError('GPU free memory below 1 GiB')
    gtrain,gtest=global24(v4,data,train,val)
    y=data['y'][train]
    raw_train=raw_bags(data,train);raw_test=raw_bags(data,val)
    neutral=np.full(len(train),.5)
    if smoke:epochs=1;selection_loss=None
    else:epochs,selection_loss=v4.select_epochs(gtrain,data['s7'][train],raw_train,neutral,y,'RGCondAttn',args,dev,seed)
    formal_train,formal_test,bg_audit,prep=capture_background(v4,raw_train,raw_test,y,args,seed+1707)
    graph_scaler=v4.fit_role_graph_scaler(data['s7'][train])
    graph_train=v4.transform_role_graphs(data['s7'][train],graph_scaler)
    graph_test=v4.transform_role_graphs(data['s7'][val],graph_scaler)
    for b in formal_train+formal_test:
        if b.ndim!=2 or b.shape[1]!=142 or not np.isfinite(b).all():raise RuntimeError('formal 142-wide Host invalid')
    variants,feature_audit=make_arms(v4,raw_train,raw_test,formal_train,formal_test,y,prep,seed)
    if any(not np.array_equal(variants[m]['train'][0][:,108:110],formal_train[0][:,108:110]) for m in variants):
        raise RuntimeError('SCBE altered Internal/Commonness')
    if smoke:
        result={'status':'PASS','train_n':len(train),'validation_n':len(val),'epochs':1,
            'feature_audit':feature_audit,'formal_width':142,'all_arms':list(variants)}
        if not (OUT/'scbe_interface_validation.json').exists():
            atomic_new(OUT/'scbe_interface_validation.json',result)
        elif json.loads((OUT/'scbe_interface_validation.json').read_text())['status']!='PASS':
            raise RuntimeError('existing smoke result not PASS')
        print('SCBE_INTERFACE_SELFTEST_PASS',flush=True)
        return
    split_dir=OUT/'models'/name/str(seed);split_dir.mkdir(parents=True,exist_ok=True)
    audit_path=OUT/'feature_audit'/name/(str(seed)+'.json')
    if not audit_path.exists():atomic_new(audit_path,feature_audit)
    for arm in ARMS:
        path=split_dir/(arm+'.pkl')
        if path.exists():
            with path.open('rb') as f:old=pickle.load(f)
            if old['arm']!=arm or old['seed']!=seed or old['split']!=name or old['ids']!=list(map(str,data['ids'][val])):
                raise RuntimeError('existing SCBE checkpoint contract mismatch '+str(path))
            print('SCBE_RESUME',name,seed,arm,flush=True);continue
        one=variants.get(arm)
        bags_train=formal_train if one is None else one['train']
        bags_test=formal_test if one is None else one['test']
        if dev.type=='cuda':v4.torch.cuda.reset_peak_memory_stats(dev)
        pred,state,cost=train_one(v4,args,dev,seed,epochs,gtrain,gtest,graph_train,graph_test,bags_train,bags_test,y)
        if arm=='M0_REPRODUCED':
            reference=original_reference(data,name,seed,val)
            if reference is not None:
                difference=float(np.max(np.abs(pred-reference)))
                if difference>1e-5:raise RuntimeError('M0 frozen checkpoint replay failed '+name+' '+str(difference))
                cost['historical_checkpoint_max_probability_difference']=difference
        obj={'arm':arm,'seed':seed,'split':name,'ids':list(map(str,data['ids'][val])),
             'train_sha_hash':hashlib.sha256('\n'.join(map(str,data['ids'][train])).encode()).hexdigest(),
             'probability':pred,'state_dict':state,'background':prep,'graph_scaler':graph_scaler,
             'original_external_mean':np.asarray([float(x[:,107].mean()) for x in formal_test]),
             'conditional_residual_strength':np.zeros(len(val)) if one is None else one['test_strength'],
             'category_direction_strength':np.zeros(len(val)) if one is None else one['test_direction'],
             'selected_epochs':epochs,'selection_loss':selection_loss,'cost':cost,'feature_audit':feature_audit,
             'train_years':sorted(set(map(int,data['years'][train]))),
             'test_years':sorted(set(map(int,data['years'][val])))}
        write_pickle(path,obj)
        print('SCBE_ARM_DONE',name,seed,arm,'epochs='+str(epochs),flush=True)

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--smoke-test',action='store_true');g.add_argument('--train-source',action='store_true')
    g.add_argument('--train-rolling',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if a.workers!=1:raise RuntimeError('fixed single-worker training to avoid concurrent workload')
    if verify()['status']!='PASS':raise RuntimeError('historical asset integrity failed')
    if a.device=='cuda':
        import torch
        # This host intermittently fails lazy CUDA initialization after loading
        # the historical feature bundle; establish the context first and fail closed.
        probe=torch.zeros(1,device='cuda')
        del probe
    from audit_data import load_data
    data=load_data()
    if a.smoke_test:jobs=[('S1',900000)]
    else:jobs=[(name,seed) for name in ((['S1','S2','S3','S4','S5'] if a.train_source else ['T1','T2','T3','T4'])) for seed in SEEDS]
    for name,seed in jobs:
        print('SCBE_START',name,seed,flush=True)
        run_split_seed(data,name,seed,a.device,a.smoke_test)
    print('SCBE_PHASE_DONE',('smoke' if a.smoke_test else 'source' if a.train_source else 'rolling'),flush=True)

if __name__=='__main__':main()
