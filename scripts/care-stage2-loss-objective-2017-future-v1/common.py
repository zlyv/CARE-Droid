#!/usr/bin/env python3
"""Shared contracts for the CARE Stage-II loss-objective experiment.

Inputs are frozen Direct-C43/W0 checkpoints, cached Host/RoleGraph tensors, and the
Manifest135 autonomous-router prediction table. Outputs are written only below the
new experiment artifact and no APK, checkpoint, Prototype Bank, or formal source is modified.
"""
from __future__ import annotations

import csv, hashlib, json, os, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
HERE=ROOT/'scripts/care-stage2-loss-objective-2017-future-v1'
OUT=ROOT/'artifacts/care_stage2_loss_objective_2017_future_v1'
SCBE=ROOT/'outputs/care_scbe_v1'
MYD=ROOT/'outputs/care_multiyear_development_v1'
AUTO=ROOT/'outputs/care_manifest_rf_autonomous_error_router_v1'
ENDPOINT=ROOT/'artifacts/care_api_manifest_fusion_2017_future_v1'
MODEL_SEEDS=(900000,900001,900002)
MANIFEST_SEEDS=(20261018,20261019,20261020)
SPLITS=('S1','S2','S3','S4','S5')
ARMS=('T0_W0_BCE','T1_FOCAL_GAMMA2','T2_BCE_RANK01')
YEARS=(2017,2018,2019,2020,2021)
YEAR_COUNTS={2017:799,2018:793,2019:399,2020:397,2021:400}

sys.path[:0]=[
 str(ROOT/'scripts/care-background-functional-attribution-v1'),
 str(ROOT/'scripts/care-scbe-v1'),
 str(ROOT/'scripts/care-multiyear-development-v1'),
]

def jv(x):
    if isinstance(x,dict):return {str(k):jv(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,set)):return [jv(v) for v in x]
    if isinstance(x,Path):return str(x)
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.floating):return None if not np.isfinite(x) else float(x)
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def atomic_json(path,value,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:raise FileExistsError(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(jv(value),ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def atomic_csv(path,rows,replace=False):
    import pandas as pd
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:raise FileExistsError(path)
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(list(rows))
    tmp=path.with_suffix(path.suffix+'.tmp');frame.to_csv(tmp,index=False);os.replace(tmp,path)

def atomic_pickle(path,value,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:raise FileExistsError(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:pickle.dump(value,f,pickle.HIGHEST_PROTOCOL)
    os.replace(tmp,path)

def read_pickle(path):
    with Path(path).open('rb') as f:return pickle.load(f)

def digest(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return {'path':str(path.resolve()),'size':path.stat().st_size,'sha256':h.hexdigest()}

def verify_manifest(items):
    changed=[];after=[]
    for old in items:
        try:now=digest(old['path']);after.append(now)
        except Exception as exc:changed.append({'path':old['path'],'error':repr(exc)});continue
        if now!=old:changed.append({'before':old,'after':now})
    return {'status':'PASS' if not changed else 'FAIL','checked':len(items),'changed':changed,'after':after}

def status(stage,state='RUNNING',**extra):
    atomic_json(OUT/'logs/run_status.json',{'status':state,'stage':stage,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**extra},replace=True)
    print('LOSS_OBJECTIVE_STAGE',stage,state,flush=True)

def load_data():
    from bfa_core import load_data_contract
    data,_=load_data_contract();return data

def metric(y,p):
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y,int);p=np.asarray(p,float);z=p>=.5
    tn=int(np.sum((y==0)&~z));fp=int(np.sum((y==0)&z));fn=int(np.sum((y==1)&~z));tp=int(np.sum((y==1)&z))
    tnr=tn/max(tn+fp,1);tpr=tp/max(tp+fn,1);prec=tp/max(tp+fp,1)
    return {'n':len(y),'BA':.5*(tnr+tpr),'Accuracy':float(np.mean(z==y)),'AUC':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'),'F1':2*prec*tpr/max(prec+tpr,1e-12),'TN':tn,'FP':fp,'FN':fn,'TP':tp,'FPR':1-tnr,'FNR':1-tpr}

def cascade(y,stage1,expert,route):
    y=np.asarray(y,int);stage1=np.asarray(stage1,float);expert=np.asarray(expert,float);route=np.asarray(route,bool)
    final=np.where(route,expert,stage1);before=(stage1>=.5)==y;after=(final>=.5)==y;m=metric(y,final)
    m.update({'routed':int(route.sum()),'route_rate':float(route.mean()),'Fixed':int(np.sum(~before&after)),'Broken':int(np.sum(before&~after)),'Net_Fixed':int(np.sum(~before&after)-np.sum(before&~after)),'stage2_BA':metric(y,expert)['BA'],'stage2_AUC':metric(y,expert)['AUC'],'stage2_F1':metric(y,expert)['F1'],'stage2_FPR':metric(y,expert)['FPR'],'stage2_FNR':metric(y,expert)['FNR']})
    return m,final

def route_rows(year):
    import pandas as pd
    q=pd.read_csv(AUTO/'per_apk_predictions.csv.gz')
    q=q[(q.year==year)&(q.strategy=='AUTO_ERROR_LR')].copy()
    if len(q)!=YEAR_COUNTS[year]*len(MANIFEST_SEEDS):raise RuntimeError('router row count '+str(year))
    return q

def holm(rows):
    order=sorted(range(len(rows)),key=lambda i:rows[i]['p_two_sided']);running=0.
    for rank,i in enumerate(order):
        adjusted=min(1.,(len(rows)-rank)*rows[i]['p_two_sided']);running=max(running,adjusted);rows[i]['p_holm']=running
    return rows
