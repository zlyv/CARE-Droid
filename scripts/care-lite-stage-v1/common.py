#!/usr/bin/env python3
"""LiteStage-V1 frozen asset/feature helpers. Reads source/future cache and new shards; writes only new LiteStage output; trains no model; reads no APK."""
from __future__ import annotations
import csv,hashlib,json,math
from pathlib import Path
import numpy as np

ROOT=Path('/home/boyuanjie/projects/tasa-droid');OUT=ROOT/'outputs/care_lite_stage_v1'
SOURCE_LINEAGE=Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/source_lineage_2017.csv')
FUTURE_LINEAGE=ROOT/'artifacts/care-litelocal-p31-diagnostics-v1/preflight/future_lineage_label_free.csv'
P31=ROOT/'artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz'
FUTURE_P31=ROOT/'artifacts/care-litelocal-p31-diagnostics-v1/future_p31_features_2018_2021.npz'
SOURCE_PRED=ROOT/'artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv'
HDBC=ROOT/'outputs/care_hdbc_v1'
MAPPING=ROOT/'scripts/care-source-recovery-stage2only-v1/extractor_repo/configs/protocol/semantic_mapping_v1.yaml'
PERM_DIM=128

def readcsv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(name,rows):
    rows=list(rows);p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise RuntimeError('refuse overwrite '+str(p))
    cols=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with p.open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,cols);w.writeheader();w.writerows(rows)
def writejson(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def writetext(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:f.write(value)
def digest(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),size=p.stat().st_size)
def source_contract():
    lin=readcsv(SOURCE_LINEAGE);pred=readcsv(SOURCE_PRED);l={r['sha256'].lower():r for r in lin};p={r['sha256'].lower():r for r in pred}
    if len(l)!=799 or set(l)!=set(p):raise RuntimeError('source identity')
    folds={i:sum(int(r['fold'])==i for r in pred) for i in range(1,6)}
    if list(folds.values())!=[160,160,160,160,159]:raise RuntimeError('fixed fold')
    y=np.array([int(r['label']) for r in pred]);
    if (sum(y==0),sum(y==1))!=(400,399):raise RuntimeError('class counts')
    for sid in l:
        expected=0 if l[sid]['metadata_label'].casefold()=='benign' else 1
        if int(p[sid]['label'])!=expected:raise RuntimeError('label mismatch '+sid)
    return l,p
def frozen_expert_source():
    rows=[r for r in readcsv(HDBC/'source2017_oof_predictions.csv') if r['arm']=='H0_ORIGINAL'];ret={r['sha256']:r for r in rows}
    if len(rows)!=799 or len(ret)!=799:raise RuntimeError('C43 source OOF')
    return ret
def frozen_expert_future():
    rows=[r for r in readcsv(HDBC/'future2018_2021_predictions.csv') if r['arm']=='H0_ORIGINAL'];ret={r['sha256']:r for r in rows}
    if len(rows)!=1989 or len(ret)!=1989:raise RuntimeError('C43 future 2017-trained identity')
    return ret
def load_shards(cohort):
    rows=readcsv(SOURCE_LINEAGE if cohort=='source' else FUTURE_LINEAGE);out={};missing=[]
    for r in rows:
        sid=r['sha256'].lower();p=OUT/'shards'/cohort/(sid+'.json')
        if not p.is_file():missing.append(sid);continue
        v=json.loads(p.read_text(encoding='utf-8'))
        if v['sha256']!=sid or v['callgraph_constructed'] or v['host_graph_constructed']:raise RuntimeError('shard invalid '+sid)
        out[sid]=v
    return out,missing
def permission_hash(names):
    a=np.zeros(PERM_DIM,float)
    for n in names:
        h=int.from_bytes(hashlib.sha256(str(n).encode()).digest()[:8],'big')%PERM_DIM;a[h]=1.
    return a
def matrix(rows,arm,role_select=None):
    r=[]
    for v in rows:
        manifest=np.asarray(v['manifest'],float);dex=np.asarray(v['dex_meta'],float)
        base=np.r_[permission_hash(v['permissions']),manifest]
        if arm=='L1_MANIFEST':pass
        else:base=np.r_[base,np.log1p(np.maximum(dex,0))]
        if arm.startswith('L3_') or arm.startswith('L4_'):
            block=np.asarray(v['role_ref_count'] if arm.startswith('L3_') else v['role_method_count'],float)
            mode=arm.split('_',1)[1]
            if mode=='PRESENCE':block=(block>0).astype(float)
            elif mode=='LOG':block=np.log1p(block)
            elif mode.startswith('TOP'):block=block[np.asarray(role_select,int)]
            elif mode!='COUNT':raise ValueError(arm)
            base=np.r_[base,block]
        r.append(base)
    x=np.asarray(r,float)
    if not np.isfinite(x).all():raise RuntimeError('nonfinite feature')
    return x
def choose_roles(rows,y,arm):
    if '_TOP' not in arm:return None
    k=int(arm.split('_TOP')[1]);a=np.array([v['role_ref_count'] if arm.startswith('L3_') else v['role_method_count'] for v in rows],float);y=np.asarray(y,int)
    # Train-only univariate rank with stable Role ID tie-break. No held-out labels.
    mean0=a[y==0].mean(0);mean1=a[y==1].mean(0);var0=a[y==0].var(0);var1=a[y==1].var(0)
    score=np.abs(mean1-mean0)/np.sqrt(var0+var1+1e-8)
    return np.lexsort((np.arange(31),-score))[:k]
def metrics(y,p):
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y,int);p=np.asarray(p,float);z=p>=.5
    tn=int(((y==0)&~z).sum());fp=int(((y==0)&z).sum());fn=int(((y==1)&~z).sum());tp=int(((y==1)&z).sum())
    return dict(n=len(y),BA=.5*(tn/max(tn+fp,1)+tp/max(tp+fn,1)),AUC=float(roc_auc_score(y,p)) if len(set(y))==2 else None,F1=2*tp/max(2*tp+fp+fn,1),accuracy=float((z==y).mean()),TN=tn,FP=fp,FN=fn,TP=tp,errors=fp+fn)
def route_by_score(ids,scores,budget):
    n=int(round(len(ids)*budget));order=np.lexsort((np.asarray(ids),-np.asarray(scores,float)));mask=np.zeros(len(ids),bool);mask[order[:n]]=True;return mask
def cascade(y,base,expert,route):
    y=np.asarray(y,int);b=np.asarray(base,float);e=np.asarray(expert,float);route=np.asarray(route,bool);f=np.where(route,e,b);before=(b>=.5)==y;after=(f>=.5)==y
    m=metrics(y,f);m.update(routed=int(route.sum()),routed_stage1_errors=int((~before&route).sum()),router_error_recall=float((~before&route).sum()/max((~before).sum(),1)),stage1_errors=int((~before).sum()),fixed=int((~before&after).sum()),broken=int((before&~after).sum()),net=int((~before&after).sum()-(before&~after).sum()))
    for d,name in [(0,'FN_TN_AUC'),(1,'TP_FP_AUC')]:
        ix=route&((b>=.5)==d);m[name]=metrics(y[ix],e[ix])['AUC'] if ix.any() else None
    return m,f
