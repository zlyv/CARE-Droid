#!/usr/bin/env python3
from __future__ import annotations
import csv,gzip,hashlib,json,os,sys
from pathlib import Path
import numpy as np

NAME='care-local-control-constraint-incremental-source-v1'
PROJECT=Path(os.environ.get('LCIP_PROJECT','/home/boyuanjie/projects/tasa-droid')).resolve()
OUTPUT=Path(os.environ.get('LCIP_OUTPUT',str(PROJECT/'outputs'/NAME))).resolve()
HERE=Path(__file__).resolve().parent
PARENT=Path(os.environ.get('LCIP_LCC_V2_OUTPUT',str(PROJECT/'outputs/care-local-control-constraint-feasibility-v2'))).resolve()
STORE=PROJECT/'outputs/care-svm-c43-representation-collision-audit-v1/representation_store.npz'
REPLAY=STORE.with_name('replay_validation.json')
FORMAL=PROJECT/'outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv'
DIRECT=PROJECT/'outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/source_predictions.csv'
P31=PROJECT/'artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz'
METHOD='P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2'

def norm(x):return (x.decode() if isinstance(x,bytes) else str(x)).strip().casefold()
def safe(path):
    root=(PROJECT/'outputs').resolve();path=Path(path).resolve()
    if root not in OUTPUT.parents or OUTPUT.name!=NAME:raise RuntimeError('unsafe output root '+str(OUTPUT))
    if path!=OUTPUT and OUTPUT not in path.parents:raise RuntimeError('write outside incremental output '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True);return path
def dump(path,obj):
    path=safe(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def write_csv(path,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status'];p=safe(path);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    tmp.replace(p)
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load_gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def asset(path,purpose):
    p=Path(path).resolve();s=p.stat();return dict(absolute_path=str(p),purpose=purpose,size_bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha256(p))
def protocol():return json.loads((HERE/'protocol.json').read_text())

def load_reference():
    rows=[r for r in read_csv(FORMAL) if norm(r.get('arm'))=='svm'];direct=read_csv(DIRECT)
    by={norm(r['sha256']):r for r in direct}
    ids=np.asarray([norm(r['sha256']) for r in rows]);y=np.asarray([int(float(r['label'])) for r in rows])
    folds=np.asarray([int(float(r['fold'])) for r in rows]);base=np.asarray([float(r['stage1_prob']) for r in rows]);route=np.asarray([int(float(r['routed'])) for r in rows],bool)
    if len(ids)!=799 or len(set(ids))!=799 or set(ids)!=set(by) or len(direct)!=799:raise RuntimeError('source SHA membership mismatch')
    if set(y.tolist())!={0,1}:raise RuntimeError('source labels are not binary')
    field=METHOD+'__expert_prob' if METHOD+'__expert_prob' in direct[0] else METHOD+'__prob'
    expert=np.asarray([float(by[s][field]) for s in ids]);final=base.copy();final[route]=expert[route]
    if not(np.isfinite(base).all() and np.isfinite(expert).all()) or np.any((base<0)|(base>1)|(expert<0)|(expert>1)):raise RuntimeError('invalid probabilities')
    return ids,y,folds,base,route,expert,final

def load_store(ids,y,folds):
    with np.load(STORE,allow_pickle=False) as a:data={k:np.asarray(a[k]) for k in a.files}
    if not np.array_equal(np.asarray([norm(v) for v in data['ids']]),ids) or not np.array_equal(data['folds'],folds):raise RuntimeError('frozen representation order differs')
    for f in range(1,6):
        ti=np.asarray(data[f'f{f}_train_index'],int);vi=np.asarray(data[f'f{f}_test_index'],int)
        if not np.array_equal(ti,np.flatnonzero(folds!=f)) or not np.array_equal(vi,np.flatnonzero(folds==f)):raise RuntimeError('outer fold index mismatch')
        if not np.array_equal(data[f'f{f}_train_y'],y[ti]):raise RuntimeError('stored training label mismatch')
        for split,index in (('train',ti),('test',vi)):
            probabilities=data[f'f{f}_{split}_base'];counts=data[f'f{f}_{split}_host_count']
            if probabilities.shape!=(len(index),) or not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any():raise RuntimeError('stored Stage-I shape/finiteness mismatch')
            if counts.shape!=(len(index),) or not np.isfinite(counts).all() or (counts<0).any():raise RuntimeError('stored Host-count shape/finiteness mismatch')
        a=data[f'f{f}_train_hidden'];b=data[f'f{f}_test_hidden']
        if a.ndim!=2 or b.ndim!=2 or a.shape[0]!=len(ti) or b.shape[0]!=len(vi) or a.shape[1]!=b.shape[1] or not(np.isfinite(a).all() and np.isfinite(b).all()):raise RuntimeError('hidden shape or finiteness mismatch')
    return data

def auc(y,p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
def metrics(y,base,p,route):
    y=np.asarray(y);bp=base>=.5;fp=p>=.5;before=bp==y;after=fp==y;fixed=~before&after;broken=before&~after
    tn=int(((y==0)&~fp).sum());tp=int(((y==1)&fp).sum());fpos=int(((y==0)&fp).sum());fneg=int(((y==1)&~fp).sum())
    return dict(n=len(y),BA=.5*(tn/(tn+fpos)+tp/(tp+fneg)),AUC=auc(y,p),F1=2*tp/max(2*tp+fpos+fneg,1),
        accuracy=float(after.mean()),TN=tn,FP=fpos,FN=fneg,TP=tp,errors=fpos+fneg,fixed=int(fixed.sum()),broken=int(broken.sum()),net=int(fixed.sum()-broken.sum()),
        fixed_FN=int((fixed&(y==1)).sum()),fixed_FP=int((fixed&(y==0)).sum()),broken_TN=int((broken&(y==0)).sum()),broken_TP=int((broken&(y==1)).sum()),
        routed=int(route.sum()),damage_rate=float((broken&route).sum()/max((before&route).sum(),1)),class_flips=int((bp!=fp).sum()))
