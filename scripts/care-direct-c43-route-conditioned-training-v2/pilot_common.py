from __future__ import annotations
import csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np

NAME = 'care-direct-c43-route-conditioned-training-v2'
HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('RCT_PROJECT', '/home/boyuanjie/projects/tasa-droid')).resolve()
OUTPUT = PROJECT / 'outputs' / NAME
FAMILY = PROJECT / 'scripts/care-p31-classifier-family-stage2-v1'
DIRECT_SCRIPT = PROJECT / 'scripts/care-svm-c43-remove-bounded-correction-v1'
FAMILY_OUTPUT = PROJECT / 'outputs/care-p31-classifier-family-stage2-v1'
FORMAL = FAMILY_OUTPUT / 'source_stage2_predictions.csv'
DIRECT = PROJECT / 'outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/source_predictions.csv'
P31 = PROJECT / 'artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz'
METHOD = 'P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2'

def protocol(): return json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
ARMS = tuple(protocol()['arms'])
def norm(x): return (x.decode() if isinstance(x,bytes) else str(x)).strip().casefold()
def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def safe(p):
    p=Path(p).resolve()
    if p!=OUTPUT and OUTPUT not in p.parents: raise RuntimeError('write outside independent output: '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    return p
def dump(p,value):
    p=safe(p); t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8'); t.replace(p)
def write_csv(p,rows):
    rows=list(rows); fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    p=safe(p); t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    t.replace(p)
def save_npz(p,**values):
    p=safe(p)
    with p.open('wb') as f: np.savez_compressed(f,**values)
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def asset(p,purpose):
    p=Path(p).resolve(); s=p.stat()
    return dict(absolute_path=str(p),size_bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha256(p),purpose=purpose)
def verify_integrity():
    before=read_json(OUTPUT/'baseline_asset_manifest_before.json'); after=[]; changed=[]
    for r in before:
        p=Path(r['absolute_path'])
        if not p.is_file(): changed.append(dict(path=str(p),reason='MISSING')); continue
        a=asset(p,r['purpose']); after.append(a)
        if any(a[k]!=r[k] for k in ('size_bytes','mtime_ns','sha256')): changed.append(dict(before=r,after=a))
    dump(OUTPUT/'baseline_asset_manifest_after.json',after)
    dump(OUTPUT/'baseline_integrity_check.json',dict(status='PASS' if not changed else 'FAIL',changed=changed,asset_count=len(before),
        scope='listed reused source inputs, scripts and protocol configuration; not a whole-project integrity claim'))
    if changed: raise RuntimeError('protected asset changed')

def reference(order=None):
    rows=[r for r in read_csv(FORMAL) if norm(r.get('arm'))=='svm']; by={norm(r['sha256']):r for r in rows}
    dr=read_csv(DIRECT); db={norm(r['sha256']):r for r in dr}
    ids=np.asarray(sorted(by) if order is None else list(order))
    if len(rows)!=799 or len(dr)!=799 or len(ids)!=799 or len(set(ids))!=799 or set(ids)!=set(by) or set(ids)!=set(db):
        raise RuntimeError('source cohort contract failed')
    y=np.asarray([int(float(by[s]['label'])) for s in ids]); folds=np.asarray([int(float(by[s]['fold'])) for s in ids])
    base=np.asarray([float(by[s]['stage1_prob']) for s in ids]); route=np.asarray([int(float(by[s]['routed'])) for s in ids],bool)
    for i,s in enumerate(ids):
        if int(float(db[s]['label']))!=y[i] or int(float(db[s]['fold']))!=folds[i]: raise RuntimeError('Direct label/fold contract')
    if METHOD+'__expert_prob' not in dr[0]: raise RuntimeError('all-APK expert probabilities required; cannot use routed mixture')
    expert=np.asarray([float(db[s][METHOD+'__expert_prob']) for s in ids])
    if not np.isfinite(base).all() or not np.isfinite(expert).all() or np.any((base<0)|(base>1)|(expert<0)|(expert>1)): raise RuntimeError('invalid probability')
    return ids,y,folds,base,route,expert

def auc(y,p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
def metrics(y,base,expert,route):
    y=np.asarray(y); route=np.asarray(route,bool); p=np.where(route,expert,base); before=(base>=.5)==y; after=(p>=.5)==y
    fixed=~before&after; broken=before&~after; tn=int(((y==0)&(p<.5)).sum()); tp=int(((y==1)&(p>=.5)).sum())
    fp=int(((y==0)&(p>=.5)).sum()); fn=int(((y==1)&(p<.5)).sum()); neg=route&(base<.5); pos=route&(base>=.5)
    return dict(n=len(y),BA=.5*(tn/max(tn+fp,1)+tp/max(tp+fn,1)),AUC=auc(y,p),F1=2*tp/max(2*tp+fp+fn,1),accuracy=float(after.mean()),
        TN=tn,FP=fp,FN=fn,TP=tp,errors=fp+fn,fixed=int(fixed.sum()),broken=int(broken.sum()),net=int(fixed.sum()-broken.sum()),
        fixed_FN=int((fixed&(y==1)).sum()),fixed_FP=int((fixed&(y==0)).sum()),broken_TN=int((broken&(y==0)).sum()),broken_TP=int((broken&(y==1)).sum()),
        routed=int(route.sum()),routed_errors=int((~before&route).sum()),routed_correct=int((before&route).sum()),
        damage_rate=float((broken&route).sum()/max((before&route).sum(),1)),
        routed_BA=conditional_ba(y[route],expert[route]),routed_AUC=auc(y[route],expert[route]),
        routed_F1=float(2*((y[route]==1)&(expert[route]>=.5)).sum()/max((y[route]==1).sum()+(expert[route]>=.5).sum(),1)),
        FN_vs_TN_AUC=auc(y[neg],expert[neg]),TP_vs_FP_AUC=auc(y[pos],expert[pos]),
        FN_TN_n=int(neg.sum()),TP_FP_n=int(pos.sum()),class_flips=int(((p>=.5)!=(base>=.5)).sum()))
def conditional_ba(y,p):
    if len(np.unique(y))!=2: return None
    return float(.5*(np.mean(p[y==0]<.5)+np.mean(p[y==1]>=.5)))
