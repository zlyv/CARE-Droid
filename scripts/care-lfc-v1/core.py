#!/usr/bin/env python3
"""LFC-V1 shared contracts and features. Inputs: read-only LiteStage shards/OOF and frozen C43; outputs: files only under LFC_OUT. No APK parsing or C43 training."""
from __future__ import annotations
import csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
LITE=ROOT/'outputs/care_lite_stage_v1'
HDBC=ROOT/'outputs/care_hdbc_v1'
BASE=ROOT/'outputs/care_lfc_v1'
OUT=Path(os.environ['LFC_OUT']).resolve()
if BASE!=OUT.parent or not OUT.name.startswith('run_'):
    raise RuntimeError('LFC_OUT must be a unique direct run_ child of outputs/care_lfc_v1')
sys.path.insert(0,str(ROOT/'scripts/care-lite-stage-v1'))
import common as lc

MAIN=('ManifestOnly','RefPresence31','RefCount31','InvokeCount31','OriginalP31RF','BestLiteStage')
NEW=('RefPresence31','RefCount31','InvokeCount31','RefLog31','RefDensity31','InvokePresence31','InvokeDensity31','RefTop16')
EXTRA=('RefLog31','RefDensity31','InvokePresence31','InvokeDensity31','RefTop16')

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def shard(cohort):
    folder=LITE/'shards'/cohort
    files=sorted(folder.glob('*.json'))
    expect=799 if cohort=='source' else 1989
    if len(files)!=expect:raise RuntimeError(f'{cohort} shards {len(files)}/{expect}')
    data={}
    for p in files:
        v=json.loads(p.read_text(encoding='utf-8'))
        if p.stem!=v['sha256'] or v['callgraph_constructed'] or v['host_graph_constructed']:raise RuntimeError('invalid shard '+p.name)
        data[p.stem]=v
    return data
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4<<20),b''):h.update(part)
    return dict(path=str(path),sha256=h.hexdigest(),bytes=Path(path).stat().st_size)
def put_json(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);s=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if p.exists():
        if p.read_text(encoding='utf-8')!=s:raise RuntimeError('existing output differs '+str(p))
    else:p.write_text(s,encoding='utf-8')
def put_text(name,s):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_text(encoding='utf-8')!=s:raise RuntimeError('existing output differs '+str(p))
    else:p.write_text(s,encoding='utf-8')
def put_csv(name,data):
    data=list(data);p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for row in data for k in row)) or ['status']
    from io import StringIO
    f=StringIO();w=csv.DictWriter(f,columns,lineterminator='\n');w.writeheader();w.writerows(data);s=f.getvalue()
    if p.exists():
        if p.read_text(encoding='utf-8')!=s:raise RuntimeError('existing output differs '+str(p))
    else:p.write_text(s,encoding='utf-8')
def feature(v,arm,selection=None):
    manifest=np.r_[lc.permission_hash(v['permissions']),np.asarray(v['manifest'],float)]
    ref=np.asarray(v['role_ref_count'],float)
    inv=np.asarray(v['role_invoke_count'],float)
    dex=np.asarray(v['dex_meta'],float)
    # dex[4] is invoke_n and MUST NOT enter reference-only features.
    if arm=='ManifestOnly':x=manifest
    elif arm=='RefPresence31':x=np.r_[manifest,(ref>0).astype(float)]
    elif arm=='RefCount31':x=np.r_[manifest,ref,ref/max(dex[2],1)*1e4,np.log1p(dex[2])]
    elif arm=='InvokeCount31':x=np.r_[manifest,inv,np.log1p(dex[4])]
    elif arm=='RefLog31':x=np.r_[manifest,np.log1p(ref)]
    elif arm=='RefDensity31':x=np.r_[manifest,ref/max(dex[2],1)*1e4,np.log1p(dex[2])]
    elif arm=='InvokePresence31':x=np.r_[manifest,(inv>0).astype(float)]
    elif arm=='InvokeDensity31':x=np.r_[manifest,inv/max(dex[4],1)*1e4,np.log1p(dex[4])]
    elif arm=='RefTop16':
        if selection is None or len(selection)!=16:raise RuntimeError('RefTop16 requires train-only selection')
        x=np.r_[manifest,ref[np.asarray(selection,int)]]
    else:raise ValueError(arm)
    if not np.isfinite(x).all():raise RuntimeError('nonfinite feature '+arm)
    return x
def role_select(train_rows,y):
    a=np.array([v['role_ref_count'] for v in train_rows],float);y=np.asarray(y,int)
    mu=a[y==1].mean(0)-a[y==0].mean(0)
    var=a[y==1].var(0)+a[y==0].var(0)
    score=np.abs(mu)/np.sqrt(var+1e-8)
    return np.lexsort((np.arange(31),-score))[:16]
def feature_matrix(data,ids,arm,selection=None):return np.array([feature(data[s],arm,selection) for s in ids],float)
def route(ids,prob,budget):return lc.route_by_score(ids,1-2*np.abs(np.asarray(prob,float)-.5),budget)
def source_index():
    lineage,pred=lc.source_contract();ids=np.array(sorted(pred));y=np.array([int(pred[s]['label']) for s in ids]);fold=np.array([int(pred[s]['fold']) for s in ids])
    expert=lc.frozen_expert_source()
    if set(ids)!=set(expert):raise RuntimeError('C43 source SHA mismatch')
    for i,s in enumerate(ids):
        if expert[s]['split']!=f'S{fold[i]}' or int(expert[s]['label'])!=y[i]:raise RuntimeError('C43 OOF fold/label mismatch '+s)
    return ids,y,fold,pred,expert
def legacy_source():
    data={}
    for r in rows(LITE/'stage1_oof_predictions.csv'):
        data[(r['arm'],r['sha256'])]=float(r['probability'])
    return data
def cost_class(arm):
    if arm=='ManifestOnly':return 'MANIFEST_ONLY'
    if arm.startswith('Ref'):return 'DEX_REFERENCE_SCAN'
    if arm=='OriginalP31RF':return 'HISTORIC_DEX_INVOKE_SCAN'
    return 'DEX_INVOKE_SCAN'
