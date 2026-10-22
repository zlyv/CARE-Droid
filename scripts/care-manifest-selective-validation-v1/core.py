#!/usr/bin/env python3
"""Source-only validation helpers. Reads historical assets; writes only MSV_OUT. No APK execution."""
import csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
BASE=ROOT/'outputs/care_manifest_selective_validation_v1'
OUT=Path(os.environ['MSV_OUT']).resolve()
if OUT.parent!=BASE or not OUT.name.startswith('run_'): raise RuntimeError('invalid isolated output')
sys.path.insert(0,str(ROOT/'scripts/care-lite-stage-v1'))
import common as lc
SEEDS=(20261018,20261019,20261020)
BUDGETS=(.10,.25)
ARMS=('ManifestOnly','RefPresence31')
BUILDER=ROOT/'scripts/care-di25-host-gatedgraph-source-v1'
PRIORITY=ROOT/'configs/protocol/sensitive_endpoint_priority_v1.yaml'
EDGE=ROOT/'configs/protocol/semantic_edge_mapping_v1.yaml'

def dump(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def table(name,data):
    data=list(data);p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    cols=list(dict.fromkeys(k for r in data for k in r)) or ['status']
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,cols);w.writeheader();w.writerows(data)
def textfile(name,text):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(text)
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def digest(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return {'path':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size}
def data():
    lin,p=lc.source_contract();ids=np.array(sorted(p));y=np.array([int(p[s]['label']) for s in ids]);fold=np.array([int(p[s]['fold']) for s in ids]);expert=lc.frozen_expert_source()
    shards,missing=lc.load_shards('source')
    if missing or set(ids)!=set(expert) or set(ids)!=set(shards):raise RuntimeError('source alignment')
    for i,s in enumerate(ids):
        if int(expert[s]['label'])!=y[i] or expert[s]['split']!=f'S{fold[i]}':raise RuntimeError('expert fold/label mismatch')
    ep=np.array([float(expert[s]['expert_probability']) for s in ids])
    if not np.isfinite(ep).all() or np.any((ep<0)|(ep>1)):raise RuntimeError('expert probability')
    return lin,p,ids,y,fold,ep,shards
def features(shards,ids,arm):
    arr=[]
    for sid in ids:
        v=shards[sid];a=np.r_[lc.permission_hash(v['permissions']),v['manifest']]
        if arm=='RefPresence31':a=np.r_[a,(np.array(v['role_ref_count'])>0).astype(float)]
        arr.append(a)
    x=np.array(arr,float)
    assert x.shape[1]==(135 if arm=='ManifestOnly' else 166) and np.isfinite(x).all()
    return x
def routes(ids,fold,p,b):
    r=np.zeros(len(ids),bool)
    for f in sorted(set(fold)):
        ix=np.flatnonzero(fold==f);r[ix]=lc.route_by_score(ids[ix],1-2*np.abs(p[ix]-.5),b)
    return r
def integrity(before):
    changed=[a['path'] for a in before if digest(a['path'])['sha256']!=a['sha256']]
    return dict(status='PASS' if not changed else 'FAIL',asset_count=len(before),changed=changed,scope='Actually used source caches, prediction tables, dependency source files, mapping and selected 20 original APKs; no whole-project or checkpoint audit (checkpoint not loaded).')
