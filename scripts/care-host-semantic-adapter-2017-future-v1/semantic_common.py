#!/usr/bin/env python3
"""Build fixed S1/S3 Host sidecars for the frozen CARE W0 bags.

Inputs are the v2 per-APK semantic caches and W0 Host bags. Outputs are
17-column sidecars (16 fixed features plus an evidence mask). No labels are
read here. The module also implements the predeclared within-APK controls.
"""
from __future__ import annotations

import gzip, hashlib, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
HERE=ROOT/'scripts/care-host-semantic-adapter-2017-future-v1'
OUT=ROOT/'artifacts/care_host_semantic_adapter_2017_future_v1'
HPRN=ROOT/'scripts/care-hprn-w0-manifest-auto-v1'
SEM_V2=ROOT/'artifacts/care_host_method_semantics_feasibility_2017_v2'
sys.path[:0]=[str(HERE),str(HPRN)]
from hprn_common import *  # frozen W0 contracts and atomic I/O

OUT=ROOT/'artifacts/care_host_semantic_adapter_2017_future_v1'
ARMS=('W0','C0_S1','C1_S1','C3_S1','C0_S3','C2_S3','C4_S3')
ADAPTER_ARMS=ARMS[1:]
BASE_WIDTH=142; FEATURE_WIDTH=16; MODEL_WIDTH=159
S1_CATEGORIES=('APP_PRIVATE_PATH','SHARED_STORAGE_PATH','MEDIA_PATH','CONTENT_URI',
 'NETWORK_ENDPOINT','INTENT_ACTION','COMPONENT_NAME','CRYPTO_ALGORITHM','STATIC_STRING',
 'INTEGER_CONSTANT','BOOLEAN_CONSTANT','CONSTANT_UNKNOWN_CATEGORY','UNKNOWN')

def canonical(value): return re.sub(r'\s+','',str(value or '')).casefold()
def stable_id(value): return int.from_bytes(hashlib.sha256(canonical(value).encode()).digest()[:8],'big')

def cache_path(sid):
    p=SEM_V2/'cache'/f'{sid}.json.gz'
    if p.is_file(): return p
    hits=list((OUT/'future_cache').glob(f'*/{sid}.json.gz'))
    if len(hits)!=1: raise FileNotFoundError(f'semantic cache missing/ambiguous {sid}: {hits}')
    return hits[0]

def load_hosts(sid):
    with gzip.open(cache_path(str(sid)),'rt',encoding='utf-8') as f: return json.load(f).get('hosts',[])

def _hash_vec(key):
    raw=hashlib.sha256(key.encode()).digest(); rng=np.random.RandomState(int.from_bytes(raw[:4],'big'))
    v=rng.choice([-1.,1.],FEATURE_WIDTH).astype(np.float32); return v/math.sqrt(FEATURE_WIDTH)

def _s1(row):
    c=row.get('s1_counts') or {}; v=np.zeros(FEATURE_WIDTH,np.float32)
    for i,k in enumerate(S1_CATEGORIES): v[i]=np.log1p(float(c.get(k,0)))
    mask=float(sum(float(x) for k,x in c.items() if k!='UNKNOWN')>0)
    return v,mask

def _s3(row,deps=None):
    deps=list(row.get('s3_dependencies') or []) if deps is None else deps
    v=np.zeros(FEATURE_WIDTH,np.float32); support=0.
    for d in deps:
        n=float(d.get('support_count',1)); support+=n
        key=f"{int(d['source_role'])}:{int(d['target_role'])}:{d.get('dependency_type','UNKNOWN')}"
        v+=np.log1p(n)*_hash_vec(key)
    if support: v/=math.sqrt(support)
    return v,float(support>0)

def _control(row,kind):
    status=str(row.get('parse_status','UNKNOWN')); ins=float(row.get('instruction_count',0)); ev=float(row.get('sensitive_event_count',0))
    partial=float(status=='PARTIAL'); success=float(status=='SUCCESS')
    if kind=='S1': _,mask=_s1(row); density=sum((row.get('s1_counts') or {}).values())
    else: _,mask=_s3(row); density=sum(float(d.get('support_count',1)) for d in row.get('s3_dependencies') or [])
    v=np.zeros(FEATURE_WIDTH,np.float32)
    v[:8]=[mask,success,partial,np.log1p(ins),np.log1p(ev),np.log1p(density),float(bool(row.get('unknown_reasons'))),float(status not in ('SUCCESS','PARTIAL'))]
    return v,mask

def _rewire(deps,seed,sid,hidx):
    expanded=[]
    for d in deps:
        expanded += [(int(d['source_role']),int(d['target_role']),str(d.get('dependency_type','UNKNOWN')))]*int(d.get('support_count',1))
    out=[]
    for typ in sorted({x[2] for x in expanded}):
        part=[x for x in expanded if x[2]==typ]; targets=[x[1] for x in part]
        token=hashlib.sha256(f'{seed}:{sid}:{hidx}:{typ}:s3-rewire'.encode()).digest()
        targets=np.asarray(targets)[np.random.RandomState(int.from_bytes(token[:4],'big')).permutation(len(targets))].tolist()
        out.extend((s,t,typ) for (s,_,_),t in zip(part,targets))
    c=Counter(out)
    return [{'source_role':s,'target_role':t,'dependency_type':typ,'support_count':n} for (s,t,typ),n in sorted(c.items())]

def build_semantic_bags(base_bags,raw_bags,ids,arm,seed):
    if arm not in ADAPTER_ARMS: raise ValueError(arm)
    result=[]; total=effective=changed=unshufflable=0
    for base,raw,sid in zip(base_bags,raw_bags,map(str,ids)):
        base=np.asarray(base,np.float32); raw=np.asarray(raw,np.float32); hosts=load_hosts(sid)
        if not hosts and len(base)==1: hosts=[{'parse_status':'NO_HOST'}]
        if len(hosts)!=len(base): raise RuntimeError(f'Host semantic order/count mismatch {sid}: {len(hosts)} != {len(base)}')
        feats=[]; masks=[]
        for hi,row in enumerate(hosts):
            if arm=='C0_S1': v,m=_control(row,'S1')
            elif arm in ('C1_S1','C3_S1'): v,m=_s1(row)
            elif arm=='C0_S3': v,m=_control(row,'S3')
            elif arm=='C2_S3': v,m=_s3(row)
            else:
                deps=row.get('s3_dependencies') or []; rw=_rewire(deps,seed,sid,hi); v,m=_s3(row,rw)
                if rw!=deps and deps: changed+=1
            feats.append(v); masks.append(m)
        feats=np.asarray(feats,np.float32); masks=np.asarray(masks,np.float32)
        if arm=='C3_S1':
            groups=defaultdict(list)
            for i,row in enumerate(hosts):
                role=tuple((raw[i,:31]>.5).astype(np.uint8).tolist()) if raw.shape[1]>=31 else ()
                endpoint_bin=int(min(4,math.log2(1+float(row.get('sensitive_event_count',0)))))
                ins_bin=int(min(6,math.log2(1+float(row.get('instruction_count',0)))))
                groups[(role,endpoint_bin,ins_bin,str(row.get('parse_status')))].append(i)
            for key,index in groups.items():
                if len(index)<2: unshufflable+=sum(masks[index]>0); continue
                token=hashlib.sha256(f'{seed}:{sid}:{key}:s1-shuffle'.encode()).digest()
                order=np.random.RandomState(int.from_bytes(token[:4],'big')).permutation(index)
                before=feats[index].copy(); feats[index]=feats[order]
                changed+=int(np.sum(np.any(before!=feats[index],axis=1)))
        total+=len(hosts); effective+=int(masks.sum())
        side=np.concatenate([feats,masks[:,None]],axis=1)
        joined=np.concatenate([base,side],axis=1).astype(np.float32)
        if joined.shape!=(len(base),MODEL_WIDTH) or not np.isfinite(joined).all(): raise RuntimeError('semantic bag contract')
        if not np.array_equal(joined[:,:BASE_WIDTH],base): raise RuntimeError('W0 columns changed')
        result.append(joined)
    return result,{'arm':arm,'hosts':total,'effective_hosts':effective,'changed_hosts':changed,
      'unshufflable_effective_hosts':unshufflable,'width':MODEL_WIDTH,'label_free':True,
      's1_binding':'aggregate category vector because v2 cache has no Role-by-category event binding',
      's3_shuffle':'within-Host aggregated target-role rewiring preserving source/target marginals per dependency type'}
