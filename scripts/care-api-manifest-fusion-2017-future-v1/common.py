#!/usr/bin/env python3
"""Shared contracts and feature helpers for the CARE API/Manifest fusion experiment.

Inputs: frozen W0 checkpoints/caches, endpoint-centered Host graphs, original APK paths,
Manifest135 shards, and autonomous-router predictions. Outputs are written only beneath
the new experiment artifact. This module never modifies formal models or source assets.
"""
from __future__ import annotations

import csv, gzip, hashlib, json, math, os, pickle, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
HERE=ROOT/'scripts/care-api-manifest-fusion-2017-future-v1'
OUT=ROOT/'artifacts/care_api_manifest_fusion_2017_future_v1'
SCBE=ROOT/'outputs/care_scbe_v1'
MYD=ROOT/'outputs/care_multiyear_development_v1'
AUTO=ROOT/'outputs/care_manifest_rf_autonomous_error_router_v1'
FOUR=ROOT/'outputs/care_manifest_four_member_disagreement_v1/run_20261019_pilot01'
SOURCE_ASSET=Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2')
FUTURE_ASSET=Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2')
SOURCE_ENDPOINT=SOURCE_ASSET/'endpoint_centered_tensor_cache_2017.jsonl.gz'
FUTURE_ENDPOINT=FUTURE_ASSET/'endpoint_centered_tensor_cache_2018_2021.jsonl.gz'
SOURCE_LINEAGE=SOURCE_ASSET/'source_lineage_2017.csv'
FUTURE_LINEAGE=ROOT/'artifacts/care-litelocal-p31-diagnostics-v1/preflight/future_lineage_label_free.csv'
MAPPING=ROOT/'scripts/care-source-recovery-stage2only-v1/extractor_repo/configs/protocol/semantic_mapping_v1.yaml'
MODEL_SEEDS=(900000,900001,900002)
MANIFEST_SEEDS=(20261018,20261019,20261020)
SPLITS=('S1','S2','S3','S4','S5')
YEARS=(2017,2018,2019,2020,2021)
YEAR_COUNTS={2017:799,2018:793,2019:399,2020:397,2021:400}
ROLE_MAP=[7,10,21,19,9,15,16,0,11,25,29,6,12,1,26,28,23,13,14,2,17,30,22,4,5,24,3,18,8,20,27]
API_DIM=512
ARMS=(
 'B0_W0','C0_CODE_CAPACITY','A0_API_STATS','A1_API_FAMILY','A2_API_ENDPOINT','A3_API_SHUFFLE',
 'M1_MANIFEST_FUSION','M2_CAPACITY','M2_MANIFEST_SHUFFLE','AM1_FAMILY_MANIFEST','AM2_ENDPOINT_MANIFEST')
TRAINED_ARMS=ARMS[1:]
PRIMARY_ARMS=('B0_W0','A1_API_FAMILY','A2_API_ENDPOINT','A3_API_SHUFFLE','M1_MANIFEST_FUSION',
              'M2_CAPACITY','AM1_FAMILY_MANIFEST','AM2_ENDPOINT_MANIFEST')
COMPARISONS=(
 ('A1_API_FAMILY','B0_W0','A1_vs_B0'),('A1_API_FAMILY','C0_CODE_CAPACITY','A1_vs_C0'),
 ('A1_API_FAMILY','A0_API_STATS','A1_vs_API_STATS'),('A2_API_ENDPOINT','B0_W0','A2_vs_B0'),
 ('A2_API_ENDPOINT','A1_API_FAMILY','A2_vs_A1'),('A2_API_ENDPOINT','A3_API_SHUFFLE','A2_vs_A3'),
 ('A2_API_ENDPOINT','A0_API_STATS','A2_vs_API_STATS'),('A2_API_ENDPOINT','C0_CODE_CAPACITY','A2_vs_C0'),
 ('M1_MANIFEST_FUSION','B0_W0','M1_vs_B0'),('M1_MANIFEST_FUSION','C0_CODE_CAPACITY','M1_vs_C0'),
 ('M1_MANIFEST_FUSION','M2_CAPACITY','M1_vs_M2_CAPACITY'),
 ('M1_MANIFEST_FUSION','M2_MANIFEST_SHUFFLE','M1_vs_M2_SHUFFLE'),
 ('AM1_FAMILY_MANIFEST','A1_API_FAMILY','AM1_vs_A1'),('AM1_FAMILY_MANIFEST','M1_MANIFEST_FUSION','AM1_vs_M1'),
 ('AM2_ENDPOINT_MANIFEST','A2_API_ENDPOINT','AM2_vs_A2'),('AM2_ENDPOINT_MANIFEST','M1_MANIFEST_FUSION','AM2_vs_M1'),
 ('AM2_ENDPOINT_MANIFEST','AM1_FAMILY_MANIFEST','AM2_vs_AM1'))

sys.path[:0]=[str(ROOT/'scripts/care-background-functional-attribution-v1'),str(ROOT/'scripts/care-scbe-v1'),
              str(ROOT/'scripts/care-multiyear-development-v1'),str(ROOT/'scripts/care-lite-stage-v1'),
              str(ROOT/'scripts/care-host-method-semantics-feasibility-2017-v2'),
              str(ROOT/'scripts/care-local-control-constraint-feasibility-v2')]

def canonical(value): return re.sub(r'\s+','',str(value or '')).casefold()
def stable_id(value): return int.from_bytes(hashlib.sha256(canonical(value).encode()).digest()[:8],'big')

def jv(x):
    if isinstance(x,dict): return {str(k):jv(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,set)): return [jv(v) for v in x]
    if isinstance(x,Path): return str(x)
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.bool_): return bool(x)
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,np.floating): return None if not np.isfinite(x) else float(x)
    if isinstance(x,float) and not np.isfinite(x): return None
    return x

def atomic_json(path,value,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace: raise FileExistsError(path)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(jv(value),ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,path)

def atomic_pickle(path,value,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace: raise FileExistsError(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:pickle.dump(value,f,pickle.HIGHEST_PROTOCOL)
    os.replace(tmp,path)

def read_pickle(path):
    with Path(path).open('rb') as f:return pickle.load(f)

def atomic_csv(path,rows,replace=False):
    import pandas as pd
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace: raise FileExistsError(path)
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(list(rows));tmp=path.with_suffix(path.suffix+'.tmp');frame.to_csv(tmp,index=False);os.replace(tmp,path)

def atomic_json_gz(path,value,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace: raise FileExistsError(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(tmp,'wt',encoding='utf-8') as f:json.dump(jv(value),f,ensure_ascii=False,separators=(',',':'))
    os.replace(tmp,path)

def read_json_gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)

def digest(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return {'path':str(path.resolve()),'size':path.stat().st_size,'sha256':h.hexdigest()}

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def status(stage,state='RUNNING',**extra):
    atomic_json(OUT/'logs/run_status.json',{'status':state,'stage':stage,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**extra},replace=True)
    print('API_MANIFEST_STAGE',stage,state,flush=True)

def load_data():
    from bfa_core import load_data_contract
    data,_=load_data_contract();return data

def lineage():
    rows=[]
    for r in read_csv(SOURCE_LINEAGE):rows.append({'sha256':r['sha256'].lower(),'year':2017,'absolute_path':r['absolute_path']})
    for r in read_csv(FUTURE_LINEAGE):rows.append({'sha256':r['sha256'].lower(),'year':int(r['year']),'absolute_path':r['absolute_path']})
    if len(rows)!=2788 or len({r['sha256'] for r in rows})!=2788:raise RuntimeError('lineage membership')
    counts=Counter(r['year'] for r in rows)
    if dict(counts)!=YEAR_COUNTS:raise RuntimeError('lineage year counts '+repr(counts))
    missing=[r['absolute_path'] for r in rows if not Path(r['absolute_path']).is_file()]
    if missing:raise RuntimeError('missing APK paths '+repr(missing[:5]))
    return rows

def manifest135(ids):
    import importlib.util
    module_path=ROOT/'scripts/care-lite-stage-v1/common.py'
    spec=importlib.util.spec_from_file_location('care_lite_stage_common_for_api_manifest',module_path)
    lc=importlib.util.module_from_spec(spec);spec.loader.exec_module(lc)
    source,sm=lc.load_shards('source');future,fm=lc.load_shards('future')
    if sm or fm:raise RuntimeError(f'Manifest shards missing source={len(sm)} future={len(fm)}')
    lookup={**source,**future};x=lc.matrix([lookup[str(s)] for s in ids],'L1_MANIFEST').astype(np.float32)
    if x.shape!=(2788,135) or not np.isfinite(x).all():raise RuntimeError('Manifest135 contract')
    return x

def frozen_view(record):
    payload=record['payload'];ids=list(map(int,payload['node_ids']));roles=payload['node_role_ids'];base=np.asarray(payload['node_base_features'],np.float32)
    if base.size==0:base=np.empty((0,15),np.float32)
    edges=payload['edge_index']
    if len(ids)!=len(set(ids)) or base.shape!=(len(ids),15) or len(edges)!=2:raise RuntimeError('endpoint schema')
    model_roles=[sorted({ROLE_MAP[int(v)] for v in (values or [])}) for values in roles]
    endpoints=set(map(int,np.flatnonzero(base[:,4]>.5)));targets=defaultdict(dict)
    for left,right in zip(edges[0],edges[1]):
        left,right=int(left),int(right)
        if right in endpoints and model_roles[right]:targets[left][str(ids[right])]=model_roles[right]
    hosts=sorted(n for n,v in targets.items() if v)
    return {'sha256':str(record['sha256']).lower(),'year':int(record['metadata_year']),'host_ids':[str(ids[n]) for n in hosts],
            'pairs_by_host':{str(ids[n]):targets[n] for n in hosts},'summary':record.get('summary',{})}

def load_endpoint_views():
    result={}
    for path in (SOURCE_ENDPOINT,FUTURE_ENDPOINT):
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                if not line.strip():continue
                row=json.loads(line);sid=str(row.get('sha256','')).lower()
                if row.get('status')!='OK' or sid in result:raise RuntimeError('endpoint row '+sid)
                result[sid]=frozen_view(row)
    if len(result)!=2788:return (_ for _ in ()).throw(RuntimeError('endpoint coverage '+str(len(result))))
    return result

def verify_host_alignment(data,views):
    ids=list(map(str,data['ids']));off=np.asarray(data['host_offsets'],int);total=0
    for i,sid in enumerate(ids):
        v=views[sid];n=int(data['real_host_count'][i])
        if len(v['host_ids'])!=n:raise RuntimeError(f'Host count mismatch {sid} {len(v["host_ids"])} {n}')
        raw=np.asarray(data['host_values'][off[i]:off[i]+n],np.float32)
        for j,hid in enumerate(v['host_ids']):
            expected=np.zeros(31,np.float32)
            for rv in v['pairs_by_host'][hid].values():expected[list(map(int,rv))]=1
            if not np.array_equal(raw[j,:31],expected):raise RuntimeError(f'SelfRole31 mismatch {sid} {j}')
        total+=n
    if total!=int(np.sum(data['real_host_count'])):raise RuntimeError('Host total')
    return total

def save_views(views):
    folder=OUT/'features/frozen_views';folder.mkdir(parents=True,exist_ok=True)
    for sid,v in views.items():
        p=folder/f'{sid}.json.gz'
        if not p.exists():atomic_json_gz(p,v)

def load_api_records(ids):
    result={}
    for sid in map(str,ids):
        p=OUT/'features/api_events'/f'{sid}.json.gz'
        if not p.is_file():raise RuntimeError('API cache missing '+sid)
        result[sid]=read_json_gz(p)
    return result

def token_counter(record,kind):
    c=Counter()
    for host in record.get('hosts',[]):
        for e in host.get('events',[]):c[e[kind]]+=int(e['count'])
    return c

def token_roles(record):
    d={}
    for host in record.get('hosts',[]):
        for e in host.get('events',[]):d[e['endpoint']]=str(e['role_key'])
    return d

def api_stats(record):
    host_counts=[sum(int(e['count']) for e in h.get('events',[])) for h in record.get('hosts',[])]
    active=[v for v in host_counts if v>0];ep=token_counter(record,'endpoint');fam=token_counter(record,'family')
    return np.asarray([sum(host_counts),len(ep),len(fam),len(active),np.mean(active) if active else 0.,max(active) if active else 0.],np.float32)

def build_vocab(records,train_ids,kind,max_tokens=API_DIM-1,min_df=2):
    df=Counter()
    for sid in map(str,train_ids):df.update(token_counter(records[sid],kind).keys())
    selected=[t for t,n in sorted(df.items(),key=lambda x:(-x[1],x[0])) if n>=min_df][:max_tokens]
    return {'kind':kind,'tokens':selected,'index':{t:i+1 for i,t in enumerate(selected)},'oov_index':0,'min_df':min_df,'dimension':API_DIM,
            'train_documents':len(train_ids),'observed_tokens':len(df)}

def shuffle_binding(records,train_ids,vocab,seed):
    groups=defaultdict(list);role={}
    for sid in map(str,train_ids):role.update(token_roles(records[sid]))
    for t in vocab['tokens']:groups[role.get(t,'UNKNOWN')].append(t)
    mapping={};movable=set()
    rng=np.random.RandomState(seed)
    for key,tokens in sorted(groups.items()):
        tokens=sorted(tokens)
        if len(tokens)<=1:
            for t in tokens:mapping[t]=t
        else:
            offset=int(rng.randint(1,len(tokens)));rot=tokens[offset:]+tokens[:offset]
            mapping.update(dict(zip(tokens,rot)));movable.update(tokens)
    return mapping,movable

def vectorize_api(records,ids,vocab,kind,mapping=None):
    x=np.zeros((len(ids),API_DIM),np.float32);total=known=movable_events=0
    idx=vocab['index']
    for i,sid in enumerate(map(str,ids)):
        c=token_counter(records[sid],kind)
        for token,n in c.items():
            total+=n;target=mapping.get(token,token) if mapping else token
            j=idx.get(target,0);x[i,j]+=n;known+=n if j else 0
        x[i]=np.log1p(x[i])
    return x,{'events':int(total),'known_events':int(known),'oov_events':int(total-known),'oov_rate':float((total-known)/max(total,1)),
              'covered_apks':int(np.sum(np.any(x>0,axis=1))),'n_apks':len(ids)}

def stats_block(records,train_ids,target_ids):
    tr=np.stack([api_stats(records[str(s)]) for s in train_ids]);te=np.stack([api_stats(records[str(s)]) for s in target_ids])
    mean=tr.mean(0);std=tr.std(0);std[std<1e-6]=1
    rng=np.random.RandomState(20261022);w=rng.normal(0,1/math.sqrt(6),(6,API_DIM)).astype(np.float32);b=rng.uniform(-math.pi,math.pi,API_DIM).astype(np.float32)
    return np.tanh(((te-mean)/std)@w+b).astype(np.float32),{'mean':mean,'std':std,'projection_sha256':hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()}

def permute_rows(values,ids,seed):
    ids=np.asarray(list(map(str,ids)));order=np.argsort([hashlib.sha256(f'{seed}:{s}:manifest-row-control'.encode()).hexdigest() for s in ids])
    if len(order)>1:order=np.roll(order,1)
    return np.asarray(values)[order]

def standardize(train,target):
    train=np.asarray(train,np.float32);target=np.asarray(target,np.float32);mean=train.mean(0);std=train.std(0);std[std<1e-6]=1
    return ((train-mean)/std).astype(np.float32),((target-mean)/std).astype(np.float32),mean,std

def feature_blocks(arm,records,train_ids,target_ids,manifest_train,manifest_target,seed):
    api_tr=np.zeros((len(train_ids),API_DIM),np.float32);api_te=np.zeros((len(target_ids),API_DIM),np.float32)
    man_tr=np.zeros((len(train_ids),135),np.float32);man_te=np.zeros((len(target_ids),135),np.float32)
    meta={'arm':arm}
    if arm in ('A0_API_STATS',):
        api_tr,st=stats_block(records,train_ids,train_ids);api_te,_=stats_block(records,train_ids,target_ids);meta['api_stats']=jv(st)
    elif arm in ('A1_API_FAMILY','AM1_FAMILY_MANIFEST'):
        vocab=build_vocab(records,train_ids,'family');api_tr,c1=vectorize_api(records,train_ids,vocab,'family');api_te,c2=vectorize_api(records,target_ids,vocab,'family');meta.update(vocab={k:v for k,v in vocab.items() if k!='index'},train_coverage=c1,target_coverage=c2)
    elif arm in ('A2_API_ENDPOINT','AM2_ENDPOINT_MANIFEST','A3_API_SHUFFLE'):
        vocab=build_vocab(records,train_ids,'endpoint');mapping=None
        if arm=='A3_API_SHUFFLE':
            mapping,movable=shuffle_binding(records,train_ids,vocab,seed)
            def movable_fraction(sample_ids):
                total=changed=0
                for sid in map(str,sample_ids):
                    for token,n in token_counter(records[sid],'endpoint').items():
                        total+=n;changed+=n if token in movable else 0
                return float(changed/max(total,1))
            meta['shuffle_movable_vocab_fraction']=len(movable)/max(len(vocab['tokens']),1)
            meta['shuffle_movable_event_fraction_train']=movable_fraction(train_ids)
            meta['shuffle_movable_event_fraction_target']=movable_fraction(target_ids)
            meta['shuffle_mapping_sha256']=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
        api_tr,c1=vectorize_api(records,train_ids,vocab,'endpoint',mapping);api_te,c2=vectorize_api(records,target_ids,vocab,'endpoint',mapping);meta.update(vocab={k:v for k,v in vocab.items() if k!='index'},train_coverage=c1,target_coverage=c2)
    if arm in ('M1_MANIFEST_FUSION','AM1_FAMILY_MANIFEST','AM2_ENDPOINT_MANIFEST'):
        man_tr,man_te,mean,std=standardize(manifest_train,manifest_target);meta['manifest_scaler']={'mean':mean,'std':std}
    elif arm=='M2_MANIFEST_SHUFFLE':
        ptr=permute_rows(manifest_train,train_ids,seed);pte=permute_rows(manifest_target,target_ids,seed+1)
        man_tr,man_te,mean,std=standardize(ptr,pte);meta['manifest_scaler']={'mean':mean,'std':std};meta['manifest_row_binding']='deterministic label-free cyclic hash permutation within train/target cohorts'
    return api_tr,api_te,man_tr,man_te,meta

def metric(y,p):
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y,int);p=np.asarray(p,float);z=p>=.5;tn=int(np.sum((y==0)&~z));fp=int(np.sum((y==0)&z));fn=int(np.sum((y==1)&~z));tp=int(np.sum((y==1)&z));tnr=tn/max(tn+fp,1);tpr=tp/max(tp+fn,1);prec=tp/max(tp+fp,1)
    return {'n':len(y),'BA':.5*(tnr+tpr),'Accuracy':float(np.mean(z==y)),'AUC':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'),'F1':2*prec*tpr/max(prec+tpr,1e-12),'TN':tn,'FP':fp,'FN':fn,'TP':tp,'FPR':1-tnr,'FNR':1-tpr}

def cascade(y,stage1,expert,route):
    y=np.asarray(y,int);stage1=np.asarray(stage1,float);expert=np.asarray(expert,float);route=np.asarray(route,bool);final=np.where(route,expert,stage1);before=(stage1>=.5)==y;after=(final>=.5)==y;m=metric(y,final);m.update({'routed':int(route.sum()),'route_rate':float(route.mean()),'Fixed':int(np.sum(~before&after)),'Broken':int(np.sum(before&~after)),'Net_Fixed':int(np.sum(~before&after)-np.sum(before&~after)),'stage1_errors':int(np.sum(~before)),'routed_stage1_errors':int(np.sum(route&~before)),'expert_all_BA':metric(y,expert)['BA'],'expert_routed_BA':metric(y[route],expert[route])['BA'] if np.any(route) and len(np.unique(y[route]))==2 else float('nan')});return m,final


