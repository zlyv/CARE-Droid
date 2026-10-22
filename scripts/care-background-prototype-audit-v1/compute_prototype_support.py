#!/usr/bin/env python3
"""Measure APK-level support of each frozen Prototype in its own training window.

Reads historical Host data and checkpoints, writes only new audit output. Uses
the formal RMS metric and formal Role-conditioned hard assignment, including
its nearest-distance fallback. No model, Prototype or threshold is fitted.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_assets import OUT, ROOT, MODEL, TAGS, SEEDS, write_csv, write_json

sys.path.insert(0, str(ROOT/'scripts/care-p31rf-c43-haig-host-ablation-source-v1'))
sys.path.insert(0, str(ROOT/'scripts'))
import haig_background as hb
import run_care_joint_dualprototype_ablation as v2

QUANTILES = (.25,.50,.75)
ARGS = SimpleNamespace(internal_background_coverage=.70)


def bag_parts(data, pos, mean, std):
    a,b = int(data['host_offsets'][pos]), int(data['host_offsets'][pos+1])
    raw = np.asarray(data['host_values'][a:b], np.float32)
    if raw.ndim != 2 or raw.shape[1] != 169 or not len(raw):
        raise RuntimeError('malformed frozen Host bag at '+str(pos))
    compact = np.concatenate([raw[:,:43],raw[:,105:]],axis=1)
    if compact.shape[1] != 107: raise RuntimeError('formal compact width changed')
    x = ((compact-mean)/std).astype(np.float32)
    role = np.maximum(raw[:,43:74],0)
    total = role.sum(axis=1,keepdims=True)
    role = np.divide(role,total,out=np.zeros_like(role),where=total>0)
    return x,role


def minima_by_assignment(dist, assigned, k):
    result = np.full(k,np.inf,dtype=np.float32)
    for j in np.unique(assigned):
        result[int(j)] = float(np.min(dist[assigned==j,int(j)]))
    return result


def transform_one(data,pos,bg,source_min=None,source_ids=None):
    state = bg['state']; sp = bg['sp']; bp = bg['bp']
    x,w = bag_parts(data,pos,bg['mean'],bg['std'])
    sd = v2.pairwise_rms(x,state['shared_centers'])
    bd = v2.pairwise_rms(x,state['benign_centers'])
    sa = hb.hard(sd/np.maximum(state['shared_scale'][None,:],1e-6),w,sp)
    ba = hb.hard(bd,w,bp)
    selected_s = sd[np.arange(len(x)),sa]
    selected_b = bd[np.arange(len(x)),ba]
    min_s = minima_by_assignment(sd,sa,state['shared_k'])
    min_b = minima_by_assignment(bd,ba,state['benign_k'])
    c = hb.cols(x,w,state,sp,bp,ARGS,v2)
    if c.shape != (len(x),4) or not np.isfinite(c).all():
        raise RuntimeError('formal background columns malformed')
    rh = np.argmax(w,axis=1); hv = w.sum(axis=1)>0
    shared_role = np.argmax(sp,axis=1); benign_role = np.argmax(bp,axis=1)
    shared_compat = hv & (sp.sum(axis=1)[sa]>0) & (rh==shared_role[sa])
    benign_compat = hv & (bp.sum(axis=1)[ba]>0) & (rh==benign_role[ba])
    if source_min is not None:
        # This is the nearest *observed* training Host, not a KMeans source ID.
        for bank,d,store in [('Shared',sd,source_min['Shared']),('External',bd,source_min['External'])]:
            local = np.argmin(d,axis=0)
            value = d[local,np.arange(d.shape[1])]
            for j in np.flatnonzero(value<store['distance']):
                store['distance'][j] = value[j]
                store['sha256'][j] = source_ids[pos]
                store['host_index'][j] = int(local[j])
    return (min_s,min_b,c.mean(axis=0),
            np.asarray([np.median(selected_s),np.median(selected_b)],np.float32),
            len(x),w.mean(axis=0),
            float(shared_compat.mean()),float(benign_compat.mean()))


def load_checkpoint(tag,seed):
    path=MODEL/'models/stage2'/tag/(str(seed)+'.pkl')
    with path.open('rb') as f: obj=pickle.load(f)
    bg=obj['background']; state=bg['state']
    if np.asarray(state['shared_centers']).shape!=(32,107) or np.asarray(state['benign_centers']).shape!=(24,107):
        raise RuntimeError('Prototype dimensions changed '+str(path))
    for key,shape in [('mean',(107,)),('std',(107,)),('sp',(32,31)),('bp',(24,31))]:
        if np.asarray(bg[key]).shape!=shape: raise RuntimeError(key+' shape changed '+str(path))
    return obj


def run_bank(data,id_position,source_ids,tag,seed,smoke=False):
    path=OUT/'bank_cache'/(tag+'_'+str(seed)+'.npz')
    if path.exists() and not smoke:
        with np.load(path,allow_pickle=False) as a:
            if a['min_shared'].shape[1]!=32 or a['min_external'].shape[1]!=24:
                raise RuntimeError('existing bank cache width changed')
        print('PROTOTYPE_BANK_RESUME',tag,seed,flush=True)
        return
    obj=load_checkpoint(tag,seed)
    train_ids=[str(x).lower() for x in obj['metadata']['train_sha']]
    val_ids=[str(x).lower() for x in obj['metadata']['validation_sha']]
    if set(train_ids)&set(val_ids): raise RuntimeError('train/validation SHA overlap')
    if any(s not in id_position for s in train_ids+val_ids):
        raise RuntimeError('checkpoint SHA not in frozen data')
    train_y=np.asarray([int(data['y'][id_position[s]]) for s in train_ids])
    expected={'C0R_2017':2017,'T2_C1':2018,'T3_C1':2019,'T4_C1':2020}[tag]
    if any(int(data['years'][id_position[s]])>expected for s in train_ids):
        raise RuntimeError('future year entered Prototype training')
    if any(int(data['years'][id_position[s]])<=expected for s in val_ids):
        raise RuntimeError('validation year is not after training')
    if len(train_ids)!=obj['audit']['train_n'] or len(val_ids)!=obj['audit']['validation_n']:
        raise RuntimeError('checkpoint lineage length mismatch')
    if smoke: val_ids=val_ids[:2];train_ids=train_ids[:2]
    ids=train_ids+val_ids; n=len(ids); train_n=len(train_ids)
    min_s=np.full((n,32),np.inf,np.float32);min_b=np.full((n,24),np.inf,np.float32)
    response=np.zeros((n,4),np.float32);medians=np.zeros((n,2),np.float32)
    host_count=np.zeros(n,np.int32);role_mean=np.zeros((n,31),np.float32)
    compatible=np.zeros((n,2),np.float32)
    nearest={bank:{'distance':np.full(k,np.inf,np.float32),'sha256':['']*k,
                   'host_index':[-1]*k} for bank,k in [('Shared',32),('External',24)]}
    start=time.monotonic()
    for i,sid in enumerate(ids):
        index=id_position[sid]
        row=transform_one(data,index,obj['background'],nearest if i<train_n else None,
                          source_ids)
        min_s[i],min_b[i],response[i],medians[i],host_count[i],role_mean[i],compatible[i,0],compatible[i,1]=row
        if (i+1)%200==0 or i+1==n:
            print('PROTOTYPE_BANK_PROGRESS',tag,seed,str(i+1)+'/'+str(n),
                  'seconds='+str(round(time.monotonic()-start,1)),flush=True)
    if smoke:
        if not np.isfinite(response).all() or response.shape!=(4,4):
            raise RuntimeError('smoke response failed')
        return
    thresholds=np.quantile(medians[:train_n],QUANTILES,axis=0).astype(np.float32)
    if thresholds.shape!=(3,2) or not np.isfinite(thresholds).all():
        raise RuntimeError('training threshold invalid')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f,ids=np.asarray(ids),train_n=train_n,labels=data['y'][[id_position[x] for x in ids]],
            years=data['years'][[id_position[x] for x in ids]],min_shared=min_s,min_external=min_b,
            response=response,median_matched_distance=medians,thresholds=thresholds,
            host_count=host_count,role_mean=role_mean,compatibility=compatible)
    provenance=[]
    for bank,k in [('Shared',32),('External',24)]:
        for j in range(k):
            provenance.append({'tag':tag,'seed':seed,'bank':bank,'prototype_id':j,
                'construction':'MiniBatchKMeans center; not a source Host',
                'source_host_identity':'UNKNOWN',
                'nearest_observed_training_apk':nearest[bank]['sha256'][j],
                'nearest_observed_host_index':nearest[bank]['host_index'][j],
                'nearest_observed_distance':nearest[bank]['distance'][j],
                'nearest_is_true_source':False})
    write_csv(OUT/'bank_cache'/(tag+'_'+str(seed)+'_provenance.csv'),provenance)
    write_json(OUT/'bank_cache'/(tag+'_'+str(seed)+'_manifest.json'),{
        'tag':tag,'seed':seed,'train_n':train_n,'validation_n':len(val_ids),
        'train_benign':int(np.sum(train_y==0)),'train_malware':int(np.sum(train_y==1)),
        'threshold_quantiles':QUANTILES,'thresholds':thresholds.tolist(),
        'role_rule':'formal Role-conditioned hard assignment with nearest fallback',
        'support_rule':'APK has >=1 assigned Host within fixed train-only distance quantile',
        'elapsed_seconds':round(time.monotonic()-start,2),
        'source_provenance':'KMeans center cannot be traced to one exact Host'})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prototype-support',action='store_true')
    p.add_argument('--smoke-test',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cpu')
    p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    if args.device!='cpu' or args.workers!=1:
        raise RuntimeError('read-only audit fixed to one low-priority CPU worker')
    if not args.prototype_support and not args.smoke_test:
        raise RuntimeError('select --smoke-test or --prototype-support')
    with np.load(MODEL/'data.npz',allow_pickle=False) as a:
        data={k:a[k] for k in ('ids','years','y','host_values','host_offsets')}
    ids=[str(x).lower() for x in data['ids']]
    if len(ids)!=len(set(ids)): raise RuntimeError('frozen data SHA duplicate')
    lookup={s:i for i,s in enumerate(ids)}
    if args.smoke_test:
        run_bank(data,lookup,ids,TAGS[0],SEEDS[0],smoke=True)
        print('PROTOTYPE_SELFTEST_PASS',flush=True)
    else:
        for tag in TAGS:
            for seed in SEEDS: run_bank(data,lookup,ids,tag,seed)
        print('PROTOTYPE_SUPPORT_CACHE_DONE',flush=True)


if __name__=='__main__':main()
