#!/usr/bin/env python3
"""Frozen Direct-C43 conditional Host-bag ablation with fixed RoleGraph context.

Role: post-hoc model sensitivity audit, never a whole-program Host deletion.
Writes only new v7 output files; reads but never edits APKs, caches or models.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
OUT=ROOT/'outputs/care_host_region_attribution_v7'
BASE=ROOT/'outputs/care_multiyear_development_v1'
SEEDS=(900000,900001,900002)


def save_json(name,obj):
    p=OUT/name
    if p.exists():raise RuntimeError('refuse overwrite '+str(p))
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def save_csv(name,rows):
    p=OUT/name
    if p.exists():raise RuntimeError('refuse overwrite '+str(p))
    rows=list(rows)
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def asset(path):
    path=Path(path).resolve(strict=True);st=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return dict(path=str(path),size=st.st_size,mtime_ns=st.st_mtime_ns,sha256=h.hexdigest())


def protected():
    names=[BASE/'data.npz',BASE/'feature_rows.pkl',BASE/'dataset_audit.json',
           BASE/'predictions/T3/router.npz',BASE/'per_apk_predictions.csv',
           ROOT/'outputs/care-2020-fn-tn-dex-review-v2/matched_pairs_restricted.csv',
           ROOT/'artifacts/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz',
           Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json'),
           ROOT/'scripts/care-di25-compact43-relationquery-source-v1/hostrel_features.py',
           ROOT/'scripts/care-multiyear-development-v1/train_stage2.py',
           ROOT/'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py',
           ROOT/'scripts/care-di25-graphglobal-path-source-v1/graphglobal_path_models.py',
           ROOT/'scripts/care-svm-c43-remove-bounded-correction-v1/model_variants.py']
    names += [BASE/'models/stage2/T3_C1'/(str(seed)+'.pkl') for seed in SEEDS]
    return names


def stable_hash_model(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def sigmoid(x):return 1/(1+math.exp(-x))


def run_pilot(device_name):
    if not OUT.is_dir() or (OUT/'all_host_effects.csv').exists():raise RuntimeError('output not fresh')
    start=time.monotonic();before=[asset(p) for p in protected()];save_json('asset_manifest_before.json',before)
    sys.path.insert(0,str(ROOT/'scripts/care-crossyear-layerwise-diagnosis-v2'))
    import diagnostic_common as dc
    mc,v4,_,data=dc.original();torch=v4.torch
    from train_stage2 import raw_bags,transform_saved
    if device_name=='cuda' and (not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<2*1024**3):
        raise RuntimeError('CUDA unavailable or <2 GiB free')
    device=torch.device(device_name)
    pairs=read_csv(ROOT/'outputs/care-2020-fn-tn-dex-review-v2/matched_pairs_restricted.csv')
    if len(pairs)!=22:raise RuntimeError('matched pair count differs')
    pair_of={}
    for k,row in enumerate(pairs,1):
        for side,key in [('FN','fn_sha256'),('TN','tn_sha256')]:
            if row[key] in pair_of:raise RuntimeError('pair reuse of APK')
            pair_of[row[key]]=dict(pair=k,pair_side=side)
    with np.load(BASE/'predictions/T3/router.npz',allow_pickle=False) as z:
        router={k:z[k] for k in z.files}
    route={str(s):bool(q) for s,q in zip(router['ids'],router['new_route'])}
    s1={str(s):float(q) for s,q in zip(router['ids'],router['s1'])}
    positions={str(s):i for i,s in enumerate(data['ids'])}
    per=read_csv(BASE/'per_apk_predictions.csv')
    frozen={r['sha256']:r for r in per if r['window']=='T3' and r['arm']=='U3_FULL_ROLLING_C1'}
    ids=sorted(sid for sid,r in frozen.items() if route.get(sid,False) and s1[sid]<.5)
    if len(ids)!=77:raise RuntimeError('2020 routed Stage-I benign count changed: '+str(len(ids)))
    member={}
    for sid in ids:
        r=frozen[sid]
        if not route[sid] or s1[sid]>=.5 or int(r['stage1_prediction'])!=0 or int(r['routed'])!=1:
            raise RuntimeError('T3 Stage-I benign routed contract mismatch')
        label=int(r['label']); expert=float(r['expert_probability'])
        category=('UNFIXED_FN' if expert<.5 else 'FIXED_FN') if label==1 else ('STABLE_TN' if expert<.5 else 'BROKEN_TN')
        member[sid]=dict(category=category,**pair_of.get(sid,dict(pair='',pair_side='')))
    if dict(Counter(v['category'] for v in member.values()))!={'UNFIXED_FN':24,'FIXED_FN':2,'STABLE_TN':48,'BROKEN_TN':3}:
        raise RuntimeError('2020 outcome counts changed')
    ix=np.asarray([positions[s] for s in ids],int)
    raw=raw_bags(data,ix)
    if any(len(b)<2 for b in raw):raise RuntimeError('empty/singleton Host in audit')
    sys.path.insert(0,str(ROOT/'scripts/care-di25-compact43-relationquery-source-v1'))
    from hostrel_features import reconstruct_host_relations
    role_mapping=np.asarray(json.loads(Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json').read_text())['payload_to_model_role'],int)
    endpoint={}
    with gzip.open(ROOT/'artifacts/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);sid=row.get('sha256')
            if sid in member and row.get('status')=='OK':
                if sid in endpoint:raise RuntimeError('duplicate endpoint '+sid)
                endpoint[sid]=row['payload']
    if len(endpoint)!=len(ids):raise RuntimeError('missing endpoint payloads')
    identity={}
    for sid,bag in zip(ids,raw):
        records=reconstruct_host_relations(endpoint[sid],role_mapping)
        if len(records)!=len(bag):raise RuntimeError('Host count mismatch '+sid)
        if not np.allclose(bag[:,:31],np.stack([r['presence'] for r in records]),atol=1e-6,rtol=0):
            raise RuntimeError('Host presence/order mismatch '+sid)
        if not np.allclose(bag[:,43:74],np.stack([r['endpoint_counts'] for r in records]),atol=1e-6,rtol=0):
            raise RuntimeError('Host endpoint count/order mismatch '+sid)
        if not np.allclose(bag[:,74:105],np.stack([r['directed_neighbor_role'] for r in records]),atol=1e-6,rtol=0):
            raise RuntimeError('Host Relation31/order mismatch '+sid)
        identity[sid]=[str(r['host_id']) for r in records]
    models=[];bags=[];graphs=[];globals_=[];metadata=[];attention=[];baseline=[];modelhash=[]
    for seed in SEEDS:
        obj=dc.checkpoint(BASE/'models/stage2/T3_C1'/(str(seed)+'.pkl'),mc,data)
        args=argparse.Namespace(**obj['metadata']['args'])
        va_pos={s:i for i,s in enumerate(obj['metadata']['validation_sha'])}
        vi=[va_pos[s] for s in ids]
        region=transform_saved(v4,raw,obj['background'],args)
        graph=v4.transform_role_graphs(data['s7'][ix],obj['graph_scaler'])
        graph=v4.graph_tensors(graph,device)
        glob=np.asarray(obj['global_validation'][vi],np.float32)
        model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args).to(device)
        model.load_state_dict(obj['state_dict']);model.eval();weight=stable_hash_model(model)
        original_attention=[];original_logits=[]
        with torch.no_grad():
            for j,sid in enumerate(ids):
                g={key:value[j] for key,value in graph.items()}
                logit,_,trace=model.forward_with_trace(torch.as_tensor(glob[j],device=device),torch.zeros((),device=device),g,torch.as_tensor(region[j],device=device))
                p=sigmoid(float(logit))
                if abs(p-float(obj['probability'][vi[j]]))>1e-6:
                    raise RuntimeError('frozen seed replay mismatch '+sid+' '+str(seed))
                a=trace['attention'].detach().cpu().numpy()
                if len(a)!=len(raw[j]) or not np.isfinite(a).all() or abs(float(a.sum())-1)>1e-5:
                    raise RuntimeError('attention shape or sum invalid')
                original_attention.append(a);original_logits.append(float(logit))
        models.append(model);bags.append(region);graphs.append(graph);globals_.append(glob)
        metadata.append((args,obj['background']));attention.append(original_attention);baseline.append(original_logits);modelhash.append(weight)
        print('HRA_V7_BASELINE',seed,len(ids),flush=True)
    # Pre-register *all* frozen Host rows before computing any effects. This
    # descriptive sweep is not a model-selection or semantics-preserving test.
    selected=[];host_out=[]
    for j,sid in enumerate(ids):
        n=len(raw[j]);mean_attention=np.mean([a[j] for a in attention],axis=0)
        order=np.lexsort((np.arange(n),-mean_attention))
        primary=np.where(raw[j][:,43:74].sum(1)>0,np.argmax(raw[j][:,43:74],axis=1),-1)
        for rank,k in enumerate(order.tolist(),1):
            selected.append(dict(sid=sid,rank=rank,index=int(k),role=int(primary[k]),host_id=identity[sid][int(k)]))
        host_out.append(dict(sha256=sid,category=member[sid]['category'],pair=member[sid]['pair'],hosts=n,
                             primary_role_counts=dict(Counter(map(int,primary)))))
    save_json('all_host_selection_protocol.json',dict(status='FROZEN_BEFORE_EFFECTS',
        cohort='all 77 T3 routed Stage-I-predicted-benign APKs; 24 unfixed FN, 2 fixed FN, 48 stable TN, 3 broken TN',
        stage1_direction='predicted benign, routed',candidate_scope='every existing Host row, no outcome-based selection',
        order='mean baseline attention across three frozen seeds, used only for traversal/rank',
        comparison='all 77 by category, plus pre-existing 22 matched FN/TN pairs',
        graph='frozen original RoleGraph and Global24; conditional Host-path ablation, NOT coherent whole-APK Host deletion',
        background='each seed uses its original frozen Shared/External prototypes and recomputes dynamic Internal on reduced bag',
        inference='full original forward re-encodes remaining Hosts and recomputes attention/logit',
        interpretation='ARTIFICIAL_INPUT_PERTURBATION, model sensitivity only; no program or classification causal claim'))
    save_json('all_host_counts.json',host_out)
    effects=[]
    lookup={sid:j for j,sid in enumerate(ids)}
    for item in selected:
        sid=item['sid'];j=lookup[sid];k=item['index'];reduced=np.delete(raw[j],k,axis=0)
        if len(reduced)<1:continue
        for m,seed in enumerate(SEEDS):
            args,prep=metadata[m]
            changed=transform_saved(v4,[reduced],prep,args)[0]
            if changed.shape!=(len(reduced),142) or not np.isfinite(changed).all():
                raise RuntimeError('transformed reduced bag invalid')
            graph={key:value[j] for key,value in graphs[m].items()}
            with torch.no_grad():
                logit,_,trace=models[m].forward_with_trace(torch.as_tensor(globals_[m][j],device=device),torch.zeros((),device=device),graph,torch.as_tensor(changed,device=device))
            after=float(logit);before_logit=baseline[m][j];delta=after-before_logit
            if not (math.isfinite(after) and math.isfinite(delta)):
                raise RuntimeError('nonfinite intervention')
            effects.append(dict(sha256=sid,pair=member[sid]['pair'],category=member[sid]['category'],label=int(frozen[sid]['label']),
                attention_rank=item['rank'],host_index=k,host_id=item['host_id'],primary_role=item['role'],
                seed=seed,host_count=len(raw[j]),removed_fraction=1/len(raw[j]),
                baseline_attention=float(attention[m][j][k]),
                logit_before=before_logit,logit_after=after,delta_logit=delta,
                probability_before=sigmoid(before_logit),probability_after=sigmoid(after),
                flipped=int((sigmoid(before_logit)>=.5)!=(sigmoid(after)>=.5)),
                internal_recomputed=1,other_background_parameters_frozen=1,
                host_encoder_attention_recomputed=1,graph_context_frozen=1,
                validity='ARTIFICIAL_INPUT_PERTURBATION'))
        if len(effects)%1500==0:print('HRA_V7_EFFECTS',len(effects),'of',len(selected)*len(SEEDS),flush=True)
    for model,pre in zip(models,modelhash):
        if stable_hash_model(model)!=pre:raise RuntimeError('frozen weights changed')
    save_csv('all_host_effects.csv',effects)
    bykey={}
    for row in effects:
        key=(row['sha256'],row['host_index'])
        bykey.setdefault(key,[]).append(row)
    summary=[]
    for key,rows in bykey.items():
        if len(rows)!=3:raise RuntimeError('missing seed effect')
        delta=np.array([r['delta_logit'] for r in rows])
        summary.append(dict(sha256=key[0],pair=rows[0]['pair'],category=rows[0]['category'],attention_rank=rows[0]['attention_rank'],
            host_index=key[1],host_id=rows[0]['host_id'],primary_role=rows[0]['primary_role'],host_count=rows[0]['host_count'],
            attention_mean=float(np.mean([r['baseline_attention'] for r in rows])),
            delta_mean=float(delta.mean()),delta_abs_mean=float(np.abs(delta).mean()),
            seed_direction_consistent=bool(np.all(delta>0) or np.all(delta<0)),
            suppressive_consistent=bool(np.all(delta>0)),promoting_consistent=bool(np.all(delta<0)),
            seed_deltas=[float(x) for x in delta],
            seed_flips=sum(int(r['flipped']) for r in rows)))
    save_csv('all_host_ensemble_effects.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in summary])
    by_sid={sid:[] for sid in ids}
    for r in summary:by_sid[r['sha256']].append(r)
    apk=[];candidates=[]
    for sid in ids:
        rows=by_sid[sid];category=member[sid]['category']
        # FN: removing a suppressive Host raises malware score. Broken TN:
        # removing a promoting Host lowers malware score. These are conditional
        # model effects, not claims that the Host is benign/malicious code.
        harmful_sign=1 if category in ('UNFIXED_FN','FIXED_FN') else -1
        ordered=sorted(rows,key=lambda r:(-harmful_sign*r['delta_mean'],r['host_index']))
        stable=[r for r in ordered if (r['suppressive_consistent'] if harmful_sign==1 else r['promoting_consistent'])]
        best=stable[0] if stable else None
        any_best=ordered[0]
        for rank,r in enumerate(stable[:5],1):
            candidates.append(dict(sha256=sid,category=category,pair=member[sid]['pair'],harmful_rank=rank,
                host_index=r['host_index'],host_id=r['host_id'],primary_role=r['primary_role'],
                attention_rank=r['attention_rank'],attention_mean=r['attention_mean'],
                harmful_delta_logit=harmful_sign*r['delta_mean'],seed_deltas=r['seed_deltas'],
                seed_flips=r['seed_flips'],scope='CONDITIONAL_HOST_PATH_ONLY'))
        apk.append(dict(sha256=sid,category=category,pair=member[sid]['pair'],hosts=len(raw[lookup[sid]]),
            direction='remove_host_raises_malware_logit' if harmful_sign==1 else 'remove_host_lowers_malware_logit',
            seed_consistent_harmful_host_count=len(stable),
            strongest_consistent_host_index='' if best is None else best['host_index'],
            strongest_consistent_host_id='' if best is None else best['host_id'],
            strongest_consistent_harmful_delta=0.0 if best is None else harmful_sign*best['delta_mean'],
            strongest_consistent_attention_rank='' if best is None else best['attention_rank'],
            strongest_consistent_seed_flips=0 if best is None else best['seed_flips'],
            strongest_any_host_index=any_best['host_index'],
            strongest_any_harmful_delta=harmful_sign*any_best['delta_mean']))
    save_csv('per_apk_host_sensitivity.csv',apk)
    save_csv('error_host_candidates.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()}
        for r in candidates if r['category'] in ('UNFIXED_FN','BROKEN_TN')])
    paired=[]
    for pair in range(1,23):
        fn=next(r for r in apk if r['pair']==pair and r['category']=='UNFIXED_FN')
        tn=next(r for r in apk if r['pair']==pair and r['category']=='STABLE_TN')
        paired.append(dict(pair=pair,fn_strongest_consistent_suppressive=fn['strongest_consistent_harmful_delta'],
            tn_strongest_consistent_promoting=tn['strongest_consistent_harmful_delta'],
            fn_host_id=fn['strongest_consistent_host_id'],tn_host_id=tn['strongest_consistent_host_id']))
    save_csv('matched_pair_host_sensitivity.csv',paired)
    after=[asset(Path(r['path'])) for r in before]
    changed=[dict(before=b,after=a) for b,a in zip(before,after) if b!=a]
    save_json('asset_manifest_after.json',after)
    save_json('baseline_integrity_check.json',dict(status='PASS' if not changed else 'FAIL',asset_count=len(before),changed=changed,scope='listed reused files only; no APK opened'))
    if changed:raise RuntimeError('protected assets changed')
    result=dict(status='CONDITIONAL_HOST_PATH_SENSITIVITY_AUDIT_COMPLETE',
        valid_model_interventions=0,artificial_input_perturbations=len(effects),
        matched_pairs=22,apks=len(ids),seeds=3,hosts=len(summary),
        category_counts=dict(Counter(r['category'] for r in apk)),
        unfixed_fn_with_consistent_suppressive=sum(r['seed_consistent_harmful_host_count']>0 for r in apk if r['category']=='UNFIXED_FN'),
        broken_tn_with_consistent_promoting=sum(r['seed_consistent_harmful_host_count']>0 for r in apk if r['category']=='BROKEN_TN'),
        matched_pairs_fn_stronger_than_tn=sum(r['fn_strongest_consistent_suppressive']>r['tn_strongest_consistent_promoting'] for r in paired),
        interpretation='All frozen Host rows have method-level identity and model sensitivity effects. Holding RoleGraph/Global24 fixed prevents attribution of the complete second-layer error to these rows. An observed logit shift is not DEX behavioral proof or program causality.',
        seconds=time.monotonic()-start,formal_integrity='PASS')
    save_json('final_decision.json',result)
    print('HRA_V7_AUDIT_DONE',json.dumps(result,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    if os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise RuntimeError('PYTHONDONTWRITEBYTECODE must be 1')
    run_pilot(a.device)


if __name__=='__main__':main()
