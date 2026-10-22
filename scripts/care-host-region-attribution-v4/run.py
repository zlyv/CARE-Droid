#!/usr/bin/env python3
"""Frozen Direct-C43 conditional Host-bag ablation with fixed RoleGraph context.

Role: post-hoc model sensitivity audit, never a whole-program Host deletion.
Writes only new v4 output files; reads but never edits APKs, caches or models.
"""
from __future__ import annotations

import argparse
import csv
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
OUT=ROOT/'outputs/care_host_region_attribution_v4'
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
    if not OUT.is_dir() or (OUT/'pilot_effects.csv').exists():raise RuntimeError('output not fresh')
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
    pairs=sorted(pairs,key=lambda r:(float(r['matching_cost']),r['fn_sha256'],r['tn_sha256']))[:8]
    ids=[];member={}
    for k,row in enumerate(pairs,1):
        for side,key in [('FN','fn_sha256'),('TN','tn_sha256')]:
            sid=row[key]
            if sid in member:raise RuntimeError('pair reuse of APK')
            ids.append(sid);member[sid]=dict(pair=k,side=side,matching_cost=float(row['matching_cost']))
    with np.load(BASE/'predictions/T3/router.npz',allow_pickle=False) as z:
        router={k:z[k] for k in z.files}
    route={str(s):bool(q) for s,q in zip(router['ids'],router['new_route'])}
    s1={str(s):float(q) for s,q in zip(router['ids'],router['s1'])}
    positions={str(s):i for i,s in enumerate(data['ids'])}
    per=read_csv(BASE/'per_apk_predictions.csv')
    frozen={r['sha256']:r for r in per if r['window']=='T3' and r['arm']=='U3_FULL_ROLLING_C1'}
    for sid in ids:
        r=frozen[sid]
        if not route[sid] or s1[sid]>=.5 or int(r['stage1_prediction'])!=0 or int(r['routed'])!=1:
            raise RuntimeError('matched sample violates T3 Stage-I benign routed contract')
        if member[sid]['side']=='FN' and not (int(r['label'])==1 and float(r['expert_probability'])<.5):
            raise RuntimeError('FN pair side no longer unfixed FN')
        if member[sid]['side']=='TN' and not (int(r['label'])==0 and float(r['expert_probability'])<.5):
            raise RuntimeError('TN pair side no longer stable TN')
    ix=np.asarray([positions[s] for s in ids],int)
    raw=raw_bags(data,ix)
    if any(len(b)<2 for b in raw):raise RuntimeError('empty/singleton Host in pilot')
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
        print('HRA_V4_BASELINE',seed,len(ids),flush=True)
    # Same candidates for every seed: mean baseline attention, fixed K=5.
    # Null Host indices are sampled without labels or perturbation outcomes.
    selected=[];host_out=[]
    for j,sid in enumerate(ids):
        n=len(raw[j]);mean_attention=np.mean([a[j] for a in attention],axis=0)
        order=np.lexsort((np.arange(n),-mean_attention))
        top=order[:min(5,n-1)].tolist()
        primary=np.where(raw[j][:,43:74].sum(1)>0,np.argmax(raw[j][:,43:74],axis=1),-1)
        rng=np.random.default_rng(int(hashlib.sha256((sid+'|HRA_V4_NULL_20').encode()).hexdigest()[:16],16))
        pool=[k for k in range(n) if k not in top]
        null=rng.choice(pool,size=min(20,len(pool)),replace=False).tolist()
        for rank,k in enumerate(top,1):
            role_alternatives=[z for z in range(n) if z!=k and int(primary[z])==int(primary[k]) and z not in top]
            role_index=role_alternatives[0] if role_alternatives else None
            selected.append(dict(sid=sid,kind='TOP_ATTENTION',rank=rank,index=int(k),role=int(primary[k])))
            if role_index is not None:selected.append(dict(sid=sid,kind='ROLE_MATCHED',rank=rank,index=int(role_index),role=int(primary[role_index]),for_index=int(k)))
        for rank,k in enumerate(null,1):selected.append(dict(sid=sid,kind='RANDOM_20',rank=rank,index=int(k),role=int(primary[k])))
        host_out.append(dict(sha256=sid,pair=member[sid]['pair'],side=member[sid]['side'],hosts=n,
                             candidates=len(top),random_controls=len(null),top_indices=top,
                             primary_role_counts=dict(Counter(map(int,primary)))))
    save_json('pilot_selection_protocol.json',dict(status='FROZEN_BEFORE_EFFECTS',
        subset='eight lowest matching-cost pairs among pre-existing 22, selected without perturbation outcomes',
        stage1_direction='predicted benign, routed',top_k=5,random_repeats=20,
        candidate_score='mean baseline Host attention across three frozen seeds; SHA/index tie break',
        comparison='matched FN/TN; same-APK random and same-primary-Role controls',
        graph='frozen original RoleGraph and Global24; conditional Host-path ablation, NOT coherent whole-APK Host deletion',
        background='each seed uses its original frozen Shared/External prototypes and recomputes dynamic Internal on reduced bag',
        inference='full original forward re-encodes remaining Hosts and recomputes attention/logit',
        interpretation='ARTIFICIAL_INPUT_PERTURBATION, model sensitivity only; no program or classification causal claim'))
    save_json('pilot_host_counts.json',host_out)
    effects=[];validity=[]
    for item in selected:
        sid=item['sid'];j=ids.index(sid);k=item['index'];reduced=np.delete(raw[j],k,axis=0)
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
            effects.append(dict(sha256=sid,pair=member[sid]['pair'],side=member[sid]['side'],label=int(frozen[sid]['label']),
                category='UNFIXED_FN' if member[sid]['side']=='FN' else 'STABLE_TN',
                kind=item['kind'],rank=item['rank'],host_index=k,primary_role=item['role'],
                control_for_host_index=item.get('for_index',''),seed=seed,host_count=len(raw[j]),removed_fraction=1/len(raw[j]),
                baseline_attention=float(attention[m][j][k]),
                logit_before=before_logit,logit_after=after,delta_logit=delta,
                probability_before=sigmoid(before_logit),probability_after=sigmoid(after),
                flipped=int((sigmoid(before_logit)>=.5)!=(sigmoid(after)>=.5)),
                internal_recomputed=1,other_background_parameters_frozen=1,
                host_encoder_attention_recomputed=1,graph_context_frozen=1,
                validity='ARTIFICIAL_INPUT_PERTURBATION'))
        if len(effects)%150==0:print('HRA_V4_EFFECTS',len(effects),flush=True)
    for model,pre in zip(models,modelhash):
        if stable_hash_model(model)!=pre:raise RuntimeError('frozen weights changed')
    save_csv('pilot_effects.csv',effects)
    bykey={}
    for row in effects:
        key=(row['sha256'],row['kind'],row['rank'],row['host_index'])
        bykey.setdefault(key,[]).append(row)
    summary=[]
    for key,rows in bykey.items():
        if len(rows)!=3:raise RuntimeError('missing seed effect')
        delta=np.array([r['delta_logit'] for r in rows])
        summary.append(dict(sha256=key[0],pair=rows[0]['pair'],side=rows[0]['side'],kind=key[1],rank=key[2],host_index=key[3],
            primary_role=rows[0]['primary_role'],host_count=rows[0]['host_count'],
            attention_mean=float(np.mean([r['baseline_attention'] for r in rows])),
            delta_mean=float(delta.mean()),delta_abs_mean=float(np.abs(delta).mean()),
            seed_direction_consistent=bool(np.all(delta>0) or np.all(delta<0)),
            seed_deltas=[float(x) for x in delta],
            any_seed_flip=bool(any(r['flipped'] for r in rows))))
    save_csv('pilot_ensemble_effects.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in summary])
    apk=[]
    for sid in ids:
        rows=[r for r in summary if r['sha256']==sid]
        top=[r for r in rows if r['kind']=='TOP_ATTENTION']
        null=[r for r in rows if r['kind']=='RANDOM_20']
        threshold=float(np.quantile([r['delta_abs_mean'] for r in null],.95)) if null else None
        candidate=max(top,key=lambda r:r['delta_abs_mean'])
        # Positive delta after removal = removed Host suppressed the malware logit.
        suppressive=max(top,key=lambda r:r['delta_mean'])
        apk.append(dict(sha256=sid,pair=member[sid]['pair'],side=member[sid]['side'],hosts=len(raw[ids.index(sid)]),
            largest_top_host_index=candidate['host_index'],largest_abs_delta=candidate['delta_abs_mean'],
            largest_seed_consistent=candidate['seed_direction_consistent'],
            random_abs_delta_q95=threshold,
            exceeds_random_q95=threshold is not None and candidate['delta_abs_mean']>threshold,
            strongest_suppressive_index=suppressive['host_index'],strongest_suppressive_delta=suppressive['delta_mean'],
            stable_suppressive=bool(suppressive['delta_mean']>0 and suppressive['seed_direction_consistent'])))
    save_csv('pilot_per_apk.csv',apk)
    paired=[]
    for pair in range(1,9):
        fn=next(r for r in apk if r['pair']==pair and r['side']=='FN')
        tn=next(r for r in apk if r['pair']==pair and r['side']=='TN')
        paired.append(dict(pair=pair,fn_max_suppressive=fn['strongest_suppressive_delta'],
            tn_max_suppressive=tn['strongest_suppressive_delta'],
            difference=fn['strongest_suppressive_delta']-tn['strongest_suppressive_delta'],
            fn_exceeds_random=fn['exceeds_random_q95'],tn_exceeds_random=tn['exceeds_random_q95']))
    save_csv('pilot_matched_fn_tn.csv',paired)
    after=[asset(Path(r['path'])) for r in before]
    changed=[dict(before=b,after=a) for b,a in zip(before,after) if b!=a]
    save_json('asset_manifest_after.json',after)
    save_json('baseline_integrity_check.json',dict(status='PASS' if not changed else 'FAIL',asset_count=len(before),changed=changed,scope='listed reused files only; no APK opened'))
    if changed:raise RuntimeError('protected assets changed')
    result=dict(status='CONDITIONAL_HOST_PATH_SENSITIVITY_PILOT_COMPLETE',
        valid_model_interventions=0,artificial_input_perturbations=len(effects),
        matched_pairs=8,apks=len(ids),seeds=3,top_hosts_per_apk=5,random_controls_per_apk=20,
        fn_stable_suppressive=sum(r['stable_suppressive'] for r in apk if r['side']=='FN'),
        tn_stable_suppressive=sum(r['stable_suppressive'] for r in apk if r['side']=='TN'),
        fn_exceeds_random=sum(r['exceeds_random_q95'] for r in apk if r['side']=='FN'),
        tn_exceeds_random=sum(r['exceeds_random_q95'] for r in apk if r['side']=='TN'),
        paired_fn_greater=sum(r['difference']>0 for r in paired),
        median_paired_difference=float(np.median([r['difference'] for r in paired])),
        any_2020_expansion_authorized=False,
        interpretation='Conditional on unchanged RoleGraph/Global24, selected Host-path input rows influence frozen logits. This cannot identify program-causal Host errors or meet the original VALID_MODEL_INTERVENTION criterion.',
        seconds=time.monotonic()-start,formal_integrity='PASS')
    save_json('pilot_summary.json',result)
    print('HRA_V4_PILOT_DONE',json.dumps(result,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    if os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise RuntimeError('PYTHONDONTWRITEBYTECODE must be 1')
    run_pilot(a.device)


if __name__=='__main__':main()
