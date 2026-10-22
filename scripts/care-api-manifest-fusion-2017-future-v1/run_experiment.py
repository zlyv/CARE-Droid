#!/usr/bin/env python3
"""Run the CARE multi-granularity API and Manifest–Code fusion experiment.

Inputs are frozen 2017-only W0 assets, exact Host/API extraction caches, Manifest135,
and the frozen autonomous router. Outputs include strict 2017 OOF fits, a source freeze,
2018–2021 frozen diagnostics, paired statistics, integrity records, and the final report.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

from common import *
from model import predict_model, train_model


def atomic_npz(path,**values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_suffix(path.suffix+'.tmp.npz');np.savez_compressed(tmp,**values);os.replace(tmp,path)

def required_assets():
    paths=[MYD/'data.npz',MYD/'feature_rows.pkl',MYD/'dataset_audit.json',SOURCE_ENDPOINT,FUTURE_ENDPOINT,SOURCE_LINEAGE,FUTURE_LINEAGE,MAPPING,
           AUTO/'per_apk_predictions.csv.gz',AUTO/'protocol.json',AUTO/'asset_integrity_check.json',FOUR/'asset_integrity_check.json',
           ROOT/'scripts/care-background-functional-attribution-v1/bfa_core.py',ROOT/'scripts/care-lite-stage-v1/common.py',
           ROOT/'scripts/care-local-control-constraint-feasibility-v2/dex_adapter.py',HERE/'common.py',HERE/'api_extract.py',HERE/'model.py',HERE/'run_experiment.py',HERE/'run_tmux.sh']
    paths += [SCBE/'models'/split/str(seed)/'M0_REPRODUCED.pkl' for split in SPLITS+('T1',) for seed in MODEL_SEEDS]
    return paths

def verify_manifest(items):
    changed=[]
    for old in items:
        try:now=digest(old['path'])
        except Exception as exc:changed.append({'path':old['path'],'error':repr(exc)});continue
        if now!=old:changed.append({'before':old,'after':now})
    return {'status':'PASS' if not changed else 'FAIL','checked':len(items),'changed':changed}

def preflight(resume=False):
    if OUT.exists() and not resume:raise RuntimeError('output exists; never overwrite '+str(OUT))
    missing=[str(p) for p in required_assets() if not Path(p).is_file()]
    if missing:raise RuntimeError('missing required assets '+repr(missing))
    if json.loads((AUTO/'asset_integrity_check.json').read_text()).get('status')!='PASS':raise RuntimeError('router integrity')
    if json.loads((FOUR/'asset_integrity_check.json').read_text()).get('status')!='PASS':raise RuntimeError('Manifest asset integrity')
    if not OUT.exists():
        for d in ('config','integrity','features','models','source_2017','future_2018','future_2019','future_2020','future_2021','statistics','reports','logs','smoke'): (OUT/d).mkdir(parents=True,exist_ok=True)
    status('preflight')
    data=load_data();ids=list(map(str,data['ids']));years=np.asarray(data['years'],int)
    if dict(Counter(map(int,years)))!=YEAR_COUNTS:raise RuntimeError('data year counts')
    x=manifest135(ids);views=load_endpoint_views();hosts=verify_host_alignment(data,views);save_views(views)
    before=[digest(p) for p in required_assets()]
    atomic_json(OUT/'config/audit_config.json',{
      'status':'PRE_REGISTERED','baseline':'frozen W0 Direct-C43','stage1':'Manifest135 RF','router':'autonomous Manifest error router P(error)>=0.5','fixed_budget':False,
      'arms':ARMS,'primary_arms':PRIMARY_ARMS,'comparisons':[list(x) for x in COMPARISONS],'api_dimension':API_DIM,'api_vocabulary':'outer-train only; min document frequency 2; max 511 plus OOV',
      'api_family':'sorted role-set plus exact declaring class descriptor','api_endpoint':'normalized full DEX method signature','api_shuffle':'within identical role-set, deterministic cyclic derangement when group size >1',
      'api_stats_control':'six size/count statistics expanded by fixed label-free random features','manifest_control':'separate zero-capacity and row-binding shuffle controls',
      'fusion':'frozen W0 logit plus zero-initialized equal-capacity residual head over frozen W0 113D, API512 and Manifest135','epochs':40,'optimizer':'AdamW lr=1e-3 weight_decay=1e-4',
      'training_years':[2017],'source_protocol':'fixed five-fold OOF, three W0 seeds','future_years':[2018,2019,2020,2021],'future_used_for_selection':False,'multiple_comparisons':'Holm over preregistered 2017 comparisons',
      'support_rule':'positive 2017 OOF BA difference with APK-cluster class-stratified CI lower bound >0 against required semantic, shuffle and capacity controls'},replace=resume)
    atomic_json(OUT/'integrity/sha256_before.json',{'formal_assets':before,'frozen_view_count':len(views),'frozen_host_count':hosts,'APK_files_opened':0},replace=resume)
    atomic_json(OUT/'config/baseline_manifest.json',{'expected_W0_end_to_end_BA':0.9194653299916458,'W0_model_seeds':MODEL_SEEDS,'manifest_seeds':MANIFEST_SEEDS,'source_apks':799,'future_counts':YEAR_COUNTS,'Host_definition':'direct caller of at least one frozen sensitive endpoint','Host_alignment':'exact stable ID and frozen Host order','Manifest135_shape':list(x.shape)},replace=resume)
    atomic_json(OUT/'baseline_manifest.json',json.loads((OUT/'config/baseline_manifest.json').read_text()),replace=resume)
    print('API_MANIFEST_PREFLIGHT_PASS apks=2788 hosts='+str(hosts),flush=True)

def trace_model(pack,bags,graphs,globals_):
    import torch
    gt=pack['v4'].graph_tensors(graphs,pack['dev']);code=[];logits=[];model=pack['model'];model.eval()
    with torch.no_grad():
        for i,b in enumerate(bags):
            one={k:v[i] for k,v in gt.items()};region=torch.as_tensor(b,dtype=torch.float32,device=pack['dev']);g=torch.as_tensor(globals_[i],dtype=torch.float32,device=pack['dev']);zero=torch.zeros((),device=pack['dev']);raw,_,_=model.forward_with_trace(g,zero,one,region);cache,gg=model.select_local_evidence(one,region);_,_,fused=model.delta_from_selected_evidence(g,zero,cache,gg)
            if fused.numel()!=113:raise RuntimeError('W0 fused width')
            check=model.delta_head(fused).reshape(())
            if float(torch.abs(check-raw).cpu())>1e-7:raise RuntimeError('W0 head trace mismatch')
            code.append(fused.detach().cpu().numpy());logits.append(float(raw.cpu()))
    return np.asarray(code,np.float32),np.asarray(logits,np.float32)

def prepare_w0(device='cuda'):
    status('w0_113d_representations')
    from bfa_core import formal_split
    data=load_data();years=np.asarray(data['years'],int)
    for split in SPLITS:
      for seed in MODEL_SEEDS:
        path=OUT/'features/w0_repr'/f'{split}_{seed}.npz'
        if path.is_file():continue
        pack=formal_split(data,split,seed,device=device,train_data=True);ct,lt=trace_model(pack,pack['bags_train'],pack['graph_train'],pack['global_train']);cv,lv=trace_model(pack,pack['bags_val'],pack['graph_val'],pack['global_val'])
        replay=float(np.max(np.abs(1/(1+np.exp(-lv))-np.asarray(pack['old']['probability']))))
        if replay>1e-6:raise RuntimeError('source W0 replay')
        atomic_npz(path,train_ids=np.asarray(data['ids'][pack['train']]),target_ids=np.asarray(data['ids'][pack['val']]),code_train=ct,code_target=cv,logit_train=lt,logit_target=lv,replay=np.asarray([replay]))
        print('W0_REPR_DONE',split,seed,flush=True);del pack
        if device=='cuda':__import__('torch').cuda.empty_cache()
    # T1 source and all future years under each full-2017 W0 checkpoint.
    from train import global24
    from train_stage2 import raw_bags,transform_saved
    for seed in MODEL_SEEDS:
        train_path=OUT/'features/w0_repr'/f'T1_{seed}_train.npz'
        pack=formal_split(data,'T1',seed,device=device,train_data=True)
        if not train_path.is_file():
            c,l=trace_model(pack,pack['bags_train'],pack['graph_train'],pack['global_train']);atomic_npz(train_path,ids=np.asarray(data['ids'][pack['train']]),code=c,logit=l)
        train=np.flatnonzero(years==2017)
        for year in YEARS[1:]:
            path=OUT/'features/w0_repr'/f'T1_{seed}_{year}.npz'
            if path.is_file():continue
            target=np.flatnonzero(years==year);_,gg=global24(pack['v4'],data,train,target);raw=raw_bags(data,target);bags=transform_saved(pack['v4'],raw,pack['old']['background'],pack['args']);graphs=pack['v4'].transform_role_graphs(data['s7'][target],pack['old']['graph_scaler']);c,l=trace_model(pack,bags,graphs,gg);atomic_npz(path,ids=np.asarray(data['ids'][target]),code=c,logit=l);print('W0_FUTURE_REPR_DONE',seed,year,flush=True)
        del pack
        if device=='cuda':__import__('torch').cuda.empty_cache()

def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def train_source(device='cuda'):
    status('source_2017_training')
    import torch
    dev=torch.device(device);data=load_data();ids=np.asarray(data['ids']).astype(str);idx={s:i for i,s in enumerate(ids)};manifest=manifest135(ids);records=load_api_records(ids);y=np.asarray(data['y'],int)
    cover=[]
    for fi,split in enumerate(SPLITS,1):
      for seed in MODEL_SEEDS:
        rep=npz(OUT/'features/w0_repr'/f'{split}_{seed}.npz');tr=np.asarray([idx[str(s)] for s in rep['train_ids']],int);te=np.asarray([idx[str(s)] for s in rep['target_ids']],int)
        for arm in TRAINED_ARMS:
            path=OUT/'models'/split/str(seed)/(arm+'.pkl')
            if path.is_file():continue
            atr,ate,mtr,mte,meta=feature_blocks(arm,records,ids[tr],ids[te],manifest[tr],manifest[te],seed+fi*1000)
            pred,_,state,cost=train_model(torch,rep['code_train'],atr,mtr,rep['logit_train'],y[tr],rep['code_target'],ate,mte,rep['logit_target'],seed+fi*1000,dev)
            atomic_pickle(path,{'arm':arm,'split':split,'seed':seed,'train_ids':list(ids[tr]),'target_ids':list(ids[te]),'probability':pred,'state_dict':state,'feature_contract':meta,'cost':cost,'train_years':[2017],'future_labels_used':False})
            if 'target_coverage' in meta:cover.append({'phase':'source_oof','split':split,'seed':seed,'arm':arm,**meta['target_coverage']})
            print('SOURCE_ARM_DONE',split,seed,arm,flush=True)
    atomic_csv(OUT/'features/feature_coverage_source.csv',cover,replace=True)

def source_probabilities(data):
    ids=np.asarray(data['ids']).astype(str);source=np.flatnonzero(np.asarray(data['years'],int)==2017);pos={ids[i]:j for j,i in enumerate(source)};values={a:np.full((len(MODEL_SEEDS),len(source)),np.nan) for a in ARMS}
    for split in SPLITS:
      for si,seed in enumerate(MODEL_SEEDS):
        rep=npz(OUT/'features/w0_repr'/f'{split}_{seed}.npz');loc=np.asarray([pos[str(s)] for s in rep['target_ids']],int);values['B0_W0'][si,loc]=1/(1+np.exp(-rep['logit_target']))
        for arm in TRAINED_ARMS:
            obj=read_pickle(OUT/'models'/split/str(seed)/(arm+'.pkl'));values[arm][si,loc]=obj['probability']
    for a,v in values.items():
        if not np.isfinite(v).all():raise RuntimeError('incomplete source '+a)
    return source,values

def bootstrap_ba(y,candidate,reference,repeats=1500,seed=20261022):
    y=np.asarray(y,int);candidate=np.asarray(candidate,float);reference=np.asarray(reference,float);rng=np.random.RandomState(seed);classes=[np.flatnonzero(y==v) for v in (0,1)];vals=[]
    for _ in range(repeats):
        ix=np.concatenate([rng.choice(v,len(v),replace=True) for v in classes]);vals.append(np.mean([metric(y[ix],candidate[s,ix])['BA']-metric(y[ix],reference[s,ix])['BA'] for s in range(candidate.shape[0])]))
    vals=np.asarray(vals);p=min(1.,2*min(float(np.mean(vals<=0)),float(np.mean(vals>=0))))
    return {'bootstrap_mean':float(vals.mean()),'ci95_low':float(np.quantile(vals,.025)),'ci95_high':float(np.quantile(vals,.975)),'p_two_sided':p,'repeats':repeats,'cluster_unit':'independent APK; class stratified'}

def holm(rows):
    order=sorted(range(len(rows)),key=lambda i:rows[i]['p_two_sided']);running=0.
    for rank,i in enumerate(order):
        adj=min(1.,(len(rows)-rank)*rows[i]['p_two_sided']);running=max(running,adj);rows[i]['p_holm']=running
    return rows

def evaluate_source():
    status('source_2017_oof_evaluation')
    data=load_data();source,probs=source_probabilities(data);ids=list(map(str,data['ids'][source]));y=np.asarray(data['y'][source],int);routes=pd.read_csv(AUTO/'per_apk_predictions.csv.gz');routes=routes[(routes.year==2017)&(routes.strategy=='AUTO_ERROR_LR')]
    performance=[];per=[];final_by={}
    for arm in ARMS:
        expert=np.mean(probs[arm],axis=0);finals=[]
        for ms in MANIFEST_SEEDS:
            q=routes[routes.manifest_seed==ms].set_index('sha256').loc[ids];stage1=q.stage1_probability.to_numpy(float);route=q.routed.to_numpy(bool);m,final=cascade(y,stage1,expert,route);performance.append({'year':2017,'arm':arm,'manifest_seed':ms,**m});finals.append(final)
            for i,sid in enumerate(ids):per.append({'year':2017,'arm':arm,'manifest_seed':ms,'sha256':sid,'label':int(y[i]),'routed':int(route[i]),'stage1_probability':stage1[i],'expert_probability':expert[i],'final_probability':final[i]})
        final_by[arm]=np.asarray(finals)
    means=[]
    for arm in ARMS:
        rows=[r for r in performance if r['arm']==arm];numeric=[k for k,v in rows[0].items() if k not in ('year','arm','manifest_seed') and isinstance(v,(int,float,np.integer,np.floating))];means.append({'year':2017,'arm':arm,**{k:float(np.nanmean([r[k] for r in rows])) for k in numeric}})
    comparisons=[]
    for i,(cand,ref,name) in enumerate(COMPARISONS):
        ci=bootstrap_ba(y,final_by[cand],final_by[ref],seed=20261022+i);cm=next(x for x in means if x['arm']==cand);rm=next(x for x in means if x['arm']==ref);comparisons.append({'comparison':name,'candidate':cand,'reference':ref,'BA_difference':cm['BA']-rm['BA'],'Net_Fixed_difference':cm['Net_Fixed']-rm['Net_Fixed'],**ci})
    holm(comparisons);lookup={r['comparison']:r for r in comparisons}
    pos=lambda n:lookup[n]['BA_difference']>0 and lookup[n]['ci95_low']>0 and lookup[n]['p_holm']<.05 and lookup[n]['Net_Fixed_difference']>0
    family=all(pos(n) for n in ('A1_vs_B0','A1_vs_C0','A1_vs_API_STATS'))
    endpoint=all(pos(n) for n in ('A2_vs_B0','A2_vs_A1','A2_vs_A3','A2_vs_API_STATS','A2_vs_C0'))
    manifest=all(pos(n) for n in ('M1_vs_B0','M1_vs_C0','M1_vs_M2_CAPACITY','M1_vs_M2_SHUFFLE'))
    joint1=all(pos(n) for n in ('AM1_vs_A1','AM1_vs_M1'));joint2=all(pos(n) for n in ('AM2_vs_A2','AM2_vs_M1','AM2_vs_AM1'));joint=joint1 or joint2
    directions={'api_family':family,'api_endpoint':endpoint,'manifest':manifest,'joint':joint}
    supported=[]
    if family:supported.append('API_FAMILY_SUPPORTED')
    if endpoint:supported.append('API_ENDPOINT_SUPPORTED')
    if manifest:supported.append('MANIFEST_FUSION_SUPPORTED')
    if joint:supported.append('JOINT_FUSION_SUPPORTED')
    overall=('JOINT_FUSION_SUPPORTED' if joint else 'API_ENDPOINT_SUPPORTED' if endpoint else 'API_FAMILY_SUPPORTED' if family else 'MANIFEST_FUSION_SUPPORTED' if manifest else 'NO_SUPPORTED_GAIN_2017_OOF')
    b0=next(x for x in means if x['arm']=='B0_W0')
    if abs(b0['BA']-0.9194653299916458)>1e-8:raise RuntimeError(f'BLOCKED_BASELINE_REPLAY BA={b0["BA"]}')
    atomic_csv(OUT/'source_2017/source_per_manifest_seed.csv',performance,replace=True);atomic_csv(OUT/'source_2017/source_arm_comparison.csv',means,replace=True);atomic_csv(OUT/'source_2017/source_oof_predictions.csv',per,replace=True);atomic_csv(OUT/'statistics/paired_bootstrap.csv',comparisons,replace=True)
    atomic_csv(OUT/'source_arm_comparison.csv',means,replace=True);atomic_csv(OUT/'source_oof_predictions.csv',per,replace=True);atomic_csv(OUT/'paired_bootstrap.csv',comparisons,replace=True)
    decision={'status':overall,'directions':directions,'supported_statuses':supported,'baseline_W0_BA':b0['BA'],'selection_years':[2017],'future_used_for_selection':False,'comparisons':comparisons}
    atomic_json(OUT/'source_2017/source_decision.json',decision,replace=True);print('SOURCE_OOF_COMPLETE',overall,flush=True)

def train_full(device='cuda'):
    status('full_2017_freeze_training_and_label_free_future_inference')
    import torch
    dev=torch.device(device);data=load_data();ids=np.asarray(data['ids']).astype(str);years=np.asarray(data['years'],int);source=np.flatnonzero(years==2017);manifest=manifest135(ids);records=load_api_records(ids);y=np.asarray(data['y'][source],int);coverage=[]
    for seed in MODEL_SEEDS:
        trrep=npz(OUT/'features/w0_repr'/f'T1_{seed}_train.npz');tr=np.asarray([np.flatnonzero(ids==str(s))[0] for s in trrep['ids']],int)
        if not np.array_equal(tr,source):raise RuntimeError('T1 train order')
        for arm in TRAINED_ARMS:
            path=OUT/'models/T1'/str(seed)/(arm+'.pkl')
            if path.is_file():continue
            atr,_,mtr,_,meta=feature_blocks(arm,records,ids[tr],ids[tr],manifest[tr],manifest[tr],seed+6000)
            _,_,state,cost=train_model(torch,trrep['code'],atr,mtr,trrep['logit'],y,trrep['code'],atr,mtr,trrep['logit'],seed+6000,dev)
            future={};future_meta={}
            for year in YEARS[1:]:
                rep=npz(OUT/'features/w0_repr'/f'T1_{seed}_{year}.npz');te=np.asarray([np.flatnonzero(ids==str(s))[0] for s in rep['ids']],int);_,ate,_,mte,m=feature_blocks(arm,records,ids[tr],ids[te],manifest[tr],manifest[te],seed+6000);future[year]=predict_model(torch,state,rep['code'],ate,mte,rep['logit'],dev);future_meta[year]=m
                if 'target_coverage' in m:coverage.append({'phase':'future','year':year,'seed':seed,'arm':arm,**m['target_coverage']})
            atomic_pickle(path,{'arm':arm,'split':'T1','seed':seed,'train_ids':list(ids[tr]),'state_dict':state,'feature_contract':meta,'cost':cost,'future_probability':future,'future_feature_contract':future_meta,'train_years':[2017],'future_labels_used':False})
            print('T1_ARM_DONE',seed,arm,flush=True)
    atomic_csv(OUT/'features/feature_oov_summary.csv',coverage,replace=True)
    atomic_csv(OUT/'feature_oov_summary.csv',coverage,replace=True)
    source_cov=pd.read_csv(OUT/'features/feature_coverage_source.csv') if (OUT/'features/feature_coverage_source.csv').is_file() else pd.DataFrame();combined_cov=pd.concat([source_cov,pd.DataFrame(coverage)],ignore_index=True,sort=False);atomic_csv(OUT/'feature_coverage.csv',combined_cov,replace=True)
    model_files=sorted((OUT/'models/T1').glob('*/*.pkl'));freeze={'status':'FROZEN_BEFORE_FUTURE_LABEL_EVALUATION','source_decision':json.loads((OUT/'source_2017/source_decision.json').read_text()),'model_checkpoints':[digest(p) for p in model_files],'arms':ARMS,'training_years':[2017],'future_years':[2018,2019,2020,2021],'future_labels_used_for_training_or_selection':False,'router_frozen':True,'stage1_frozen':True}
    atomic_json(OUT/'source_freeze_manifest.json',freeze,replace=True)

def evaluate_future():
    status('future_2018_2021_frozen_evaluation')
    if json.loads((OUT/'source_freeze_manifest.json').read_text()).get('status')!='FROZEN_BEFORE_FUTURE_LABEL_EVALUATION':raise RuntimeError('source freeze missing')
    data=load_data();ids=np.asarray(data['ids']).astype(str);years=np.asarray(data['years'],int);labels=np.asarray(data['y'],int);routes=pd.read_csv(AUTO/'per_apk_predictions.csv.gz');routes=routes[routes.strategy=='AUTO_ERROR_LR'];performance=[];per=[];means=[]
    for year in YEARS[1:]:
        target=np.flatnonzero(years==year);sids=list(ids[target]);y=labels[target];probs={a:[] for a in ARMS}
        for seed in MODEL_SEEDS:
            rep=npz(OUT/'features/w0_repr'/f'T1_{seed}_{year}.npz');probs['B0_W0'].append(1/(1+np.exp(-rep['logit'])))
            for arm in TRAINED_ARMS:probs[arm].append(np.asarray(read_pickle(OUT/'models/T1'/str(seed)/(arm+'.pkl'))['future_probability'][year],float))
        for arm in ARMS:
            expert=np.mean(np.asarray(probs[arm]),axis=0)
            for ms in MANIFEST_SEEDS:
                q=routes[(routes.year==year)&(routes.manifest_seed==ms)].set_index('sha256').loc[sids];stage1=q.stage1_probability.to_numpy(float);route=q.routed.to_numpy(bool);m,final=cascade(y,stage1,expert,route);performance.append({'year':year,'arm':arm,'manifest_seed':ms,**m})
                for i,sid in enumerate(sids):per.append({'year':year,'arm':arm,'manifest_seed':ms,'sha256':sid,'label':int(y[i]),'routed':int(route[i]),'stage1_probability':stage1[i],'expert_probability':expert[i],'final_probability':final[i]})
        atomic_csv(OUT/f'future_{year}/predictions.csv',[r for r in per if r['year']==year],replace=True)
    for year in YEARS[1:]:
      for arm in ARMS:
        rows=[r for r in performance if r['year']==year and r['arm']==arm];numeric=[k for k,v in rows[0].items() if k not in ('year','arm','manifest_seed') and isinstance(v,(int,float,np.integer,np.floating))];means.append({'year':year,'arm':arm,**{k:float(np.nanmean([r[k] for r in rows])) for k in numeric}})
    future_avg={a:float(np.mean([r['BA'] for r in means if r['arm']==a])) for a in ARMS}
    atomic_csv(OUT/'future_year_comparison.csv',means,replace=True);atomic_csv(OUT/'fixed_broken_analysis.csv',[{k:r[k] for k in ('year','arm','manifest_seed','Fixed','Broken','Net_Fixed','routed','route_rate')} for r in performance],replace=True)
    source=json.loads((OUT/'source_2017/source_decision.json').read_text());final={'status':source['status'],'direction_decisions':source['directions'],'supported_statuses':source['supported_statuses'],'source_2017_primary':True,'future_development_only':True,'future_average_BA':future_avg,'future_used_for_selection':False}
    atomic_json(OUT/'reports/final_decision.json',final,replace=True)
    atomic_json(OUT/'final_decision.json',final,replace=True)
    src=pd.read_csv(OUT/'source_2017/source_arm_comparison.csv');lines=['# CARE-Droid 多粒度 API 与 Manifest–Code 联合融合实验','',f"最终源域裁决：**{source['status']}**。",'',f"方向裁决：`{source['directions']}`。",'','2017 固定五折 OOF 是唯一裁决依据；2018—2021 仅为源域冻结后的时间泛化诊断。Stage-I、P(error)≥0.5 自主路由、W0 主干、背景和 Prototype Bank 均未修改。','', '| 年份 | Arm | BA | AUC | F1 | FPR | FNR | Fixed | Broken | Net |','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _,r in src.iterrows():lines.append(f"| 2017 | {r.arm} | {r.BA:.4f} | {r.AUC:.4f} | {r.F1:.4f} | {r.FPR:.4f} | {r.FNR:.4f} | {r.Fixed:.1f} | {r.Broken:.1f} | {r.Net_Fixed:.1f} |")
    for r in means:lines.append(f"| {int(r['year'])} | {r['arm']} | {r['BA']:.4f} | {r['AUC']:.4f} | {r['F1']:.4f} | {r['FPR']:.4f} | {r['FNR']:.4f} | {r['Fixed']:.1f} | {r['Broken']:.1f} | {r['Net_Fixed']:.1f} |")
    lines+=['','## 四年等权平均 BA','']+[f'- {a}: {v:.4f}' for a,v in future_avg.items()]+['','具体 API 来源于冻结 Host 内真实 invoke 签名；未从 SelfRole31 反推。多角色 Endpoint 仍按一个调用事件计数。所有词表与归一化只由对应 2017 训练分区构造。','']
    (OUT/'reports/final_report.md').write_text('\n'.join(lines),encoding='utf-8')

def final_integrity():
    status('final_integrity')
    before=json.loads((OUT/'integrity/sha256_before.json').read_text())['formal_assets'];check=verify_manifest(before)
    required=['baseline_manifest.json','source_freeze_manifest.json','feature_coverage.csv','feature_oov_summary.csv','source_oof_predictions.csv','source_arm_comparison.csv','paired_bootstrap.csv','future_year_comparison.csv','fixed_broken_analysis.csv','final_report.md','final_decision.json','reports/final_report.md','reports/final_decision.json']
    missing=[x for x in required if not (OUT/x).is_file()];check['missing_outputs']=missing
    check['status']='PASS' if check['status']=='PASS' and not missing else 'FAIL';atomic_json(OUT/'asset_integrity.json',check,replace=True)
    if check['status']!='PASS':raise RuntimeError('final integrity failed')
    d=json.loads((OUT/'reports/final_decision.json').read_text());status('complete','COMPLETE',decision=d['status'],apks=2788,source=799,future=1989)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('preflight','w0','train-source','evaluate-source','train-full','evaluate-future','integrity'));p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cuda');a=p.parse_args();os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if a.action=='preflight':preflight(a.resume)
    elif a.action=='w0':prepare_w0(a.device)
    elif a.action=='train-source':train_source(a.device)
    elif a.action=='evaluate-source':evaluate_source()
    elif a.action=='train-full':train_full(a.device)
    elif a.action=='evaluate-future':evaluate_future()
    else:final_integrity()
if __name__=='__main__':main()


