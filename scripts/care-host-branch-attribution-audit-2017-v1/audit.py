#!/usr/bin/env python3
"""Frozen CARE-Droid Critical/Attended Host attribution audit (2017 OOF).

Inputs are the v3 aligned formal endpoint, SCBE W0 checkpoints, prepared Host
and RoleGraph caches, and canonical frozen Host IDs. Outputs are readout traces,
strictly local counterfactuals, matching/cluster-bootstrap diagnostics, and a
Chinese report. No model is fitted, no APK is opened, and no formal asset is
modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
HERE=ROOT/'scripts/care-host-branch-attribution-audit-2017-v1'
OUT=ROOT/'artifacts/care_host_branch_attribution_audit_2017_v1'
V3=ROOT/'artifacts/care_stage2_error_complementarity_audit_2017_v3'
SCBE=ROOT/'outputs/care_scbe_v1'; MYD=ROOT/'outputs/care_multiyear_development_v1'
FROZEN=ROOT/'artifacts/care_host_method_semantics_feasibility_2017_v2/frozen'
MODEL_SEEDS=(900000,900001,900002); MANIFEST_SEEDS=(20261018,20261019,20261020); SPLITS=('S1','S2','S3','S4','S5')
TOL=1e-6; C_SL=slice(57,81); A_SL=slice(81,105); ROLE_SL=slice(25,57); SUMMARY_SL=slice(105,113)
sys.path[:0]=[str(HERE/'.deps'),str(ROOT/'scripts/care-background-functional-attribution-v1'),str(ROOT/'scripts/care-scbe-v1'),str(ROOT/'scripts/care-multiyear-development-v1')]

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

def wjson(path,value,replace=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace: raise FileExistsError(path)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(jv(value),ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)

def wcsv(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    frame=value if isinstance(value,pd.DataFrame) else pd.DataFrame(list(value));tmp=path.with_suffix(path.suffix+'.tmp');frame.to_csv(tmp,index=False);os.replace(tmp,path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return {'path':str(path),'size':Path(path).stat().st_size,'sha256':h.hexdigest()}

def status(stage,state='RUNNING',**extra):
    wjson(OUT/'logs/run_status.json',{'status':state,'stage':stage,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**extra},replace=True);print('BRANCH_AUDIT_STAGE',stage,state,flush=True)

def sigmoid(x): return float(1/(1+math.exp(-float(np.clip(x,-80,80)))))

def cosine(a,b):
    na=np.linalg.norm(a);nb=np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na>0 and nb>0 else float('nan')

def assets():
    p=[V3/'predictions/aligned_2017_oof.csv',V3/'integrity/replay_check.json',V3/'integrity/validation_results.json',V3/'integrity/asset_sha256_after.json',V3/'reports/final_decision.json',MYD/'data.npz',MYD/'feature_rows.pkl',MYD/'dataset_audit.json',ROOT/'scripts/care-background-functional-attribution-v1/bfa_core.py',ROOT/'scripts/care-di25-graphglobal-path-source-v1/graphglobal_path_models.py',ROOT/'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_models.py',ROOT/'scripts/care-svm-c43-remove-bounded-correction-v1/model_variants.py']
    p += [SCBE/f'models/{s}/{seed}/M0_REPRODUCED.pkl' for s in SPLITS for seed in MODEL_SEEDS]
    return p

def preflight():
    missing=[str(p) for p in assets() if not p.is_file()]
    if missing:raise RuntimeError('missing '+repr(missing))
    if OUT.exists():raise RuntimeError('output exists; never overwrite '+str(OUT))
    v=json.loads((V3/'integrity/validation_results.json').read_text());a=json.loads((V3/'integrity/asset_sha256_after.json').read_text())
    if v.get('status')!='PASS' or a.get('status')!='PASS':raise RuntimeError('v3 integrity not PASS')
    d=pd.read_csv(V3/'predictions/aligned_2017_oof.csv')
    if len(d)!=2397 or d.sha256.nunique()!=799 or set(d.seed)!=set(MANIFEST_SEEDS):raise RuntimeError('v3 inventory mismatch')
    try: import pyarrow
    except Exception as e: raise RuntimeError('isolated parquet dependency unavailable: '+repr(e))
    print('PREFLIGHT_PASS rows=2397 apks=799',flush=True)

def load_data():
    with np.load(MYD/'data.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with (MYD/'feature_rows.pkl').open('rb') as f:data['feature_rows']=pickle.load(f)
    src=np.flatnonzero(data['years']==2017);pos={str(data['ids'][i]):int(i) for i in src}
    roles=json.loads((MYD/'dataset_audit.json').read_text())['role_names']
    frozen={sid:json.loads((FROZEN/f'{sid}.json').read_text()) for sid in pos}
    for sid,v in frozen.items():
        if str(v.get('sha256'))!=sid or len(v['host_ids'])!=int(data['real_host_count'][pos[sid]]):raise RuntimeError('Host identity mismatch '+sid)
    return data,src,pos,roles,frozen

def static_covariates(data,pos):
    off=np.asarray(data['host_offsets'],int);rows=[]
    for sid,i in pos.items():
        raw=np.asarray(data['host_values'][off[i]:off[i+1]],float);n=int(data['real_host_count'][i]);r=raw[:n,:31] if n else np.empty((0,31));rc=(r>0).sum(1) if n else np.asarray([])
        rows.append({'sha256':sid,'real_host_count':n,'no_real_host':int(n==0),'active_role_count':int(np.any(r>0,axis=0).sum()) if n else 0,'multi_role_host_ratio':float(np.mean(rc>1)) if n else 0.})
    return pd.DataFrame(rows)

def training_references(pack):
    import torch
    gt=pack['v4'].graph_tensors(pack['graph_train'],pack['dev']);cs=[];ats=[];model=pack['model']
    with torch.no_grad():
        for j,b in enumerate(pack['bags_train']):
            one={k:v[j] for k,v in gt.items()};region=torch.as_tensor(b,dtype=torch.float32,device=pack['dev']);cache,_=model.select_local_evidence(one,region);cs.append(cache.critical.detach().cpu().numpy());ats.append(cache.attended.detach().cpu().numpy())
    c=np.mean(cs,axis=0).astype(np.float32);a=np.mean(ats,axis=0).astype(np.float32)
    if c.shape!=(24,) or a.shape!=(24,) or not np.isfinite(c).all() or not np.isfinite(a).all():raise RuntimeError('training reference invalid')
    return c,a,len(cs)

def head_logit(model,fused): return model.delta_head(fused).reshape(())

def replay_and_intervene(data,pos,roles,frozen):
    status('2_9_frozen_replay_and_interventions')
    from bfa_core import formal_split
    import torch
    emb=[];sim=[];cf=[];grad=[];topk=[];pool=[];checks=[];layout_seen=[]
    for fi,split in enumerate(SPLITS,1):
      for seed in MODEL_SEEDS:
        pack=formal_split(data,split,seed,device='cuda',train_data=True);model=pack['model'];model.eval()
        if any(m.training for m in model.modules()):raise RuntimeError('model not fully eval')
        for p in model.parameters():p.requires_grad_(False)
        c0,a0,nref=training_references(pack);gt=pack['v4'].graph_tensors(pack['graph_val'],pack['dev'])
        max_raw_diff=0.;max_prob_diff=0.;max_mutation_leak=0.
        for j,idx in enumerate(pack['val']):
            sid=str(data['ids'][idx]);bag=np.asarray(pack['bags_val'][j],np.float32);raw=np.asarray(pack['raw_val'][j],np.float32);one={k:v[j] for k,v in gt.items()};region=torch.as_tensor(bag,dtype=torch.float32,device=pack['dev']);g=torch.as_tensor(pack['global_val'][j],dtype=torch.float32,device=pack['dev']);zero=torch.zeros((),device=pack['dev'])
            with torch.no_grad():
                original,diag,trace=model.forward_with_trace(g,zero,one,region);cache,gg=model.select_local_evidence(one,region);_,_,fused=model.delta_from_selected_evidence(g,zero,cache,gg);raw_head=head_logit(model,fused)
                local=model.region_encoder(region[:,:111]);att=cache.attention;score=cache.score
            if fused.numel()!=113 or tuple(fused[C_SL].shape)!=(24,) or tuple(fused[A_SL].shape)!=(24,):raise RuntimeError('readout layout mismatch')
            if not torch.allclose(fused[24:25],torch.zeros(1,device=pack['dev']),atol=0,rtol=0):raise RuntimeError('Direct zero slot changed')
            max_raw_diff=max(max_raw_diff,float(torch.abs(original-raw_head).cpu()));p= float(torch.sigmoid(original).cpu());expected=float(pack['old']['probability'][j]);max_prob_diff=max(max_prob_diff,abs(p-expected))
            if abs(p-expected)>TOL:raise RuntimeError('W0 probability mismatch '+sid)
            c=fused[C_SL].detach().cpu().numpy();a=fused[A_SL].detach().cpu().numpy();at=att.detach().cpu().numpy();sc=score.detach().cpu().numpy();loc=local.detach().cpu().numpy();rank=np.lexsort((np.arange(len(at)),-at));crit=int(np.argmax(sc));real=int(data['real_host_count'][idx]);hostids=list(map(str,frozen[sid]['host_ids'])) if real else ['EMPTY_PLACEHOLDER']
            if len(hostids)!=len(at):raise RuntimeError('attention/Host stable ID mismatch '+sid)
            entropy=float(-np.sum(at*np.log(np.clip(at,1e-12,1))));nent=entropy/math.log(max(len(at),2));neff=float(1/np.sum(at*at));top3i=rank[:min(3,len(rank))];top5i=rank[:min(5,len(rank))]
            base={'sha256':sid,'outer_fold':fi,'model_seed':seed,'original_w0_probability':expected,'original_logit':float(original.cpu()),'real_host_count':real,'no_real_host':int(real==0),'critical_host_stable_id':hostids[crit],'critical_host_attention':float(at[crit]),'critical_host_local_score':float(sc[crit]),'attention_entropy':entropy,'normalized_attention_entropy':nent,'attention_max':float(at.max()),'attention_top3_sum':float(at[top3i].sum()),'attention_top5_sum':float(at[top5i].sum()),'effective_host_count':neff,'training_reference_n':nref}
            er={**base,**{f'critical_{k:02d}':float(c[k]) for k in range(24)},**{f'attended_{k:02d}':float(a[k]) for k in range(24)}};emb.append(er)
            sim.append({**base,'critical_norm':float(np.linalg.norm(c)),'attended_norm':float(np.linalg.norm(a)),'cosine_similarity':cosine(c,a),'euclidean_distance':float(np.linalg.norm(c-a)),'norm_ratio_critical_over_attended':float(np.linalg.norm(c)/max(np.linalg.norm(a),1e-12)),'mean_abs_dimension_difference':float(np.mean(np.abs(c-a)))})
            tc0=torch.as_tensor(c0,device=pack['dev']);ta0=torch.as_tensor(a0,device=pack['dev']);z=torch.zeros(24,device=pack['dev'])
            def repl(cs,as_):
                x=fused.detach().clone();x[C_SL]=cs;x[A_SL]=as_;return float(head_logit(model,x).detach().cpu())
            l11=float(original.detach().cpu());l10=repl(fused[C_SL],ta0);l01=repl(tc0,fused[A_SL]);l00=repl(tc0,ta0);z10=repl(fused[C_SL],z);z01=repl(z,fused[A_SL]);z00=repl(z,z)
            phi_c=.5*((l10-l00)+(l11-l01));phi_a=.5*((l01-l00)+(l11-l10));inter=l11-l10-l01+l00
            cf.append({**base,'reference':'outer_training_mean_same_fold_seed','L11':l11,'L10':l10,'L01':l01,'L00':l00,'phi_critical':phi_c,'phi_attended':phi_a,'interaction':inter,'abs_interaction':abs(inter),'P11':sigmoid(l11),'P10':sigmoid(l10),'P01':sigmoid(l01),'P00':sigmoid(l00),'zero_L10':z10,'zero_L01':z01,'zero_L00':z00})
            leaf=fused.detach().clone().requires_grad_(True);lg=head_logit(model,leaf);ggra=torch.autograd.grad(lg,leaf)[0].detach().cpu().numpy();gc=ggra[C_SL];ga=ggra[A_SL]
            grad.append({**base,'critical_grad_l1':float(np.abs(gc).sum()),'critical_grad_l2':float(np.linalg.norm(gc)),'critical_grad_max_abs':float(np.abs(gc).max()),'critical_grad_input_dot':float(np.dot(gc,c)),'attended_grad_l1':float(np.abs(ga).sum()),'attended_grad_l2':float(np.linalg.norm(ga)),'attended_grad_max_abs':float(np.abs(ga).max()),'attended_grad_input_dot':float(np.dot(ga,a)),'critical_attended_grad_l2_ratio':float(np.linalg.norm(gc)/max(np.linalg.norm(ga),1e-12))})
            for rr,k in enumerate(rank[:min(5,len(rank))],1):
                x=fused.detach().clone();x[C_SL]=local[k];le=float(head_logit(model,x).detach().cpu());topk.append({**base,'attention_rank':rr,'candidate_host_stable_id':hostids[int(k)],'candidate_attention':float(at[k]),'candidate_local_score':float(sc[k]),'candidate_is_original_critical':int(k==crit),'counterfactual_logit':le,'logit_delta':le-l11,'counterfactual_probability':sigmoid(le),'probability_delta':sigmoid(le)-sigmoid(l11),'prediction_flip':int((le>=0)!=(l11>=0))})
            alternatives={'A0_ORIGINAL':fused[A_SL],'A1_UNIFORM':local.mean(0),'A2_TOP5_RENORMALIZED':torch.sum((att[top5i]/att[top5i].sum())[:,None]*local[top5i],0),'A3_TOP1':local[rank[0]]}
            for name,av in alternatives.items():
                x=fused.detach().clone();x[A_SL]=av;le=float(head_logit(model,x).detach().cpu());pool.append({**base,'pooling_arm':name,'counterfactual_logit':le,'logit_delta':le-l11,'counterfactual_probability':sigmoid(le),'probability_delta':sigmoid(le)-sigmoid(l11),'prediction_flip':int((le>=0)!=(l11>=0))})
            probe=fused.detach().clone();probe[C_SL]+=1.;leak=torch.cat([probe[:57]-fused[:57],probe[81:]-fused[81:]]);max_mutation_leak=max(max_mutation_leak,float(torch.max(torch.abs(leak)).cpu()))
            if not layout_seen:layout_seen=[{'name':'Global24','start':0,'end':24,'dim':24},{'name':'zero1','start':24,'end':25,'dim':1},{'name':'RoleSlot32','start':25,'end':57,'dim':32},{'name':'CriticalHost24','start':57,'end':81,'dim':24},{'name':'AttendedHost24','start':81,'end':105,'dim':24},{'name':'Summary8','start':105,'end':113,'dim':8}]
        checks.append({'split':split,'model_seed':seed,'rows':len(pack['val']),'training_reference_n':nref,'formal_split_replay_diff':pack['replay_diff'],'max_direct_head_logit_diff':max_raw_diff,'max_probability_diff':max_prob_diff,'max_noncritical_mutation_leak':max_mutation_leak})
        del pack,gt,model
        torch.cuda.empty_cache();print('BRANCH_REPLAY',split,seed,flush=True)
    e=pd.DataFrame(emb);s=pd.DataFrame(sim);c=pd.DataFrame(cf);g=pd.DataFrame(grad);t=pd.DataFrame(topk);p=pd.DataFrame(pool)
    if any(x.sha256.nunique()!=799 for x in (e,s,c,g,p)) or any(len(x)!=2397 for x in (e,s,c,g)):raise RuntimeError('trace inventory incomplete')
    current=e.sort_values(['sha256','model_seed']).groupby('sha256').critical_host_stable_id.agg(lambda x:'|'.join(map(str,x)))
    prior=pd.read_csv(V3/'stage2/difficult_fn_fp_diagnostics.csv',dtype={'sha256':str});prior=prior.drop_duplicates('sha256').set_index('sha256');common=current.index.intersection(prior.index)
    mismatch=[sid for sid in common if str(current.loc[sid])!=str(prior.loc[sid,'critical_host_ids'])]
    if mismatch:raise RuntimeError('v3 Critical Host alignment mismatch '+repr(mismatch[:5]))
    alignment={'v3_difficult_rows_checked':len(common),'critical_host_id_mismatches':len(mismatch),'canonical_all_799_checked':True}
    e.to_parquet(OUT/'features/host_branch_embeddings.parquet',index=False,engine='pyarrow');wcsv(OUT/'features/critical_attended_similarity.csv',s);wcsv(OUT/'interventions/branch_2x2_counterfactuals.csv',c);wcsv(OUT/'interventions/branch_gradient_sensitivity.csv',g);wcsv(OUT/'interventions/critical_topk_substitution.csv',t);wcsv(OUT/'interventions/attended_pooling_interventions.csv',p)
    return e,s,c,g,t,p,checks,layout_seen,alignment

def aggregate_model_seeds(sim,cf,grad,topk,pool):
    key=['sha256','outer_fold']
    simcols=['critical_norm','attended_norm','cosine_similarity','euclidean_distance','norm_ratio_critical_over_attended','mean_abs_dimension_difference','attention_entropy','normalized_attention_entropy','attention_max','attention_top3_sum','attention_top5_sum','effective_host_count','real_host_count','no_real_host']
    cfcols=['L11','L10','L01','L00','phi_critical','phi_attended','interaction','abs_interaction','P11','P10','P01','P00','zero_L10','zero_L01','zero_L00']
    grcols=['critical_grad_l1','critical_grad_l2','critical_grad_max_abs','critical_grad_input_dot','attended_grad_l1','attended_grad_l2','attended_grad_max_abs','attended_grad_input_dot','critical_attended_grad_l2_ratio']
    a=sim.groupby(key)[simcols].mean().reset_index()
    a=a.merge(cf.groupby(key)[cfcols].mean().reset_index(),on=key,how='inner')
    a=a.merge(grad.groupby(key)[grcols].mean().reset_index(),on=key,how='inner')
    tk=topk.groupby(key+['attention_rank'])[['logit_delta','probability_delta','prediction_flip']].mean().reset_index();wide=tk.pivot_table(index=key,columns='attention_rank',values=['logit_delta','probability_delta','prediction_flip']).reset_index();wide.columns=['sha256','outer_fold']+[f'top{int(k)}_{v}' for v,k in wide.columns[2:]]
    a=a.merge(wide,on=key,how='left')
    pp=pool.groupby(key+['pooling_arm'])[['logit_delta','probability_delta','prediction_flip']].mean().reset_index();pw=pp.pivot_table(index=key,columns='pooling_arm',values=['logit_delta','probability_delta','prediction_flip']).reset_index();pw.columns=['sha256','outer_fold']+[f'{str(k).lower()}_{v}' for v,k in pw.columns[2:]]
    return a.merge(pw,on=key,how='left')

def describe_groups(groups):
    metrics=['cosine_similarity','euclidean_distance','critical_norm','attended_norm','phi_critical','phi_attended','interaction','abs_interaction','critical_grad_l2','attended_grad_l2','critical_attended_grad_l2_ratio','normalized_attention_entropy','attention_max','effective_host_count','top2_logit_delta','top3_logit_delta','top5_logit_delta','a1_uniform_logit_delta','a2_top5_renormalized_logit_delta','a3_top1_logit_delta']
    rows=[]
    for (grp,label),q0 in groups.groupby(['outcome_group','true_label']):
        q=q0.drop_duplicates('sha256')
        for m in metrics:
            if m not in q:continue
            x=q[m].dropna();rows.append({'outcome_group':grp,'true_label':label,'feature':m,'n_seed_apk':len(q0),'n_independent_apk':q.sha256.nunique(),'mean':x.mean(),'median':x.median(),'q25':x.quantile(.25),'q75':x.quantile(.75),'p90':x.quantile(.9)})
    wcsv(OUT/'groups/class_stratified_results.csv',rows)
    wcsv(OUT/'features/attention_descriptive_stats.csv',[r for r in rows if r['feature'] in ['normalized_attention_entropy','attention_max','effective_host_count','cosine_similarity']])
    return pd.DataFrame(rows)

def smd(a,b):
    den=math.sqrt((float(np.var(a,ddof=1))+float(np.var(b,ddof=1)))/2) if len(a)>1 and len(b)>1 else 0
    return float((np.mean(a)-np.mean(b))/den) if den>0 else 0.

def fixed_matching(groups,treat,control,name):
    cov=['log_host_count','active_role_count','rf_confidence','auto_error_probability','multi_role_host_ratio'];q=groups[groups.outcome_group.isin([treat,control])].copy()
    mu=q[cov].mean();sd=q[cov].std().replace(0,1);z=(q[cov]-mu)/sd
    candidates=[]
    for strata,block in q.groupby(['true_label','outer_fold','seed']):
        ti=block.index[block.outcome_group==treat];ci=block.index[block.outcome_group==control]
        for i in ti:
            for j in ci:
                delta=np.abs(z.loc[i].to_numpy()-z.loc[j].to_numpy());dist=float(np.linalg.norm(delta))
                if float(delta.max())<=.75 and dist<=1.5:candidates.append((dist,i,j,strata))
    used_sha=set();used_rows=set();pairs=[]
    for dist,i,j,strata in sorted(candidates,key=lambda x:(x[0],str(q.loc[x[1],'sha256']),str(q.loc[x[2],'sha256']))):
        si,sj=str(q.loc[i,'sha256']),str(q.loc[j,'sha256'])
        if i in used_rows or j in used_rows or si in used_sha or sj in used_sha:continue
        used_rows|={i,j};used_sha|={si,sj};pairs.append({'comparison':name,'pair_id':len(pairs),'distance':dist,'treat_index':i,'control_index':j,'treat_sha256':si,'control_sha256':sj,'true_label':strata[0],'outer_fold':strata[1],'seed':strata[2]})
    metrics=['phi_critical','phi_attended','interaction','abs_interaction','critical_grad_l2','attended_grad_l2','cosine_similarity','normalized_attention_entropy','top2_logit_delta','top3_logit_delta','top5_logit_delta','a1_uniform_logit_delta','a2_top5_renormalized_logit_delta','a3_top1_logit_delta']
    comps=[]
    for p in pairs:
        a=q.loc[p['treat_index']];b=q.loc[p['control_index']];row={k:v for k,v in p.items() if not k.endswith('_index')}
        for m in metrics:row[m+'_difference']=float(a[m]-b[m])
        comps.append(row)
    balance=[]
    rawt=q[q.outcome_group==treat];rawc=q[q.outcome_group==control]
    mt=q.loc[[p['treat_index'] for p in pairs]] if pairs else q.iloc[:0];mc=q.loc[[p['control_index'] for p in pairs]] if pairs else q.iloc[:0]
    for c in cov:balance.append({'comparison':name,'covariate':c,'raw_smd':smd(rawt[c],rawc[c]),'matched_smd':smd(mt[c],mc[c]) if pairs else np.nan,'raw_treat_n':len(rawt),'raw_control_n':len(rawc),'matched_pairs':len(pairs),'matched_independent_apks':len(used_sha),'caliper_max_abs_z':.75,'caliper_euclidean':1.5,'exact_match':'true_label+outer_fold+seed','status':'PASS' if len(pairs)>=20 else 'MATCHING_UNDERPOWERED'})
    return comps,balance

def bootstrap_pairs(matched,repeats=2000):
    rng=np.random.RandomState(20261021);rows=[]
    for comp,q in matched.groupby('comparison'):
        metrics=[c for c in q if c.endswith('_difference')]
        for m in metrics:
            x=q[m].to_numpy(float);vals=np.asarray([np.mean(rng.choice(x,len(x),replace=True)) for _ in range(repeats)]) if len(x) else np.asarray([np.nan])
            rows.append({'analysis':'matched','comparison':comp,'metric':m,'pairs':len(x),'estimate':float(np.mean(x)) if len(x) else np.nan,'median':float(np.median(x)) if len(x) else np.nan,'ci95_low':float(np.quantile(vals,.025)) if len(x) else np.nan,'ci95_high':float(np.quantile(vals,.975)) if len(x) else np.nan,'bootstrap_unit':'matched pair with globally unique APKs','status':'PASS' if len(x)>=20 else 'MATCHING_UNDERPOWERED'})
    return rows

def bootstrap_raw(groups,repeats=1000):
    rng=np.random.RandomState(20261022);rows=[]
    specs=[('FIXED_vs_BOTH_WRONG','G1_FIXED','G3_BOTH_WRONG'),('BROKEN_vs_BOTH_CORRECT','G2_BROKEN','G4_BOTH_CORRECT')]
    metrics=['phi_critical','phi_attended','interaction','abs_interaction','critical_grad_l2','attended_grad_l2','cosine_similarity','normalized_attention_entropy','top2_logit_delta','top3_logit_delta','top5_logit_delta','a1_uniform_logit_delta','a2_top5_renormalized_logit_delta','a3_top1_logit_delta']
    for name,treat,control in specs:
        q=groups[groups.outcome_group.isin([treat,control])].copy();blocks={sid:b for sid,b in q.groupby('sha256')};classes={lab:q[q.true_label==lab].sha256.unique() for lab in (0,1)};vals={m:[] for m in metrics}
        for _ in range(repeats):
            sample=np.concatenate([rng.choice(v,len(v),replace=True) for v in classes.values() if len(v)]);b=pd.concat([blocks[s] for s in sample],ignore_index=True)
            for m in metrics:
                x=b[b.outcome_group==treat][m];y=b[b.outcome_group==control][m];vals[m].append(float(x.median()-y.median()) if len(x) and len(y) else np.nan)
        for m in metrics:
            x=q[q.outcome_group==treat][m];y=q[q.outcome_group==control][m];v=np.asarray(vals[m]);v=v[np.isfinite(v)]
            rows.append({'analysis':'raw_clustered','comparison':name,'metric':m+'_difference','pairs':np.nan,'estimate':float(x.median()-y.median()),'median':float(np.median(v)) if len(v) else np.nan,'ci95_low':float(np.quantile(v,.025)) if len(v) else np.nan,'ci95_high':float(np.quantile(v,.975)) if len(v) else np.nan,'bootstrap_unit':'independent APK, class-stratified; seed rows retained as correlated','status':'EXPLORATORY_RAW'})
    return rows

def summaries(groups,matched,balance,boots):
    # Branch contribution summaries use independent APK deduplication inside each formal outcome group.
    metrics=['phi_critical','phi_attended','interaction','abs_interaction','critical_grad_l2','attended_grad_l2','cosine_similarity']
    rows=[]
    for grp,q0 in groups.groupby('outcome_group'):
        q=q0.drop_duplicates('sha256')
        for m in metrics:
            x=q[m].dropna();rows.append({'outcome_group':grp,'metric':m,'n_seed_apk':len(q0),'n_independent_apk':q.sha256.nunique(),'mean':x.mean(),'median':x.median(),'q25':x.quantile(.25),'q75':x.quantile(.75)})
    wcsv(OUT/'interventions/branch_contribution_summary.csv',rows)
    wcsv(OUT/'groups/matched_group_comparison.csv',matched);wcsv(OUT/'groups/matching_balance_report.csv',balance);wcsv(OUT/'statistics/bootstrap_intervals.csv',boots)
    stab=[]
    for (seed,fold),q in groups.groupby(['seed','outer_fold']):
        for grp,x in q.groupby('outcome_group'):
            stab.append({'manifest_seed':seed,'outer_fold':fold,'outcome_group':grp,'n':len(x),'n_apk':x.sha256.nunique(),'phi_critical_median':x.phi_critical.median(),'phi_attended_median':x.phi_attended.median(),'interaction_median':x.interaction.median(),'cosine_median':x.cosine_similarity.median(),'uniform_abs_delta_median':x.a1_uniform_logit_delta.abs().median(),'top5_abs_delta_median':x.a2_top5_renormalized_logit_delta.abs().median()})
    wcsv(OUT/'statistics/fold_seed_stability.csv',stab)
    return rows,stab

def decide(groups,matched,balance,boots):
    under=any(r['status']=='MATCHING_UNDERPOWERED' for r in balance)
    b=pd.DataFrame(boots);b=b[b.analysis=='matched'] if len(b) and 'analysis' in b else b;supported=[]
    if not under:
        for comp in b.comparison.unique():
            q=b[b.comparison==comp]
            def sig(name):
                x=q[q.metric==name];return len(x)==1 and (float(x.ci95_low.iloc[0])>0 or float(x.ci95_high.iloc[0])<0)
            if sig('top3_logit_delta_difference') or sig('top5_logit_delta_difference'):supported.append('CRITICAL_SELECTION_SENSITIVITY')
            if sig('a1_uniform_logit_delta_difference') or sig('a2_top5_renormalized_logit_delta_difference'):supported.append('ATTENDED_AGGREGATION_SENSITIVITY')
            if sig('interaction_difference'):supported.append('READOUT_BRANCH_INTERACTION')
    if under:decision='INCONCLUSIVE'
    elif len(set(supported))==1:decision=supported[0]
    elif not supported:decision='NO_STABLE_BRANCH_GAP'
    else:decision='INCONCLUSIVE'
    nextd='ONE_MINIMAL_EXPERIMENT_JUSTIFIED' if decision in ['CRITICAL_SELECTION_SENSITIVITY','ATTENDED_AGGREGATION_SENSITIVITY','READOUT_BRANCH_INTERACTION'] and not under else 'NO_JUSTIFIED_NEXT_EXPERIMENT'
    return decision,nextd,under,sorted(set(supported))

def report(groups,matched,balance,boots,checks,layout):
    decision,nextd,under,supported=decide(groups,matched,balance,boots)
    routed=groups.drop_duplicates(['sha256','seed']);med=lambda c:float(routed[c].median())
    by={g:q.drop_duplicates('sha256') for g,q in routed.groupby('outcome_group')}
    def gm(g,c):return float(by[g][c].median()) if g in by and len(by[g]) else float('nan')
    matched_frame=matched if isinstance(matched,pd.DataFrame) else pd.DataFrame(matched)
    pairs=matched_frame.groupby('comparison').size().to_dict() if not matched_frame.empty else {}
    obj={'status':decision,'next_experiment_decision':nextd,'matching_underpowered':under,'matched_pairs':pairs,'supported_mechanism_flags':supported,'claim_boundary':'frozen-readout sensitivity only; not causal Host truth or deployable performance'}
    wjson(OUT/'reports/final_decision.json',obj)
    evidence=[{'mechanism':x,'supported':x in supported,'matching_underpowered':under,'decision':decision,'next_experiment_decision':nextd,'interpretation':'frozen model sensitivity only; not causal or deployable'} for x in ['CRITICAL_SELECTION_SENSITIVITY','ATTENDED_AGGREGATION_SENSITIVITY','READOUT_BRANCH_INTERACTION','NO_STABLE_BRANCH_GAP']]
    wcsv(OUT/'statistics/mechanism_evidence_summary.csv',evidence)
    text=f'''# CARE-Droid Critical / Attended Host 判别贡献审计（2017 OOF）

## 裁决

- 机制状态：**{decision}**
- 下一实验：**{nextd}**
- 匹配状态：**{'MATCHING_UNDERPOWERED' if under else 'ADEQUATE'}**，配对数：{pairs}

本审计完成 15 个 fold×seed 冻结 W0 回放，只使用 2017 OOF；未训练模型、未扫描 APK、未使用未来年份标签。

## Q1–Q12

**Q1. Critical 与 Attended 是否高度相似？** 全部实际路由 APK 的余弦相似度中位数为 {med('cosine_similarity'):.4f}，欧氏距离中位数为 {med('euclidean_distance'):.4f}。二者来自同一 Host 集合，Attended 通常包含 Critical 的贡献，因此相似性不代表冗余。

**Q2. Attention 集中与分散时是否不同？** 原始归一化熵中位数为 {med('normalized_attention_entropy'):.4f}；相似性和熵的完整关系见 `features/critical_attended_similarity.csv` 与分层结果。不能把集中程度直接解释为质量。

**Q3. Critical 对 logit 的敏感性？** 训练折均值参考下，Critical Shapley 型贡献中位数 {med('phi_critical'):.4f}，梯度 L2 中位数 {med('critical_grad_l2'):.4f}。

**Q4. Attended 对 logit 的敏感性？** Attended 贡献中位数 {med('phi_attended'):.4f}，梯度 L2 中位数 {med('attended_grad_l2'):.4f}。

**Q5. 非线性交互？** 交互项中位数 {med('interaction'):.4f}，绝对交互中位数 {med('abs_interaction'):.4f}。这是相对训练折参考的冻结 head 敏感性。

**Q6. FIXED 与 BOTH_WRONG 是否不同？** Critical 贡献中位数分别为 {gm('G1_FIXED','phi_critical'):.4f}/{gm('G3_BOTH_WRONG','phi_critical'):.4f}；Attended 为 {gm('G1_FIXED','phi_attended'):.4f}/{gm('G3_BOTH_WRONG','phi_attended'):.4f}；交互为 {gm('G1_FIXED','interaction'):.4f}/{gm('G3_BOTH_WRONG','interaction'):.4f}。

**Q7. 匹配后是否仍存在？** 固定规则精确匹配类别、fold、seed，并对 Host 数、Role 数、RF 置信度、路由风险和多 Role 比例设置预声明 caliper。结果为 {'样本不足，不能形成稳定结论' if under else '详见匹配 Bootstrap 区间'}。

**Q8. BROKEN 与 BOTH_CORRECT？** 原始及匹配差异均在 `groups/matched_group_comparison.csv` 与 `statistics/bootstrap_intervals.csv`；未将类别构成差异冒充聚合机制。

**Q9. 替换 Critical 为其他高 Attention Host？** Top-1 至 Top-5 的冻结读出响应、翻转和 stable ID 见 `interventions/critical_topk_substitution.csv`。未按标签挑选最佳候选。

**Q10. 均匀、Top-5、Top-1 聚合影响？** 三种预定义反事实的 logit/概率变化和翻转见 `interventions/attended_pooling_interventions.csv`。这些输入可能分布外，不能当作新策略性能。

**Q11. 更支持哪种问题？** 当前状态为 **{decision}**；支持标志 {supported}。Attention 分散本身没有被当成错误原因。

**Q12. 是否足以开展新聚合训练？** **{nextd}**。{'停止继续增加聚合网络复杂度；先补充独立样本或更强的机制证据。' if nextd=='NO_JUSTIFIED_NEXT_EXPERIMENT' else '只允许设计一个带容量和选择反事实对照的最小实验，本轮不训练。'}

## 解释边界

Attention 不是恶意 Host 真值；最大权重 Host 不一定是恶意代码；更集中不一定更好。Critical 与 Attended 不是独立数据源。冻结读出干预可能分布外，梯度仅表示局部敏感性，预测翻转不能作为部署性能，不能用测试标签选择 Top-K 或聚合策略。
'''
    (OUT/'reports/final_report.md').write_text(text,encoding='utf-8')
    return obj

def integrity_before():
    status('1_asset_integrity');items=[sha(p) for p in assets()];frozen=[sha(p) for p in sorted(FROZEN.glob('*.json'))]
    if len(frozen)!=799:raise RuntimeError('frozen Host files !=799')
    m={'formal_assets':items,'canonical_host_files':frozen,'APK_files_opened':0};wjson(OUT/'config/asset_manifest.json',m);wjson(OUT/'integrity/sha256_before.json',m);return m

def integrity_after(before):
    changed=[]
    for old in before['formal_assets']+before['canonical_host_files']:
        now=sha(Path(old['path']))
        if now!=old:changed.append({'before':old,'after':now})
    r={'status':'PASS' if not changed else 'FAIL','checked':len(before['formal_assets'])+len(before['canonical_host_files']),'changed':changed};wjson(OUT/'integrity/sha256_after.json',r)
    if changed:raise RuntimeError('assets changed')

def run():
    if not OUT.exists():raise RuntimeError('launcher must create owned output')
    config={'scope':'2017 strict OOF only','model':'frozen W0 Direct-C43','training':False,'APK_scan':False,'future_labels':False,'readout_slices':{'global24':[0,24],'zero1':[24,25],'role32':[25,57],'critical24':[57,81],'attended24':[81,105],'summary8':[105,113]},'reference':'same checkpoint outer-training mean per branch','matching':{'exact':['label','fold','manifest_seed'],'covariates':['log1p_host_count','active_roles','RF_confidence','router_risk','multi_role_ratio'],'max_abs_z':.75,'euclidean':1.5,'no_APK_reuse':True},'bootstrap_repeats':2000}
    wjson(OUT/'config/audit_config.json',config);before=integrity_before();data,src,pos,roles,frozen=load_data();e,s,c,g,t,p,checks,layout,alignment=replay_and_intervene(data,pos,roles,frozen)
    formal_all=pd.read_csv(V3/'predictions/aligned_2017_oof.csv');ens=e.groupby('sha256').original_w0_probability.mean();endpoint=formal_all.merge(ens.rename('replayed_stage2'),left_on='sha256',right_index=True,validate='many_to_one');endpoint_stage2_diff=float(np.max(np.abs(endpoint.stage2_probability-endpoint.replayed_stage2)));route_all=endpoint.routed if pd.api.types.is_bool_dtype(endpoint.routed) else endpoint.routed.astype(str).str.lower().eq('true');endpoint_final=np.where(route_all,endpoint.replayed_stage2,endpoint.p_rf);endpoint_final_diff=float(np.max(np.abs(endpoint.final_probability-endpoint_final)))
    maxdiff=max(x['max_probability_diff'] for x in checks);replay_status='PASS' if max(maxdiff,endpoint_stage2_diff,endpoint_final_diff)<=TOL else 'FAIL';wjson(OUT/'integrity/w0_replay_check.json',{'status':replay_status,'tolerance':TOL,'checks':checks,'max_probability_diff':maxdiff,'endpoint_stage2_max_diff':endpoint_stage2_diff,'endpoint_final_max_diff':endpoint_final_diff,'host_alignment':alignment})
    if replay_status!='PASS':raise RuntimeError('endpoint replay mismatch')
    wjson(OUT/'integrity/readout_layout_check.json',{'status':'PASS','actual_layout':layout,'width':113,'head':'delta_head raw logit under DirectStage2','dropout_eval':True,'mutation_only_selected_slice':True,'normalization':'Global/RoleGraph/Host transformations are frozen upstream; final fused tensor is passed directly to MLP head'})
    agg=aggregate_model_seeds(s,c,g,t,p);formal=pd.read_csv(V3/'predictions/aligned_2017_oof.csv');route_mask=formal.routed if pd.api.types.is_bool_dtype(formal.routed) else formal.routed.astype(str).str.lower().eq('true');formal=formal[route_mask&formal.outcome_group.isin(['G1_FIXED','G2_BROKEN','G3_BOTH_WRONG','G4_BOTH_CORRECT'])].copy();cov=static_covariates(data,pos);formal['rf_confidence']=np.abs(formal.p_rf-.5);formal['log_host_count']=np.log1p(formal.sha256.map(cov.set_index('sha256').real_host_count));groups=formal.merge(cov,on='sha256',validate='many_to_one').merge(agg,on=['sha256','outer_fold'],validate='many_to_one')
    wcsv(OUT/'groups/outcome_groups.csv',groups);describe_groups(groups)
    m1,b1=fixed_matching(groups,'G1_FIXED','G3_BOTH_WRONG','FIXED_vs_BOTH_WRONG');m2,b2=fixed_matching(groups,'G2_BROKEN','G4_BOTH_CORRECT','BROKEN_vs_BOTH_CORRECT');matched=pd.DataFrame(m1+m2);balance=b1+b2;boots=bootstrap_raw(groups)+(bootstrap_pairs(matched) if len(matched) else [])
    summaries(groups,matched,balance,boots);status('10_12_matching_statistics_report');decision=report(groups,matched,balance,boots,checks,layout)
    required=['config/audit_config.json','config/baseline_manifest.json','config/asset_manifest.json','integrity/sha256_before.json','integrity/w0_replay_check.json','integrity/readout_layout_check.json','features/host_branch_embeddings.parquet','features/critical_attended_similarity.csv','features/attention_descriptive_stats.csv','interventions/branch_2x2_counterfactuals.csv','interventions/branch_contribution_summary.csv','interventions/branch_gradient_sensitivity.csv','interventions/critical_topk_substitution.csv','interventions/attended_pooling_interventions.csv','groups/outcome_groups.csv','groups/matched_group_comparison.csv','groups/matching_balance_report.csv','groups/class_stratified_results.csv','statistics/bootstrap_intervals.csv','statistics/fold_seed_stability.csv','statistics/mechanism_evidence_summary.csv','reports/final_report.md','reports/final_decision.json']
    baseline={'status':'PASS','source_endpoint':str(V3),'rows':2397,'apks':799,'manifest_seeds':MANIFEST_SEEDS,'W0_model_seeds':MODEL_SEEDS,'folds':SPLITS,'endpoint_BA':json.loads((V3/'config/baseline_manifest.json').read_text())['mean_2017_end_to_end_BA'],'max_W0_probability_diff':maxdiff,'different_historical_endpoint_used':False};wjson(OUT/'config/baseline_manifest.json',baseline)
    missing=[x for x in required if not (OUT/x).is_file()]
    if missing:raise RuntimeError('missing outputs '+repr(missing))
    integrity_after(before);status('complete','COMPLETE',decision=decision['status'],next_experiment_decision=decision['next_experiment_decision'],rows=2397,apks=799)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args();preflight() if a.preflight_only else run()
