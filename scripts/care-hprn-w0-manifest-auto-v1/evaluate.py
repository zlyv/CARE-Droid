#!/usr/bin/env python3
"""Evaluate W0 and HPRN under the frozen Manifest135 autonomous router."""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from hprn_common import (ARMS, AUTO, MANIFEST_SEEDS, OUT, RELATION_ARMS,
                         SCBE, SEEDS, SOURCE_SPLITS, YEARS, atomic_csv,
                         atomic_json, build_relation_bags, load_auto_routes,
                         read_pickle, verify_assets)
from model import make_model


def metric(y, probability):
    y=np.asarray(y,int); p=np.asarray(probability,float); z=p>=.5
    tn=int(np.sum((y==0)&(~z))); fp=int(np.sum((y==0)&z)); fn=int(np.sum((y==1)&(~z))); tp=int(np.sum((y==1)&z))
    tnr=tn/max(tn+fp,1); tpr=tp/max(tp+fn,1)
    try:
        from sklearn.metrics import roc_auc_score
        auc=float(roc_auc_score(y,p))
    except ValueError: auc=float('nan')
    precision=tp/max(tp+fp,1); recall=tpr; f1=2*precision*recall/max(precision+recall,1e-12)
    return {'BA':.5*(tnr+tpr),'AUC':auc,'F1':f1,'FPR':1-tnr,'FNR':1-tpr,
            'TN':tn,'FP':fp,'FN':fn,'TP':tp,'Accuracy':float(np.mean(z==y))}


def cascade(y,stage1,expert,route):
    y=np.asarray(y,int); stage1=np.asarray(stage1,float); expert=np.asarray(expert,float); route=np.asarray(route,bool)
    final=np.where(route,expert,stage1); base_correct=(stage1>=.5)==y; final_correct=(final>=.5)==y
    err=~base_correct; fixed=int(np.sum(err&final_correct)); broken=int(np.sum(base_correct&(~final_correct)))
    result=metric(y,final); result.update({'routed':int(route.sum()),'route_rate':float(route.mean()),
        'stage1_errors':int(err.sum()),'routed_stage1_errors':int(np.sum(route&err)),
        'stage1_error_coverage':float(np.sum(route&err)/max(err.sum(),1)),
        'route_error_precision':float(np.sum(route&err)/max(route.sum(),1)),
        'Fixed':fixed,'Broken':broken,'Net_Fixed':fixed-broken,
        'expert_all_BA':metric(y,expert)['BA']})
    if np.any(route) and len(np.unique(y[route]))==2: result['expert_routed_BA']=metric(y[route],expert[route])['BA']
    else: result['expert_routed_BA']=float('nan')
    return result,final


def npz(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}


def source_probabilities(data):
    source=np.flatnonzero(np.asarray(data['years'],int)==2017); pos={int(v):i for i,v in enumerate(source)}
    values={arm:np.full((len(SEEDS),len(source)),np.nan) for arm in ARMS}
    for split in SOURCE_SPLITS:
        for si,seed in enumerate(SEEDS):
            base=read_pickle(SCBE/'models'/split/str(seed)/'M0_REPRODUCED.pkl')
            ids=list(map(str,base['ids'])); index={str(data['ids'][i]):int(i) for i in source}
            loc=np.asarray([pos[index[sid]] for sid in ids],int); values[ARMS[0]][si,loc]=np.asarray(base['probability'],float)
            for arm in RELATION_ARMS:
                obj=read_pickle(OUT/'models'/split/str(seed)/(arm+'.pkl'))
                if obj['ids']!=ids: raise RuntimeError('source SHA mismatch')
                values[arm][si,loc]=np.asarray(obj['probability'],float)
    for arm,v in values.items():
        if not np.isfinite(v).all(): raise RuntimeError('source OOF incomplete '+arm)
    return source,values


def future_probabilities(data,device):
    from bfa_core import load_data_contract
    from myd_common import runtime
    from train import global24
    from train_stage2 import raw_bags,transform_saved
    v4,_,args=runtime(); args.device=device; args.keep_weight=args.fix_weight=1.; args.keep_consistency=0.
    dev=v4.resolve_device(device); years=np.asarray(data['years'],int); train=np.flatnonzero(years==2017)
    fixed=npz(OUT.parent/'care_multiyear_development_v1/predictions/C0R_2017/expert.npz')
    fixed_pos={str(s):i for i,s in enumerate(fixed['ids'])}; result={}; replay=[]; runtime=[]
    for year in YEARS[1:]:
        target=np.flatnonzero(years==year); ids=list(map(str,data['ids'][target])); loc=np.asarray([fixed_pos[s] for s in ids],int)
        probs={ARMS[0]:np.asarray(fixed['seed_probability'][:,loc],float)}
        _,global_target=global24(v4,data,train,target); raw_target=raw_bags(data,target)
        for si,seed in enumerate(SEEDS):
            base=read_pickle(SCBE/'models/T1'/str(seed)/'M0_REPRODUCED.pkl')
            bags=transform_saved(v4,raw_target,base['background'],args)
            graph=v4.transform_role_graphs(data['s7'][target],base['graph_scaler'])
            if year==2018:
                diff=float(np.max(np.abs(probs[ARMS[0]][si]-np.asarray(base['probability'],float))))
                replay.append({'year':year,'arm':ARMS[0],'seed':seed,'max_probability_difference':diff})
                if diff>1e-6: raise RuntimeError(f'W0 T1 replay mismatch {seed}: {diff}')
            for arm in RELATION_ARMS:
                obj=read_pickle(OUT/'models/T1'/str(seed)/(arm+'.pkl'))
                relation,audit=build_relation_bags(bags,raw_target,ids,base['background'],arm,seed)
                model=make_model(v4,args,arm,base['state_dict']).to(dev); model.load_state_dict(obj['state_dict']); model.eval()
                started=time.perf_counter(); p=v4.predict_model(model,global_target,np.full(len(target),.5),graph,relation,dev)[0]
                runtime.append({'year':year,'arm':arm,'seed':seed,'inference_seconds':time.perf_counter()-started,
                                'hosts':audit['hosts']})
                probs.setdefault(arm,[]).append(np.asarray(p,float))
                if year==2018:
                    diff=float(np.max(np.abs(np.asarray(p)-np.asarray(obj['probability'],float))))
                    replay.append({'year':year,'arm':arm,'seed':seed,'max_probability_difference':diff})
                    if diff>1e-6: raise RuntimeError(f'HPRN T1 replay mismatch {seed}/{arm}: {diff}')
                del model
                if dev.type=='cuda': v4.torch.cuda.empty_cache()
        for arm in RELATION_ARMS: probs[arm]=np.asarray(probs[arm],float)
        result[year]=(target,probs)
    return result,replay,runtime


def bootstrap_ba(y,candidate,reference,repeats=1000,seed=20261021):
    """Resample independent APKs; all Manifest-seed repeats for an APK remain together."""
    y=np.asarray(y,int); candidate=np.asarray(candidate,float); reference=np.asarray(reference,float)
    if candidate.shape!=reference.shape or candidate.shape[1]!=len(y): raise RuntimeError('bootstrap shape')
    rng=np.random.RandomState(seed); by=[np.flatnonzero(y==label) for label in (0,1)]; values=[]
    for _ in range(repeats):
        index=np.concatenate([rng.choice(v,size=len(v),replace=True) for v in by])
        delta=[]
        for s in range(candidate.shape[0]): delta.append(metric(y[index],candidate[s,index])['BA']-metric(y[index],reference[s,index])['BA'])
        values.append(float(np.mean(delta)))
    return {'mean':float(np.mean(values)),'ci95_low':float(np.quantile(values,.025)),
            'ci95_high':float(np.quantile(values,.975)),'repeats':repeats,'cluster_unit':'APK'}


def run(device):
    from bfa_core import load_data_contract
    if verify_assets()['status']!='PASS': raise RuntimeError('assets changed before evaluation')
    data,_=load_data_contract(); routes=load_auto_routes(); source,source_probs=source_probabilities(data)
    future,replay,runtime=future_probabilities(data,device); years=np.asarray(data['years'],int)
    distributions={2017:(source,source_probs)}; distributions.update(future)
    performance=[]; per=[]; final_by={}
    for year in YEARS:
        target,probabilities=distributions[year]; ids=list(map(str,data['ids'][target])); y=np.asarray(data['y'][target],int)
        for arm in ARMS:
            expert=np.mean(np.asarray(probabilities[arm],float),axis=0); final_seeds=[]
            for manifest_seed in MANIFEST_SEEDS:
                q=routes[(routes.year==year)&(routes.manifest_seed==manifest_seed)].set_index('sha256').loc[ids]
                if not np.array_equal(np.asarray(q.label,int),y): raise RuntimeError('route label alignment')
                stage1=np.asarray(q.stage1_probability,float); route=np.asarray(q.routed,bool)
                result,final=cascade(y,stage1,expert,route); final_seeds.append(final)
                performance.append({'year':year,'arm':arm,'manifest_seed':manifest_seed,'n':len(y),**result})
                for i,sid in enumerate(ids): per.append({'year':year,'arm':arm,'manifest_seed':manifest_seed,'sha256':sid,
                    'label':int(y[i]),'auto_error_probability':float(q.auto_error_probability.iloc[i]),
                    'routed':int(route[i]),'stage1_probability':float(stage1[i]),'expert_probability':float(expert[i]),
                    'final_probability':float(final[i]),'stage1_correct':int((stage1[i]>=.5)==y[i]),
                    'final_correct':int((final[i]>=.5)==y[i])})
            final_by[(year,arm)]=np.asarray(final_seeds,float)
    means=[]
    for year in YEARS:
        for arm in ARMS:
            rows=[r for r in performance if r['year']==year and r['arm']==arm]
            numeric=[k for k,v in rows[0].items() if k not in ('year','arm','manifest_seed') and isinstance(v,(int,float,np.integer,np.floating))]
            means.append({'year':year,'arm':arm,**{k:float(np.nanmean([r[k] for r in rows])) for k in numeric}})
    paired=[]
    for year in YEARS:
        y=np.asarray(data['y'][distributions[year][0]],int)
        for ref in (ARMS[0],ARMS[1],ARMS[3],ARMS[4]):
            ci=bootstrap_ba(y,final_by[(year,ARMS[2])],final_by[(year,ref)],seed=20261021+year+ARMS.index(ref))
            candidate=next(r for r in means if r['year']==year and r['arm']==ARMS[2]); reference=next(r for r in means if r['year']==year and r['arm']==ref)
            paired.append({'year':year,'candidate':ARMS[2],'reference':ref,
                'mean_BA_difference':candidate['BA']-reference['BA'],
                'mean_Net_Fixed_difference':candidate['Net_Fixed']-reference['Net_Fixed'],**ci})
    source_cmp=[r for r in paired if r['year']==2017]
    supported=all(r['mean_BA_difference']>0 and r['mean_Net_Fixed_difference']>0 and r['ci95_low']>0 for r in source_cmp)
    status='HPRN_SUPPORTED_2017_OOF' if supported else 'HPRN_NOT_SUPPORTED_2017_OOF'
    # APK-level difficult-sample table; Manifest seeds stay visible through aggregate counts.
    difficult=[]
    per_lookup={}
    for row in per: per_lookup.setdefault((row['year'],row['arm'],row['sha256']),[]).append(row)
    for year in YEARS:
        ids=list(map(str,data['ids'][distributions[year][0]]))
        for sid in ids:
            r0=per_lookup[(year,ARMS[0],sid)]; r2=per_lookup[(year,ARMS[2],sid)]
            difficult.append({'year':year,'sha256':sid,'label':r0[0]['label'],
                'difficulty_score':float(np.mean([r['auto_error_probability'] for r in r0])),
                'route_rate':float(np.mean([r['routed'] for r in r0])),
                'stage1_error_count':int(sum(1-r['stage1_correct'] for r in r0)),
                'R0_final_error_count':int(sum(1-r['final_correct'] for r in r0)),
                'R2_final_error_count':int(sum(1-r['final_correct'] for r in r2)),
                'R2_minus_R0_expert_probability':float(np.mean([r['expert_probability'] for r in r2])-np.mean([r['expert_probability'] for r in r0]))})
    difficult.sort(key=lambda r:(-r['difficulty_score'],r['year'],r['sha256']))
    integrity=verify_assets()
    if integrity['status']!='PASS': raise RuntimeError('assets changed during evaluation')
    atomic_csv(OUT/'performance_per_manifest_seed.csv',performance); atomic_csv(OUT/'performance_seed_mean.csv',means)
    atomic_csv(OUT/'per_apk_predictions.csv',per); atomic_csv(OUT/'paired_apk_cluster_bootstrap.csv',paired)
    atomic_csv(OUT/'difficult_sample_ranking.csv',difficult); atomic_csv(OUT/'runtime_cost_future.csv',runtime)
    atomic_json(OUT/'replay_validation.json',{'status':'PASS','tolerance':1e-6,'details':replay,
        'maximum_difference':max(r['max_probability_difference'] for r in replay)})
    atomic_json(OUT/'baseline_integrity_check.json',integrity)
    future_average={}
    for arm in ARMS:
        future_average[arm]=float(np.mean([r['BA'] for r in means if r['arm']==arm and r['year']>2017]))
    atomic_json(OUT/'final_decision.json',{'status':status,'baseline':'W0','candidate':ARMS[2],
        'source_comparisons':source_cmp,'future_average_BA_development_only':future_average,
        'selection_years':[2017],'future_years_development_only':[2018,2019,2020,2021],
        'future_used_for_selection':False,'Stage1':'Manifest135 RF','router':'autonomous P(error)>=0.5',
        'equal_capacity_R1_R4':True,'W0_backbone_frozen':True,
        'next_action':'independent validation before integration' if supported else 'stop HPRN integration and retain W0'})
    lines=['# W0 + Host–Prototype Relation Network（HPRN）','',f'最终状态：**{status}**。','',
        'R0是W0；R1—R4参数量与训练预算一致。W0、三背景原型、RoleGraph、Global24、Manifest135 RF和自主路由全部冻结。',
        '模型选择只依据2017五折OOF；2018—2021为冻结后的开发年份诊断。','',
        '| 年份 | 方案 | BA | AUC | F1 | FPR | FNR | Fixed | Broken | Net | 路由比例 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for year in YEARS:
        for arm in ARMS:
            r=next(x for x in means if x['year']==year and x['arm']==arm)
            lines.append(f"| {year} | {arm} | {r['BA']:.4f} | {r['AUC']:.4f} | {r['F1']:.4f} | {r['FPR']:.4f} | {r['FNR']:.4f} | {r['Fixed']:.1f} | {r['Broken']:.1f} | {r['Net_Fixed']:.1f} | {r['route_rate']:.3f} |")
    lines += ['','## 2017固定裁决比较','']
    for r in source_cmp:
        lines.append(f"- R2 相对 {r['reference']}：平均BA差 {r['mean_BA_difference']:+.5f}，Net差 {r['mean_Net_Fixed_difference']:+.2f}，APK聚类bootstrap 95% CI [{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}]。")
    lines += ['','## 未来年份平均BA（仅描述）','']
    for arm,value in future_average.items(): lines.append(f'- {arm}: {value:.4f}')
    lines += ['','零残差时概率、Attention和Critical Host已严格重放W0。TB-FRS只做APK级聚合Fisher信息探针，与本实验的逐Host关系残差不同。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('HPRN_EVALUATION_DONE status='+status,flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--device',choices=('cpu','cuda'),default='cuda'); args=parser.parse_args()
    os.environ['PYTHONDONTWRITEBYTECODE']='1'; run(args.device)


if __name__=='__main__': main()
