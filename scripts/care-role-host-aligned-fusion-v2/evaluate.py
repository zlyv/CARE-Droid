#!/usr/bin/env python3
"""APK-level evaluation for the frozen Direct-C43 baseline and static aligned-fusion controls."""
from __future__ import annotations

import pickle
import numpy as np

from rhaf_common import ARMS,OUT,SCBE,SEEDS,SPLITS,atomic_csv,atomic_json,verify_assets


def load_model(split,seed,arm):
    path=(SCBE/'models'/split/str(seed)/'M0_REPRODUCED.pkl') if arm==ARMS[0] else (OUT/'models'/split/str(seed)/(arm+'.pkl'))
    if not path.is_file(): raise RuntimeError('missing checkpoint '+str(path))
    with path.open('rb') as handle: obj=pickle.load(handle)
    expected='M0_REPRODUCED' if arm==ARMS[0] else arm
    if obj['arm']!=expected or obj['split']!=split or obj['seed']!=seed: raise RuntimeError('checkpoint identity mismatch')
    return obj,path


def run():
    from bfa_core import load_data_contract
    from metrics import cascade,paired_intervals
    from train import split_contract
    if verify_assets()['status']!='PASS': raise RuntimeError('assets changed before evaluation')
    data,source=load_data_contract(); results=[]; per=[]; runtime=[]; distributions={}; scales=[]
    source_index=np.flatnonzero(data['years']==2017); source_pos={int(v):i for i,v in enumerate(source_index)}
    source_base=np.empty(len(source_index)); source_route=np.empty(len(source_index),bool)
    source_probability={a:np.empty(len(source_index)) for a in ARMS}; source_seen={a:np.zeros(len(source_index),bool) for a in ARMS}
    for split in SPLITS:
        train,val,base,route,score=split_contract(data,split,source); ids=list(map(str,data['ids'][val])); y=np.asarray(data['y'][val],int)
        for arm in ARMS:
            probabilities=[]
            for seed in SEEDS:
                obj,path=load_model(split,seed,arm)
                if obj['ids']!=ids: raise RuntimeError('SHA order mismatch '+str(path))
                p=np.asarray(obj['probability'],float); probabilities.append(p); metric=cascade(y,base,p,route)
                results.append({'split':split,'year':int(data['years'][val[0]]),'arm':arm,'seed':seed,
                                'train_n':len(train),'test_n':len(val),**metric})
                cost=obj['cost']; runtime.append({'split':split,'arm':arm,'seed':seed,
                    'selected_epochs':obj['selected_epochs'],'parameters':cost['parameters'],
                    'training_seconds':cost['training_seconds'],'inference_seconds':cost['inference_seconds'],
                    'gpu_peak_mib':cost['gpu_peak_mib']})
                if arm!=ARMS[0]: scales.append({'split':split,'arm':arm,'seed':seed,**obj['fusion_scales']})
            p=np.mean(probabilities,0); metric=cascade(y,base,p,route)
            results.append({'split':split,'year':int(data['years'][val[0]]),'arm':arm,'seed':'ENSEMBLE',
                            'train_n':len(train),'test_n':len(val),**metric})
            distributions[(split,arm)]=(val,y,base,route,p); final=np.where(route,p,base)
            for i,sid in enumerate(ids): per.append({'split':split,'year':int(data['years'][val[i]]),'arm':arm,
                'sha256':sid,'label':int(y[i]),'routed':int(route[i]),'stage1_probability':float(base[i]),
                'expert_probability':float(p[i]),'final_probability':float(final[i]),
                'final_correct':int(int(final[i]>=.5)==int(y[i]))})
            if split.startswith('S'):
                positions=np.asarray([source_pos[int(i)] for i in val]); source_probability[arm][positions]=p; source_seen[arm][positions]=True
                if arm==ARMS[0]: source_base[positions]=base; source_route[positions]=route
    for arm in ARMS:
        if not source_seen[arm].all(): raise RuntimeError('source OOF incomplete')
        y=np.asarray(data['y'][source_index],int); metric=cascade(y,source_base,source_probability[arm],source_route)
        results.append({'split':'SOURCE_2017_OOF','year':2017,'arm':arm,'seed':'ENSEMBLE','train_n':'five folds','test_n':799,**metric})
        distributions[('SOURCE_2017_OOF',arm)]=(source_index,y,source_base,source_route,source_probability[arm])
    def metric(split,arm): return next(r for r in results if r['split']==split and r['arm']==arm and r['seed']=='ENSEMBLE')
    paired=[]
    for split in ('SOURCE_2017_OOF','T1','T2','T3','T4'):
        _,y,base,route,_=distributions[(split,ARMS[0])]
        p=distributions[(split,ARMS[1])][4]
        for reference in (ARMS[0],ARMS[2],ARMS[3]):
            q=distributions[(split,reference)][4]
            for row in paired_intervals(y,base,route,p,q,repeats=500):
                paired.append({'split':split,'candidate':ARMS[1],'reference':reference,**row})
    dev_splits=('T2','T3','T4')
    def comparison(reference):
        return {'mean_BA_gain':float(np.mean([metric(s,ARMS[1])['BA']-metric(s,reference)['BA'] for s in dev_splits])),
                'Net_gain':int(sum(metric(s,ARMS[1])['Net']-metric(s,reference)['Net'] for s in dev_splits)),
                'positive_BA_years':int(sum(metric(s,ARMS[1])['BA']>metric(s,reference)['BA'] for s in dev_splits)),
                'mean_FN_TN_AUC_gain':float(np.mean([metric(s,ARMS[1])['FN_vs_TN_AUC']-metric(s,reference)['FN_vs_TN_AUC'] for s in dev_splits])),
                'mean_TP_FP_AUC_gain':float(np.mean([metric(s,ARMS[1])['TP_vs_FP_AUC']-metric(s,reference)['TP_vs_FP_AUC'] for s in dev_splits])),
                'source_BA_gain':float(metric('SOURCE_2017_OOF',ARMS[1])['BA']-metric('SOURCE_2017_OOF',reference)['BA'])}
    comparisons={ref:comparison(ref) for ref in (ARMS[0],ARMS[2],ARMS[3])}; base=comparisons[ARMS[0]]
    control_gain=min(comparisons[ARMS[2]]['mean_BA_gain'],comparisons[ARMS[3]]['mean_BA_gain'])
    supported=(base['mean_BA_gain']>0 and base['Net_gain']>0 and base['positive_BA_years']>=2 and
               base['source_BA_gain']>=-.01 and base['mean_FN_TN_AUC_gain']>=-.01 and
               base['mean_TP_FP_AUC_gain']>=-.01 and control_gain>=.001)
    status='ROLE_HOST_ALIGNED_FUSION_SUPPORTED_ON_DEVELOPMENT' if supported else 'ROLE_HOST_ALIGNED_FUSION_NOT_SUPPORTED'
    integrity=verify_assets()
    if integrity['status']!='PASS': raise RuntimeError('assets changed during evaluation')
    atomic_csv(OUT/'performance.csv',results); atomic_csv(OUT/'per_apk_predictions.csv',per)
    atomic_csv(OUT/'paired_intervals.csv',paired); atomic_csv(OUT/'runtime_cost.csv',runtime); atomic_csv(OUT/'fusion_scales.csv',scales)
    atomic_json(OUT/'baseline_integrity_check.json',integrity)
    atomic_json(OUT/'final_decision.json',{'status':status,'comparisons':comparisons,'criteria':{
        'positive_vs_baseline':base['mean_BA_gain']>0 and base['Net_gain']>0,
        'at_least_two_positive_development_years':base['positive_BA_years']>=2,
        'source_BA_loss_no_more_than_0.01':base['source_BA_gain']>=-.01,
        'both_conditional_AUC_mean_losses_no_more_than_0.01':base['mean_FN_TN_AUC_gain']>=-.01 and base['mean_TP_FP_AUC_gain']>=-.01,
        'mean_BA_gain_over_each_equal_capacity_control_at_least_0.001':control_gain>=.001},
        'dynamic_analysis':False,'new_APKs':False,'future_years_are_development':True,
        'prior_compact_HBI_result':'HOST_BINDING_INFORMATION_NOT_SUPPORTED',
        'next_action':'Independent dataset validation before integration.' if supported else 'Stop Role-Host fusion redesign; retain formal Direct-C43.'})
    lines=['# CARE-Droid静态Role–Host对齐融合实验 v2','',f'最终状态：**{status}**。','',
           '本实验使用现有静态Host自有Role31，不读取APK、不进行动态分析、不修改Stage-I、Router或三背景。','',
           '| 窗口 | 方案 | BA | AUC | FN | FP | Fixed | Broken | Net | FN-vs-TN AUC | TP-vs-FP AUC |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('SOURCE_2017_OOF','T1','T2','T3','T4'):
        for arm in ARMS:
            r=metric(split,arm); lines.append(f"| {split} | {arm} | {r['BA']:.4f} | {r['AUC']:.4f} | {r['FN']} | {r['FP']} | {r['Fixed']} | {r['Broken']} | {r['Net']} | {r['FN_vs_TN_AUC']:.4f} | {r['TP_vs_FP_AUC']:.4f} |")
    lines += ['','## 固定比较','']
    for ref,value in comparisons.items(): lines.append(f"- F1相对{ref}：T2—T4平均BA差{value['mean_BA_gain']:+.5f}，Net差{value['Net_gain']:+d}，FN/TN AUC差{value['mean_FN_TN_AUC_gain']:+.5f}，TP/FP AUC差{value['mean_TP_FP_AUC_gain']:+.5f}。")
    lines += ['','F1、F2、F3使用完全相同的网络和参数量；F2保留APK内Host角色行集合但打乱其与Host特征的绑定，F3只保留APK角色边际。若F1不能胜过两者，不继续调融合。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('RHAF_EVALUATION_DONE status='+status,flush=True)


if __name__=='__main__': run()
