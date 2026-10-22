#!/usr/bin/env python3
"""APK-level evaluation for the frozen baseline and three same-capacity arms; reads no APKs and writes only new result tables."""
from __future__ import annotations

import argparse
import pickle

import numpy as np

from cfgwa_common import (ARMS, OUT, SCBE, SEEDS, SPLITS, atomic_csv,
                          atomic_json, verify_assets)


def load_model(split, seed, arm):
    if arm == ARMS[0]: path = SCBE/'models'/split/str(seed)/'M0_REPRODUCED.pkl'
    else: path = OUT/'models'/split/str(seed)/(arm+'.pkl')
    if not path.is_file(): raise RuntimeError('missing checkpoint '+str(path))
    with path.open('rb') as handle: obj=pickle.load(handle)
    expected = 'M0_REPRODUCED' if arm==ARMS[0] else arm
    if obj['arm'] != expected or obj['split'] != split or obj['seed'] != seed: raise RuntimeError('checkpoint identity')
    return obj,path


def run():
    from bfa_core import load_data_contract
    from metrics import cascade, paired_intervals
    from train import split_contract
    if verify_assets()['status']!='PASS': raise RuntimeError('assets changed before evaluation')
    data, source = load_data_contract(); results=[]; per=[]; runtime=[]; distributions={}
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
                p=np.asarray(obj['probability'],float); probabilities.append(p)
                metric=cascade(y,base,p,route)
                results.append({'split':split,'year':int(data['years'][val[0]]),'arm':arm,'seed':seed,
                                'train_n':len(train),'test_n':len(val),**metric})
                cost=obj['cost']; runtime.append({'split':split,'arm':arm,'seed':seed,
                    'selected_epochs':obj['selected_epochs'],'parameters':cost['parameters'],
                    'training_seconds':cost['training_seconds'],'inference_seconds':cost['inference_seconds'],
                    'gpu_peak_mib':cost['gpu_peak_mib']})
            p=np.mean(probabilities,0); metric=cascade(y,base,p,route)
            results.append({'split':split,'year':int(data['years'][val[0]]),'arm':arm,'seed':'ENSEMBLE',
                            'train_n':len(train),'test_n':len(val),**metric})
            distributions[(split,arm)]=(val,y,base,route,p)
            final=np.where(route,p,base)
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
        for candidate in ARMS[1:]:
            p=distributions[(split,candidate)][4]
            for reference in (ARMS[0], ARMS[2], ARMS[3]):
                if candidate==reference: continue
                q=distributions[(split,reference)][4]
                for row in paired_intervals(y,base,route,p,q,repeats=500):
                    paired.append({'split':split,'candidate':candidate,'reference':reference,**row})
    splits=('T2','T3','T4')
    def score(arm,reference):
        return {'mean_BA_gain':float(np.mean([metric(s,arm)['BA']-metric(s,reference)['BA'] for s in splits])),
                'Net_gain':int(sum(metric(s,arm)['Net']-metric(s,reference)['Net'] for s in splits)),
                'positive_BA_years':int(sum(metric(s,arm)['BA']>metric(s,reference)['BA'] for s in splits)),
                'mean_FN_TN_AUC_gain':float(np.mean([metric(s,arm)['FN_vs_TN_AUC']-metric(s,reference)['FN_vs_TN_AUC'] for s in splits])),
                'source_BA_gain':float(metric('SOURCE_2017_OOF',arm)['BA']-metric('SOURCE_2017_OOF',reference)['BA'])}
    comparisons={ref:score(ARMS[1],ref) for ref in (ARMS[0],ARMS[2],ARMS[3])}
    b=comparisons[ARMS[0]]; beyond=min(comparisons[ARMS[2]]['mean_BA_gain'],comparisons[ARMS[3]]['mean_BA_gain'])
    supported=(b['mean_BA_gain']>0 and b['Net_gain']>0 and b['positive_BA_years']>=2 and
               b['mean_FN_TN_AUC_gain']>=0 and b['source_BA_gain']>=-.01 and beyond>=.001)
    status='CFG_WITNESS_INCREMENT_SUPPORTED_ON_DEVELOPMENT' if supported else 'CFG_WITNESS_INCREMENT_NOT_SUPPORTED'
    integrity=verify_assets()
    if integrity['status']!='PASS': raise RuntimeError('assets changed during evaluation')
    atomic_csv(OUT/'performance.csv',results); atomic_csv(OUT/'per_apk_predictions.csv',per)
    atomic_csv(OUT/'paired_intervals.csv',paired); atomic_csv(OUT/'runtime_cost.csv',runtime)
    atomic_json(OUT/'baseline_integrity_check.json',integrity)
    decision={'status':status,'comparisons':comparisons,'screen':{
        'positive_vs_baseline':b['mean_BA_gain']>0 and b['Net_gain']>0,
        'at_least_two_positive_years':b['positive_BA_years']>=2,
        'FN_TN_AUC_not_worse':b['mean_FN_TN_AUC_gain']>=0,
        'source_BA_loss_no_more_than_0.01':b['source_BA_gain']>=-.01,
        'mean_BA_gain_over_each_equal_capacity_control_at_least_0.001':beyond>=.001},
        'future_years_are_development':True,'independent_validation':False,
        'Stage1_or_Router_changed':False,'formal_model_changed':False,
        'next_action':'Independent dataset validation before integration.' if supported else 'Stop CFG Witness integration; retain the formal Direct-C43.'}
    atomic_json(OUT/'final_decision.json',decision)
    lines=['# CARE-Droid CFG Witness固定容量增量实验','',f"最终状态：**{status}**。",'',
           '2018—2021已经参与Witness候选冻结，因此本结果仅为开发集验证。','',
           '| 窗口 | 方案 | BA | AUC | FN | FP | Fixed | Broken | Net | FN-vs-TN AUC | TP-vs-FP AUC |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('SOURCE_2017_OOF','T1','T2','T3','T4'):
        for arm in ARMS:
            r=metric(split,arm); lines.append(f"| {split} | {arm} | {r['BA']:.4f} | {r['AUC']:.4f} | {r['FN']} | {r['FP']} | {r['Fixed']} | {r['Broken']} | {r['Net']} | {r['FN_vs_TN_AUC']:.4f} | {r['TP_vs_FP_AUC']:.4f} |")
    lines += ['', '## 固定比较','']
    for ref,value in comparisons.items(): lines.append(f"- W1相对{ref}：T2—T4平均BA差{value['mean_BA_gain']:+.5f}，Net差{value['Net_gain']:+d}，FN/TN AUC差{value['mean_FN_TN_AUC_gain']:+.5f}，源域BA差{value['source_BA_gain']:+.5f}。")
    lines += ['', 'W1、W2、W3均为27维且训练协议相同；W0为原24维冻结基线。静态CFG可达不代表运行时执行或恶意意图。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('CFGWA_EVALUATION_DONE status='+status,flush=True)


if __name__=='__main__': run()
