#!/usr/bin/env python3
"""Evaluate fixed-route SCBE ablations and freeze a Chinese report; writes only outputs/care_scbe_v1, no APK access."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
HERE=ROOT/'scripts/care-scbe-v1';OUT=ROOT/'outputs/care_scbe_v1';OLD=ROOT/'outputs/care_multiyear_development_v1'
sys.path[:0]=[str(HERE),str(ROOT/'scripts/care-multiyear-development-v1')]
from preflight import verify,atomic_new
from train import ARMS,SEEDS,load_source_contract,split_contract
from audit_data import load_data
from metrics import cascade,paired_intervals

SPLITS=('S1','S2','S3','S4','S5','T1','T2','T3','T4')

def new_csv(path,rows):
    rows=list(rows);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def load_models():
    result={}
    for split in SPLITS:
        for seed in SEEDS:
            for arm in ARMS:
                path=OUT/'models'/split/str(seed)/(arm+'.pkl')
                if not path.is_file():raise RuntimeError('model missing '+str(path))
                with path.open('rb') as f:obj=pickle.load(f)
                if obj['arm']!=arm or obj['split']!=split or obj['seed']!=seed:
                    raise RuntimeError('checkpoint metadata mismatch '+str(path))
                result[(split,seed,arm)]=(obj,path)
    return result

def frozen_reference(data,source):
    # Existing formal source Direct-C43 replay; only descriptive comparison.
    path=ROOT/'outputs/care-direct-c43-joint-contribution-audit-v1/replay/source_predictions.csv'
    out={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            key='P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2__RepresentationReplay__expert_prob'
            out[r['sha256']]=float(r[key])
    if set(out)!=set(source):raise RuntimeError('formal source Direct-C43 SHA set changed')
    return out

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--evaluate',action='store_true');g.add_argument('--ablation',action='store_true');g.add_argument('--collect',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    if verify()['status']!='PASS':raise RuntimeError('historical asset modified')
    if a.resume and (OUT/'final_decision.json').is_file():
        print('SCBE_EVALUATION_RESUME_COMPLETE',flush=True);return
    if (OUT/'final_decision.json').exists():raise RuntimeError('final result exists; use --resume')
    data=load_data();source=load_source_contract(data);models=load_models()
    rows=[];per=[];fixed=[];conds=[];runtime=[];weights=[];predictor=[];support=[];dep=[];replay=[];paired=[]
    all_by_arm=defaultdict(list);by_seed=defaultdict(list)
    for split in SPLITS:
        tr,va,base,route,score=split_contract(data,split,source)
        ids=list(map(str,data['ids'][va]));y=np.asarray(data['y'][va],int)
        for arm in ARMS:
            seed_prob=[]
            for seed in SEEDS:
                obj,path=models[(split,seed,arm)]
                if obj['ids']!=ids or sorted(set(map(int,data['years'][tr])))!=obj['train_years']:
                    raise RuntimeError('model and fixed split differ '+str(path))
                prob=np.asarray(obj['probability'],float)
                if prob.shape!=(len(va),) or not np.isfinite(prob).all():raise RuntimeError('invalid expert probability')
                seed_prob.append(prob)
                m=cascade(y,base,prob,route)
                row={'split':split,'year':int(data['years'][va][0]),'arm':arm,'seed':seed,
                    'train_n':len(tr),'test_n':len(va),'checkpoint':str(path),**m}
                rows.append(row);by_seed[(split,seed,arm)].append(row)
                runtime.append({'split':split,'arm':arm,'seed':seed,'train_n':len(tr),'test_n':len(va),
                    'checkpoint':str(path),'selected_epochs':obj['selected_epochs'],
                    'SCBE_feature_seconds':obj['feature_audit']['fit_seconds'],**obj['cost']})
                if arm=='M0_REPRODUCED' and 'historical_checkpoint_max_probability_difference' in obj['cost']:
                    replay.append({'split':split,'seed':seed,
                        'max_probability_difference':obj['cost']['historical_checkpoint_max_probability_difference']})
                final=np.where(route,prob,base)
                for j,sid in enumerate(ids):
                    per.append({'split':split,'year':int(data['years'][va][j]),'arm':arm,'seed':seed,
                        'sha256':sid,'label':int(y[j]),'stage1_probability':float(base[j]),
                        'router_score':float(score[j]) if np.isfinite(score[j]) else None,
                        'routed':int(route[j]),'expert_probability':float(prob[j]),
                        'final_probability':float(final[j]),'final_correct':int((final[j]>=.5)==y[j]),
                        'original_external_mean':float(obj['original_external_mean'][j]),
                        'conditional_residual_strength':float(obj['conditional_residual_strength'][j]),
                        'category_direction_strength':float(obj['category_direction_strength'][j]),
                        'checkpoint':str(path)})
            ensemble=np.mean(seed_prob,axis=0)
            m=cascade(y,base,ensemble,route)
            row={'split':split,'year':int(data['years'][va][0]),'arm':arm,'seed':'ENSEMBLE',
                'train_n':len(tr),'test_n':len(va),'checkpoint':'mean of three same-split seed checkpoints',**m}
            rows.append(row);all_by_arm[(split,arm)].append((va,base,route,ensemble))
            final=np.where(route,ensemble,base)
            for j,sid in enumerate(ids):
                per.append({'split':split,'year':int(data['years'][va][j]),'arm':arm,'seed':'ENSEMBLE',
                    'sha256':sid,'label':int(y[j]),'stage1_probability':float(base[j]),
                    'router_score':float(score[j]) if np.isfinite(score[j]) else None,
                    'routed':int(route[j]),'expert_probability':float(ensemble[j]),
                    'final_probability':float(final[j]),'final_correct':int((final[j]>=.5)==y[j]),
                    'checkpoint':'ensemble'})
            fixed.append({'split':split,'year':int(data['years'][va][0]),'arm':arm,
                'seed':'ENSEMBLE','routed':m['routed'],'routed_errors':m['routed_errors'],
                'Fixed':m['Fixed'],'Broken':m['Broken'],'Net':m['Net'],
                'FN_fixed':m['Fixed_FN'],'FP_fixed':m['Fixed_FP'],
                'Broken_TN':m['Broken_TN'],'Broken_TP':m['Broken_TP']})
            conds.append({'split':split,'year':int(data['years'][va][0]),'arm':arm,'seed':'ENSEMBLE',
                'FN_vs_TN_AUC':m['FN_vs_TN_AUC'],'TP_vs_FP_AUC':m['TP_vs_FP_AUC'],
                'Routed_AUC':m['Routed_AUC'],'FN_TN_n':m['FN_TN_n'],'TP_FP_n':m['TP_FP_n']})
    # Source OOF is assembled by fixed fold, not by averaging fold metrics.
    source_val=np.flatnonzero(data['years']==2017);source_position={int(v):i for i,v in enumerate(source_val)}
    ysrc=np.asarray(data['y'][source_val],int);source_base=np.zeros(len(source_val));source_route=np.zeros(len(source_val),bool)
    for arm in ARMS:
        oof=np.zeros(len(source_val));seen=np.zeros(len(source_val),bool)
        for split in SPLITS[:5]:
            va,base,route,pred=all_by_arm[(split,arm)][0]
            pos=np.asarray([source_position[int(v)] for v in va]);oof[pos]=pred;seen[pos]=True
            if arm==ARMS[0]:source_base[pos]=base;source_route[pos]=route
            elif not np.array_equal(source_base[pos],base) or not np.array_equal(source_route[pos],route):
                raise RuntimeError('source upstream changed across arms')
        if not seen.all():raise RuntimeError('2017 OOF missing sample')
        m=cascade(ysrc,source_base,oof,source_route)
        rows.append({'split':'SOURCE_2017_OOF','year':2017,'arm':arm,'seed':'ENSEMBLE',
            'train_n':'outer-fold-specific','test_n':len(ysrc),'checkpoint':'five fixed outer folds × three seeds',**m})
        fixed.append({'split':'SOURCE_2017_OOF','year':2017,'arm':arm,'seed':'ENSEMBLE',
            'routed':m['routed'],'routed_errors':m['routed_errors'],'Fixed':m['Fixed'],
            'Broken':m['Broken'],'Net':m['Net']})
        conds.append({'split':'SOURCE_2017_OOF','year':2017,'arm':arm,'seed':'ENSEMBLE',
            'FN_vs_TN_AUC':m['FN_vs_TN_AUC'],'TP_vs_FP_AUC':m['TP_vs_FP_AUC'],
            'Routed_AUC':m['Routed_AUC'],'FN_TN_n':m['FN_TN_n'],'TP_FP_n':m['TP_FP_n']})
        all_by_arm[('SOURCE_2017_OOF',arm)]=[(source_val,source_base,source_route,oof)]
    formal=frozen_reference(data,source)
    fprob=np.asarray([formal[str(data['ids'][i])] for i in source_val])
    formal_metrics=cascade(ysrc,source_base,fprob,source_route)
    reproduced=[r for r in rows if r['split']=='SOURCE_2017_OOF' and r['arm']=='M0_REPRODUCED'][0]
    replay.append({'split':'SOURCE_2017_OOF','seed':'ENSEMBLE',
        'formal_BA':formal_metrics['BA'],'reproduced_BA':reproduced['BA'],
        'formal_Fixed':formal_metrics['Fixed'],'reproduced_Fixed':reproduced['Fixed'],
        'formal_Broken':formal_metrics['Broken'],'reproduced_Broken':reproduced['Broken'],
        'interpretation':'descriptive protocol check; source code path not identical to historical source fit'})
    # Pairwise APK-level uncertainty on ensemble probabilities; seeds never treated as independent APKs.
    for split in ('SOURCE_2017_OOF','T1','T2','T3','T4'):
        va,base,route,m0=all_by_arm[(split,'M0_REPRODUCED')][0]
        y=np.asarray(data['y'][va],int)
        for against in ('M0_REPRODUCED','M1_CLASS_SUPPORT','M2_CONDITION_ONLY','M4_ROLE_SHUFFLE','M5_CAPACITY_PLACEBO'):
            _,b,r,other=all_by_arm[(split,against)][0]
            if not np.array_equal(b,base) or not np.array_equal(r,route):raise RuntimeError('paired upstream differs')
            cand=all_by_arm[(split,'M3_SCBE')][0][3]
            paired.extend({'split':split,'comparison':'M3_SCBE_minus_'+against,**x} for x in paired_intervals(y,base,route,cand,other,repeats=1000))
    for split in SPLITS:
        for seed in SEEDS:
            audit=models[(split,seed,'M3_SCBE')][0]['feature_audit']
            predictor.append({'split':split,'seed':seed,'train_APKs':audit['train_apks'],
                'train_Hosts':audit['train_host_count'],'test_Hosts':audit['test_host_count'],
                'conditional_R2':audit['test_conditional_r2'],'fit_seconds':audit['fit_seconds']})
            dep.append({'split':split,'seed':seed,'APK_level_abs_Eperp_vs_formal_Residual_Pearson':audit['per_APK_abs_conditional_residual_vs_old_residual_Pearson'],
                'mathematical_equivalence_to_formal_Residual':False,'bank_self_source_bias_remains':True})
            w=np.asarray(audit['full_weight'])
            weights.append({'split':split,'seed':seed,'positive_centers':int((w>0).sum()),
                'negative_centers':int((w<0).sum()),'mean_abs_weight':float(np.mean(np.abs(w))),
                'max_abs_weight':float(np.max(np.abs(w))),'role_shuffle_changed':audit['role_shuffle_changed_prototypes'],
                'role_shuffle_singletons':audit['role_shuffle_singletons'],
                'cross_bank_prototype_index_alignment':'NOT_VALID'})
            ntr=int(audit['train_apks']);nbenign=int(np.sum(data['y'][split_contract(data,split,source)[0]]==0));nmal=ntr-nbenign
            for j,(cb,cm,wj) in enumerate(zip(audit['positive_support_benign_total'],audit['positive_support_malware_total'],w)):
                support.append({'split':split,'seed':seed,'prototype_id':j,'train_benign_APKs':nbenign,
                    'train_malware_APKs':nmal,'benign_support_APKs':cb,'malware_support_APKs':cm,
                    'q_B_smoothed':(cb+2)/(nbenign+4),'q_M_smoothed':(cm+2)/(nmal+4),
                    'w_shrunk_clipped':float(wj),'own_label_used_in_own_OOF_weight':False,
                    'bank_self_source_bias_remains':True})
    # Fixed test-set selection criteria. These are development diagnostics only.
    get=lambda split,arm: next(r for r in rows if r['split']==split and r['arm']==arm and r['seed']=='ENSEMBLE')
    average_gain=float(np.mean([get(t,'M3_SCBE')['BA']-get(t,'M0_REPRODUCED')['BA'] for t in ('T2','T3','T4')]))
    net_gain=int(sum(get(t,'M3_SCBE')['Net']-get(t,'M0_REPRODUCED')['Net'] for t in ('T2','T3','T4')))
    positive_years=int(sum(get(t,'M3_SCBE')['BA']>get(t,'M0_REPRODUCED')['BA'] for t in ('T2','T3','T4')))
    control_better=all(np.mean([get(t,'M3_SCBE')['BA']-get(t,c)['BA'] for t in ('T2','T3','T4')])>0
        for c in ('M1_CLASS_SUPPORT','M2_CONDITION_ONLY','M4_ROLE_SHUFFLE','M5_CAPACITY_PLACEBO'))
    source_gap=get('SOURCE_2017_OOF','M3_SCBE')['BA']-get('SOURCE_2017_OOF','M0_REPRODUCED')['BA']
    seed_direction=[]
    for seed in SEEDS:
        gain=np.mean([next(r['BA'] for r in rows if r['split']==t and r['arm']=='M3_SCBE' and r['seed']==seed)-
                      next(r['BA'] for r in rows if r['split']==t and r['arm']=='M0_REPRODUCED' and r['seed']==seed)
                      for t in ('T2','T3','T4')])
        seed_direction.append(float(gain))
    criteria={'T2_T4_mean_BA_gain_at_least_0_005':average_gain>=.005,
        'T2_T4_Net_gain_at_least_6':net_gain>=6,'at_least_two_years_BA_positive':positive_years>=2,
        'M3_mean_BA_above_all_controls':control_better,
        'source_BA_not_degraded_over_0_01':source_gap>=-.01,
        'all_three_seed_mean_BA_gains_nonnegative':all(x>=0 for x in seed_direction),
        '2020_FN_not_severely_worse':get('T3','M3_SCBE')['FN']<=get('T3','M0_REPRODUCED')['FN']+2}
    supported=all(criteria.values())
    status='SCBE_MECHANISM_SUPPORTED_ON_DEVELOPMENT_DATA' if supported else 'SCBE_NO_RELIABLE_INCREMENT'
    decision={'status':status,'criteria':criteria,'T2_T4_mean_BA_gain':average_gain,
        'T2_T4_Net_gain':net_gain,'positive_years':positive_years,'seed_mean_BA_gains':seed_direction,
        'source_BA_gain':source_gap,'formal_asset_integrity':verify()['status'],
        'no_formal_model_change':True,'future_years_are_development_not_independent_test':True,
        'Prototype_Bank_self_source_bias_remains':True,
        'next_action':'保留正式三背景；仅在全部对照和预设条件成立时，另行设计独立样本验证。' if not supported else '先在全新独立样本冻结验证，不能直接替换正式模型'}
    new_csv(OUT/'source2017_fivefold_results.csv',[r for r in rows if r['split'].startswith('S') or r['split']=='SOURCE_2017_OOF'])
    new_csv(OUT/'rolling_crossyear_results.csv',[r for r in rows if r['split'].startswith('T')])
    new_csv(OUT/'ablation_results.csv',paired)
    new_csv(OUT/'per_apk_predictions.csv',per)
    new_csv(OUT/'fixed_broken_analysis.csv',fixed)
    new_csv(OUT/'conditional_auc_analysis.csv',conds)
    new_csv(OUT/'capacity_control_results.csv',[r for r in rows if r['arm'] in ('M3_SCBE','M5_CAPACITY_PLACEBO')])
    new_csv(OUT/'conditional_predictor_metrics.csv',predictor)
    new_csv(OUT/'prototype_class_support.csv',support)
    new_csv(OUT/'scbe_weight_stability.csv',weights)
    new_csv(OUT/'residual_dependency_analysis.csv',dep)
    new_csv(OUT/'runtime_cost.csv',runtime)
    atomic_new(OUT/'baseline_replay.json',{'status':'PASS' if all(r['max_probability_difference']<=1e-5 for r in replay if 'max_probability_difference' in r) else 'FAIL',
        'rolling_seed_replays':replay,'source_formal_comparison_is_descriptive':True})
    atomic_new(OUT/'final_decision.json',decision)
    report=['# SCBE受控实验 V1','',f'最终状态：**{status}**。','',
        '所有结果均来自历史2017—2021开发数据；冻结Stage-I、Top25%路由及0.5阈值，未新增APK或修改正式模型。',
        f'T2—T4平均BA差（M3−M0）：{average_gain:+.6f}；Net差合计：{net_gain:+d}；正向年份：{positive_years}/3。',
        f'2017源域BA差：{source_gap:+.6f}；三种子T2—T4平均BA差：{seed_direction}。','',
        '## 关键对照（各窗口三种子概率平均）','',
        '| 窗口 | 方案 | BA | AUC | FN | FP | Fixed | Broken | Net | FN-vs-TN AUC | TP-vs-FP AUC |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in ('SOURCE_2017_OOF','T1','T2','T3','T4'):
        for arm in ARMS:
            r=get(t,arm)
            report.append(f"| {t} | {arm} | {r['BA']:.4f} | {r['AUC']:.4f} | {r['FN']} | {r['FP']} | {r['Fixed']} | {r['Broken']} | {r['Net']} | {r['FN_vs_TN_AUC']:.4f} | {r['TP_vs_FP_AUC']:.4f} |")
    report += ['', '## 机制边界','',
        'SCBE的24维条件残差并非正式Residual的代数重命名；其冻结模型响应相关性、条件预测R²和类别权重见对应CSV。',
        'M0与历史滚动checkpoint的逐概率比较见`baseline_replay.json`。三种子是训练重复，不当作独立APK。',
        '全部Bank由对应训练窗口拟合；内部APK交叉拟合排除了自身标签进入自身w，但Bank仍含该APK的Host，自来源偏差未完全消除。',
        '本轮2018—2021曾被多次开发使用；即使有正向结果，也只能称开发集机制支持，不能宣称独立泛化。',
        f'30项命名历史文件完整性：{decision["formal_asset_integrity"]}。','']
    atomic_new(OUT/'final_report.md','\n'.join(report))
    atomic_new(OUT/'reproduce_commands.md','''# 复现\n\n此V1结果不可覆盖；仅能在全新版本目录重跑。\n\n`PYTHONDONTWRITEBYTECODE=1 bash scripts/care-scbe-v1/run.sh --audit-only`\n`CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 bash scripts/care-scbe-v1/run.sh --smoke-test --device cuda --workers 1`\n`CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 bash scripts/care-scbe-v1/run.sh --train-source --resume --device cuda --workers 1`\n`CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 bash scripts/care-scbe-v1/run.sh --train-rolling --resume --device cuda --workers 1`\n`PYTHONDONTWRITEBYTECODE=1 bash scripts/care-scbe-v1/run.sh --evaluate --resume`\n''')
    atomic_new(OUT/'asset_integrity_final.json',verify())
    print('SCBE_FINAL',status,flush=True)

if __name__=='__main__':main()
