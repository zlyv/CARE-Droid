#!/usr/bin/env python3
"""Manifest-only four-member Stage-I and router audit. Reads existing feature/
prediction assets, never APKs. Writes only to fresh/resumable MFD_OUT files.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
OUT = Path(os.environ['MFD_OUT']).resolve()
if OUT.parent != ROOT/'outputs/care_manifest_four_member_disagreement_v1' or not OUT.name.startswith('run_'):
    raise RuntimeError('BLOCKED_BY_CONTRACT: output must be a fresh isolated run directory')
sys.path.insert(0, str(ROOT/'scripts/care-lite-stage-v1'))
import common as lc
sys.path.insert(0, str(ROOT/'scripts/care-p31-stage1-rf-selection-source-v1'))
from stage1_core import fit_calibrated_member

BASE_OUT = ROOT/'outputs/care_manifest_routing_coverage_v1/run_20261019_091224'
CASCADE_OUT = ROOT/'outputs/care_manifest_acquisition_cascade_audit_v1/run_20261019_pilot01'
NAMES = ('LR','RF','XGB','SVM')
SEEDS = (20261018,20261019,20261020)
STRATEGIES = ('STD25_REF','OTHER3_OP25_REF','ANCHOR_UNC25_REF',
              'STD25_MATCHED_UNC','OTHER3_OP25_MATCHED_UNC')

def write_json(path, value):
    path = OUT/path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_csv(name, rows):
    rows = list(rows)
    if not rows: raise RuntimeError('BLOCKED_BY_CONTRACT: no rows for '+name)
    with (OUT/name).open('x', encoding='utf-8', newline='') as handle:
        w = csv.DictWriter(handle, list(rows[0]))
        w.writeheader(); w.writerows(rows)

def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4<<20), b''): h.update(block)
    return {'path':str(path), 'sha256':h.hexdigest()}

def integrity():
    before = read_json(OUT/'asset_manifest_before.json')
    changed = [r['path'] for r in before if digest(r['path'])['sha256'] != r['sha256']]
    return {'status':'FAIL' if changed else 'PASS','changed':changed,'asset_count':len(before),
            'scope':'Exact reused feature shards, source lineage, prior RF/LR probabilities, frozen C43 prediction table and implementation files; no APK files/checkpoints opened.'}

def inputs():
    _, contract = lc.source_contract()
    ids = np.array(sorted(contract))
    y = np.array([int(contract[s]['label']) for s in ids])
    fold = np.array([int(contract[s]['fold']) for s in ids])
    source, missing = lc.load_shards('source')
    future, future_missing = lc.load_shards('future')
    fi = np.array(sorted(future))
    year_map = {r['sha256']:int(r['year']) for r in lc.readcsv(lc.FUTURE_LINEAGE)}
    assert not missing and not future_missing and len(ids)==799 and len(fi)==1989
    assert set(fi)==set(year_map) and not set(ids)&set(fi)
    x = lc.matrix([source[s] for s in ids], 'L1_MANIFEST')
    z = lc.matrix([future[s] for s in fi], 'L1_MANIFEST')
    years = np.array([year_map[s] for s in fi])
    assert x.shape==(799,135) and z.shape==(1989,135)
    assert {v:int((years==v).sum()) for v in (2018,2019,2020,2021)}=={2018:793,2019:399,2020:397,2021:400}
    return ids,y,fold,x,fi,years,z

def preflight():
    if (OUT/'protocol.json').exists():
        assert integrity()['status']=='PASS'
        return
    ids,y,fold,x,fi,years,z = inputs()
    assert (OUT/'final_decision.json').exists() is False
    assert len(set(ids))==799 and len(set(fi))==1989
    assert read_json(BASE_OUT/'final_decision.json')['asset_integrity']=='PASS'
    assert read_json(CASCADE_OUT/'final_decision.json')['asset_integrity']=='PASS'
    paths = [lc.SOURCE_LINEAGE, lc.SOURCE_PRED, lc.FUTURE_LINEAGE,
             BASE_OUT/'final_decision.json',CASCADE_OUT/'final_decision.json',
             CASCADE_OUT/'per_apk_predictions.csv',Path(__file__),Path(__file__).with_name('run.sh'),
             ROOT/'scripts/care-lite-stage-v1/common.py',
             ROOT/'scripts/care-p31-stage1-rf-selection-source-v1/stage1_core.py']
    paths += list((lc.OUT/'shards/source').glob('*.json'))
    paths += list((lc.OUT/'shards/future').glob('*.json'))
    paths += list((BASE_OUT/'frozen').glob('*.json'))
    paths += list((BASE_OUT/'crossfit').glob('*.json'))
    write_json('asset_manifest_before.json', [digest(p) for p in sorted(set(paths))])
    write_json('protocol.json', {
        'status':'FROZEN_BEFORE_NEW_FITS','source':799,'future':1989,'seeds':SEEDS,
        'features':'Manifest135 only; no DEX, Host, graph or C43 input used for routing',
        'anchors':NAMES,'threshold':0.5,'source_outer_folds':5,'router_reference_inner_folds':3,
        'score_primary':'population std of four calibrated probabilities',
        'score_secondary':'mean probability of the other three opposing anchor class',
        'control':'anchor uncertainty; exact same-count uncertainty for each candidate',
        'reference_budget':0.25,'threshold_fit':'75th percentile order statistic of train-only inner OOF score; frozen before target labels',
        'future_fit':'all 2017 only; threshold derived from crossfit 2017, then unchanged for 2018-2021',
        'expert':'same frozen Direct-C43 HDBC H0 source and SCBE M0 T1 future as prior cascade',
        'model_selection':'none; all arms reported, no future-selected winner',
        'cost_scope':'classifier inference not yet separately measured; all members share pre-existing Manifest extraction',
        'data_boundary':'historical development years, not independent final test'})
    print('MFD_PREFLIGHT_PASS source=799 future=1989 assets='+str(len(paths)),flush=True)

def uncertainty(p): return 1-2*np.abs(np.asarray(p)-.5)
def scores(members, anchor_index):
    anchor = members[:,anchor_index]
    other = np.delete(members, anchor_index, axis=1)
    opposition = np.where(anchor[:,None]>=.5, 1-other, other).mean(axis=1)
    return {'STD25_REF':members.std(axis=1), 'OTHER3_OP25_REF':opposition,
            'ANCHOR_UNC25_REF':uncertainty(anchor)}
def cutoff(v, budget=.25):
    k = max(1, round(len(v)*budget))
    return float(np.sort(v)[-k])
def top(ids, score, count):
    order = np.lexsort((ids, -score))
    mask = np.zeros(len(ids),bool)
    mask[order[:int(count)]] = True
    return mask

def fit_window(seed, f, ids, y, fold, x, fi, z):
    path = OUT/f'windows/{seed}_{f}.json'
    if path.exists():
        obj = read_json(path)
        assert obj['seed']==seed and obj['fold']==f
        return obj
    t0=time.perf_counter()
    tr=np.flatnonzero(fold!=f) if f else np.arange(len(ids))
    te=np.flatnonzero(fold==f) if f else np.arange(len(fi))
    target=x[te] if f else z
    target_ids=ids[te] if f else fi
    prior=read_json(BASE_OUT/f'frozen/{seed}_{f}.json')
    assert prior['train_sha']==list(ids[tr]) and prior['target_sha']==list(target_ids)
    target_member=np.zeros((len(target_ids),4))
    target_member[:,0]=prior['lr_probability']
    target_member[:,1]=prior['rf_probability']
    for j,name in ((2,'XGB'),(3,'SVM')):
        target_member[:,j]=fit_calibrated_member(name,x[tr],y[tr],target,seed+f,inner_folds=3,jobs=2)
    inner_member=np.full((len(tr),4),np.nan)
    skf=StratifiedKFold(3,shuffle=True,random_state=seed+f)
    for k,(it,iv) in enumerate(skf.split(x[tr],y[tr]),1):
        for j,name in enumerate(('LR','RF')):
            prior_inner=read_json(BASE_OUT/f'crossfit/{seed}_{f}_{k}_{name}.json')
            assert prior_inner['train_sha']==list(ids[tr[it]]) and prior_inner['test_sha']==list(ids[tr[iv]])
            inner_member[iv,j]=prior_inner['probability']
        for j,name in ((2,'XGB'),(3,'SVM')):
            inner_member[iv,j]=fit_calibrated_member(name,x[tr[it]],y[tr[it]],x[tr[iv]],
                                                       seed+f+100000*k,inner_folds=3,jobs=2)
    assert np.isfinite(inner_member).all() and np.isfinite(target_member).all()
    assert np.all((inner_member>=0)&(inner_member<=1)) and np.all((target_member>=0)&(target_member<=1))
    rules={}
    for j,name in enumerate(NAMES):
        rules[name]={s:cutoff(v) for s,v in scores(inner_member,j).items()}
    obj={'seed':seed,'fold':f,'train_sha':list(ids[tr]),'target_sha':list(target_ids),
         'members':target_member.tolist(),'rules':rules,'fit_seconds':time.perf_counter()-t0}
    write_json(str(path.relative_to(OUT)),obj)
    print(f'MFD_WINDOW_DONE seed={seed} fold={f} fit_seconds={obj["fit_seconds"]:.1f}',flush=True)
    return obj

def expert_reference():
    rows=read_csv(CASCADE_OUT/'per_apk_predictions.csv')
    ref={}
    for row in rows:
        if row['strategy']!='U_REF10':continue
        key=(int(row['seed']),int(row['year']),row['sha256'])
        assert key not in ref
        ref[key]=row
    assert len(ref)==len(SEEDS)*(799+1989)
    return ref

def metrics(y,p,expert,route):
    final=np.where(route,expert,p)
    before=(p>=.5)==y; after=(final>=.5)==y
    tn,fp,fn,tp=confusion_matrix(y,final>=.5,labels=[0,1]).ravel()
    err=~before
    return dict(n=len(y),routed=int(route.sum()),stage1_errors=int(err.sum()),
                routed_stage1_errors=int((route&err).sum()),unrouted_stage1_errors=int((~route&err).sum()),
                routed_FN=int((route&err&(y==1)).sum()),routed_FP=int((route&err&(y==0)).sum()),
                stage1_BA=float(balanced_accuracy_score(y,p>=.5)),
                stage1_AUC=float(roc_auc_score(y,p)),
                BA=float(balanced_accuracy_score(y,final>=.5)),AUC=float(roc_auc_score(y,final)),
                ACC=float(accuracy_score(y,final>=.5)),F1=float(f1_score(y,final>=.5)),
                TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
                fixed=int((err&after).sum()),broken=int((before&~after).sum()),
                net=int((err&after).sum()-(before&~after).sum()))

def evaluate(ids,y,fold,fi,years,windows):
    # Future labels and expert outcomes are not opened until all rules and fits are frozen.
    ref=expert_reference()
    result=[]; stage=[]; predictions=[]; parity=[]
    for seed in SEEDS:
        for f in (1,2,3,4,5,0):
            obj=windows[(seed,f)]
            ids0=ids[fold==f] if f else fi
            members=np.asarray(obj['members'],float)
            assert obj['target_sha']==list(ids0) and members.shape==(len(ids0),4)
            year0=np.full(len(ids0),2017) if f else years
            for yr in sorted(set(year0)):
                ix=np.flatnonzero(year0==yr)
                sha=ids0[ix];part=members[ix]
                records=[ref[(seed,int(yr),str(s))] for s in sha]
                yy=np.array([int(r['label']) for r in records])
                expert=np.array([float(r['expert_probability']) for r in records])
                if f:
                    assert np.array_equal(yy,y[fold==f])
                frozen_rf=np.array([float(r['stage1_probability']) for r in records])
                parity.append(dict(seed=seed,fold=f,year=int(yr),RF_max_difference=float(np.max(np.abs(part[:,1]-frozen_rf)))))
                if parity[-1]['RF_max_difference']>1e-8:raise RuntimeError('BLOCKED_BY_CONTRACT: RF baseline parity failed')
                for j,anchor in enumerate(NAMES):
                    p=part[:,j]; score=scores(part,j)
                    masks={k:score[k]>=obj['rules'][anchor][k] for k in ('STD25_REF','OTHER3_OP25_REF','ANCHOR_UNC25_REF')}
                    unc=uncertainty(p)
                    for key in ('STD25_REF','OTHER3_OP25_REF'):
                        masks[key.replace('_REF','_MATCHED_UNC')]=top(sha,unc,int(masks[key].sum()))
                    for strategy in STRATEGIES:
                        r=masks[strategy]
                        row=dict(seed=seed,year=int(yr),fold=f,anchor=anchor,strategy=strategy,**metrics(yy,p,expert,r))
                        result.append(row)
                        for k,s in enumerate(sha):
                            predictions.append(dict(seed=seed,year=int(yr),fold=f,sha256=str(s),anchor=anchor,strategy=strategy,
                                                    label=int(yy[k]),stage1_probability=float(p[k]),expert_probability=float(expert[k]),
                                                    final_probability=float(expert[k] if r[k] else p[k]),routed=int(r[k]),
                                                    std_score=float(score['STD25_REF'][k]),other3_opposition=float(score['OTHER3_OP25_REF'][k])))
                    stage.append(dict(seed=seed,year=int(yr),fold=f,anchor=anchor,
                                      n=len(yy),BA=float(balanced_accuracy_score(yy,p>=.5)),
                                      AUC=float(roc_auc_score(yy,p)),F1=float(f1_score(yy,p>=.5)),
                                      ACC=float(accuracy_score(yy,p>=.5))))
    # Aggregate source folds by seed, anchor, strategy before comparing years.
    source=[r for r in result if r['year']==2017]
    future=[r for r in result if r['year']!=2017]
    for seed in SEEDS:
        for anchor in NAMES:
            for strategy in STRATEGIES:
                a=[r for r in source if (r['seed'],r['anchor'],r['strategy'])==(seed,anchor,strategy)]
                assert len(a)==5
                selected=[r for r in predictions if (r['seed'],r['year'],r['anchor'],r['strategy'])==(seed,2017,anchor,strategy)]
                yy=np.array([r['label'] for r in selected]);p=np.array([r['stage1_probability'] for r in selected]);e=np.array([r['expert_probability'] for r in selected]);route=np.array([bool(r['routed']) for r in selected])
                future.append(dict(seed=seed,year=2017,fold='OOF',anchor=anchor,strategy=strategy,**metrics(yy,p,e,route)))
    write_csv('yearly_metrics.csv',future)
    write_csv('source_fold_metrics.csv',source)
    write_csv('per_apk_predictions.csv',predictions)
    write_csv('baseline_parity.csv',parity)
    write_csv('stage1_per_fold_year.csv',stage)
    summary=[]
    for year in (2017,2018,2019,2020,2021):
        for anchor in NAMES:
            values=[]
            for seed in SEEDS:
                t=[r for r in predictions if r['seed']==seed and r['year']==year and r['anchor']==anchor and r['strategy']=='STD25_REF']
                assert len(t)==(799 if year==2017 else {2018:793,2019:399,2020:397,2021:400}[year])
                yy=np.array([r['label'] for r in t]);pp=np.array([r['stage1_probability'] for r in t])
                values.append(dict(BA=float(balanced_accuracy_score(yy,pp>=.5)),
                                   AUC=float(roc_auc_score(yy,pp)),
                                   F1=float(f1_score(yy,pp>=.5)),
                                   ACC=float(accuracy_score(yy,pp>=.5))))
            summary.append(dict(year=year,anchor=anchor,n=len(t),
                                BA=float(np.mean([r['BA'] for r in values])),
                                AUC=float(np.mean([r['AUC'] for r in values])),
                                F1=float(np.mean([r['F1'] for r in values])),
                                ACC=float(np.mean([r['ACC'] for r in values])),
                                seeds=','.join(map(str,SEEDS))))
    write_csv('stage1_summary.csv',summary)
    print('MFD_EVALUATION_DONE metrics='+str(len(future))+' predictions='+str(len(predictions)),flush=True)

def report():
    rows=read_csv(OUT/'yearly_metrics.csv')
    lines=['# CARE-Droid Manifest四成员第一层与分歧路由实验','',
           '四个第一层分别为LR/RF/XGB/SVM，全部只读取同一Manifest135维。2017固定外层OOF，2018—2021仅2017训练后冻结评估。',
           '主要路由STD25_REF为四成员概率标准差达到2017交叉拟合25%参考阈值；OTHER3_OP25_REF为其它三成员反对当前第一层决定的平均概率。ANCHOR_UNC25_REF是当前第一层不确定性对照。',
           'MATCHED_UNC对照在同一年/折送入完全相同数量的APK；所有路由仅使用第一层/委员概率，不读取C43结果或测试标签。',
           '', '|年份|第一层|路由|深查占比|Stage-I BA|Final BA|错误覆盖|Fixed|Broken|Net|',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for year in (2017,2018,2019,2020,2021):
        for anchor in NAMES:
            for strategy in STRATEGIES:
                a=[r for r in rows if int(r['year'])==year and r['anchor']==anchor and r['strategy']==strategy]
                n=np.mean([int(r['n']) for r in a]);routed=np.mean([int(r['routed']) for r in a]);errors=np.mean([int(r['stage1_errors']) for r in a]);capt=np.mean([int(r['routed_stage1_errors']) for r in a])
                lines.append(f"|{year}|{anchor}|{strategy}|{routed/n:.1%}|{np.mean([float(r['stage1_BA']) for r in a]):.2%}|{np.mean([float(r['BA']) for r in a]):.2%}|{capt/errors:.1%}|{np.mean([int(r['fixed']) for r in a]):.1f}|{np.mean([int(r['broken']) for r in a]):.1f}|{np.mean([int(r['net']) for r in a]):.1f}|")
    lines += ['', '## 证据边界',
              '这是一轮预定义的开发数据描述性实验。四模型同用Manifest特征，不会新增DEX提取成本，但四次模型推理与校准成本需要另行实测。标准差或投票分歧不是错误概率，也不保证C43会修复。',
              '四成员共享同一特征空间，无法解决Manifest共同盲区；四种第一层并列评估不意味着根据未来年份选出的最好者具有无偏性能。',
              '未来年份已经被历史实验反复观察，不能根据此表选择路由阈值、成员、模型或预算，然后再把同一未来数据称作最终验证。',
              '本实验没有修改正式模型或历史数据。']
    with (OUT/'final_report.md').open('x',encoding='utf-8') as h:h.write('\n'.join(lines)+'\n')

def main():
    if (OUT/'final_decision.json').exists():raise RuntimeError('completed run preserved; use a fresh output directory')
    preflight()
    ids,y,fold,x,fi,years,z=inputs()
    windows={}
    for seed in SEEDS:
        for f in (1,2,3,4,5,0):
            windows[(seed,f)]=fit_window(seed,f,ids,y,fold,x,fi,z)
    if not (OUT/'all_models_and_rules_frozen.json').exists():
        write_json('all_models_and_rules_frozen.json',{'status':'PASS','windows':18,'future_labels_read_before_freeze':False})
    evaluate(ids,y,fold,fi,years,windows)
    report()
    check=integrity();write_json('asset_integrity_check.json',check)
    if check['status']!='PASS':raise RuntimeError('historical input assets changed')
    write_json('final_decision.json',{'status':'FOUR_MANIFEST_ANCHORS_DISAGREEMENT_AUDIT_COMPLETE',
                                      'integrity':'PASS','future_model_selection':False,'new_APK_scans':0,
                                      'new_C43_training':False,'next_action':'Review all four anchors and route controls; do not select a winner using historical future years.'})
    print('MFD_DONE '+str(OUT),flush=True)

if __name__=='__main__':main()
