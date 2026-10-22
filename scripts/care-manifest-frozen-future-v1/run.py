#!/usr/bin/env python3
"""2017-only first-stage training and frozen 2018-2021 evaluation. Reads historical features/predictions, no APKs. Writes only MFF_OUT; never alters or trains C43."""
import argparse, ast, csv, hashlib, json, os, pickle, sys, time, traceback
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
BASE=ROOT/'outputs/care_manifest_frozen_future_v1'
OUT=Path(os.environ['MFF_OUT']).resolve()
if OUT.parent!=BASE or not OUT.name.startswith('run_'):raise RuntimeError('invalid new output directory')
sys.path.insert(0,str(ROOT/'scripts/care-lite-stage-v1'))
import common as lc
sys.path.insert(0,str(ROOT/'scripts/care-p31-stage1-rf-selection-source-v1'))
from stage1_core import make_member,raw_score,fit_calibrated_member,MEMBERS
SEEDS=(20261018,20261019,20261020)
ARMS=('ManifestOnly','RefPresence31')
BUDGETS=(.10,.25)
YEARS=(2018,2019,2020,2021)

def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def table(name,rows):
    rows=list(rows);p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,columns);w.writeheader();w.writerows(rows)
def digest(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)
def matrix(shards,ids,arm):
    x=lc.matrix([shards[s] for s in ids],'L1_MANIFEST')
    if arm=='RefPresence31':x=np.c_[x,np.array([np.array(shards[s]['role_ref_count'])>0 for s in ids],float)]
    assert x.shape==(len(ids),135 if arm=='ManifestOnly' else 166) and np.isfinite(x).all()
    return x
def fit(x,y,seed):
    # Exact seed offsets and calibration of unchanged stage1_core.fit_calibrated_member.
    j=MEMBERS.index('RF');oof=np.full(len(y),np.nan)
    for k,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(x,y),1):
        m=make_member('RF',seed+10000*j+100*k,2);m.fit(x[tr],y[tr]);oof[va]=raw_score(m,x[va])
    assert np.isfinite(oof).all()
    c=LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000,random_state=seed+10000*j+999);c.fit(oof[:,None],y)
    m=make_member('RF',seed+10000*j,2);m.fit(x,y)
    return m,c
def infer(bundle,x):return np.clip(bundle[1].predict_proba(raw_score(bundle[0],x)[:,None])[:,1],1e-8,1-1e-8)
def load_inputs():
    _,p=lc.source_contract();ids=np.array(sorted(p));y=np.array([int(p[s]['label']) for s in ids])
    source,miss=lc.load_shards('source');future,fmiss=lc.load_shards('future')
    if miss or fmiss or set(source)!=set(ids) or len(future)!=1989:raise RuntimeError('feature coverage')
    fi=np.array(sorted(future));lin=lc.readcsv(lc.FUTURE_LINEAGE);lookup={r['sha256']:r for r in lin}
    if len(lin)!=1989 or set(lookup)!=set(fi) or set(ids)&set(fi):raise RuntimeError('SHA duplicate/overlap')
    yr=np.array([int(lookup[s]['year']) for s in fi])
    if {a:int((yr==a).sum()) for a in YEARS}!={2018:793,2019:399,2020:397,2021:400}:raise RuntimeError('year counts')
    return ids,y,source,fi,yr,future
def integrity():
    a=read('asset_manifest_before.json');changed=[r['path'] for r in a if digest(r['path'])['sha256']!=r['sha256']]
    return dict(status='PASS' if not changed else 'FAIL',changed=changed,asset_count=len(a),scope='Used source/future shards, lineages, frozen prediction tables and protocol/dependency scripts; no original APK or C43 checkpoint opened.')
def preflight():
    ids,y,source,fi,yr,future=load_inputs()
    frozen=json.loads((lc.HDBC/'final2017_frozen.json').read_text())
    if frozen['train_years']!=[2017] or frozen['original_C43']!='SCBE M0 T1':raise RuntimeError('expert provenance')
    # Synthetic parity test, not model selection; no future labels involved.
    rng=np.random.RandomState(17);x=rng.normal(size=(36,5));yy=np.tile([0,1],18);z=rng.normal(size=(8,5))
    a=infer(fit(x,yy,17),z);b=fit_calibrated_member('RF',x,yy,z,17,inner_folds=3,jobs=2)
    if not np.allclose(a,b,rtol=0,atol=1e-12):raise RuntimeError('saved calibrator parity')
    for arm in ARMS:matrix(source,ids,arm);matrix(future,fi,arm)
    paths=[lc.SOURCE_LINEAGE,lc.SOURCE_PRED,lc.FUTURE_LINEAGE,lc.HDBC/'final2017_frozen.json',lc.HDBC/'future2018_2021_predictions.csv',ROOT/'scripts/care-hdbc-v1/main.py',ROOT/'scripts/care-lite-stage-v1/common.py',ROOT/'scripts/care-p31-stage1-rf-selection-source-v1/stage1_core.py']
    paths+=list((lc.OUT/'shards/source').glob('*.json'))+list((lc.OUT/'shards/future').glob('*.json'))+list(Path(__file__).parent.glob('*'))
    save('asset_manifest_before.json',[digest(p) for p in sorted(set(paths)) if p.is_file()])
    save('protocol.json',dict(status='FROZEN_BEFORE_TRAINING',train_years=[2017],train_n=799,future_n=1989,years=list(YEARS),arms=list(ARMS),seeds=list(SEEDS),budgets=list(BUDGETS),threshold=.5,router='1-2*abs(p-.5); top fraction per year; SHA tie break; round(N*budget); NOT MeanOpposition',expert='HDBC H0 mean of three SCBE M0 T1 experts trained only on2017, frozen predictions reused',calibration='3fold training-only OOF Platt; unchanged stage1_core RF300',new_APKs=False,within_year_training=False,rolling_training=False,model_selection=False,cost_claim='No new extraction or end-to-end timing in this run',evidence_boundary='Historical repeatedly inspected development years, not independent blind testing; seeds are not independent samples.',selftest_probability_max_diff=float(np.max(np.abs(a-b)))))
    print('MFF_PREFLIGHT_PASS source=799 future=1989 calibration_parity=PASS',flush=True)
def train_predict():
    ids,y,source,fi,yr,future=load_inputs();audits=[]
    for arm in ARMS:
        x=matrix(source,ids,arm);z=matrix(future,fi,arm)
        for seed in SEEDS:
            name=f'checkpoints/{arm}_{seed}.pkl';p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                with p.open('rb') as f:obj=pickle.load(f)
                if obj['train_sha']!=list(ids) or obj['seed']!=seed or obj['arm']!=arm:raise RuntimeError('checkpoint resume contract')
            else:
                start=time.perf_counter();bundle=fit(x,y,seed);obj=dict(arm=arm,seed=seed,train_sha=list(ids),train_years=[2017],bundle=bundle,fit_s=time.perf_counter()-start)
                with p.open('xb') as f:pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
            pr=f'predictions/{arm}_{seed}.json'
            if not (OUT/pr).exists():
                start=time.perf_counter();q=infer(obj['bundle'],z);save(pr,dict(sha256=list(fi),probability=q.tolist(),inference_s=time.perf_counter()-start))
            r=read(pr)
            if r['sha256']!=list(fi):raise RuntimeError('prediction SHA alignment')
            audits.append(dict(arm=arm,seed=seed,train_n=len(ids),width=x.shape[1],fit_s=obj['fit_s'],inference_s=r['inference_s'],checkpoint_sha256=digest(p)['sha256']))
            print('MFF_TRAIN2017_PREDICT_DONE',arm,seed,flush=True)
    table('training_audit.csv',audits)
    save('models_frozen_before_evaluation.json',dict(status='FROZEN',models=audits,future_labels_used_to_fit=False))
def evaluate():
    _,_,_,ids,years,_=load_inputs();e=lc.frozen_expert_future()
    if set(e)!=set(ids):raise RuntimeError('expert coverage')
    for s,yr in zip(ids,years):
        if int(e[s]['year'])!=yr or e[s]['label'] not in ('0','1'):raise RuntimeError('expert year/label')
    y=np.array([int(e[s]['label']) for s in ids]);ep=np.array([float(e[s]['expert_probability']) for s in ids]);assert np.isfinite(ep).all() and np.all((ep>=0)&(ep<=1))
    metrics=[];per=[];pooled=[];macro=[]
    for arm in ARMS:
        for seed in SEEDS:
            q=np.array(read(f'predictions/{arm}_{seed}.json')['probability']);assert np.isfinite(q).all()
            previous=None
            for budget in BUDGETS:
                allroute=np.zeros(len(ids),bool)
                for yr in YEARS:
                    ix=np.flatnonzero(years==yr);route=lc.route_by_score(ids[ix],1-2*np.abs(q[ix]-.5),budget);allroute[ix]=route
                    m,final=lc.cascade(y[ix],q[ix],ep[ix],route);st=lc.metrics(y[ix],q[ix]);before=(q[ix]>=.5)==y[ix];after=(final>=.5)==y[ix]
                    assert m['errors']==st['errors']-m['net'] and np.array_equal(final[~route],q[ix][~route])
                    fixed=~before&after;broken=before&~after
                    row=dict(arm=arm,seed=seed,budget=budget,year=yr,stage1_BA=st['BA'],stage1_AUC=st['AUC'],stage1_F1=st['F1'],stage1_FN=st['FN'],stage1_FP=st['FP'],gain_pp=100*(m['BA']-st['BA']),fixed_FN=int((fixed&(y[ix]==1)).sum()),fixed_FP=int((fixed&(y[ix]==0)).sum()),broken_TN=int((broken&(y[ix]==0)).sum()),broken_TP=int((broken&(y[ix]==1)).sum()),unrouted_errors=int((~route&~before).sum()),routed_unfixed=int((route&~before&~after).sum()),**m)
                    metrics.append(row)
                    for j,i in enumerate(ix):per.append(dict(arm=arm,seed=seed,budget=budget,year=int(yr),sha256=ids[i],label=int(y[i]),stage1_probability=float(q[i]),expert_probability=float(ep[i]),routed=int(route[j]),final_probability=float(final[j]),fixed=int(fixed[j]),broken=int(broken[j])))
                if previous is not None:assert np.all(~previous|allroute)
                previous=allroute
                m,_=lc.cascade(y,q,ep,allroute);pooled.append(dict(arm=arm,seed=seed,budget=budget,scope='pooled 1989 APK; not year macro',**m))
                rr=[r for r in metrics if r['arm']==arm and r['seed']==seed and r['budget']==budget];macro.append(dict(arm=arm,seed=seed,budget=budget,stage1_macro_BA=float(np.mean([r['stage1_BA'] for r in rr])),final_macro_BA=float(np.mean([r['BA'] for r in rr])),positive_net_years=sum(r['net']>0 for r in rr),fixed=sum(r['fixed'] for r in rr),broken=sum(r['broken'] for r in rr),net=sum(r['net'] for r in rr)))
    table('future_yearly_metrics.csv',metrics);table('future_per_apk_predictions.csv',per);table('future_pooled_metrics.csv',pooled);table('future_macro_metrics.csv',macro)
    lines=['# Manifest第一层＋冻结Direct-C43跨年份诊断','', '2017全部799个APK训练第一层，训练内三折校准；2018—2021逐年评价。不是年份内部交叉验证，也不是滚动训练。仅复用已有特征，无新增APK、无第二层重训练。','', '第一层三个随机种子分别报告；没有挑选最好种子，也没有新增集成。路由为概率不确定度，非MeanOpposition；阈值0.5，预算固定10%/25%。第二层复用2017-only T1冻结专家概率。','', '|候选|种子|年份|预算|第一层BA|两层BA|提升pp|Fixed|Broken|Net|捕获错误/第一层错误|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in metrics:lines.append(f"|{r['arm']}|{r['seed']}|{r['year']}|{r['budget']:.0%}|{r['stage1_BA']:.6f}|{r['BA']:.6f}|{r['gain_pp']:+.3f}|{r['fixed']}|{r['broken']}|{r['net']}|{r['routed_stage1_errors']}/{r['stage1_errors']}|")
    lines+=['','## 边界','', '这些年份已反复用于开发，本轮是冻结配置的历史跨年诊断，不是独立盲测。仅统计每年实际表现，不用未来标签改变模型、预算、阈值或选择种子。逐年结果优先；另提供年度宏平均与合并样本指标，二者不混用。种子不能视为独立APK。本轮不验证端到端成本，不重新进行特征提取。','', 'FN/TN和TP/FP条件AUC根据本次第一层预测方向及路由集合计算；不同第一层形成不同集合，不能直接解释为同一批困难样本的排序改善。', '', '未来结果不触发自动重训或新的优化。']
    with (OUT/'final_report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    return dict(status='FROZEN_FUTURE_EVALUATION_COMPLETE',train_years=[2017],evaluated_years=list(YEARS),new_APKs_used=False,C43_retrained=False,within_year_experiment=False,automatic_tuning=False,manifest10_by_seed=[r for r in macro if r['arm']=='ManifestOnly' and r['budget']==.1],interpretation='Descriptive historical development evaluation; no independent generalization or end-to-end cost claim')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    if (OUT/'final_decision.json').exists():raise RuntimeError('completed output preserved')
    if not (OUT/'protocol.json').exists():preflight()
    elif integrity()['status']!='PASS':raise RuntimeError('historical assets changed')
    if args.preflight_only:return
    start=time.perf_counter();failure=None;decision=None
    try:
        if not (OUT/'models_frozen_before_evaluation.json').exists():train_predict()
        print('MFF_EVALUATE_START frozen_models=true',flush=True);decision=evaluate()
    except Exception as exc:failure=repr(exc);traceback.print_exc()
    check=integrity();save('integrity_'+str(time.time_ns())+'.json',check)
    if failure or check['status']!='PASS':
        save('failure_'+str(time.time_ns())+'.json',dict(error=failure,integrity=check));raise RuntimeError(failure or 'integrity failed')
    save('baseline_integrity_check.json',check);decision.update(asset_integrity=check['status'],elapsed_s=time.perf_counter()-start);save('final_decision.json',decision)
    print('MFF_DONE',OUT,flush=True)
if __name__=='__main__':main()
