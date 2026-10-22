#!/usr/bin/env python3
"""Manifest routing-only diagnosis. Reads frozen features/Stage-I predictions; never reads APKs or uses expert scores. Writes only MRC_OUT. Source thresholds strictly nested; future labels evaluation-only."""
import argparse,csv,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
ROOT=Path('/home/boyuanjie/projects/tasa-droid')
OUT=Path(os.environ['MRC_OUT']).resolve()
if OUT.parent!=ROOT/'outputs/care_manifest_routing_coverage_v1' or not OUT.name.startswith('run_'):raise RuntimeError('unsafe output')
sys.path.insert(0,str(ROOT/'scripts/care-lite-stage-v1'))
import common as lc
sys.path.insert(0,str(ROOT/'scripts/care-p31-stage1-rf-selection-source-v1'))
from stage1_core import fit_calibrated_member
OLD=ROOT/'outputs/care_manifest_selective_validation_v1/run_20261018_205020'
FUT=ROOT/'outputs/care_manifest_frozen_future_v1/run_20261018_234850'
LABELS=lc.HDBC/'future2018_2021_predictions.csv'
SEEDS=(20261018,20261019,20261020)
def save(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def read(name):return json.loads((OUT/name).read_text())
def table(name,rs):
    rs=list(rs);p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    cols=list(dict.fromkeys(k for r in rs for k in r)) or ['status']
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,cols);w.writeheader();w.writerows(rs)
def digest(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest())
def check():
    before=read('asset_manifest_before.json');changed=[r['path'] for r in before if digest(r['path'])['sha256']!=r['sha256']]
    return dict(status='FAIL' if changed else 'PASS',changed=changed,asset_count=len(before),scope='Used feature shards, Stage-I probabilities, label source and implementation files only; no APKs/checkpoints opened.')
def inputs():
    _,p=lc.source_contract();ids=np.array(sorted(p));y=np.array([int(p[s]['label']) for s in ids]);fold=np.array([int(p[s]['fold']) for s in ids])
    a,missing=lc.load_shards('source');b,fmissing=lc.load_shards('future');fi=np.array(sorted(b));lin=lc.readcsv(lc.FUTURE_LINEAGE);ymap={r['sha256']:int(r['year']) for r in lin}
    assert not missing and not fmissing and len(a)==799 and len(b)==1989 and not set(ids)&set(fi) and set(fi)==set(ymap)
    x=lc.matrix([a[s] for s in ids],'L1_MANIFEST');z=lc.matrix([b[s] for s in fi],'L1_MANIFEST');years=np.array([ymap[s] for s in fi])
    assert x.shape==(799,135) and z.shape==(1989,135)
    assert {int(yr):int((years==yr).sum()) for yr in set(years)}=={2018:793,2019:399,2020:397,2021:400}
    return ids,y,fold,x,fi,years,z
def reject(p,lo,hi):return (p>=lo)&(p<hi)
def cutoff(p,k,side):
    # Half-open [lo,hi) makes the zero-width refusal region exactly empty.
    v=np.sort(p[p<.5])[::-1] if side=='low' else np.sort(p[p>=.5])
    if k<=0 or len(v)==0:return .5
    if k>=len(v):return 0. if side=='low' else float(np.nextafter(1.,2.))
    return float((v[k-1]+v[k])/2)
def thresholds(p,y):
    err=(p>=.5)!=y;u=1-2*np.abs(p-.5);out={}
    for budget in (.1,.25):
        suffix=str(round(100*budget));k=round(len(p)*budget)
        tau=float(np.sort(u)[::-1][max(0,k-1)])
        out['ABS'+suffix]=dict(tau=tau,source_reference_budget=budget,source_routed=int((u>=tau).sum()),source_captured=int(((u>=tau)&err).sum()),source_ties_may_exceed_reference=True)
        candidates=[]
        for allocation in np.linspace(0,1,11):
            kl=int(round(k*allocation));kh=k-kl;lo=cutoff(p,kl,'low');hi=cutoff(p,kh,'high');r=reject(p,lo,hi)
            if r.sum()<=k:candidates.append((int((r&err).sum()),-int(r.sum()),-abs(allocation-.5),lo,hi))
        candidates.append((0,0,-1.,.5,.5))
        best=max(candidates);lo,hi=best[-2:];r=reject(p,lo,hi)
        out['DUAL'+suffix]=dict(low=lo,high=hi,source_reference_budget=budget,source_routed=int(r.sum()),source_captured=int((r&err).sum()),allocation_grid='0,.1,...,1; maximize captured source crossfit errors; tie fewer routed then symmetric')
    return out
def top(ids,p,k):
    order=np.lexsort((ids,-(1-2*np.abs(p-.5))));r=np.zeros(len(ids),bool);r[order[:k]]=True;return r
def masks(ids,p,lr,rules):
    out={'TOP10':top(ids,p,round(len(ids)*.1)),'TOP25':top(ids,p,round(len(ids)*.25))}
    for name,v in rules.items():out[name]=(1-2*np.abs(p-.5)>=v['tau']) if name.startswith('ABS') else reject(p,v['low'],v['high'])
    out['DISAGREE']=(p>=.5)!=(lr>=.5)
    out['ABS10_OR_DISAGREE']=out['ABS10']|out['DISAGREE']
    return out
def preflight():
    ids,y,fold,x,fi,years,z=inputs()
    p=np.array([.01,.1,.2,.4,.49,.5,.51,.6,.8,.9,.99]);yy=np.array([0,1,0,1,0,1,0,1,1,0,1]);rules=thresholds(p,yy)
    r=masks(np.array([str(i) for i in range(len(p))]),p,1-p,rules)
    assert not reject(p,.5,.5).any() and r['DISAGREE'].sum()==10 and np.all(~r['ABS10']|r['ABS10_OR_DISAGREE'])
    for n,v in rules.items():
        if n.startswith('DUAL'):assert reject(p,v['low'],v['high']).sum()<=round(len(p)*v['source_reference_budget'])
    paths=[lc.SOURCE_LINEAGE,lc.SOURCE_PRED,lc.FUTURE_LINEAGE,LABELS,ROOT/'scripts/care-lite-stage-v1/common.py',ROOT/'scripts/care-p31-stage1-rf-selection-source-v1/stage1_core.py']
    paths+=list((lc.OUT/'shards/source').glob('*.json'))+list((lc.OUT/'shards/future').glob('*.json'))
    paths+=list((OLD/'fold_predictions').glob('ManifestOnly_*.json'))+list((FUT/'predictions').glob('ManifestOnly_*.json'))+list(Path(__file__).parent.glob('*'))
    assert len(list((OLD/'fold_predictions').glob('ManifestOnly_*.json')))==15
    for seed in SEEDS:
        for f in range(1,6):
            obj=json.loads((OLD/f'fold_predictions/ManifestOnly_{seed}_{f}.json').read_text());assert obj['test_sha']==list(ids[fold==f]) and obj['train_sha']==list(ids[fold!=f])
        obj=json.loads((FUT/f'predictions/ManifestOnly_{seed}.json').read_text());assert obj['sha256']==list(fi)
    save('asset_manifest_before.json',[digest(p) for p in sorted(set(paths)) if p.is_file()])
    save('protocol.json',dict(status='FROZEN',seeds=list(SEEDS),features='Manifest135 only',StageI='unchanged prior RF probabilities',StageII='NOT INVOKED, probabilities not used, fixed/broken not computed',strategies=list(r),threshold_reference_budgets=[.1,.25],dual_allocation_grid=list(np.linspace(0,1,11)),objective='maximize captured training-crossfit Stage-I errors under source reference cap; not predicted expert benefit',source_protocol='same five outer folds; generate fully calibrated 3fold crossfit RF/LR within each outer train; thresholds see outer-train only',future_protocol='rules fit on 2017 crossfit only; frozen full2017 Stage-I probabilities; LR trained/calibrated on2017; evaluate every future year separately',labels_boundary='future label table projected to SHA/year/label only after predictions/rules frozen; no expert scores consumed',cost_control='same NUMBER routed uncertainty comparator per evaluation fold/year; not equal actual runtime',no_APK_reads=True,no_graph_features=True,no_automatic_further_training=True,interpretation='historical development diagnosis, not independent blind testing; 10/25 suffix on ABS/DUAL means source reference budget, NOT fixed future fraction'))
    print('MRC_PREFLIGHT_PASS source=799 future=1989 StageII_DISABLED',flush=True)
def build():
    ids,y,fold,x,fi,years,z=inputs();timing=[]
    for seed in SEEDS:
        for f in [1,2,3,4,5,0]:
            name=f'frozen/{seed}_{f}.json'
            if (OUT/name).exists():continue
            tr=np.flatnonzero(fold!=f) if f else np.arange(len(ids));te=np.flatnonzero(fold==f) if f else np.arange(len(fi))
            target=x[te] if f else z;started=time.perf_counter();cf=np.full((len(tr),2),np.nan)
            split_seed=seed+f
            for k,(it,iv) in enumerate(StratifiedKFold(3,shuffle=True,random_state=split_seed).split(x[tr],y[tr]),1):
                for j,model in enumerate(('RF','LR')):
                    cache=f'crossfit/{seed}_{f}_{k}_{model}.json'
                    if not (OUT/cache).exists():
                        vals=fit_calibrated_member(model,x[tr[it]],y[tr[it]],x[tr[iv]],split_seed+100000*k,inner_folds=3,jobs=2)
                        save(cache,dict(train_sha=list(ids[tr[it]]),test_sha=list(ids[tr[iv]]),probability=vals.tolist()))
                    a=read(cache);assert a['test_sha']==list(ids[tr[iv]]) and a['train_sha']==list(ids[tr[it]])
                    cf[iv,j]=a['probability']
                print('MRC_INNER_DONE seed',seed,'outer',f,'inner',k,flush=True)
            assert np.isfinite(cf).all();rule=thresholds(cf[:,0],y[tr])
            # Fit calibrated auxiliary LR on exactly the same legal training fold/window.
            lr=fit_calibrated_member('LR',x[tr],y[tr],target,split_seed,inner_folds=3,jobs=2)
            old=json.loads((OLD/f'fold_predictions/ManifestOnly_{seed}_{f}.json').read_text()) if f else json.loads((FUT/f'predictions/ManifestOnly_{seed}.json').read_text())
            targetids=ids[te] if f else fi;actual=old['test_sha'] if f else old['sha256'];assert actual==list(targetids)
            save(name,dict(seed=seed,fold=f,train_sha=list(ids[tr]),target_sha=list(targetids),rf_probability=old['probability'],lr_probability=lr.tolist(),rules=rule,elapsed_s=time.perf_counter()-started))
            print('MRC_RULES_FROZEN seed',seed,'outer',f,'seconds',round(time.perf_counter()-started,2),flush=True)
    save('all_rules_frozen.json',dict(status='PASS',count=18,future_labels_used_for_selection=False))
def measure(y,p,r,matched):
    err=(p>=.5)!=y;fn=err&(y==1);fp=err&(y==0);n=len(y);k=int(r.sum());capt=int((err&r).sum());e=int(err.sum())
    return dict(n=n,stage1_errors=e,routed=k,route_rate=k/n,captured_errors=capt,error_recall=capt/e if e else None,route_precision=capt/k if k else None,routed_correct=int((~err&r).sum()),unrouted_errors=int((err&~r).sum()),FN=int(fn.sum()),captured_FN=int((fn&r).sum()),FP=int(fp.sum()),captured_FP=int((fp&r).sum()),matched_count_captured=int((err&matched).sum()),captured_gain_vs_same_count=int((err&r).sum()-(err&matched).sum()),new_errors_vs_same_count=int((err&r&~matched).sum()),lost_errors_vs_same_count=int((err&~r&matched).sum()),uniform_random_expected_captured=k*e/n,high_confidence_errors=int((err&(np.maximum(p,1-p)>=.9)).sum()),captured_high_confidence_errors=int((err&r&(np.maximum(p,1-p)>=.9)).sum()))
def evaluate():
    ids,y,fold,x,fi,years,z=inputs()
    # Only project reference labels, never use Stage-II prediction or routed columns.
    labels={}
    with LABELS.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['arm']=='H0_ORIGINAL':
                s=row['sha256'];assert s not in labels;labels[s]=(int(row['year']),int(row['label']))
    assert set(labels)==set(fi)
    fy=np.array([labels[s][1] for s in fi]);assert all(labels[s][0]==int(yr) for s,yr in zip(fi,years))
    results=[];per=[];detail=[];threshold_rows=[]
    for seed in SEEDS:
        q=np.full(len(ids),np.nan);lr=q.copy();routes={};matches={}
        for f in range(1,6):
            obj=read(f'frozen/{seed}_{f}.json');ix=np.flatnonzero(fold==f);p=np.array(obj['rf_probability']);lp=np.array(obj['lr_probability']);q[ix]=p;lr[ix]=lp
            rr=masks(ids[ix],p,lp,obj['rules'])
            for name,r in rr.items():
                routes.setdefault(name,np.zeros(len(ids),bool))[ix]=r;matched=top(ids[ix],p,int(r.sum()));matches.setdefault(name,np.zeros(len(ids),bool))[ix]=matched
                detail.append(dict(seed=seed,year=2017,fold=f,strategy=name,**measure(y[ix],p,r,matched)))
            for name,rule in obj['rules'].items():threshold_rows.append(dict(seed=seed,outer_fold=f,strategy=name,**rule))
        for name,r in routes.items():
            results.append(dict(seed=seed,year=2017,strategy=name,**measure(y,q,r,matches[name])))
            for i,s in enumerate(ids):per.append(dict(seed=seed,year=2017,sha256=s,fold=int(fold[i]),strategy=name,label=int(y[i]),stage1_probability=float(q[i]),aux_LR_probability=float(lr[i]),routed=int(r[i]),matched_count_routed=int(matches[name][i])))
        obj=read(f'frozen/{seed}_0.json');q=np.array(obj['rf_probability']);lr=np.array(obj['lr_probability'])
        for name,rule in obj['rules'].items():threshold_rows.append(dict(seed=seed,outer_fold='FULL2017',strategy=name,**rule))
        for yr in (2018,2019,2020,2021):
            ix=np.flatnonzero(years==yr);rr=masks(fi[ix],q[ix],lr[ix],obj['rules'])
            for name,r in rr.items():
                matched=top(fi[ix],q[ix],int(r.sum()));results.append(dict(seed=seed,year=yr,strategy=name,**measure(fy[ix],q[ix],r,matched)))
                for j,i in enumerate(ix):per.append(dict(seed=seed,year=yr,sha256=fi[i],fold='FROZEN2017',strategy=name,label=int(fy[i]),stage1_probability=float(q[i]),aux_LR_probability=float(lr[i]),routed=int(r[j]),matched_count_routed=int(matched[j])))
    table('routing_yearly_metrics.csv',results);table('source_fold_metrics.csv',detail);table('per_apk_routing.csv',per);table('frozen_thresholds.csv',threshold_rows)
    summary=[]
    for yr in (2017,2018,2019,2020,2021):
        for name in routes:
            rr=[r for r in results if r['year']==yr and r['strategy']==name]
            summary.append(dict(year=yr,strategy=name,mean_route_rate=float(np.mean([r['route_rate'] for r in rr])),mean_error_recall=float(np.mean([r['error_recall'] for r in rr])),mean_captured_errors=float(np.mean([r['captured_errors'] for r in rr])),mean_routed_correct=float(np.mean([r['routed_correct'] for r in rr])),same_count_capture_gains=','.join(str(r['captured_gain_vs_same_count']) for r in rr)))
    table('routing_seed_summary.csv',summary)
    lines=['# Manifest路由错误覆盖审计','', '只运行第一层与廉价路由；不执行第二层，不报告Fixed/Broken或最终两层BA。2017严格外层OOF；阈值由每个outer-train内部重新交叉拟合产生。未来统一用全部2017交叉拟合确定的规则，禁止用未来标签选择阈值。','', 'ABS10/25和DUAL10/25后缀是2017训练参照预算，未来实际深查比例不固定。TOP10/25才是每批固定比例。DUAL只在11个预注册分配候选中选择训练交叉拟合错误捕获最多的区间；不估计第二层收益。DISAGREE是RF/LR分类分歧，ABS10_OR_DISAGREE是固定并集对照。','', '|年份|策略|平均深查比例|平均错误召回|平均捕获错误|平均送入正确APK|同数量对照捕获增量（三种子）|','|---|---|---:|---:|---:|---:|---|']
    for r in summary:lines.append(f"|{r['year']}|{r['strategy']}|{r['mean_route_rate']:.2%}|{r['mean_error_recall']:.2%}|{r['mean_captured_errors']:.2f}|{r['mean_routed_correct']:.2f}|{r['same_count_capture_gains']}|")
    lines+=['','## 必须保留的边界','', '更多捕获不等于更会选择：必须结合送入总数、正确APK数和同数量不确定度对照看。对称ABS阈值与不确定度排序在同数量下通常相同，它改变数量而非排序；分数并列时因SHA打破并列可能不同。','', '同数量不是同实际分析成本，APK深查耗时不同；本轮未测端到端延迟。新捕获错误是否能被第二层修复、正确APK是否会被误伤，均未测试。','', '此前年份已用于开发，因此是历史开发诊断而非独立盲测。三种子不是独立APK；不将种子合并放大样本量。高置信度>=0.9仅为预注册描述，不能把校准分数视为真实错误概率。', '', '原RF及原始数据未改变。未来结果不触发阈值重搜、专家训练或后续自动运行。']
    with (OUT/'final_report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    return dict(status='ROUTING_COVERAGE_EVALUATION_COMPLETE',years=[2017,2018,2019,2020,2021],StageII_invoked=False,expert_scores_used=False,correction_capacity='UNTESTED',deployable_router_selected=False,future_tuning=False,next_action='Review yearly capture versus added correct APKs and same-count baseline; no automatic classification or tuning.')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    if (OUT/'final_decision.json').exists():raise RuntimeError('completed run preserved')
    if not (OUT/'protocol.json').exists():preflight()
    elif check()['status']!='PASS':raise RuntimeError('assets changed')
    if a.preflight_only:return
    t=time.perf_counter();failure=None;decision=None
    try:
        if not (OUT/'all_rules_frozen.json').exists():build()
        decision=evaluate()
    except Exception as exc:failure=repr(exc);traceback.print_exc()
    status=check();save('integrity_'+str(time.time_ns())+'.json',status)
    if failure or status['status']!='PASS':save('failure_'+str(time.time_ns())+'.json',dict(error=failure,integrity=status));raise RuntimeError(failure or 'integrity failure')
    save('baseline_integrity_check.json',status);decision.update(elapsed_s=time.perf_counter()-t,asset_integrity=status['status']);save('final_decision.json',decision);print('MRC_DONE',OUT,flush=True)
if __name__=='__main__':main()
