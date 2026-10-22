#!/usr/bin/env python3
"""Frozen routing + C43 prediction replay and component-cost sensitivity audit.
Reads historical predictions, protocols and timing only; no APK/checkpoint access or fits.
Writes only a fresh MCA_OUT directory. No selection/tuning based on future results.
"""
import os,sys,csv,json,time,hashlib,traceback,argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path('/home/boyuanjie/projects/tasa-droid')
OUT=Path(os.environ['MCA_OUT']).resolve()
assert OUT.parent==ROOT/'outputs/care_manifest_acquisition_cascade_audit_v1' and OUT.name.startswith('run_')
PREV=ROOT/'outputs/care_manifest_support_acquisition_v1/run_20261019_pilot01'
COST=ROOT/'outputs/care_manifest_selective_validation_v1/run_20261018_205020'
FUT=ROOT/'outputs/care_manifest_frozen_future_v1/run_20261018_234850'
sys.path.insert(0,str(ROOT/'scripts/care-lite-stage-v1'))
import common as lc
SEEDS=(20261018,20261019,20261020)
ALPHAS=(0.,.01,.05,.1,.2,.5,1.)
PRIMARY='U_REF10'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def table(n,rows):
    rows=list(rows);cols=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with (OUT/n).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,cols);w.writeheader();w.writerows(rows)
def text(n,s):
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(s)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest())
def integrity():
    rows=read(OUT/'asset_manifest_before.json');bad=[r['path'] for r in rows if digest(r['path'])['sha256']!=r['sha256']]
    return dict(status='FAIL' if bad else 'PASS',changed=bad,asset_count=len(rows),scope='Actual reused routing/prediction/protocol/timing tables and local implementation; previous asset manifest checked before replay. APKs/checkpoints not opened, not a whole-server integrity claim.')
def route_top(ids,p,fold,counts):
    out=np.zeros(len(p),bool)
    for f in sorted(set(fold)):
        ix=np.flatnonzero(fold==f);k=min(len(ix),int(counts[int(f)]));order=np.lexsort((ids[ix],-(1-2*np.abs(p[ix]-.5))))
        out[ix[order[:k]]]=True
    return out
def score(y,p,e,r):
    result,final=lc.cascade(y,p,e,r)
    err=(p>=.5)!=y;ef=(e>=.5)==y
    for label,mask in [('FN',err&(y==1)),('FP',err&(y==0))]:
        result['captured_'+label]=int((r&mask).sum());result['fixed_'+label]=int((r&mask&ef).sum())
    result['broken_TN']=int((r&~err&~ef&(y==0)).sum());result['broken_TP']=int((r&~err&~ef&(y==1)).sum())
    result['unfixed_routed']=int((r&err&~ef).sum());result['unrouted_errors']=int((~r&err).sum())
    assert result['errors']==int(err.sum())-result['net']
    assert result['fixed']==result['fixed_FN']+result['fixed_FP']
    assert result['broken']==result['broken_TN']+result['broken_TP']
    assert np.array_equal(final[~r],p[~r])
    return result,final
def selftest():
    y=np.array([1,0,1,0]);p=np.array([.2,.8,.8,.2]);e=np.array([.8,.2,.2,.8]);r=np.array([1,1,1,0],bool)
    m,z=score(y,p,e,r);assert (m['fixed'],m['broken'],m['net'])==(2,1,1)
    ids=np.array(['b','a','c','d']);f=np.ones(4,int);rt=route_top(ids,np.full(4,.5),f,{1:2});assert rt.tolist()==[True,True,False,False]
    print('MCA_SELFTEST_PASS',flush=True)
def preflight():
    selftest();assert read(PREV/'final_decision.json')['asset_integrity']=='PASS'
    previous=read(PREV/'asset_manifest_before.json')
    changed=[r['path'] for r in previous if digest(r['path'])['sha256']!=r['sha256']]
    if changed:raise RuntimeError('prior assets changed '+repr(changed[:5]))
    frozen=read(lc.HDBC/'final2017_frozen.json')
    assert frozen['train_years']==[2017] and frozen['original_C43']=='SCBE M0 T1'
    paths=[PREV/n for n in ('protocol.json','final_decision.json','asset_manifest_before.json','per_apk_routing.csv','routing_yearly_metrics.csv')]
    paths+=list((PREV/'frozen').glob('*.json'))
    paths+=[lc.SOURCE_PRED,lc.SOURCE_LINEAGE,lc.HDBC/'source2017_oof_predictions.csv',lc.HDBC/'future2018_2021_predictions.csv',lc.HDBC/'final2017_frozen.json']
    paths+=[COST/n for n in ('cost_measurements.csv','cost_summary.csv','source_metrics.csv','protocol.json')]
    paths+=[FUT/'future_yearly_metrics.csv',FUT/'protocol.json',ROOT/'scripts/care-lite-stage-v1/common.py',ROOT/'scripts/care-manifest-selective-validation-v1/worker.py']
    paths+=list(Path(__file__).parent.glob('*'))
    save('asset_manifest_before.json',[digest(p) for p in sorted(set(paths)) if p.is_file()])
    save('protocol.json',dict(status='FROZEN_BEFORE_EXPERT_REPLAY',primary=PRIMARY,secondary='All prior frozen routes retained; no best-route selection',seeds=list(SEEDS),years=[2017,2018,2019,2020,2021],stage1='Unchanged Manifest RF cached probabilities',expert='Source matched outerfold HDBC H0 ensemble; future full2017 SCBE M0 T1 ensemble',threshold=.5,new_fits=0,new_APKs=0,comparisons=['source-frozen ABS10/ABS25','same-number uncertainty per source fold/future year','Manifest-only auxiliary control','fixed Top10/25 parity','equal assumed cost sensitivity'],cost_alpha=list(ALPHAS),alpha_meaning='Additional reference acquisition per candidate / complete stageII per routed APK. Hypothetical homogeneous costs, not actual latency.',cost_proxy='20-APK source component means only, repeats averaged within APK; standalone repeat-Manifest vs reuse-difference scenarios; excludes C43 preprocessing/forward and routing inference; future extrapolation assumption',interpretation='Historical development replay. No future tuning; seeds not independent APKs; no significance claim; no automatic downstream run.'))
    save('previous_integrity_validation.json',dict(status='PASS',count=len(previous)))
    print('MCA_PREFLIGHT_PASS',flush=True)

def inputs():
    _,contract=lc.source_contract();src=lc.frozen_expert_source();fut=lc.frozen_expert_future()
    assert set(src)==set(contract) and len(fut)==1989 and not set(src)&set(fut)
    experts={**src,**fut};groups=defaultdict(list)
    for row in lc.readcsv(PREV/'per_apk_routing.csv'):groups[(int(row['seed']),int(row['year']),row['strategy'])].append(row)
    counts={2017:799,2018:793,2019:399,2020:397,2021:400};allsets={};audits=[]
    for (seed,yr,name),rs in groups.items():
        assert len(rs)==counts[yr] and len({r['sha256'] for r in rs})==len(rs)
        rs.sort(key=lambda r:r['sha256']);key=(seed,yr);identity=[(r['sha256'],r['label'],r['probability'],r['fold']) for r in rs]
        if key in allsets:assert allsets[key]==identity
        else:allsets[key]=identity
        for r in rs:
            e=experts[r['sha256']];assert int(e['label'])==int(r['label']) and int(e['year'])==yr
            ep=float(e['expert_probability']);assert np.isfinite(ep) and 0<=ep<=1
            assert abs(ep-np.mean([float(e['seed_'+str(s)]) for s in (900000,900001,900002)]))<1e-12
            if yr==2017:assert e['split']=='S'+r['fold'] and int(contract[r['sha256']]['fold'])==int(r['fold'])
        ids=[r['sha256'] for r in rs]
        if yr==2017:
            frozen={s:p for f in range(1,6) for s,p in zip(read(PREV/f'frozen/{seed}_{f}.json')['target_sha'],read(PREV/f'frozen/{seed}_{f}.json')['probability'])}
        else:
            a=read(PREV/f'frozen/{seed}_0.json');frozen=dict(zip(a['target_sha'],a['probability']))
        assert all(float(r['probability'])==frozen[r['sha256']] for r in rs)
        audits.append(dict(seed=seed,year=yr,strategy=name,n=len(rs),alignment='PASS'))
    assert len(groups)==3*5*14
    table('alignment_audit.csv',audits)
    return groups,experts

def costs():
    timing=lc.readcsv(COST/'cost_measurements.csv');items=defaultdict(list)
    for r in timing:
        assert r['status']=='PASS';items[(r['sha256'],r['kind'])].append(float(r['wall_s']))
    means={}
    for kind in ('ManifestOnly','RefPresence31','compact','endpoint'):
        vals=[np.mean(v) for (s,k),v in items.items() if k==kind];assert len(vals)==20;means[kind]=float(np.mean(vals))
    graph=means['compact']+means['endpoint']
    save('cost_evidence.json',dict(APK_n=20,component_mean_s=means,graph_component_sum_s=graph,standalone_ref_to_graph_proxy_ratio=means['RefPresence31']/graph,warning='Source sample, OS cache uncontrolled, process startup included, no actual new E2E benchmark. Graph sum excludes background transform/C43 forward. Manifest reuse subtraction is hypothetical, not measured incremental latency.'))
    return means,items

def evaluate():
    groups,experts=inputs();cm,ct=costs();metrics=[];pairs=[];per=[];marginal=[];scenarios=[];proxies=[];samplecost=[];parity=[]
    for (seed,yr,name),rs in sorted(groups.items()):
        ids=np.array([r['sha256'] for r in rs]);y=np.array([int(r['label']) for r in rs]);p=np.array([float(r['probability']) for r in rs]);e=np.array([float(experts[s]['expert_probability']) for s in ids]);f=np.array([int(r['fold']) for r in rs]);r=np.array([int(q['routed']) for q in rs],bool);a=np.array([int(q['acquired']) for q in rs],bool)
        m,final=score(y,p,e,r);base=lc.metrics(y,p);counts={int(k):int(r[f==k].sum()) for k in set(f)};matched=route_top(ids,p,f,counts);ctrl,cfinal=score(y,p,e,matched)
        metrics.append(dict(seed=seed,year=yr,strategy=name,route_rate=float(r.mean()),acquired=int(a.sum()),acquisition_rate=float(a.mean()),stage1_BA=base['BA'],**m))
        pairs.append(dict(seed=seed,year=yr,strategy=name,comparator='SAME_COUNT_UNCERTAINTY',n=len(y),routed=int(r.sum()),BA_delta=m['BA']-ctrl['BA'],fixed_delta=m['fixed']-ctrl['fixed'],broken_delta=m['broken']-ctrl['broken'],net_delta=m['net']-ctrl['net'],candidate_fixed=m['fixed'],candidate_broken=m['broken'],control_fixed=ctrl['fixed'],control_broken=ctrl['broken']))
        before=(p>=.5)==y;expertok=(e>=.5)==y
        for bucket,mask in [('ADDED',r&~matched),('DROPPED',~r&matched),('COMMON',r&matched)]:
            mm,_=score(y,p,e,mask);marginal.append(dict(seed=seed,year=yr,strategy=name,bucket=bucket,n_bucket=int(mask.sum()),stage1_errors_in_bucket=int((mask&~before).sum()),fixed=mm['fixed'],broken=mm['broken'],net=mm['net'],fixed_FN=mm['fixed_FN'],fixed_FP=mm['fixed_FP']))
        for i,s in enumerate(ids):
            state='UNROUTED_CORRECT' if before[i] else 'UNROUTED_ERROR'
            if r[i]:state=('KEPT_CORRECT' if expertok[i] else 'BROKEN') if before[i] else ('FIXED' if expertok[i] else 'UNFIXED')
            per.append(dict(seed=seed,year=yr,fold=int(f[i]),sha256=s,strategy=name,label=int(y[i]),stage1_probability=float(p[i]),expert_probability=float(e[i]),final_probability=float(final[i]),routed=int(r[i]),acquired=int(a[i]),same_count_routed=int(matched[i]),state=state))
        if name in ('U_REF10','US_REF10','U_REF25','US_REF25'):
            for alpha in ALPHAS:
                budget={int(k):min(int((f==k).sum()),int(np.floor(r[f==k].sum()+alpha*a[f==k].sum()))) for k in set(f)}
                mask=route_top(ids,p,f,budget);b,_=score(y,p,e,mask)
                scenarios.append(dict(seed=seed,year=yr,strategy=name,alpha=alpha,candidate_routed=int(r.sum()),acquired=int(a.sum()),control_routed=int(mask.sum()),candidate_cost_units=float(r.sum()+alpha*a.sum()),control_cost_units=int(mask.sum()),BA_delta=m['BA']-b['BA'],net_delta=m['net']-b['net'],fixed_delta=m['fixed']-b['fixed'],broken_delta=m['broken']-b['broken'],scope='HYPOTHETICAL_HOMOGENEOUS_COST_NOT_MEASURED'))
        for mode,increment in [('standalone_ref',cm['RefPresence31']),('hypothetical_manifest_reuse',max(0,cm['RefPresence31']-cm['ManifestOnly']))]:
            proxies.append(dict(seed=seed,year=yr,strategy=name,mode=mode,manifest_component_s=len(y)*cm['ManifestOnly'],extra_reference_component_s=int(a.sum())*increment,graph_component_s=int(r.sum())*(cm['compact']+cm['endpoint']),total_component_proxy_s=len(y)*cm['ManifestOnly']+int(a.sum())*increment+int(r.sum())*(cm['compact']+cm['endpoint']),scope='UNVALIDATED_SOURCE20_MEAN_EXTRAPOLATION_NOT_E2E'))
        if yr==2017:
            sample=[i for i,s in enumerate(ids) if (s,'ManifestOnly') in ct];assert len(sample)==20
            samplecost.append(dict(seed=seed,strategy=name,sampled_APKs=20,sample_acquired=int(a[sample].sum()),sample_routed=int(r[sample].sum()),sample_standalone_component_sum_s=sum(np.mean(ct[(ids[i],'ManifestOnly')])+a[i]*np.mean(ct[(ids[i],'RefPresence31')])+r[i]*(np.mean(ct[(ids[i],'compact')])+np.mean(ct[(ids[i],'endpoint')])) for i in sample),scope='Only20_previous_benchmark_APKs_not_population_estimate'))
        if name=='ABS10':
            old=lc.readcsv(COST/'source_metrics.csv' if yr==2017 else FUT/'future_yearly_metrics.csv')
            for budget in (.1,.25):
                mask=route_top(ids,p,f,{int(k):round(int((f==k).sum())*budget) for k in set(f)});b,_=score(y,p,e,mask)
                candidates=[q for q in old if q.get('arm')=='ManifestOnly' and int(q['seed'])==seed and abs(float(q['budget'])-budget)<1e-9 and (yr==2017 or int(q['year'])==yr)]
                assert len(candidates)==1
                q=candidates[0]
                assert all(abs(float(q[k])-b[k])<1e-10 for k in ('BA','fixed','broken','net'))
                parity.append(dict(seed=seed,year=yr,budget=budget,status='PASS'))
    table('yearly_metrics.csv',metrics);table('same_count_comparison.csv',pairs);table('incremental_route_outcomes.csv',marginal);table('per_apk_predictions.csv',per);table('cost_sensitivity.csv',scenarios);table('component_cost_proxy.csv',proxies);table('sample20_cost_replay.csv',samplecost);table('baseline_replay_parity.csv',parity)
    summary=[]
    for yr in (2017,2018,2019,2020,2021):
        for name in sorted({r['strategy'] for r in metrics}):
            rs=[r for r in metrics if r['year']==yr and r['strategy']==name];pp=[r for r in pairs if r['year']==yr and r['strategy']==name]
            v=dict(year=yr,strategy=name)
            for k in ('stage1_BA','BA','AUC','F1','FN','FP','fixed','broken','net','fixed_FN','fixed_FP','broken_TN','broken_TP','routed','acquired','route_rate','acquisition_rate'):v[k]=float(np.mean([r[k] for r in rs]))
            v['same_count_net_deltas']=','.join(str(r['net_delta']) for r in pp);summary.append(v)
    table('seed_summary.csv',summary)
    lines=['# 补充观测路由与冻结C43净收益核验','', '主要方案U_REF10与次要US_REF10、预算敏感性及全部历史路由一起报告，不根据未来结果挑选参数。第一层分类阈值与C43均为0.5，不重新训练。2017对应原外层OOF专家，2018—2021使用仅2017训练的固定专家。','', '|年份|策略|第一层BA|两层BA|Fixed|Broken|Net|同数量Net增量（三种子）|','|---|---|---:|---:|---:|---:|---:|---|']
    for v in summary:
        if v['strategy'] in ('ABS10','ABS25','SUPPORT10','U_MANIFEST_CTRL10','U_REF10','US_REF10'):
            lines.append(f"|{v['year']}|{v['strategy']}|{v['stage1_BA']:.2%}|{v['BA']:.2%}|{v['fixed']:.2f}|{v['broken']:.2f}|{v['net']:.2f}|{v['same_count_net_deltas']}|")
    lines+=['','## 成本边界','已有20个APK的组件计时，未重新运行APK解析或图构建。component_cost_proxy仅将源域均值外推，未测路由集合条件成本、未来APK耗时、C43背景预处理和前向推理，不能声称完整端到端加速或同实际成本胜出。sample20_cost_replay只适用于这20个历史APK。','', 'cost_sensitivity令alpha为补充观测成本/完整二层成本，按预设0、0.01、0.05、0.1、0.2、0.5、1比较允许概率基线多分析一些APK后的收益。它是假定均匀成本的敏感性分析，不是现实延迟测量。没有使用标签搜索alpha。','', '补充观测策略与同数量概率基线相比仍多付获取成本。若只胜同数量、不能稳健胜成本敏感性对照，仍不足以支持部署。新增错误捕获与最终净纠错分别看；新增/删除路由桶见incremental_route_outcomes.csv。','', '此前年份已多次观察，为开发数据描述性结果。种子只衡量第一层/路由随机性，专家固定；不将三种子当独立APK，不声称显著性。对未知错误根因不作推断。']
    text('final_report.md','\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args()
    if (OUT/'final_decision.json').exists():raise RuntimeError('completed run preserved')
    if not (OUT/'protocol.json').exists():preflight()
    assert integrity()['status']=='PASS'
    if args.preflight_only:return
    start=time.perf_counter()
    try:
        evaluate();a=integrity();save('baseline_integrity_check.json',a);assert a['status']=='PASS'
        save('final_decision.json',dict(status='FROZEN_CASCADE_AND_COMPONENT_COST_AUDIT_COMPLETE',asset_integrity=a['status'],new_training=0,new_APK_reads=0,threshold_tuning=False,full_end_to_end_cost='UNVERIFIED',independent_final_validation=False,elapsed_s=time.perf_counter()-start,next_action='Review yearly net gains and cost sensitivity; no automatic tuning or model replacement.'))
        print('MCA_DONE',OUT,flush=True)
    except Exception as exc:
        save('failure_'+str(time.time_ns())+'.json',dict(error=repr(exc),integrity=integrity()));traceback.print_exc();raise
if __name__=='__main__':main()
