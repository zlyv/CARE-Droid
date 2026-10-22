#!/usr/bin/env python3
"""Fixed source OOF stability and paired static-cost pilot. Reads APKs only for cost; writes isolated MSV_OUT. Never trains C43 or reads future cohorts."""
import argparse, gzip, subprocess, time, traceback, resource, ast
from core import *
sys.path.insert(0,str(ROOT/'scripts/care-p31-stage1-rf-selection-source-v1'))
from stage1_core import fit_calibrated_member
from scipy.stats import binomtest

def preflight():
    lin,p,ids,y,fold,ep,shards=data()
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text())
    # Feature contracts: invoke/body counts can never leak into the cheap arms.
    sid=ids[0];a=features(shards,[sid],'RefPresence31');copy=dict(shards[sid]);copy['dex_meta']=[999999]*5;copy['role_invoke_count']=[999999]*31
    assert np.array_equal(a,features({sid:copy},[sid],'RefPresence31'))
    assert np.array_equal(features(shards,[sid],'ManifestOnly'),lc.matrix([shards[sid]],'L1_MANIFEST'))
    for b,n in [(.1,80),(.25,200)]:assert routes(ids,fold,np.full(799,.5),b).sum()==n
    assert np.all(~routes(ids,fold,ep,.1)|routes(ids,fold,ep,.25))
    selected=[]
    for f in range(1,6):
        order=sorted(ids[fold==f],key=lambda s:(int(shards[s]['apk_bytes']),s))
        for size_bin,group in enumerate(np.array_split(order,4)):
            # No labels/predictions in cost sampling: lowest seeded SHA hash within size stratum.
            sid=min(group,key=lambda s:hashlib.sha256(('MSV_COST_20261018'+s).encode()).hexdigest())
            row=dict(lin[sid]);row['sha256']=sid;row.pop('metadata_label',None)
            row.update(fold=f,size_bin=size_bin,stratum_population=len(group))
            if not Path(row['absolute_path']).is_file():raise RuntimeError('benchmark APK missing '+sid)
            selected.append(row)
    assets=[lc.SOURCE_LINEAGE,lc.SOURCE_PRED,lc.HDBC/'source2017_oof_predictions.csv',lc.P31,lc.MAPPING,PRIORITY,EDGE]
    assets+=list((lc.OUT/'shards/source').glob('*.json'))
    for directory in ['care-lite-stage-v1','care-p31-stage1-rf-selection-source-v1','care-litelocal-p31-diagnostics-v1','care-di25-host-gatedgraph-source-v1']:
        assets+=list((ROOT/'scripts'/directory).glob('*.py'))
    assets+=list((ROOT/'src/tasa_droid').rglob('*.py'))
    assets+=[Path(r['absolute_path']) for r in selected]
    assets+=list(Path(__file__).parent.glob('*'))
    before=[digest(v) for v in sorted(set(assets)) if v.is_file()]
    dump('asset_manifest_before.json',before)
    for r in selected:
        expected=r['sha256'];actual=next(a['sha256'] for a in before if a['path']==r['absolute_path'])
        if actual!=expected:raise RuntimeError('original APK identity differs '+expected)
    dump('cost_sample.json',selected)
    dump('protocol.json',dict(status='FROZEN_BEFORE_NEW_FITS',source_n=799,class_counts=[400,399],fold_sizes=[160,160,160,160,159],seeds=list(SEEDS),arms=list(ARMS),budgets=list(BUDGETS),primary='ManifestOnly Top10%',control='Top25%, RefPresence31; historical P31 RF',threshold=.5,router='1-2*abs(p-0.5), fold-relative top budget; SHA tie break; NOT MeanOpposition',calibration='RF300 + 3fold training-only Platt, unchanged stage1_core',expert='HDBC H0_ORIGINAL same-outer-fold frozen all-APK probabilities, fixed ensemble',future_rows_loaded=0,new_downloaded_APKs_used=0,cost_sample_n=20,cost_selection='5 fixed folds x 4 size quartiles, label-free deterministic hash',cost_timeout_per_component_seconds=1200,cost_memory_limit_gib=20,component_cost_boundary='Manifest, reference, P31, compact and endpoint subprocess wall times. C43 transforms/forward and routing overhead NOT measured: full end-to-end cost UNVERIFIED.',interpretation='Exploratory repeat on previously inspected source. Seeds are not independent APKs. No candidate or budget selection. No automatic future execution.'))
    print('MSV_PREFLIGHT_PASS source=799 sampled_APK=20 future=0',flush=True)

def interval(y,a,b,fold):
    rng=np.random.RandomState(20261018);delta=[]
    strata=[np.flatnonzero((fold==f)&(y==c)) for f in range(1,6) for c in (0,1)]
    for _ in range(2000):
        ix=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]);delta.append(lc.metrics(y[ix],a[ix])['BA']-lc.metrics(y[ix],b[ix])['BA'])
    ca=(a>=.5)==y;cb=(b>=.5)==y;win=int((ca&~cb).sum());loss=int((~ca&cb).sum());ci=np.quantile(delta,[.025,.975])
    return dict(BA_difference=lc.metrics(y,a)['BA']-lc.metrics(y,b)['BA'],CI_low=float(ci[0]),CI_high=float(ci[1]),win=win,loss=loss,mcnemar_p=float(binomtest(win,win+loss,.5).pvalue) if win+loss else 1.)

def source():
    lin,formal,ids,y,fold,ep,shards=data();results=[];folds=[];predictions=[];statistics=[]
    baseline=np.array([float(formal[s]['p_P31__RF']) for s in ids]);dump('historical_P31_RF_reference.json',lc.metrics(y,baseline))
    for arm in ARMS:
        x=features(shards,ids,arm)
        for seed in SEEDS:
            q=np.full(799,np.nan)
            for f in range(1,6):
                name=f'fold_predictions/{arm}_{seed}_{f}.json';tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
                if not (OUT/name).exists():
                    started=time.perf_counter();v=fit_calibrated_member('RF',x[tr],y[tr],x[te],seed+f,inner_folds=3,jobs=2)
                    dump(name,dict(arm=arm,seed=seed,fold=f,train_sha=list(ids[tr]),test_sha=list(ids[te]),probability=v.tolist(),fit_predict_s=time.perf_counter()-started))
                r=read(name)
                if r['test_sha']!=list(ids[te]) or r['train_sha']!=list(ids[tr]):raise RuntimeError('resume fold contract')
                q[te]=r['probability'];print('MSV_SOURCE_FOLD_DONE',arm,seed,f,flush=True)
            if not np.isfinite(q).all():raise RuntimeError('OOF incomplete')
            stage=lc.metrics(y,q); finals={}
            for budget in BUDGETS:
                route=routes(ids,fold,q,budget);m,final=lc.cascade(y,q,ep,route);finals[budget]=final
                assert m['errors']==m['stage1_errors']-m['net'] and np.array_equal(final[~route],q[~route])
                results.append(dict(arm=arm,seed=seed,budget=budget,stage1_BA=stage['BA'],stage1_AUC=stage['AUC'],**m))
                for f in range(1,6):
                    ix=fold==f;fm,_=lc.cascade(y[ix],q[ix],ep[ix],route[ix]);folds.append(dict(arm=arm,seed=seed,budget=budget,fold=f,**fm))
                for i,s in enumerate(ids):predictions.append(dict(arm=arm,seed=seed,budget=budget,sha256=s,fold=int(fold[i]),label=int(y[i]),stage1_probability=float(q[i]),C43_probability=float(ep[i]),routed=int(route[i]),final_probability=float(final[i])))
                for name,ref in [('own_stage1',q),('historical_P31_RF',baseline)]:statistics.append(dict(arm=arm,seed=seed,budget=budget,comparison='cascade vs '+name,**interval(y,final,ref,fold)))
            statistics.append(dict(arm=arm,seed=seed,budget=.1,comparison='Top10 vs Top25',**interval(y,finals[.1],finals[.25],fold)))
            print('MSV_SOURCE_SEED_DONE',arm,seed,flush=True)
    # Holm across reported McNemar comparisons; bootstrap intervals are descriptive, not simultaneous.
    order=sorted(range(len(statistics)),key=lambda i:statistics[i]['mcnemar_p']);last=0
    for k,i in enumerate(order):last=max(last,min(1.,(len(order)-k)*statistics[i]['mcnemar_p']));statistics[i]['mcnemar_holm_p']=last
    table('source_fold_metrics.csv',folds);table('source_per_apk.csv',predictions);table('paired_statistics.csv',statistics);table('source_metrics.csv',results)
    print('MSV_SOURCE_DONE no_future=true',flush=True)

def child_limits():
    resource.setrlimit(resource.RLIMIT_AS,(20*1024**3,20*1024**3))

def costs():
    selected=read('cost_sample.json');records=[];_,_,_,_,_,_,shards=data()
    for i,row in enumerate(selected):
        sid=row['sha256'];folder=f'cost/{sid}';(OUT/folder).mkdir(parents=True,exist_ok=True)
        if not (OUT/f'{folder}/row.json').exists():dump(f'{folder}/row.json',row)
        if not (OUT/f'{folder}/lineage.csv').exists():table(f'{folder}/lineage.csv',[row])
        kinds=['ManifestOnly','RefPresence31','P31','compact','endpoint'];kinds=kinds[i%5:]+kinds[:i%5]
        for kind in kinds:
            rounds=2 if kind in ('ManifestOnly','RefPresence31','P31') else 1
            for repeat in range(rounds):
                recname=f'{folder}/{kind}_{repeat}_timing.json'
                if (OUT/recname).exists():records.append(read(recname));continue
                # Each builder's output is fresh; never resume partial extractor files as a fresh timing.
                dest=OUT/f'{folder}/{kind}_{repeat}_assets'
                result=f'{folder}/{kind}_{repeat}_values.json'
                if dest.exists() or (OUT/result).exists():raise RuntimeError('interrupted cost item: preserve it and use a fresh run')
                if kind in ('compact','endpoint'):
                    builder=BUILDER/('run_compact_feature_extraction_ag.py' if kind=='compact' else 'run_endpoint_centered_tensor_cache_ag.py')
                    cmd=[sys.executable,'-B',str(builder),'--lineage',str(OUT/f'{folder}/lineage.csv'),'--mapping',str(lc.MAPPING),'--priority',str(PRIORITY),'--output',str(dest),'--min-free-gb','5','--progress-every','1']
                    if kind=='endpoint':cmd+=['--edge-semantics',str(EDGE),'--limit','0','--upstream-hops','2','--downstream-hops','1','--K-in','8','--K-out','4','--endpoint-cap','256','--node-cap','2048','--edge-cap','4096','--reachability-max-states','100000']
                else:cmd=[sys.executable,'-B',str(Path(__file__).parent/'worker.py'),'--row',str(OUT/f'{folder}/row.json'),'--kind',kind,'--result',result]
                print('MSV_COST_START',i+1,'/20',kind,repeat,sid,flush=True)
                t=time.perf_counter();status='PASS';error='';extract_s=None;parity='NOT_APPLICABLE'
                try:
                    with (OUT/f'{folder}/{kind}_{repeat}.log').open('x') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200,preexec_fn=child_limits)
                    if kind=='compact':
                        rr=lc.readcsv(dest/'features.csv');assert len(rr)==1 and rr[0]['sha256'].lower()==sid and rr[0]['status']=='OK'
                    elif kind=='endpoint':
                        with gzip.open(dest/'endpoint_centered_tensor_cache.jsonl.gz','rt') as f:rr=[json.loads(line) for line in f if line.strip()]
                        assert len(rr)==1 and rr[0]['sha256'].lower()==sid and rr[0]['status']=='OK'
                    else:
                        rr=read(result);v=rr['values'];extract_s=rr['extract_s'];old=shards[sid]
                        if kind=='P31':assert np.allclose(v['p31'],old['role_prevalence'],atol=1e-7,rtol=0)
                        else:
                            assert v['permissions']==old['permissions'] and v['manifest']==old['manifest']
                            if kind=='RefPresence31':assert v['role_ref_count']==old['role_ref_count']
                        parity='PASS'
                except Exception as exc:status='FAILED_OR_TIMEOUT';error=repr(exc)
                r=dict(sha256=sid,fold=row['fold'],size_bin=row['size_bin'],stratum_population=row['stratum_population'],apk_bytes=Path(row['absolute_path']).stat().st_size,kind=kind,repeat=repeat,status=status,wall_s=time.perf_counter()-t,extract_s=extract_s,feature_parity=parity,error=error,cache_state='UNCONTROLLED_OS_CACHE; original APK prehashed; not cold-start')
                dump(recname,r);records.append(r);print('MSV_COST_DONE',kind,status,round(r['wall_s'],3),flush=True)
    table('cost_measurements.csv',records)
    summary=[]
    for kind in ('ManifestOnly','RefPresence31','P31','compact','endpoint'):
        for repeat in sorted(set(r['repeat'] for r in records if r['kind']==kind)):
            rr=[r for r in records if r['kind']==kind and r['repeat']==repeat];ok=[r for r in rr if r['status']=='PASS'];v=np.array([r['wall_s'] for r in ok])
            summary.append(dict(kind=kind,repeat=repeat,planned=len(rr),successful=len(ok),failed=len(rr)-len(ok),wall_mean_s=float(v.mean()) if len(v) else None,wall_median_s=float(np.median(v)) if len(v) else None,wall_p90_s=float(np.quantile(v,.9)) if len(v) else None,scope='Component wall time including startup; graph components exclude C43 preprocessing/forward; no end-to-end claim'))
    table('cost_summary.csv',summary)

def report():
    rows=lc.readcsv(OUT/'source_metrics.csv');foldrows=lc.readcsv(OUT/'source_fold_metrics.csv');cost=lc.readcsv(OUT/'cost_summary.csv');checks=read('baseline_integrity_check.json')
    main=[r for r in rows if r['arm']=='ManifestOnly' and float(r['budget'])==.1]
    positive=all(int(r['net'])>0 for r in main);costok=all(int(r['failed'])==0 for r in cost)
    status='MANIFEST10_INCREMENT_REPEATED_ON_SOURCE' if positive else 'MANIFEST10_INCREMENT_NOT_STABLE_ACROSS_SEEDS'
    if checks['status']!='PASS':status='BLOCKED_BY_ASSET_INTEGRITY'
    dump('final_decision.json',dict(status=status,source_only=True,seeds=list(SEEDS),manifest10_nets=[int(r['net']) for r in main],cost_components_complete=costok,end_to_end_cost_verified=False,C43_forward_and_preprocessing_cost='NOT_MEASURED',independent_generalization_verified=False,model_or_budget_selected=False,future_run_authorized=False,asset_integrity=checks['status'],next_action='Review fixed-seed stability and paired component costs; do not claim deployed total cost or independent confirmation. No automatic future training or tuning.'))
    lines=['# Manifest低成本第一层与选择性Direct-C43验证','',f'状态：`{status}`。仅2017已有799个APK，无新增数据、无未来年份，无第二层重训练。','',
    '## 固定协议','', 'ManifestOnly主方案为135维，RefPresence31为166维对照；RF与训练内校准复用正式函数。三种子分别评价，不把种子视为独立APK，也不额外创建三RF集成部署。10%和25%按每折第一层概率不确定度排序，**不是原MeanOpposition**。C43使用同折冻结专家全样本预测，未路由样本严格保留第一层输出。','',
    '## 实际性能','', '|候选|种子|预算|第一层BA|两层BA|Fixed|Broken|Net|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['arm']}|{r['seed']}|{r['budget']}|{float(r['stage1_BA']):.6f}|{float(r['BA']):.6f}|{r['fixed']}|{r['broken']}|{r['net']}|")
    lines+=['','逐折结果见source_fold_metrics.csv；逐APK结果见source_per_apk.csv。paired_statistics.csv按折×类别分层进行APK配对bootstrap，并报告McNemar及Holm校正。区间未含训练集重采样不确定性，种子共享同一批APK。','',
    '## 实际成本边界','', '20个APK按固定折×大小四分位无标签抽样；每个APK独立运行Manifest、引用扫描、P31、compact及endpoint构建。前3项重复2次，构建次序循环交错。原APK只读，不执行。OS页缓存未清空，哈希预检已读过文件，不能称冷缓存计时。超时/失败全部保留，不能只用成功子集声称全面低成本。','',
    '|组件|轮次|成功/计划|均值秒|中位秒|P90秒|','|---|---:|---:|---:|---:|---:|']
    for r in cost:lines.append(f"|{r['kind']}|{r['repeat']}|{r['successful']}/{r['planned']}|{r['wall_mean_s']}|{r['wall_median_s']}|{r['wall_p90_s']}|")
    lines+=['','compact+endpoint按现有提取器配置测量，仍不包含C43背景变换、模型前向、排队与路由开销；本轮**不宣称端到端延迟已经验证，也不把10%乘全样本平均深查耗时冒充真实路由成本**。20个分层样本是成本Pilot，不代表所有APK。','',
    '## 研究结论边界','', '10%方案源于之前已看过的OOF结果，本轮是稳定性复核而非独立确认或无偏模型选择。增益仅指相对该低成本第一层；是否值得替代P31必须同时考虑最终准确率与完整计算成本。不得故意削弱第一层以放大增益。保留历史P31参考，但其种子和选择协议不同，不能当作同种子严格头对头结论。', '',f"完整性：{checks['status']}；覆盖{checks['asset_count']}个实际使用文件。未加载checkpoint，未对整个项目宣称完整性。",'', '无论结果好坏均在报告后停止，不自动调参、扩展预算、运行未来年份。']
    textfile('final_report.md','\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    if not (OUT/'protocol.json').exists():preflight()
    else:
        check=integrity(read('asset_manifest_before.json'))
        if check['status']!='PASS':raise RuntimeError('assets changed before resume')
    if args.preflight_only:return
    if (OUT/'final_decision.json').exists():raise RuntimeError('completed run preserved; no overwrite')
    failed=None;started=time.perf_counter()
    try:
        for name,fn,done in [('source',source,'source_metrics.csv'),('cost',costs,'cost_summary.csv')]:
            if (OUT/done).exists():print('MSV_PHASE_RESUME',name,flush=True);continue
            print('MSV_PHASE_START',name,flush=True);fn();print('MSV_PHASE_DONE',name,flush=True)
    except Exception as exc:
        failed=repr(exc);traceback.print_exc()
    check=integrity(read('asset_manifest_before.json'))
    if not (OUT/'baseline_integrity_check.json').exists():dump('baseline_integrity_check.json',check)
    if failed or check['status']!='PASS':
        dump('failure_'+str(time.time_ns())+'.json',dict(error=failed,integrity=check));raise RuntimeError(failed or 'asset integrity failure')
    dump('runtime.json',dict(total_s=time.perf_counter()-started));report();print('MSV_DONE',OUT,flush=True)
if __name__=='__main__':main()
