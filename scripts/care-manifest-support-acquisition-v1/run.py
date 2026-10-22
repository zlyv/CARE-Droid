#!/usr/bin/env python3
"""Source-nested routing pilot. Reads frozen feature/score caches, never APKs or expert scores.
Writes exclusively to the new MSA_OUT run directory; fits diagnostic reference/LR models only.
"""
import os, sys, json, csv, time, hashlib, argparse, traceback, importlib.util
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import pairwise_distances

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
OUT=Path(os.environ['MSA_OUT']).resolve()
assert OUT.parent==ROOT/'outputs/care_manifest_support_acquisition_v1' and OUT.name.startswith('run_')
PRIOR=ROOT/'outputs/care_manifest_routing_coverage_v1/run_20261019_091224'
os.environ['MRC_OUT']=str(PRIOR)
spec=importlib.util.spec_from_file_location('routing_baseline',ROOT/'scripts/care-manifest-routing-coverage-v1/run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
lc=base.lc
SEEDS=base.SEEDS

def save(name,obj):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def table(name,rows):
    rows=list(rows);cols=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with (OUT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,cols);w.writeheader();w.writerows(rows)
def write(name,s):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(s)
def integrity():
    rows=read(OUT/'asset_manifest_before.json')
    changed=[r['path'] for r in rows if base.digest(r['path'])['sha256']!=r['sha256']]
    return dict(status='FAIL' if changed else 'PASS',changed=changed,asset_count=len(rows),scope='Actual reused feature shards, frozen probabilities, label sources, source code and extractor definitions. No original APK or expert checkpoint opened.')
def load():
    ids,y,fold,x,fi,years,z=base.inputs()
    a,_=lc.load_shards('source');b,_=lc.load_shards('future')
    xr=np.asarray([np.asarray(a[s]['role_ref_count'])>0 for s in ids],float)
    zr=np.asarray([np.asarray(b[s]['role_ref_count'])>0 for s in fi],float)
    assert xr.shape==(799,31) and zr.shape==(1989,31)
    return ids,y,fold,x,fi,years,z,np.c_[x,xr],np.c_[z,zr]

class Support:
    """Trust-style high-density reference, NOT a verbatim Trust Score/conformal guarantee.
    Fixed metric: 1/2 permission Jaccard + 1/2 capped log-count standardized L1.
    Drop 10% sparsest points per training class; mean five-neighbor class distance.
    """
    def distance(self,a,b):
        aa=a[:,:128]>0;bb=b[:,:128]>0
        inter=aa.astype(float)@bb.astype(float).T
        union=aa.sum(1)[:,None]+bb.sum(1)[None,:]-inter
        jac=1-np.divide(inter,union,out=np.ones_like(inter),where=union>0)
        an=np.log1p(np.maximum(a[:,128:],0))/self.scale
        bn=np.log1p(np.maximum(b[:,128:],0))/self.scale
        numeric=sum(np.minimum(np.abs(an[:,j,None]-bn[None,:,j]),1) for j in range(7))/7
        return .5*jac+.5*numeric
    def fit(self,x,y):
        q=np.log1p(np.maximum(x[:,128:],0));self.scale=np.maximum(np.quantile(q,.75,axis=0)-np.quantile(q,.25,axis=0),1.)
        self.refs=[]
        for c in (0,1):
            xx=x[y==c];d=self.distance(xx,xx);np.fill_diagonal(d,np.inf)
            density=np.sort(d,axis=1)[:,:min(5,len(xx)-1)].mean(1)
            keep=np.argsort(density,kind='stable')[:max(5,int(np.ceil(.9*len(xx))))]
            self.refs.append(xx[keep])
        return self
    def score(self,x,p):
        # Streaming target batches; no unbounded global Host/APK matrix.
        ds=[]
        for xx in np.array_split(x,max(1,int(np.ceil(len(x)/256)))):
            ds.append(np.column_stack([np.sort(self.distance(xx,r),axis=1)[:,:5].mean(1) for r in self.refs]))
        d=np.concatenate(ds);pred=(p>=.5).astype(int);ii=np.arange(len(p))
        return np.log((d[ii,pred]+1e-6)/(d[ii,1-pred]+1e-6))

def uncertainty(p):return 1-2*np.abs(p-.5)
def opposition(p,aux):return np.where(p>=.5,1-aux,aux)
def tau(score,budget,pool=None):
    v=score if pool is None else score[pool]
    k=min(len(v),max(1,round(len(score)*budget)))
    return float(np.sort(v)[-k]) if k else 1e100
def prob_ncm(p,ref):
    # Class-conditional empirical rank control; not a conformal validity claim.
    out=np.zeros(len(p))
    for c in (0,1):
        ix=(p>=.5)==bool(c);r=uncertainty(ref[(ref>=.5)==bool(c)])
        out[ix]=np.searchsorted(np.sort(r),uncertainty(p[ix]),side='right')/max(1,len(r))
    return out
def rules(p,s,refp,ctrlp):
    u=uncertainty(p);pools={'U':u>=tau(u,.25),'US':(u>=tau(u,.25))|(s>=tau(s,.1))}
    d={}
    for b in (.1,.25):
        suffix=str(round(b*100));d['ABS'+suffix]=tau(u,b);d['SUPPORT'+suffix]=tau(s,b)
        d['PROB_RANK'+suffix]=tau(prob_ncm(p,p),b)
        for poolname,pool in pools.items():
            d[poolname+'_REF'+suffix]=tau(opposition(p,refp),b,pool)
            d[poolname+'_MANIFEST_CTRL'+suffix]=tau(opposition(p,ctrlp),b,pool)
    return d
def masks(p,s,refp,ctrlp,r,cfp):
    u=uncertainty(p);pools={'U':u>=r['ABS25'],'US':(u>=r['ABS25'])|(s>=r['SUPPORT10'])}
    ret={};acq={}
    for b in (10,25):
        for pre,score in [('ABS',u),('SUPPORT',s),('PROB_RANK',prob_ncm(p,cfp))]:
            n=pre+str(b);ret[n]=score>=r[n];acq[n]=np.zeros(len(p),bool)
        for pn,pool in pools.items():
            for feat,ap in [('REF',refp),('MANIFEST_CTRL',ctrlp)]:
                n=pn+'_'+feat+str(b);ret[n]=pool&(opposition(p,ap)>=r[n]);acq[n]=pool if feat=='REF' else np.zeros(len(p),bool)
    return ret,acq,pools

def selftest():
    rng=np.random.default_rng(42);x=rng.integers(0,2,(40,135)).astype(float);y=np.tile([0,1],20)
    model=Support().fit(x,y);d=model.distance(x,x)
    assert np.allclose(d,d.T) and np.allclose(np.diag(d),0) and d.min()>=0
    p=np.linspace(.01,.99,40);s=model.score(x,p)
    assert np.allclose(s,-model.score(x,1-p)) and np.isfinite(s).all()
    r=rules(p,s,1-p,p);mm,ac,po=masks(p,s,1-p,p,r,p)
    for n in mm:
        if n.startswith('U_'):assert not (mm[n]&~po['U']).any()
        if n.startswith('US_'):assert not (mm[n]&~po['US']).any()
    assert np.all(prob_ncm(np.array([.1,.2]),p)[1:]>=prob_ncm(np.array([.1,.2]),p)[:-1])
    print('MSA_SELFTEST_PASS',flush=True)

def preflight():
    selftest();ids,y,fold,x,fi,years,z,xr,zr=load()
    assert read(PRIOR/'final_decision.json')['asset_integrity']=='PASS'
    assets=[Path(a['path']) for a in read(PRIOR/'asset_manifest_before.json')]
    assets+=list((PRIOR/'crossfit').glob('*.json'))+list((PRIOR/'frozen').glob('*.json'))
    assets+=[PRIOR/'protocol.json',PRIOR/'final_decision.json']
    assets+=list(Path(__file__).parent.glob('*'))
    assets+=list((ROOT/'scripts/care-manifest-selective-validation-v1').glob('*.py'))
    save('asset_manifest_before.json',[base.digest(p) for p in sorted(set(assets)) if p.is_file()])
    save('protocol.json',dict(status='FROZEN_BEFORE_FITS',source=799,future=1989,seeds=list(SEEDS),stage1='Unchanged cached Manifest RF probabilities',stage2='NOT INVOKED; expert probabilities NOT USED',method_A='Trust-style class support reference, fixed 10% density trim, k=5, log distance ratio; not an exact reproduction of published Trust Score',metric='0.5 hashed permission Jaccard + 0.5 capped training-IQR log-count L1; floor IQR=1',method_B='Sequential reference-presence acquisition, 31 MethodId Role presence bits plus Manifest -> calibrated LR -> opposition to frozen Manifest prediction',combined='US candidate pool = source-ABS25 union source-SUPPORT10',control='U pool uses source-ABS25 only; same calibrated LR using Manifest only; class-conditional empirical probability rank; same-count uncertainty and uniform-random expectation',budgets=[.1,.25],pool_budget=.25,thresholds='Unsupervised source crossfit score quantiles; frozen before future labels, ties may expand route fraction',source_protocol='Fixed outer5 folds; inner3 CF references and calibrated LR using legal inner-train only; identical inner RF predictions from previous validated cache',future_protocol='All2017 crossfit determines thresholds, all2017 references/LR predict2018-2021, labels evaluation-only after all rules frozen',feature_boundary='Only manifest, permissions, role_ref_count>0. No invoke/body counts, graph, Host, or P31 prevalence',deployment='Auxiliary LR/reference model is routing-only, never replaces Stage-I classification',cost_boundary='Cached replay. Acquisition fraction reported; DEX extraction and full E2E latency NOT MEASURED. Reference feature computation for all rows in replay does not count as deployment cost.',validity='No conformal coverage guarantee under drift; years already development; seeds not independent APKs',stopping='No future tuning, no automatic StageII or subsequent experiments'))
    print('MSA_PREFLIGHT_PASS',flush=True)

def build(smoke=False):
    ids,y,fold,x,fi,years,z,xr,zr=load()
    for seed in SEEDS:
        for f in (1,2,3,4,5,0):
            path=OUT/f'frozen/{seed}_{f}.json'
            if path.exists():continue
            t=time.perf_counter();tr=np.flatnonzero(fold!=f) if f else np.arange(len(ids));te=np.flatnonzero(fold==f) if f else np.arange(len(fi))
            xt=x[te] if f else z;rt=xr[te] if f else zr;targetids=ids[te] if f else fi
            cf=np.full((len(tr),4),np.nan);ss=seed+f
            for k,(it,iv) in enumerate(StratifiedKFold(3,shuffle=True,random_state=ss).split(x[tr],y[tr]),1):
                rf=read(PRIOR/f'crossfit/{seed}_{f}_{k}_RF.json');lr=read(PRIOR/f'crossfit/{seed}_{f}_{k}_LR.json')
                for a in (rf,lr):assert a['train_sha']==list(ids[tr[it]]) and a['test_sha']==list(ids[tr[iv]])
                p=np.asarray(rf['probability']);cf[iv,0]=p;cf[iv,3]=lr['probability']
                cf[iv,1]=Support().fit(x[tr[it]],y[tr[it]]).score(x[tr[iv]],p)
                cf[iv,2]=base.fit_calibrated_member('LR',xr[tr[it]],y[tr[it]],xr[tr[iv]],ss+100000*k,inner_folds=3,jobs=2)
            assert np.isfinite(cf).all();rule=rules(cf[:,0],cf[:,1],cf[:,2],cf[:,3])
            frozen=read(PRIOR/f'frozen/{seed}_{f}.json');assert frozen['target_sha']==list(targetids) and frozen['train_sha']==list(ids[tr])
            p=np.asarray(frozen['rf_probability']);ctrl=np.asarray(frozen['lr_probability'])
            infer=time.perf_counter();s=Support().fit(x[tr],y[tr]).score(xt,p);support_fit_and_score_s=time.perf_counter()-infer
            ref=base.fit_calibrated_member('LR',xr[tr],y[tr],rt,ss,inner_folds=3,jobs=2)
            save(str(path.relative_to(OUT)),dict(seed=seed,fold=f,train_sha=list(ids[tr]),target_sha=list(targetids),probability=p.tolist(),support=s.tolist(),ref_probability=ref.tolist(),manifest_control=ctrl.tolist(),crossfit_probability=cf[:,0].tolist(),rules=rule,elapsed_s=time.perf_counter()-t,support_fit_plus_target_score_s=support_fit_and_score_s))
            print('MSA_WINDOW_DONE seed',seed,'fold',f,'seconds',round(time.perf_counter()-t,2),flush=True)
            if smoke:return
    save('all_rules_frozen.json',dict(status='PASS',windows=18,future_labels_used=False))

def evaluate():
    assert (OUT/'all_rules_frozen.json').exists()
    ids,y,fold,x,fi,years,z,xr,zr=load();labels={}
    for row in lc.readcsv(base.LABELS):
        if row['arm']=='H0_ORIGINAL':
            s=row['sha256'];assert s not in labels;labels[s]=(int(row['year']),int(row['label']))
    assert set(labels)==set(fi) and all(labels[s][0]==int(yr) for s,yr in zip(fi,years))
    fy=np.array([labels[s][1] for s in fi]);rows=[];per=[];timings=[]
    for seed in SEEDS:
        source_parts={}
        for f in (1,2,3,4,5,0):
            obj=read(OUT/f'frozen/{seed}_{f}.json');p=np.array(obj['probability']);s=np.array(obj['support']);ref=np.array(obj['ref_probability']);ctrl=np.array(obj['manifest_control'])
            rr,ac,pools=masks(p,s,ref,ctrl,obj['rules'],np.array(obj['crossfit_probability']))
            tid=ids[fold==f] if f else fi;yy=y[fold==f] if f else fy;yrs=np.full(len(tid),2017) if f else years
            timings.append(dict(seed=seed,fold=f,fit_and_score_s=obj['elapsed_s'],support_fit_plus_score_s=obj['support_fit_plus_target_score_s'],extraction_timing='NOT_MEASURED_CACHE_REPLAY'))
            for yr in sorted(set(yrs)):
                ix=yrs==yr
                for name,mask in rr.items():
                    r=mask[ix];mat=base.top(tid[ix],p[ix],int(r.sum()));m=base.measure(yy[ix],p[ix],r,mat)
                    m.update(seed=seed,year=int(yr),fold=f,strategy=name,acquired=int(ac[name][ix].sum()),acquisition_rate=float(ac[name][ix].mean()))
                    if f:source_parts.setdefault(name,[]).append(m)
                    else:rows.append(m)
                    for j in np.flatnonzero(ix):per.append(dict(seed=seed,year=int(yr),fold=f,sha256=tid[j],strategy=name,label=int(yy[j]),probability=float(p[j]),support_score=float(s[j]),routed=int(mask[j]),acquired=int(ac[name][j]),aux_probability=float(ref[j]) if ac[name][j] else None))
        sumkeys=['n','stage1_errors','routed','captured_errors','routed_correct','unrouted_errors','FN','captured_FN','FP','captured_FP','matched_count_captured','captured_gain_vs_same_count','new_errors_vs_same_count','lost_errors_vs_same_count','uniform_random_expected_captured','high_confidence_errors','captured_high_confidence_errors','acquired']
        for name,parts in source_parts.items():
            m={k:sum(a[k] for a in parts) for k in sumkeys};m.update(seed=seed,year=2017,fold='OOF',strategy=name,route_rate=m['routed']/m['n'],error_recall=m['captured_errors']/m['stage1_errors'],route_precision=m['captured_errors']/max(1,m['routed']),acquisition_rate=m['acquired']/m['n']);rows.append(m)
    table('routing_yearly_metrics.csv',rows);table('per_apk_routing.csv',per);table('runtime_cost.csv',timings)
    summary=[]
    for yr in (2017,2018,2019,2020,2021):
        for name in sorted({r['strategy'] for r in rows}):
            rs=[r for r in rows if r['year']==yr and r['strategy']==name]
            summary.append(dict(year=yr,strategy=name,route_rate=float(np.mean([r['route_rate'] for r in rs])),acquisition_rate=float(np.mean([r['acquisition_rate'] for r in rs])),error_recall=float(np.mean([r['error_recall'] for r in rs])),captured_errors=float(np.mean([r['captured_errors'] for r in rs])),captured_FN=float(np.mean([r['captured_FN'] for r in rs])),captured_FP=float(np.mean([r['captured_FP'] for r in rs])),same_count_gains=','.join(str(r['captured_gain_vs_same_count']) for r in rs)))
    table('routing_seed_summary.csv',summary)
    lines=['# Manifest支持度与低成本补充观测联合验证','', '仅路由错误捕获试验，不调用C43，不报告纠错收益。第一层预测完全复用。SUPPORT是Trust-style参考支持度，不是原论文逐项复现。REF为新增31维DEX引用存在性；MANIFEST_CTRL为同算法Manifest对照。U候选池只用源域ABS25；US候选池为ABS25与SUPPORT10并集。后缀10/25是训练期参考预算，不固定未来比例。','', '|年份|策略|送入二层比例|补充DEX观测比例|错误召回|捕获FN|捕获FP|同数量增量（三种子）|','|---|---|---:|---:|---:|---:|---:|---|']
    for r in summary:lines.append(f"|{r['year']}|{r['strategy']}|{r['route_rate']:.2%}|{r['acquisition_rate']:.2%}|{r['error_recall']:.2%}|{r['captured_FN']:.2f}|{r['captured_FP']:.2f}|{r['same_count_gains']}|")
    lines+=['','## 结论边界','训练期交叉拟合分数用于冻结阈值；未来标签只作最终评价。距离、k、密度裁剪、候选池和LR配置未按未来结果搜索。历史年份已经用于开发，不是独立盲测。三种子不是三份独立APK。','', '同数量基线不是同真实成本。所有特征来自历史缓存，未重新扫描APK。额外观测仅在声明候选池中可用于路由，回放虽然预先计算了辅助概率，但池外概率不参与选择。提取耗时、支持度推理单独耗时和端到端成本尚未验证；runtime表的支持度时间含参考拟合，不可当作纯推理时间。','', '概率经验秩仅作为控制，不宣称任意漂移下conformal保证。SUPPORT不把距离直接解释为恶意语义。新增捕获错误能否被C43修复尚未测试，不自动运行后续分类。']
    write('final_report.md','\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke-test',action='store_true');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    if (OUT/'final_decision.json').exists():raise RuntimeError('completed run preserved')
    if not (OUT/'protocol.json').exists():preflight()
    assert integrity()['status']=='PASS'
    t=time.perf_counter()
    try:
        if not (OUT/'all_rules_frozen.json').exists():build(a.smoke_test)
        if a.smoke_test:
            save('smoke_test.json',dict(status='PASS',completed_windows=len(list((OUT/'frozen').glob('*.json'))),seconds=time.perf_counter()-t));return
        evaluate()
        report=integrity();save('baseline_integrity_check.json',report);assert report['status']=='PASS'
        save('final_decision.json',dict(status='SUPPORT_ACQUISITION_ROUTING_EVALUATION_COMPLETE',stage2_invoked=False,APK_scans=0,future_tuning=False,deployable_router_selected=False,asset_integrity=report['status'],elapsed_s=time.perf_counter()-t,next_action='Review same-count and acquisition-cost tradeoffs; no automatic StageII or tuning.'))
        print('MSA_DONE',OUT,flush=True)
    except Exception as exc:
        save('failure_'+str(time.time_ns())+'.json',dict(error=repr(exc),integrity=integrity()));traceback.print_exc();raise
if __name__=='__main__':main()
