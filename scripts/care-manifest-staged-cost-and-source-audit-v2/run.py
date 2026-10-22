#!/usr/bin/env python3
"""Benchmarks frozen router candidates and audits the 2017 routing loss.
Reads original APKs but never executes/rewrites them. All writes are under MSC_OUT.
No models are fitted and no prediction or route is changed.
"""
import os,sys,csv,json,hashlib,time,subprocess,traceback,gzip,resource
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np,yaml
ROOT=Path('/home/boyuanjie/projects/tasa-droid')
OUT=Path(os.environ['MSC_OUT']).resolve()
assert OUT.parent==ROOT/'outputs/care_manifest_staged_cost_source_audit_v2' and OUT.name.startswith('run_')
PREV=ROOT/'outputs/care_manifest_acquisition_cascade_audit_v1/run_20261019_pilot01'
ROUTE=ROOT/'outputs/care_manifest_support_acquisition_v1/run_20261019_pilot01'
BUILDER=ROOT/'scripts/care-di25-host-gatedgraph-source-v1'
PY=Path('/home/boyuanjie/envs/tif/bin/python')
sys.path.insert(0,str(ROOT/'scripts/care-lite-stage-v1'))
import common as lc
import extract as lite

def readcsv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(n,v):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def table(n,rows):
    rows=list(rows);p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def write(n,v):
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(v)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def integrity():
    a=load('baseline_before.json');bad=[r['path'] for r in a if sha(r['path'])!=r['sha256']]
    return dict(status='FAIL' if bad else 'PASS',changed=bad,asset_count=len(a),scope='Selected original APKs, reused CSV/protocols, extractor/config/builder code. Historical checkpoints and all other APKs outside this scope.')
def limit():resource.setrlimit(resource.RLIMIT_AS,(20*1024**3,20*1024**3))
def preflight():
    assert json.loads((PREV/'final_decision.json').read_text())['asset_integrity']=='PASS'
    assert json.loads((ROUTE/'final_decision.json').read_text())['asset_integrity']=='PASS'
    routes=readcsv(ROUTE/'per_apk_routing.csv');chosen=defaultdict(list)
    for r in routes:
        if r['strategy']=='U_REF10' and r['acquired']=='1':chosen[(int(r['year']),r['sha256'])].append(r)
    source={r['sha256'].lower():r for r in readcsv(lc.SOURCE_LINEAGE)}
    future={r['sha256'].lower():r for r in readcsv(lc.FUTURE_LINEAGE)}
    assert len(source)==799 and len(future)==1989
    sample=[]
    for year in (2017,2018,2019,2020,2021):
        data=source if year==2017 else future
        candidates=[]
        for (yr,sid),rr in chosen.items():
            if yr!=year:continue
            row=data[sid];p=Path(row['absolute_path'])
            if not p.is_file():continue
            candidates.append(dict(year=year,sha256=sid,absolute_path=str(p),apk_bytes=p.stat().st_size,fold=int(rr[0]['fold']) if year==2017 else 0,acquired_seeds=len(rr)))
        assert len(candidates)>=20,(year,len(candidates))
        if year==2017:
            for f in (1,2,3,4,5):
                group=sorted((r for r in candidates if r['fold']==f),key=lambda r:(r['apk_bytes'],r['sha256']))
                assert group
                row=group[len(group)//2];row['sampling_stratum']='fold_'+str(f);sample.append(row)
        else:
            parts=np.array_split(sorted(candidates,key=lambda r:(r['apk_bytes'],r['sha256'])),4)
            for i,g in enumerate(parts):
                assert len(g)
                row=sorted(g,key=lambda r:hashlib.sha256((f'MSC_V1_{year}_SIZE_{i}_'+r['sha256']).encode()).hexdigest())[0]
                row['sampling_stratum']='size_quartile_'+str(i);sample.append(row)
    assert len(sample)==21 and len({r['sha256'] for r in sample})==21
    baseline=[PREV/n for n in ('final_decision.json','per_apk_predictions.csv','incremental_route_outcomes.csv','baseline_integrity_check.json')]
    baseline+=[ROUTE/'per_apk_routing.csv',lc.SOURCE_LINEAGE,lc.FUTURE_LINEAGE,lc.MAPPING,ROOT/'configs/protocol/sensitive_endpoint_priority_v1.yaml',ROOT/'configs/protocol/semantic_edge_mapping_v1.yaml']
    baseline+=[ROOT/'scripts/care-lite-stage-v1/extract.py',ROOT/'scripts/care-lite-stage-v1/common.py']
    baseline+=list(BUILDER.glob('*.py'))+list((ROOT/'src/tasa_droid/p2').glob('*.py'))+list(Path(__file__).parent.glob('*'))
    baseline += [Path(r['absolute_path']) for r in sample]
    checks=[]
    for p in sorted(set(baseline)):
        if not p.is_file():continue
        h=sha(p)
        if any(p==Path(s['absolute_path']) and h!=s['sha256'] for s in sample):raise RuntimeError('APK checksum mismatch '+str(p))
        checks.append(dict(path=str(p),sha256=h))
    save('baseline_before.json',checks);save('sample_contract.json',sample)
    save('protocol.json',dict(status='FROZEN_BEFORE_TIMING',sampling='21 U_REF10 acquisition-pool APKs, five 2017 folds one median size each; 2018-2021 four size quartiles each, SHA-hash deterministic within quartile; labels never used for sample selection',seeds='union of 3 frozen routers',staged='same-process L1 Manifest then L3 DEX RefPresence on same original APK; two repeats; no cache flush, includes repeat Manifest parse in L3',graph='original compact and endpoint builders per sampled APK in fresh isolated output, one repeat',cost_scope='Component wall clock only; still excludes frozen C43 representation/background processing and forward',source_error='No feature or threshold tuning; disjoint added/dropped route errors and damage by original Stage-I prediction and corrected status',no_APK_execution=True,no_model_training=True,no_future_threshold_selection=True,resource_limits='20GiB per graph subprocess; 1200s timeout; 5GiB free guard'))
    print('MSC_PREFLIGHT_PASS selected=21',flush=True)

def replay_source():
    if (OUT/'source_changed_apks.csv').exists() and (OUT/'source_change_anatomy.csv').exists():return
    if (OUT/'source_changed_apks.csv').exists() or (OUT/'source_change_anatomy.csv').exists():raise RuntimeError('partial source audit kept; use a fresh run directory')
    rows=readcsv(PREV/'per_apk_predictions.csv');items=defaultdict(dict)
    for r in rows:
        if int(r['year'])!=2017 or r['strategy'] not in ('U_REF10','ABS10'):continue
        items[(int(r['seed']),r['sha256'])][r['strategy']]=r
    assert len(items)==3*799 and all(len(v)==2 for v in items.values())
    transitions=[];cases=[]
    for (seed,sid),v in sorted(items.items()):
        a=v['ABS10'];b=v['U_REF10'];assert a['label']==b['label'] and a['stage1_probability']==b['stage1_probability']
        if a['routed']==b['routed']:continue
        state='ADDED' if b['routed']=='1' else 'DROPPED';y=int(a['label']);p=float(a['stage1_probability']);e=float(a['expert_probability']);before=(p>=.5)==y;after=(e>=.5)==y
        typ='FN' if not before and y==1 else 'FP' if not before else 'TN' if y==0 else 'TP'
        outcome='FIXED' if not before and after else 'UNFIXED' if not before else 'BROKEN' if not after else 'KEPT_CORRECT'
        cases.append(dict(seed=seed,sha256=sid,route_change=state,stage1_type=typ,expert_outcome=outcome,label=y,stage1_probability=p,expert_probability=e,stage1_margin=abs(p-.5),expert_margin=abs(e-.5),fold=a['fold']))
    for seed in (20261018,20261019,20261020):
        for state in ('ADDED','DROPPED'):
            part=[r for r in cases if r['seed']==seed and r['route_change']==state];counts=Counter((r['stage1_type'],r['expert_outcome']) for r in part)
            transitions.append(dict(seed=seed,route_change=state,n=len(part),stage1_errors=sum(r['stage1_type'] in ('FN','FP') for r in part),fixed=counts[('FN','FIXED')]+counts[('FP','FIXED')],broken=counts[('TN','BROKEN')]+counts[('TP','BROKEN')],fixed_FN=counts[('FN','FIXED')],fixed_FP=counts[('FP','FIXED')],broken_TN=counts[('TN','BROKEN')],broken_TP=counts[('TP','BROKEN')],correct_unchanged=counts[('TN','KEPT_CORRECT')]+counts[('TP','KEPT_CORRECT')]))
    table('source_changed_apks.csv',cases);table('source_change_anatomy.csv',transitions)

def run_one(row,roles,rules):
    sid=row['sha256'];folder=OUT/'bench'/sid;folder.mkdir(parents=True,exist_ok=True)
    old=ROOT/'outputs/care_lite_stage_v1/shards'/('source' if row['year']==2017 else 'future')/(sid+'.json')
    prior=json.loads(old.read_text());rec=[]
    for rep in range(2):
        target=folder/f'staged_{rep}.json'
        if target.exists():rec+=load(str(target.relative_to(OUT)))['measurements'];continue
        p={'sha256':sid,'absolute_path':row['absolute_path']};t=time.perf_counter();v1=lite.extract(p,rules,roles,'L1');a=time.perf_counter()-t
        t=time.perf_counter();v3=lite.extract(p,rules,roles,'L3');b=time.perf_counter()-t
        assert v1['permissions']==v3['permissions']==prior['permissions']
        assert v1['manifest']==v3['manifest']==prior['manifest']
        assert v3['role_ref_count']==prior['role_ref_count']
        rr=[dict(year=row['year'],sha256=sid,stratum=row['sampling_stratum'],apk_bytes=row['apk_bytes'],repeat=rep,component='manifest_L1',seconds=a),dict(year=row['year'],sha256=sid,stratum=row['sampling_stratum'],apk_bytes=row['apk_bytes'],repeat=rep,component='incremental_L3_after_L1',seconds=b)]
        save(str(target.relative_to(OUT)),dict(status='PASS',measurements=rr,feature_parity='PASS'));rec+=rr
        print('MSC_STAGED',row['year'],sid[:8],rep,round(a,3),round(b,3),flush=True)
    return rec

def graph_one(row):
    sid=row['sha256'];folder=OUT/'bench'/sid;folder.mkdir(parents=True,exist_ok=True)
    lineage=folder/'lineage.csv'
    if not lineage.exists():
        with lineage.open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,['sha256','absolute_path']);w.writeheader();w.writerow(dict(sha256=sid,absolute_path=row['absolute_path']))
    rec=[]
    for component in ('compact','endpoint'):
        marker=folder/(component+'_timing.json')
        if marker.exists():rec.append(load(str(marker.relative_to(OUT))));continue
        dest=folder/(component+'_assets');log=folder/(component+'.log')
        if dest.exists() or log.exists():raise RuntimeError('partial graph result kept; use a new run directory: '+str(dest))
        script=BUILDER/('run_compact_feature_extraction_ag.py' if component=='compact' else 'run_endpoint_centered_tensor_cache_ag.py')
        cmd=[str(PY),'-B',str(script),'--lineage',str(lineage),'--mapping',str(lc.MAPPING),'--priority',str(ROOT/'configs/protocol/sensitive_endpoint_priority_v1.yaml'),'--output',str(dest),'--min-free-gb','5','--progress-every','1']
        if component=='endpoint':cmd+=['--edge-semantics',str(ROOT/'configs/protocol/semantic_edge_mapping_v1.yaml'),'--limit','0','--upstream-hops','2','--downstream-hops','1','--K-in','8','--K-out','4','--endpoint-cap','256','--node-cap','2048','--edge-cap','4096','--reachability-max-states','100000']
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(OUT/'tmp'),XDG_CACHE_HOME=str(OUT/'cache'),PYTHONPATH=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH',''))
        t=time.perf_counter()
        with log.open('x') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200,preexec_fn=limit)
        elapsed=time.perf_counter()-t
        if component=='compact':
            rr=readcsv(dest/'features.csv');assert len(rr)==1 and rr[0]['sha256'].lower()==sid and rr[0]['status']=='OK'
        else:
            with gzip.open(dest/'endpoint_centered_tensor_cache.jsonl.gz','rt') as f:rr=[json.loads(line) for line in f if line.strip()]
            assert len(rr)==1 and rr[0]['sha256'].lower()==sid and rr[0]['status']=='OK'
        v=dict(year=row['year'],sha256=sid,stratum=row['sampling_stratum'],apk_bytes=row['apk_bytes'],component=component,seconds=elapsed,status='PASS',log=str(log))
        save(str(marker.relative_to(OUT)),v);rec.append(v)
        print('MSC_GRAPH',row['year'],sid[:8],component,round(elapsed,2),flush=True)
    return rec

def summarize():
    samples=load('sample_contract.json');data=[]
    for row in samples:
        folder=OUT/'bench'/row['sha256'];measure=[]
        for k in (0,1):measure+=load(str((folder/f'staged_{k}.json').relative_to(OUT)))['measurements']
        d={k:float(np.mean([r['seconds'] for r in measure if r['component']==k])) for k in ('manifest_L1','incremental_L3_after_L1')}
        for comp in ('compact','endpoint'):d[comp]=float(load(str((folder/(comp+'_timing.json')).relative_to(OUT)))['seconds'])
        d.update(year=row['year'],sha256=row['sha256'],stratum=row['sampling_stratum'],apk_bytes=row['apk_bytes'],light_plus_graph=d['manifest_L1']+d['incremental_L3_after_L1']+d['compact']+d['endpoint'],graph_sum=d['compact']+d['endpoint'],extra_to_graph_ratio=d['incremental_L3_after_L1']/max(1e-9,d['compact']+d['endpoint']))
        data.append(d)
    table('candidate_component_costs.csv',data)
    agg=[]
    for yr in (2017,2018,2019,2020,2021):
        rs=[r for r in data if r['year']==yr];assert len(rs)==(5 if yr==2017 else 4)
        for component in ('manifest_L1','incremental_L3_after_L1','compact','endpoint','graph_sum','light_plus_graph','extra_to_graph_ratio'):
            v=np.array([r[component] for r in rs]);agg.append(dict(year=yr,n=len(rs),component=component,mean=float(v.mean()),median=float(np.median(v)),p90=float(np.quantile(v,.9)),min=float(v.min()),max=float(v.max())))
    for component in ('manifest_L1','incremental_L3_after_L1','compact','endpoint','graph_sum','light_plus_graph','extra_to_graph_ratio'):
        v=np.array([r[component] for r in data]);agg.append(dict(year='all21',n=21,component=component,mean=float(v.mean()),median=float(np.median(v)),p90=float(np.quantile(v,.9)),min=float(v.min()),max=float(v.max())))
    table('component_cost_summary.csv',agg)
    source=readcsv(OUT/'source_change_anatomy.csv')
    lines=['# Manifest→DEX引用→图构建：实际候选组件计时与2017错误解剖','', '冻结的U_REF10候选池中，2017按五个外层折各取1个中位大小APK，2018—2021各年按文件大小四分位各取1个SHA哈希确定APK；共21个。选择不看标签和第二层结果。','', '同一进程先运行Manifest L1，再运行DEX Ref L3，各做两轮；L3重复解析Manifest，故计时是当前现有实现的保守实际开销，并非已优化增量。compact与endpoint两个图构建器各在新输出目录运行一次。所有样本通过已有缓存特征一致性检查。','', '|年份|样本数|Manifest均值秒|候选新增DEX均值秒|两个图构建均值秒|新增DEX/图构建比例均值|','|---|---:|---:|---:|---:|---:|']
    for yr in (2017,2018,2019,2020,2021,'all21'):
        q={r['component']:r for r in agg if r['year']==yr}
        lines.append(f"|{yr}|{q['manifest_L1']['n']}|{q['manifest_L1']['mean']:.3f}|{q['incremental_L3_after_L1']['mean']:.3f}|{q['graph_sum']['mean']:.3f}|{q['extra_to_graph_ratio']['mean']:.3f}|")
    lines+=['','## 2017新增/删去路由的实际纠错','', '|种子|变化|APK|第一层错误|修复|误伤|净纠错|','|---:|---|---:|---:|---:|---:|---:|']
    for r in source:lines.append(f"|{r['seed']}|{r['route_change']}|{r['n']}|{r['stage1_errors']}|{r['fixed']}|{r['broken']}|{int(r['fixed'])-int(r['broken'])}|")
    lines+=['','这些是错误位置统计，不能仅凭预测概率断言DEX语义或概念漂移根因。逐APK与FN、FP、TN、TP分解见source_changed_apks.csv。','', '## 成本与证据边界','', '图构建仍不包含C43背景处理和神经网络推理；不是完整端到端总耗时。样本每年4—5个，OS缓存未清理，运行前APK哈希校验会预热文件，进程启动方式也与在线服务不同。因此这些计时不能外推为部署总体均值。未来年份已经用于开发；此轮未选择/调整路由，也未重新训练模型。']
    write('final_report.md','\n'.join(lines)+'\n')

def main():
    if (OUT/'final_decision.json').exists():raise RuntimeError('completed run preserved')
    if not (OUT/'protocol.json').exists():preflight()
    if integrity()['status']!='PASS':raise RuntimeError('asset integrity changed')
    try:
        replay_source();mapping=yaml.safe_load(lc.MAPPING.read_text());roles={r:i for i,r in enumerate(mapping['roles'])};rules=[r for r in mapping['rules'] if r.get('rule_kind','api')=='api']
        assert len(roles)==31
        for i,row in enumerate(load('sample_contract.json'),1):
            run_one(row,roles,rules);graph_one(row)
            print('MSC_SAMPLE_DONE',i,'/21',row['year'],row['sha256'][:8],flush=True)
        summarize();check=integrity();save('baseline_integrity_check.json',check)
        if check['status']!='PASS':raise RuntimeError('asset integrity changed after benchmark')
        save('final_decision.json',dict(status='CANDIDATE_COST_AND_SOURCE_ERROR_AUDIT_COMPLETE',asset_integrity='PASS',sampled_APKs=21,actual_APK_reads=True,APK_rewrites=False,full_end_to_end_cost='UNVERIFIED',new_model_training=False,route_changes=False,elapsed_note='See component_cost_summary.csv',next_action='Assess cost samples and 2017 source failure; do not choose router from future development data.'))
        print('MSC_DONE',OUT,flush=True)
    except Exception as exc:
        save('failure_'+str(time.time_ns())+'.json',dict(error=repr(exc),integrity=integrity()));traceback.print_exc();raise
if __name__=='__main__':main()
