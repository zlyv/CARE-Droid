#!/usr/bin/env python3
"""CARE-EAD-V1: read frozen CSV/NPZ/checkpoints, write only EAD output; diagnostic 2017 probes; no APK reads."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
import numpy as np

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
OUT=ROOT/'outputs/care_error_audit_v1'
sys.path[:0]=[str(ROOT/'scripts/care-background-functional-attribution-v1'),str(ROOT/'scripts/care-dce-v1'),str(ROOT/'scripts/care-scbe-v1'),str(ROOT/'scripts/care-multiyear-development-v1')]
from bfa_core import load_data_contract,formal_split
from train import split_contract
from train_stage2 import raw_bags
from metrics import classification,cascade
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

SPLITS=[f'S{i}' for i in range(1,6)]
SRC=ROOT/'outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv'
CLF=ROOT/'outputs/care-p31-classifier-family-v1/source_predictions.csv'
HDBC=ROOT/'outputs/care_hdbc_v1'
DCE=ROOT/'outputs/care_dce_v1_r3/nested_stage1_oof_directions.csv'
ASSETS=[SRC,CLF,HDBC/'source2017_oof_predictions.csv',HDBC/'future2018_2021_predictions.csv',HDBC/'features/2017.npz',HDBC/'graph_coverage_2017.csv',DCE,ROOT/'outputs/care_multiyear_development_v1/data.npz',ROOT/'outputs/care_multiyear_development_v1/feature_rows.pkl']

def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(name,rows,cols=None):
    rows=list(rows);p=OUT/name
    if p.exists():raise RuntimeError('refuse overwrite '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    cols=cols or list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
def writejson(name,obj):
    p=OUT/name
    if p.exists():raise RuntimeError('refuse overwrite '+str(p))
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def writetext(name,s):
    p=OUT/name
    if p.exists():raise RuntimeError('refuse overwrite '+str(p))
    p.write_text(s,encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)
def num(x):return float(x)
def safeauc(y,p):return float(roc_auc_score(y,p)) if len(set(y))==2 else None
def metric(y,p):
    if not len(y):return dict(n=0,positives=0,auc=None,pr_auc=None,ba=None)
    y=np.asarray(y,int);p=np.asarray(p,float)
    return dict(n=len(y),positives=int(y.sum()),auc=safeauc(y,p),pr_auc=float(average_precision_score(y,p)) if len(set(y))==2 else None,ba=float(balanced_accuracy_score(y,p>=.5)) if len(set(y))==2 else None)

def assets():
    paths=ASSETS+[ROOT/'scripts/care-background-functional-attribution-v1/bfa_core.py',ROOT/'scripts/care-scbe-v1/train.py',ROOT/'scripts/care-multiyear-development-v1/train_stage2.py']
    return [sha(p) for p in paths]
def verify_assets():
    manifest=json.loads((OUT/'asset_manifest_before.json').read_text());changed=[]
    for r in manifest:
        if sha(Path(r['path']))!=r:changed.append(r['path'])
    if changed:raise RuntimeError('historical asset changed '+repr(changed))
    return dict(status='PASS',checked=len(manifest),changed=changed,scope='named reused historical files only; excludes APKs and whole project')

def source_rows():
    all_h=readcsv(HDBC/'source2017_oof_predictions.csv');h=[r for r in all_h if r['arm']=='H0_ORIGINAL']
    if len(h)!=799:raise RuntimeError('H0 must have 799 OOF rows')
    by={(r['sha256'],r['split']):r for r in h}
    if len(by)!=799:raise RuntimeError('H0 duplicate SHA/fold')
    s=[r for r in readcsv(SRC) if r['arm']=='SVM'];rf=[r for r in readcsv(CLF) if r['arm']=='RF'];clsvm=[r for r in readcsv(CLF) if r['arm']=='SVM']
    for rows,name in [(s,'source SVM'),(rf,'RF'),(clsvm,'classifier SVM')]:
        if len(rows)!=799 or len({r['sha256'] for r in rows})!=799:raise RuntimeError(name+' identity')
    return h,s,rf,clsvm

def protocol():
    data,source=load_data_contract();h,s,rf,clsvm=source_rows()
    idx={r['sha256']:r for r in s};rfidx={r['sha256']:r for r in rf};cidx={r['sha256']:r for r in clsvm}
    years={int(y):int(np.sum(data['years']==y)) for y in np.unique(data['years'])}
    if years!={2017:799,2018:793,2019:399,2020:397,2021:400}:raise RuntimeError('year counts')
    folds={s:0 for s in SPLITS};aligned=[]
    for r in h:
        sid=r['sha256'];f=int(r['split'][1:]);q=idx[sid];u=rfidx[sid];v=cidx[sid]
        if f!=int(q['fold']) or f!=int(u['fold']) or f!=int(v['fold']):raise RuntimeError('fold mismatch '+sid)
        if any(int(t['label'])!=int(r['label']) for t in (q,u,v)):raise RuntimeError('label mismatch '+sid)
        if abs(num(q['stage1_prob'])-num(r['stage1_probability']))>1e-10:raise RuntimeError('Stage-I prob mismatch '+sid)
        if int(q['routed'])!=int(r['routed']):raise RuntimeError('route mismatch '+sid)
        if abs(num(v['probability'])-num(q['stage1_prob']))>1e-10:raise RuntimeError('SVM family mismatch '+sid)
        if int(r['routed'])==0 and abs(num(r['final_probability'])-num(r['stage1_probability']))>1e-10:raise RuntimeError('unrouted modified '+sid)
        folds[r['split']]+=1
        y=int(r['label']);b=num(r['stage1_probability']);e=num(r['expert_probability']);z=num(r['final_probability']);route=int(r['routed']);bwrong=int((b>=.5)!=y);fwrong=int((z>=.5)!=y)
        group='ROUTER_MISSED' if bwrong and not route else 'ROUTED_UNFIXED' if bwrong and fwrong else 'FIXED' if bwrong else 'BROKEN' if fwrong else 'ROUTED_CORRECT' if route else 'ACCEPTED_CORRECT'
        aligned.append(dict(year=2017,sha256=sid,fold=r['split'],label=y,stage1_probability=b,router_score='UNKNOWN_SOURCE_OOF_SCORE_NOT_SAVED',routed=route,expert_probability=e,final_probability=z,stage1_wrong=bwrong,final_wrong=fwrong,group=group,rf_probability=num(u['probability']),stage1_direction=int(b>=.5)))
    if sorted(folds.values())!=[159,160,160,160,160]:raise RuntimeError('fold sizes')
    aligned.sort(key=lambda r:r['sha256']);y=np.array([r['label'] for r in aligned]);b=np.array([r['stage1_probability'] for r in aligned]);z=np.array([r['final_probability'] for r in aligned]);e=np.array([r['expert_probability'] for r in aligned]);route=np.array([r['routed'] for r in aligned],bool);rp=np.array([r['rf_probability'] for r in aligned])
    m=cascade(y,b,e,route);rfc=classification(y,rp);sc=classification(y,b)
    if (sc['FP']+sc['FN'],rfc['FP']+rfc['FN'],m['Fixed'],m['Broken'],m['FP']+m['FN'])!=(97,58,45,15,67):raise RuntimeError('historical accounting mismatch')
    writecsv('aligned_oof_predictions.csv',aligned);writecsv('error_group_assignments.csv',aligned)
    writecsv('baseline_identity_matrix.csv',[dict(arm='P31_RF',n=799,errors=58,BA=rfc['BA'],source=str(CLF),model='RF',fold='same fixed fivefold',threshold=.5),dict(arm='P31_SVM_HDBC_H0',n=799,errors=97,BA=sc['BA'],source=str(SRC),model='calibrated SVM',fold='same fixed fivefold',threshold=.5),dict(arm='HDBC_H0_DIRECT_C43',n=799,errors=67,BA=m['BA'],source=str(HDBC/'source2017_oof_predictions.csv'),model='SCBE M0 Direct-C43',fold='same fixed fivefold',threshold=.5)])
    writecsv('stage1_comparison.csv',[dict(model='RF',**rfc),dict(model='SVM',**sc)])
    writecsv('fixed_broken_consistency.csv',[dict(stage1_errors=97,final_errors=67,fixed=45,broken=15,identity_holds=97-45+15==67,routed=int(route.sum()),routed_errors=m['routed_errors'],missed=m['unrouted_errors'],**{k:m[k] for k in ('Fixed_FN','Fixed_FP','Broken_TN','Broken_TP')})])
    writetext('protocol_audit_report.md',f'# 2017协议审计\n\nRF与SVM使用相同799个SHA、同一五折、0.5阈值；逐行Fold、标签和SVM概率完全对齐。58个错误来自P31-RF（BA={rfc["BA"]:.6f}），不是P31-SVM。97个错误来自P31-SVM（BA={sc["BA"]:.6f}），是HDBC H0的第一层。H0基于SCBE M0冻结Direct-C43，已路由{m["routed"]}，捕获{m["routed_errors"]}，漏掉{m["unrouted_errors"]}；Fixed={m["Fixed"]}、Broken={m["Broken"]}、最终错误={m["FP"]+m["FN"]}，恒等式97−45+15=67成立。\n\n分类器家族报告的RF与SVM为不同第一层模型，不能把RF的58与SVM+C43的Fixed/Broken横向相减。Source MeanOpposition逐行路由一致，但源域路由分数未在该H0表中保存，标UNKNOWN。2017数据为五折OOF；H0种子900000/900001/900002平均。未来为2017训练模型的冻结历史推理，不等同OOF。\n')
    writejson('protocol.json',dict(years=years,folds=folds,source_classes={str(x):int((y==x).sum()) for x in (0,1)},stage1_svm_errors=97,stage1_rf_errors=58,legal_baseline='P31_SVM_HDBC_H0',h0=m,train_year=2017,future_role='Historical Development Diagnostics, Not Independent Blind Testing'))
    print('EAD_PROTOCOL_PASS RF=58 SVM=97 H0_fixed=45 broken=15',flush=True)

def raw_features(data):
    ids=list(map(str,data['ids']));bags=raw_bags(data,np.arange(len(ids)));p31=np.asarray(data['p31'],float);s7=np.asarray(data['s7'],float)
    if p31.shape[1]!=31 or s7.shape[1]<1546:raise RuntimeError('raw feature widths '+repr((p31.shape,s7.shape)))
    out={};ev=[]
    cover={r['sha256']:r for r in readcsv(HDBC/'graph_coverage_2017.csv')}
    for i,sid in enumerate(ids):
        bag=np.asarray(bags[i],float);hc=len(bag);role=bag[:,43:74] if hc else np.empty((0,31));core=bag[:,:43] if hc else np.empty((0,43));active=(role>0).sum(axis=1) if hc else np.array([])
        host=np.array([hc,int((active>0).sum()),int((active>1).sum()),float(np.mean(active)) if hc else 0,float(np.std(active)) if hc else 0],float)
        c43=np.r_[core.mean(axis=0),core.max(axis=0)] if hc else np.zeros(86)
        g=s7[i];rolegraph=np.r_[g[:120],g[120:585],np.array([np.count_nonzero(g[585:]),np.sum(np.abs(g[585:])),np.max(np.abs(g[585:]))])]
        gr=cover.get(sid);status='MISSING_CACHE' if int(data['years'][i])==2017 and gr is None else 'UNRESOLVED_GRAPH' if gr and int(gr['graph_available'])==0 else 'ZERO_OBSERVED' if gr and int(gr['usable_relation'])==0 else 'PRESENT' if gr else 'UNKNOWN'
        external='UNKNOWN_NOT_REPLAYED';internal='UNKNOWN_NOT_REPLAYED';shared='UNKNOWN_NOT_REPLAYED'
        ev.append(dict(sha256=sid,year=int(data['years'][i]),host_count=hc,valid_role_hosts=int((active>0).sum()),multi_role_hosts=int((active>1).sum()),active_p31_roles=int((p31[i]>0).sum()),p31_status='PRESENT' if np.any(p31[i]) else 'ZERO_OBSERVED',host_status='PRESENT' if hc else 'ZERO_OBSERVED',rolegraph_status=status,rolegraph_nonzero=int(np.count_nonzero(g)),interz_nonzero=int(np.count_nonzero(g[585:])),direct_edges=gr['direct_method_edges'] if gr else 'UNKNOWN',one_hop_pairs=gr['one_hop_host_pairs'] if gr else 'UNKNOWN',two_hop_pairs=gr['two_hop_host_pairs'] if gr else 'UNKNOWN',shared_status=shared,external_status=external,internal_status=internal))
        out[sid]=dict(B1_P31=p31[i],B2_RAW43=c43,B3_HOST=host,B4_ROLEGRAPH=rolegraph)
    return out,ev

def evidence():
    data,_=load_data_contract();out,ev=raw_features(data);src={r['sha256']:r for r in readcsv(OUT/'aligned_oof_predictions.csv')}
    rows=[]
    for r in ev:
        if r['year']!=2017:continue
        v=src[r['sha256']];rows.append(dict(**r,fold=v['fold'],label=v['label'],group=v['group'],stage1_wrong=v['stage1_wrong'],routed=v['routed']))
    writecsv('evidence_availability_analysis.csv',rows);writecsv('host_role_coverage.csv',rows)
    grouped=[]
    for group in sorted({r['group'] for r in rows}):
        z=[r for r in rows if r['group']==group];grouped.append(dict(group=group,n=len(z),host_zero=sum(r['host_status']=='ZERO_OBSERVED' for r in z),graph_unresolved=sum(r['rolegraph_status']=='UNRESOLVED_GRAPH' for r in z),graph_zero=sum(r['rolegraph_status']=='ZERO_OBSERVED' for r in z),graph_present=sum(r['rolegraph_status']=='PRESENT' for r in z),median_hosts=float(np.median([int(r['host_count']) for r in z]))))
    writecsv('evidence_missingness_summary.csv',grouped)
    writetext('evidence_missingness_report.md','# 静态证据可用性\n\n逐APK P31、原始Host43及冻结S7来自已有缓存；RoleGraph可用性来自HDBC已冻结调用图覆盖。PRESENT只证明结构存在，不证明恶意语义。ZERO_OBSERVED表示成功读到缓存但值为零。MISSING_CACHE/UNRESOLVED_GRAPH与真实零严格分开。三背景响应须同一Outer Fold/种子前向重放；本轮若未成功重放，保持UNKNOWN，不用原型距离冒充。\n')
    print('EAD_EVIDENCE_PASS',len(rows),flush=True)

def probe():
    data,_=load_data_contract();raw,_=raw_features(data);test={r['sha256']:r for r in readcsv(OUT/'aligned_oof_predictions.csv')};nested=readcsv(DCE)
    trmap={(r['split'],r['sha256']):r for r in nested}
    if len(trmap)!=len(nested):raise RuntimeError('nested duplicate')
    designs=['B0_STAGE1','B1_P31','B2_RAW43','B3_HOST','B4_ROLEGRAPH','B6_COMBINED','B6_SHUFFLE_LABEL','B6_MLP']
    def vec(sid,kind,prob):
        if kind=='B0_STAGE1':return np.array([prob])
        if kind=='B6_COMBINED' or kind.startswith('B6_'):return np.r_[raw[sid]['B1_P31'],raw[sid]['B2_RAW43'],raw[sid]['B3_HOST'],raw[sid]['B4_ROLEGRAPH']]
        return raw[sid][kind]
    rows=[];folds=[];sourceids={s for s in test}
    for split in SPLITS:
        va=[r for r in test.values() if r['fold']==split];tr=[r for (sp,sid),r in trmap.items() if sp==split]
        if len(va) not in (159,160) or len(tr)!=799-len(va):raise RuntimeError('nested fold size '+split)
        if {r['sha256'] for r in tr}&{r['sha256'] for r in va}:raise RuntimeError('nested leakage')
        if any(int(r['outer_test_excluded'])!=1 for r in tr):raise RuntimeError('nested outside outer train')
        if {r['sha256'] for r in tr}|{r['sha256'] for r in va}!=sourceids:raise RuntimeError('nested source coverage')
        for direction in (0,1):
            train=[r for r in tr if int(r['stage1_oof_direction'])==direction];testpart=[r for r in va if int(r['stage1_direction'])==direction]
            yt=np.array([int(r['label']) for r in train]);yv=np.array([int(r['label']) for r in testpart]);print('EAD_PROBE',split,direction,'train',len(train),'test',len(testpart),flush=True)
            for kind in designs:
                if len(np.unique(yt))<2 or not len(testpart):
                    folds.append(dict(fold=split,direction=direction,arm=kind,status='UNDEFINED_CLASS_SUPPORT',train_n=len(train),test_n=len(testpart)));continue
                x=np.vstack([vec(r['sha256'],kind,float(r['stage1_oof_probability'])) for r in train]);v=np.vstack([vec(r['sha256'],kind,float(r['stage1_probability'])) for r in testpart]);yfit=yt.copy()
                if kind=='B6_SHUFFLE_LABEL':yfit=np.random.RandomState(2718+int(split[1:])+direction).permutation(yfit)
                if kind=='B6_MLP':clf=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),MLPClassifier(hidden_layer_sizes=(16,),alpha=1,max_iter=120,early_stopping=True,random_state=2718))
                else:clf=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=500,random_state=2718))
                clf.fit(x,yfit);pred=clf.predict_proba(v)[:,1]
                m=metric(yv,pred);folds.append(dict(fold=split,direction=direction,arm=kind,status='STRICT_OUTER_OOF',train_n=len(train),test_n=len(testpart),**m))
                for r,p in zip(testpart,pred):rows.append(dict(sha256=r['sha256'],fold=split,label=int(r['label']),direction=direction,arm=kind,probability=float(p),routed=int(r['routed'])))
    # Frozen Direct-C43 is a score reference, never an in-sample probe feature.
    for r in test.values():rows.append(dict(sha256=r['sha256'],fold=r['fold'],label=int(r['label']),direction=int(r['stage1_direction']),arm='B7_FROZEN_DIRECT_C43_SCORE',probability=float(r['expert_probability']),routed=int(r['routed'])))
    writecsv('probe_oof_predictions.csv',rows);writecsv('raw_feature_probe_results.csv',folds)
    summary=[]
    for arm in designs+['B7_FROZEN_DIRECT_C43_SCORE']:
      for direction in (0,1):
       for subset in ('all','routed','unrouted'):
        z=[r for r in rows if r['arm']==arm and int(r['direction'])==direction and (subset=='all' or (int(r['routed'])==1)==(subset=='routed'))]
        summary.append(dict(arm=arm,direction=direction,subset=subset,**metric([int(r['label']) for r in z],[float(r['probability']) for r in z])))
    writecsv('conditional_auc_results.csv',summary)
    print('EAD_PROBES_DONE',len(rows),flush=True)

def routing_cases():
    a=readcsv(OUT/'aligned_oof_predictions.csv');ev={r['sha256']:r for r in readcsv(OUT/'evidence_availability_analysis.csv')}
    groups={g:sum(r['group']==g for r in a) for g in set(r['group'] for r in a)};err=sum(int(r['stage1_wrong']) for r in a);route=sum(int(r['routed']) for r in a)
    report=dict(stage1_errors=err,route=route,routed_errors=groups.get('FIXED',0)+groups.get('ROUTED_UNFIXED',0),missed=groups.get('ROUTER_MISSED',0),fixed=groups.get('FIXED',0),unfixed=groups.get('ROUTED_UNFIXED',0),broken=groups.get('BROKEN',0),oracle_fixed_router_errors=groups.get('ROUTER_MISSED',0),oracle_perfect_router_errors=max(0,err-route),non_deployable_oracle=True)
    if report['routed_errors']!=84 or report['missed']!=13:raise RuntimeError('route accounting')
    writecsv('routing_error_analysis.csv',[report]);writecsv('routing_error_budget.csv',[report]);writecsv('stage2_fixed_broken_analysis.csv',[dict(group=g,count=n) for g,n in sorted(groups.items())])
    writetext('routing_oracle_bounds.md',f'# 非部署Oracle\n\n当前Top25%路由漏掉{report["missed"]}个Stage-I错误。固定路由完美专家仍至少留下{report["missed"]}个错误；固定200名额完美选取错误并完美分类理论上可修复全部97个错误。两者均用真实标签事后计算，不可部署，也不意味着Direct-C43会修复所有新增路由。\n')
    chosen=[]
    for group in ('ROUTER_MISSED','ROUTED_UNFIXED','FIXED','BROKEN'):
      for y in (0,1):
       z=[r for r in a if r['group']==group and int(r['label'])==y]
       z.sort(key=lambda r:(abs(float(r['final_probability'])-.5),r['sha256']))
       for r in z[:3]:
        e=ev[r['sha256']];chosen.append(dict(**r,host_count=e['host_count'],p31_roles=e['active_p31_roles'],graph_status=e['rolegraph_status'],interz_nonzero=e['interz_nonzero'],background='UNKNOWN_NOT_REPLAYED',dex_call_witness='NOT_VERIFIED_FROM_SUMMARY_CACHE',candidate='ROUTER_MISSED' if group=='ROUTER_MISSED' else 'SECOND_STAGE_OVER_CORRECTION' if group=='BROKEN' else 'UNDETERMINED'))
    writecsv('hard_case_analysis.csv',chosen)
    writetext('hard_case_evidence.md','# 困难案例证据边界\n\n案例按2017 OOF类别、与0.5距离及SHA固定排序抽取，不使用未来标签选例。HDBC图覆盖和InterZ摘要是可验证缓存统计，但当前汇总文件不包含原始方法签名和DEX偏移，因此不得宣称具体调用链或行为发生；相关列明确标记NOT_VERIFIED。\n')
    print('EAD_ROUTING_CASES_DONE',flush=True)

def representation():
    # The existing representation store is a distinct Direct-C43 checkpoint branch.
    # It cannot be silently paired with HDBC H0; leave the probe explicitly unrun.
    writecsv('embedding_probe_results.csv',[dict(status='NOT_COMPARABLE_WITH_HDBC_H0',reason='available representation_store.npz belongs to another Direct-C43 checkpoint; SCBE M0 train embeddings are in-sample and strict inner cross-fit is unavailable',n=0)])
    writecsv('representation_discriminability.csv',[dict(comparison='raw vs frozen score',status='SEE_CONDITIONAL_AUC_RESULTS',causal_claim='NONE'),dict(comparison='raw vs embedding',status='UNDEFINED_NO_STRICT_CROSS_FIT',causal_claim='NONE')])
    writecsv('paired_statistical_analysis.csv',[dict(comparison='raw vs embedding',status='NOT_RUN',reason='checkpoint/embedding protocol mismatch; no fabricated confidence interval')])
    writetext('representation_gap_report.md','# 表示层结论\n\nHDBC H0与已有representation_store.npz并非同一冻结checkpoint分支；SCBE M0的外层训练Embedding是拟合内，测试Embedding为折外。不能把两者直接训练/评价后称作严格OOF表示Probe。本轮保留合法原始特征Probe与冻结H0分数的同APK诊断比较，但不作“表征丢失”或“分类头未利用”的断言。\n')
    print('EAD_REPRESENTATION_BOUNDARY_RECORDED',flush=True)

def future():
    rows=[r for r in readcsv(HDBC/'future2018_2021_predictions.csv') if r['arm']=='H0_ORIGINAL'];out=[]
    for year in (2018,2019,2020,2021):
      z=[r for r in rows if int(r['year'])==year];y=np.array([int(r['label']) for r in z]);b=np.array([float(r['stage1_probability']) for r in z]);e=np.array([float(r['expert_probability']) for r in z]);route=np.array([int(r['routed']) for r in z],bool)
      expected={2018:793,2019:399,2020:397,2021:400}[year]
      if len(z)!=expected or len({r['sha256'] for r in z})!=len(z):raise RuntimeError('future cohort identity')
      m=cascade(y,b,e,route);out.append(dict(year=year,scope='Historical Development Diagnostics, Not Independent Blind Testing',**m))
    writecsv('future_frozen_diagnostics.csv',out);print('EAD_FUTURE_DONE',flush=True)

def finish():
    a=readcsv(OUT/'aligned_oof_predictions.csv');ev={r['sha256']:r for r in readcsv(OUT/'evidence_availability_analysis.csv')};groups={g:sum(r['group']==g for r in a) for g in set(r['group'] for r in a)};cond=readcsv(OUT/'conditional_auc_results.csv');future_rows=readcsv(OUT/'future_frozen_diagnostics.csv')
    def find(arm,d,sub):
      z=[r for r in cond if r['arm']==arm and int(r['direction'])==d and r['subset']==sub];return z[0] if z else None
    matrix=[dict(cause='Router遗漏',n=groups.get('ROUTER_MISSED',0),support='direct OOF route status',causal='operational exclusion'),dict(cause='路由内未修复',n=groups.get('ROUTED_UNFIXED',0),support='direct OOF decisions',causal='mechanism unknown'),dict(cause='第二层误伤',n=groups.get('BROKEN',0),support='direct OOF decisions',causal='intervention changes decision, why unknown'),dict(cause='图证据缺失/未解',n=sum(ev[r['sha256']]['rolegraph_status'] in ('MISSING_CACHE','UNRESOLVED_GRAPH') for r in a if r['group'] in ('ROUTED_UNFIXED','BROKEN')),support='frozen graph coverage',causal='association only'),dict(cause='表示损失',n='UNKNOWN',support='strict cross-fitted embedding unavailable',causal='not established')]
    writecsv('error_attribution_matrix.csv',matrix)
    writejson('asset_integrity_report.json',verify_assets())
    status='NO_DOMINANT_CAUSE_IDENTIFIED';rawfn=find('B6_COMBINED',0,'routed');scorefn=find('B7_FROZEN_DIRECT_C43_SCORE',0,'routed')
    report=f'''# CARE-EAD-V1 错误归因与证据可分性\n\n状态：{status}。2017唯一诊断Probe训练来源；2018—2021为 **Historical Development Diagnostics, Not Independent Blind Testing**。未修改正式模型。\n\n## 协议与错误\n\n58个Stage-I错误属于P31-RF，不属于当前P31-SVM；同799 SHA与五折、阈值0.5。当前HDBC H0合法基线是P31-SVM 97错，路由200，捕获84，漏13；Direct-C43修复45、破坏15、净+30，最终67错。历史不同分支的Direct-C43数字不能互换。\n\n## 可用证据与条件可分性\n\n2017证据表逐APK区分PRESENT、ZERO_OBSERVED、MISSING_CACHE、UNRESOLVED_GRAPH和UNKNOWN。三背景没有安全重放，保持UNKNOWN；没有将“无数据”计作“零证据”。严格嵌套Stage-I方向来自DCE已保存的outer-train内层OOF，原始Probe仅2017外层训练，评价外层测试。多特征组与标签置乱为多重探索比较，不得据此宣称新检测器增益。路由内FN/TN原始组合AUC={rawfn['auc'] if rawfn else 'UNKNOWN'}，冻结H0分数AUC={scorefn['auc'] if scorefn else 'UNKNOWN'}；具体样本量见conditional_auc_results.csv。\n\n## 表示、路由及限制\n\n严格交叉拟合的同checkpoint表示不存在；现成representation_store属于另一模型版本，故不报告Raw-vs-Embedding差异，不推断表示损失或分类头瓶颈。固定路由完美专家的事后下界仍留13错；完美200名额Router+专家是非部署Oracle。当前实验不能判断漏路由错误是否可由真实第二层修复。案例仅使用缓存结构统计；没有DEX方法签名/指令偏移验证的行为不作语义声称。\n\n## 下一步\n\n维持Direct-C43。若需证明表示瓶颈，先在2017外层训练集内部构造同checkpoint严格交叉拟合Embedding；若需研究漏路由收益，须为漏路由APK取得合法OOF专家预测。现有结果不足以授权新模块。\n'''
    writetext('final_report.md',report)
    writejson('final_decision.json',dict(status=status,secondary=['PROTOCOL_MISMATCH_FOUND'],stage1_rf_errors=58,stage1_svm_errors=97,routed=200,missed=13,fixed=45,broken=15,net=30,strict_embedding_gap_test='NOT_AVAILABLE',future_used_for_training=False,formal_asset_integrity='PASS',next_action='retain frozen Direct-C43; obtain same-checkpoint strictly cross-fitted embeddings before representation claim'))
    writetext('reproduce_commands.md','`bash scripts/care-error-audit-v1/run.sh --resume`\n`bash scripts/care-error-audit-v1/status.sh`\n')
    print('EAD_FINAL_DONE',flush=True)

def selftest():
    data,source=load_data_contract();src=np.flatnonzero(data['years']==2017)
    if len(src)!=799 or (sum(data['y'][src]==0),sum(data['y'][src]==1))!=(400,399):raise RuntimeError('2017 inventory')
    folds=[]
    for split in SPLITS:
      tr,va,b,route,_=split_contract(data,split,source);folds.append(len(va))
      if len(tr)+len(va)!=799 or route.sum()!=40 or set(data['ids'][tr])&set(data['ids'][va]):raise RuntimeError('fold/route')
    if sorted(folds)!=[159,160,160,160,160]:raise RuntimeError('fold count')
    h,s,rf,clsvm=source_rows();n=readcsv(DCE)
    if len(n)!=5*799-799 or any(int(r['outer_test_excluded'])!=1 for r in n):raise RuntimeError('nested direction')
    x=np.array([[0],[1],[2],[3]],float);y=np.array([0,0,1,1]);SimpleImputer(strategy='median',keep_empty_features=True)
    clf=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression()).fit(x,y)
    if clf.predict_proba([[1.5]]).shape!=(1,2) or safeauc([0,1],[.1,.9])!=1:raise RuntimeError('probe/auc')
    if classification(y,np.array([.1,.2,.8,.9]))['BA']!=1:raise RuntimeError('metric')
    if not OUT.is_dir() or not os.access(OUT,os.W_OK):raise RuntimeError('output not writable')
    print('EAD_SELFTEST_PASS 799 fixed folds nested directions trainable probe AUC output resume',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['selftest','protocol','evidence','probe','representation','routing','future','finish'],required=True);args=ap.parse_args();globals()[args.phase if args.phase!='routing' else 'routing_cases']()
if __name__=='__main__':main()
