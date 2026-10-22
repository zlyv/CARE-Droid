#!/usr/bin/env python3
"""HDBC-V1 runner: reads frozen graph/CARE caches, never APKs; writes only HDBC_OUTPUT."""
from __future__ import annotations
import argparse,csv,json,os,pickle,sys,time
from pathlib import Path
import numpy as np
ROOT=Path('/home/boyuanjie/projects/tasa-droid')
HERE=ROOT/'scripts/care-hdbc-v1'
OUT=Path(os.environ.get('HDBC_OUTPUT',ROOT/'outputs/care_hdbc_v1')).resolve()
if ROOT/'outputs' not in OUT.parents:raise RuntimeError('HDBC_OUTPUT must be under project outputs')
sys.path[:0]=[str(HERE),str(ROOT/'scripts/care-dce-v1'),str(ROOT/'scripts/care-background-functional-attribution-v1'),
 str(ROOT/'scripts/care-scbe-v1'),str(ROOT/'scripts/care-multiyear-development-v1')]
from bfa_core import SEEDS,digest,formal_split,load_data_contract,load_m0
from train import split_contract,global24
from train_stage2 import raw_bags,transform_saved
from metrics import cascade,paired_intervals
from dce_core import extract,sigmoid
from graph_features import cache_rows,extract_one,select_vocab,selftest,DIRECT_N,UNDIR_N
from fusion import ARMS,fit,predict
SPLITS=('S1','S2','S3','S4','S5')
YEARS={2017:799,2018:793,2019:399,2020:397,2021:400}
SOURCE_CACHE=ROOT/'artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz'
FUTURE_CACHE=ROOT/'artifacts/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz'

def serial(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def jwrite(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,default=serial,allow_nan=False)
def cwrite(name,rows):
    rows=list(rows);p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    cols=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,cols);w.writeheader();w.writerows(rows)
def cread(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def pwrite(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
def pread(path):
    with path.open('rb') as f:return pickle.load(f)
def asset_paths():
    old=ROOT/'outputs/care_multiyear_development_v1'
    paths=[HERE/'main.py',HERE/'graph_features.py',HERE/'fusion.py',ROOT/'scripts/care-dce-v1/dce_core.py',
      ROOT/'scripts/care-background-functional-attribution-v1/bfa_core.py',
      ROOT/'scripts/care-scbe-v1/train.py',ROOT/'scripts/care-multiyear-development-v1/train_stage2.py',
      ROOT/'scripts/care-multiyear-development-v1/metrics.py',
      ROOT/'scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py',
      ROOT/'scripts/run_clipped_intra_inter_structure_ablation_2017.py',
      ROOT/'scripts/audit_host_role_null_model_assets.py',SOURCE_CACHE,FUTURE_CACHE,
      old/'data.npz',old/'feature_rows.pkl',old/'sample_manifest.csv',
      ROOT/'outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv',
      ROOT/'outputs/care-svm-direct-c43-future-v1/predictions_label_free.csv']
    for split in SPLITS+('T1',):
      for seed in SEEDS:paths.append(ROOT/'outputs/care_scbe_v1/models'/split/str(seed)/'M0_REPRODUCED.pkl')
    return paths
def integrity():
    before=json.loads((OUT/'asset_manifest_before.json').read_text())
    bad=[]
    for r in before:
      try:
        if digest(r['path'])!=r:bad.append(r['path'])
      except OSError:bad.append(r['path'])
    return dict(status='PASS' if not bad else 'FAIL',checked=len(before),changed=bad,
      scope='named reused source, graph caches and C43 checkpoints only; no APKs opened; not whole server')
def preflight(resume,allow_future=False):
    if OUT.exists():
      if (OUT/'asset_manifest_before.json').exists():
        if not resume:raise RuntimeError('occupied HDBC result; use --resume')
      elif resume or any(p.name not in ('run.log','launch_info.txt','run_state.json','phase_status.json') for p in OUT.iterdir()):
        raise RuntimeError('occupied output without HDBC manifest')
    else:OUT.mkdir(parents=True,exist_ok=False)
    if not (OUT/'asset_manifest_before.json').exists():jwrite('asset_manifest_before.json',[digest(p) for p in asset_paths()])
    if integrity()['status']!='PASS':raise RuntimeError('reused asset changed')
    data,source=load_data_contract()
    if {y:int(np.sum(data['years']==y)) for y in YEARS}!=YEARS:raise RuntimeError('year count mismatch')
    src=np.flatnonzero(data['years']==2017)
    if len(src)!=799 or tuple(int(np.sum(data['y'][src]==v)) for v in (0,1))!=(400,399):raise RuntimeError('source label contract')
    folds=[]
    for split in SPLITS:
      tr,va,base,route,_=split_contract(data,split,source)
      if len(va) not in (159,160) or int(route.sum())!=40 or not np.all(data['years'][tr]==2017):raise RuntimeError('frozen fold/route')
      folds.extend(dict(sha256=str(data['ids'][i]),fold=split,year=2017) for i in va)
    if len(folds)!=799 or len(set(r['sha256'] for r in folds))!=799:raise RuntimeError('fold identity')
    if not (OUT/'dataset_protocol.json').exists():
      jwrite('dataset_protocol.json',dict(year_counts=YEARS,training_years=[2017],source_class_counts=[400,399],
        folds=[160,160,160,160,159],seeds=SEEDS,route='frozen SVM MeanOpposition Top25%',threshold=.5,
        directed_features='DIRECT_CALL only; 1/2-hop; Host anchored',head_features=128,
        original_encoder_frozen=True,train_embedding_is_in_sample_for_frozen_C43=True,
        future_is_historical_development_diagnostic=True))
      cwrite('source2017_fold_manifest.csv',folds)
      (OUT/'background_code_equivalence.md').write_text(
        '# 与旧RoleGraph的代码级关系\n\n正式S7的InterZ961由`run_clipped_intra_inter_structure_ablation_2017.py:inter_counts`'
        '对Host->Host有向边和Host Role组合计算；因此HDBC的1-hop有向Role计数不是全新证据。'
        '本实验仅将缓存中的DIRECT_CALL方法边严格对齐Host，增加显式2-hop见证、分跳数编码与方向/无向/置乱对照。'
        '若H4不超过H2/H3/H5，不得宣称新增有向类别信息。\n',encoding='utf-8')
      prior=ROOT/'artifacts/care-p31rf-c43-haig-host-ablation-source-v1/alignment_probe.json'
      if prior.is_file():
        jwrite('host_method_mapping_audit.json',dict(
          prior_exact_identity_alignment=json.loads(prior.read_text()),
          hdbc_host_definition='cached ordinary method directly calls Role-bearing endpoint via DIRECT_CALL',
          method_id='endpoint cache node_ids, SHA256(signature) first unsigned 64 bits',
          path_proof='cached directed edge_index with edge_features[2]==0',
          caveat='hashed ID is reproducible within cached graph; raw DEX signature not persisted'))
    if not allow_future:
      data['y']=data['y'].copy();data['y'][data['years']>2017]=-1
    print('HDBC_PREFLIGHT_PASS',flush=True)
    return data,source

def feature_path(year):return OUT/'features'/f'{year}.npz'
def features(year,ids=None):
    with np.load(feature_path(year),allow_pickle=False) as a:
      all_ids=list(map(str,a['ids']))
      ret={k:a[k].copy() for k in ('directed','undirected','shuffled','diag','coverage','pairraw')}
    if ids is None:return all_ids,ret
    ix={s:i for i,s in enumerate(all_ids)}
    if len(ix)!=len(all_ids) or any(s not in ix for s in ids):raise RuntimeError('feature SHA coverage '+str(year))
    return {k:v[[ix[s] for s in ids]] for k,v in ret.items()}
def make_features(data,year):
    if feature_path(year).exists():return
    start=time.perf_counter()
    target=np.flatnonzero(data['years']==year);ids=list(map(str,data['ids'][target]))
    cache=cache_rows(SOURCE_CACHE if year==2017 else FUTURE_CACHE)
    if any(s not in cache for s in ids):raise RuntimeError('graph cache missing '+str(year))
    direct=[];und=[];shuffle=[];diag=[];cover=[];evidence=[]
    for n,sid in enumerate(ids,1):
      payload=cache[sid]['payload']
      d,u,g,c,p=extract_one(sid,payload)
      ds,_,_,cs,_=extract_one(sid,payload,shuffled=True)
      if c['host_count']!=cs['host_count'] or c['direct_method_edges']!=cs['direct_method_edges']:
        raise RuntimeError('H5 graph invariant')
      direct.append(d);und.append(u);shuffle.append(ds);diag.append(g);cover.append(c);evidence.extend(p)
      if n%200==0:print('HDBC_GRAPH',year,n,'/',len(ids),flush=True)
    pairraw=np.maximum(data['s7'][target,120:585],0).astype(np.float32)
    if pairraw.shape!=(len(ids),465):raise RuntimeError('PairRaw465 input contract')
    feature_path(year).parent.mkdir(parents=True,exist_ok=True)
    with feature_path(year).open('xb') as f:
      np.savez_compressed(f,ids=np.asarray(ids),directed=np.asarray(direct),undirected=np.asarray(und),
        shuffled=np.asarray(shuffle),diag=np.asarray(diag),coverage=np.asarray([r['usable_relation'] for r in cover]),pairraw=pairraw)
    cwrite(f'graph_coverage_{year}.csv',cover)
    cwrite(f'directed_evidence_{year}.csv',evidence)
    cwrite(f'feature_runtime_{year}.csv',[dict(year=year,apks=len(ids),seconds=time.perf_counter()-start,
      proof_rows=len(evidence),directed_width=DIRECT_N,undirected_width=UNDIR_N)])
    print('HDBC_FEATURE_YEAR_DONE',year,flush=True)

def block_features(feature,arm,train,all_train,selected=None):
    if arm==ARMS[1]:
      x=np.zeros((len(feature['diag']),128),np.float32);x[:,:4]=feature['diag'];return x,np.arange(128),np.zeros(128,int)
    if arm==ARMS[2]:key='pairraw'
    elif arm==ARMS[3]:key='undirected'
    elif arm==ARMS[4]:key='directed'
    elif arm==ARMS[5]:key='shuffled'
    else:raise ValueError(arm)
    matrix=feature[key]
    if selected is None:
      # H5 uses H4's vocabulary, so both arms represent the same Role pairs.
      select_matrix=feature['directed'] if arm==ARMS[5] else matrix
      selected,support=select_vocab(select_matrix,train,128)
    else:support=np.count_nonzero(matrix[train][:,selected]>0,axis=0)
    x=matrix[:,selected].astype(np.float32)
    if not np.isfinite(x).all():raise RuntimeError('nonfinite structural feature')
    return x,selected,support

def model_path(split,seed,arm):return OUT/'models'/split/str(seed)/(arm+'.pkl')
def source_train(data,source,device,resume,smoke=False):
    selected_splits=SPLITS[:1] if smoke else SPLITS
    vocab_rows=[]
    for split in selected_splits:
      tr,va,_,_,_=split_contract(data,split,source)
      allidx=np.r_[tr,va];ids=list(map(str,data['ids'][allidx]));feat=features(2017,ids)
      itr=np.arange(len(tr));ival=np.arange(len(tr),len(allidx))
      if smoke:itr=np.r_[itr[data['y'][tr]==0][:24],itr[data['y'][tr]==1][:24]];ival=ival[:6]
      for seed in SEEDS[:1] if smoke else SEEDS:
        saved=[model_path(split,seed,arm) for arm in ARMS[1:]]
        if resume and not smoke and all(p.exists() for p in saved):
          print('HDBC_SOURCE_RESUME',split,seed,flush=True);continue
        pack=formal_split(data,split,seed,device)
        ztr,btr=extract(pack['v4'],pack['model'],pack['global_train'],pack['graph_train'],pack['bags_train'],pack['dev'])
        zva,bva=extract(pack['v4'],pack['model'],pack['global_val'],pack['graph_val'],pack['bags_val'],pack['dev'])
        if np.max(np.abs(sigmoid(bva)-np.asarray(pack['old']['probability'])))>1e-6:
          raise RuntimeError('frozen C43 replay')
        z=np.concatenate([ztr,zva]);b=np.r_[btr,bva];y=np.asarray(data['y'][tr],int)
        if smoke:y=y[itr]
        h4_vocab=None
        for arm in ARMS[1:]:
          path=model_path(split,seed,arm)
          if path.exists():
            if not resume:raise RuntimeError('existing checkpoint')
            print('HDBC_ARM_RESUME',split,seed,arm,flush=True);continue
          x,vocab,support=block_features(feat,arm,itr,allidx,selected=h4_vocab if arm==ARMS[5] else None)
          if arm==ARMS[4]:h4_vocab=vocab.copy()
          if not smoke and seed==SEEDS[0]:
            vocab_rows.extend(dict(split=split,arm=arm,rank=j,feature_index=int(k),train_APK_support=int(s))
              for j,(k,s) in enumerate(zip(vocab,support),1))
          start=time.perf_counter()
          obj=fit(z[itr],x[itr],b[itr],y,seed,float(pack['args'].learning_rate),float(pack['args'].weight_decay))
          valp=predict(obj,z[ival],x[ival],b[ival]);obj.update(arm=arm,split=split,seed=seed,
            ids=[ids[j] for j in ival],probability=valp,feature_vocabulary=vocab,
            vocabulary_support=support,train_n=len(itr),seconds=time.perf_counter()-start)
          if not smoke:pwrite(path,obj)
          print('HDBC_ARM_DONE',split,seed,arm,flush=True)
        if smoke:
          jwrite('smoke_test.json',dict(status='PASS',source_fold=split,seed=seed,
            real_directed_edges=True,one_two_hop_path_proofs=True,all_five_controls_forward_backward=True,
            frozen_C43_replay=True,train_rows=len(itr),validation_rows=len(ival),future_rows_loaded=0))
          print('HDBC_SMOKE_PASS',flush=True);return
    if not (OUT/'hdbc_vocabulary_by_fold.csv').exists():
      vocab_rows=[]
      for split in SPLITS:
        for arm in ARMS[1:]:
          obj=pread(model_path(split,SEEDS[0],arm))
          vocab_rows.extend(dict(split=split,arm=arm,rank=j,feature_index=int(k),train_APK_support=int(s))
            for j,(k,s) in enumerate(zip(obj['feature_vocabulary'],obj['vocabulary_support']),1))
      cwrite('hdbc_vocabulary_by_fold.csv',vocab_rows)

def evaluate_source(data,source):
    rows=[]
    for split in SPLITS:
      tr,va,base,route,_=split_contract(data,split,source)
      ids=list(map(str,data['ids'][va]));y=data['y'][va]
      for arm in ARMS:
        pp=[]
        for seed in SEEDS:
          obj=load_m0(split,seed) if arm==ARMS[0] else pread(model_path(split,seed,arm))
          if obj['ids']!=ids:raise RuntimeError('checkpoint identity '+split+arm)
          pp.append(np.asarray(obj['probability'],float))
        p=np.mean(pp,axis=0)
        for i,s in enumerate(ids):rows.append(dict(year=2017,split=split,sha256=s,label=int(y[i]),arm=arm,
          stage1_probability=float(base[i]),routed=int(route[i]),expert_probability=float(p[i]),
          final_probability=float(p[i] if route[i] else base[i]),
          seed_900000=float(pp[0][i]),seed_900001=float(pp[1][i]),seed_900002=float(pp[2][i])))
    cwrite('source2017_oof_predictions.csv',rows)
    summarize(rows,'source2017_oof_metrics.csv','source2017_conditional_auc.csv','source2017_paired.csv')
    print('HDBC_SOURCE_EVAL_DONE',flush=True)

def summarize(rows,metric_name,condition_name,paired_name):
    metrics=[];conds=[];paired=[]
    for year in sorted(set(int(r['year']) for r in rows)):
      subset=[r for r in rows if int(r['year'])==year]
      rr={arm:sorted([r for r in subset if r['arm']==arm],key=lambda r:r['sha256']) for arm in ARMS}
      ids=[r['sha256'] for r in rr[ARMS[0]]]
      y=np.asarray([int(r['label']) for r in rr[ARMS[0]]]);base=np.asarray([float(r['stage1_probability']) for r in rr[ARMS[0]]]);
      route=np.asarray([int(r['routed'])==1 for r in rr[ARMS[0]]]);p={}
      for arm in ARMS:
        if [r['sha256'] for r in rr[arm]]!=ids:raise RuntimeError('APK alignment')
        p[arm]=np.asarray([float(r['expert_probability']) for r in rr[arm]])
        m=cascade(y,base,p[arm],route);metrics.append(dict(year=year,arm=arm,**m))
        conds.append(dict(year=year,arm=arm,FN_vs_TN_AUC=m['FN_vs_TN_AUC'],TP_vs_FP_AUC=m['TP_vs_FP_AUC'],
          FN_TN_n=m['FN_TN_n'],TP_FP_n=m['TP_FP_n']))
      for arm in ARMS[:4]+ARMS[5:]:
        for row in paired_intervals(y,base,route,p[ARMS[4]],p[arm],repeats=300):
          paired.append(dict(year=year,candidate=ARMS[4],reference=arm,**row))
    cwrite(metric_name,metrics);cwrite(condition_name,conds);cwrite(paired_name,paired)

def final_train(data,device,resume):
    from myd_common import runtime
    v4,_,args=runtime();args.device=device;args.keep_weight=args.fix_weight=1.;args.keep_consistency=0.;dev=v4.resolve_device(device)
    tr=np.flatnonzero(data['years']==2017);ids=list(map(str,data['ids'][tr]));feat=features(2017,ids)
    gx,_=global24(v4,data,tr,tr[:8]);raw=raw_bags(data,tr);result=[]
    for seed in SEEDS:
      paths=[model_path('FINAL2017',seed,arm) for arm in ARMS[1:]]
      if resume and all(p.exists() for p in paths):
        result.extend(dict(arm=arm,seed=seed,**{k:pread(p)[k] for k in ('best_epoch','parameter_n','seconds')}) for arm,p in zip(ARMS[1:],paths));continue
      old=load_m0('T1',seed)
      if old['train_years']!=[2017]:raise RuntimeError('not 2017-only baseline')
      bags=transform_saved(v4,raw,old['background'],args);graphs=v4.transform_role_graphs(data['s7'][tr],old['graph_scaler'])
      model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args).to(dev);model.load_state_dict(old['state_dict']);model.eval()
      z,b=extract(v4,model,gx,graphs,bags,dev)
      h4_vocab=None
      for arm in ARMS[1:]:
        path=model_path('FINAL2017',seed,arm)
        if path.exists():
          if not resume:raise RuntimeError('existing final head')
          obj=pread(path)
          if arm==ARMS[4]:h4_vocab=obj['feature_vocabulary']
        else:
          x,vocab,support=block_features(feat,arm,np.arange(len(tr)),tr,selected=h4_vocab if arm==ARMS[5] else None)
          if arm==ARMS[4]:h4_vocab=vocab.copy()
          start=time.perf_counter();obj=fit(z,x,b,data['y'][tr],seed,float(args.learning_rate),float(args.weight_decay))
          obj.update(arm=arm,split='FINAL2017',seed=seed,ids=ids,feature_vocabulary=vocab,
            vocabulary_support=support,train_n=799,seconds=time.perf_counter()-start)
          pwrite(path,obj)
        result.append(dict(arm=arm,seed=seed,best_epoch=obj['best_epoch'],parameter_n=obj['parameter_n'],seconds=obj['seconds']))
        print('HDBC_FINAL_ARM_DONE',seed,arm,flush=True)
    cwrite('final2017_training_results.csv',result)
    manifest=[digest(model_path('FINAL2017',seed,arm)) for seed in SEEDS for arm in ARMS[1:]]
    jwrite('final2017_frozen.json',dict(status='FROZEN',train_years=[2017],models=manifest,
      feature_vocab_source='2017 only',original_C43='SCBE M0 T1'))

def future_eval(data,device):
    frozen=json.loads((OUT/'final2017_frozen.json').read_text())
    if any(digest(r['path'])!=r for r in frozen['models']):raise RuntimeError('final model changed after freeze')
    stage={r['sha256']:r for r in cread(ROOT/'outputs/care-svm-direct-c43-future-v1/predictions_label_free.csv')}
    if len(stage)!=1989:raise RuntimeError('future Stage-I count')
    from myd_common import runtime
    v4,_,args=runtime();args.device=device;args.keep_weight=args.fix_weight=1.;args.keep_consistency=0.;dev=v4.resolve_device(device)
    src=np.flatnonzero(data['years']==2017);rows=[]
    for year in (2018,2019,2020,2021):
      make_features(data,year)
      va=np.flatnonzero(data['years']==year);ids=list(map(str,data['ids'][va]));feat=features(year,ids)
      if any(s not in stage or int(stage[s]['year'])!=year for s in ids):raise RuntimeError('future SHA')
      base=np.asarray([float(stage[s]['stage1_prob']) for s in ids]);route=np.asarray([int(stage[s]['routed']) for s in ids]);
      if int(route.sum())!=int(round(.25*len(va))):raise RuntimeError('frozen route')
      gz=global24(v4,data,src,va)[1];raw=raw_bags(data,va);allp={arm:[] for arm in ARMS}
      for seed in SEEDS:
        old=load_m0('T1',seed);bags=transform_saved(v4,raw,old['background'],args)
        graphs=v4.transform_role_graphs(data['s7'][va],old['graph_scaler'])
        model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args).to(dev);model.load_state_dict(old['state_dict']);model.eval()
        z,b=extract(v4,model,gz,graphs,bags,dev);p0=sigmoid(b)
        if year==2018 and (old['ids']!=ids or np.max(np.abs(p0-old['probability']))>1e-6):raise RuntimeError('T1 2018 replay')
        allp[ARMS[0]].append(p0)
        for arm in ARMS[1:]:
          obj=pread(model_path('FINAL2017',seed,arm));x,_,_=block_features(feat,arm,np.arange(len(va)),va,selected=obj['feature_vocabulary'])
          allp[arm].append(predict(obj,z,x,b))
        del model
        if dev.type=='cuda':v4.torch.cuda.empty_cache()
      y=data['y'][va] # labels touched only after all model/feature parameters frozen
      for arm in ARMS:
        mem=np.asarray(allp[arm]);p=mem.mean(0)
        for i,s in enumerate(ids):rows.append(dict(year=year,sha256=s,label=int(y[i]),arm=arm,
          stage1_probability=float(base[i]),routed=int(route[i]),router_score=float(stage[s]['router_score']),
          host_count=int(round(np.expm1(feat['diag'][i,0]))),coverage=int(feat['coverage'][i]),
          expert_probability=float(p[i]),final_probability=float(p[i] if route[i] else base[i]),
          seed_900000=float(mem[0,i]),seed_900001=float(mem[1,i]),seed_900002=float(mem[2,i])))
      print('HDBC_FUTURE_DONE',year,flush=True)
    cwrite('future2018_2021_predictions.csv',rows)
    summarize(rows,'future2018_2021_metrics.csv','future_conditional_auc.csv','future_paired.csv')

def collect():
    check=integrity()
    if check['status']!='PASS':raise RuntimeError('asset integrity failed')
    jwrite('asset_integrity_report.json',check)
    src=cread(OUT/'source2017_oof_metrics.csv');fut=cread(OUT/'future2018_2021_metrics.csv')
    cwrite('host_graph_coverage.csv',[r for year in YEARS for r in cread(OUT/f'graph_coverage_{year}.csv')])
    cwrite('feature_extraction_runtime.csv',[r for year in YEARS for r in cread(OUT/f'feature_runtime_{year}.csv')])
    model_cost=[]
    for split in SPLITS+('FINAL2017',):
      for seed in SEEDS:
        for arm in ARMS[1:]:
          o=pread(model_path(split,seed,arm))
          model_cost.append(dict(split=split,seed=seed,arm=arm,seconds=o['seconds'],parameters=o['parameter_n'],
            best_epoch=o['best_epoch'],train_n=o['train_n']))
    cwrite('model_runtime_cost.csv',model_cost)
    cwrite('fixed_broken_analysis.csv',[dict(year=r['year'],arm=r['arm'],Fixed=r['Fixed'],Broken=r['Broken'],Net=r['Net'],
      Fixed_FN=r['Fixed_FN'],Fixed_FP=r['Fixed_FP'],Broken_TN=r['Broken_TN'],Broken_TP=r['Broken_TP']) for r in src+fut])
    cwrite('directed_vs_undirected_results.csv',[r for r in src+fut if r['arm'] in (ARMS[3],ARMS[4])])
    cwrite('direction_shuffle_control.csv',[r for r in src+fut if r['arm'] in (ARMS[4],ARMS[5])])
    cwrite('paired_statistical_analysis.csv',cread(OUT/'source2017_paired.csv')+cread(OUT/'future_paired.csv'))
    vocab={seed:set(map(int,pread(model_path('FINAL2017',seed,ARMS[4]))['feature_vocabulary'])) for seed in SEEDS}
    if len({tuple(sorted(s)) for s in vocab.values()})!=1:raise RuntimeError('final H4 vocabulary differs across seeds')
    selected=next(iter(vocab.values()))
    evidence=[]
    for year in YEARS:
      for r in cread(OUT/f'directed_evidence_{year}.csv'):
        index=(int(r['hop'])-1)*961+int(r['source_role'])*31+int(r['target_role'])
        r['year']=year;r['used_by_final_H4']=int(index in selected)
        evidence.append(r)
    cwrite('per_apk_host_evidence.csv',evidence)
    # Frozen 2020 FN/TN comparison; no re-matching or feature selection.
    future_rows=cread(OUT/'future2018_2021_predictions.csv')
    case=[]
    for r in future_rows:
      if r['year']!='2020' or r['arm'] not in (ARMS[0],ARMS[4]):continue
      base=float(r['stage1_probability']);y=int(r['label'])
      if base>=.5 or int(r['routed'])!=1:continue
      case.append(dict(sha256=r['sha256'],arm=r['arm'],label=y,
        stage1_error_type='FN' if y==1 else 'TN',expert_probability=r['expert_probability'],
        final_prediction=int(float(r['final_probability'])>=.5),host_count=r['host_count'],relation_coverage=r['coverage']))
    cwrite('fn_tn_case_analysis.csv',case)
    by={r['arm']:r for r in src};h4=by[ARMS[4]];h0=by[ARMS[0]]
    val=lambda r,k:float(r[k]);delta=lambda a,b,k:val(a,k)-val(b,k)
    criteria=dict(BA_gain_0p5pp=delta(h4,h0,'BA')>=.005,Net_gain_6=delta(h4,h0,'Net')>=6,
      FN_decrease=delta(h4,h0,'FN')<0,FN_TN_AUC_not_down=delta(h4,h0,'FN_vs_TN_AUC')>=-.01,
      better_role_pair=delta(h4,by[ARMS[2]],'BA')>0,
      better_undirected=delta(h4,by[ARMS[3]],'BA')>0,
      better_shuffle=delta(h4,by[ARMS[5]],'BA')>0,
      broken_not_large=val(h4,'Broken')<=val(h0,'Broken')+3)
    flags=[]
    if not criteria['better_role_pair']:flags.append('HDBC_GAIN_EXPLAINED_BY_ROLE_PAIRS')
    if not criteria['better_undirected']:flags.append('HDBC_DIRECTION_NOT_DISTINCT_FROM_UNDIRECTED')
    if not criteria['better_shuffle']:flags.append('HDBC_DIRECTION_SHUFFLE_CONTROL_NOT_BEATEN')
    if delta(h4,h0,'FN')>0:flags.append('HDBC_INCREASES_FALSE_NEGATIVES')
    if delta(h4,h0,'FN_vs_TN_AUC')<-.01 or delta(h4,h0,'TP_vs_FP_AUC')<-.01:flags.append('HDBC_HURTS_CONDITIONAL_AUC')
    if delta(h4,h0,'BA')<=0:flags.append('HDBC_NO_RELIABLE_INCREMENT')
    status='HDBC_DIRECTED_EVIDENCE_SUPPORTED_ON_DEVELOPMENT' if all(criteria.values()) else 'HDBC_NO_RELIABLE_INCREMENT'
    def table(data):
      lines=['| 年份 | Arm | BA | FN | FP | Fixed | Broken | Net | FN/TN AUC | TP/FP AUC |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
      for r in data:lines.append(f"| {r['year']} | {r['arm']} | {float(r['BA']):.4f} | {r['FN']} | {r['FP']} | {r['Fixed']} | {r['Broken']} | {r['Net']} | {float(r['FN_vs_TN_AUC']):.4f} | {float(r['TP_vs_FP_AUC']):.4f} |")
      return lines
    report=['# CARE-Droid HDBC-V1 冻结开发集实验','',
      '结论：'+status+'；附加标记：'+(', '.join(flags) if flags else '无')+'。','',
      '## 证据与边界','',
      '原RoleGraph的InterZ961已经由Host->Host有向边计算，HDBC 1-hop与其部分重合；本次独立候选为仅DIRECT_CALL的Host锚定1/2-hop分层及路径见证。',
      '静态调用方向不是执行时间先后、数据流或恶意行为证明。H5是反转边的合成负对照，不是真实程序关系。',
      '主实验冻结旧C43编码器，在拟合内训练嵌入上训练同容量融合头；所有H1-H5共享这一偏差，结果不能宣称排除表示域偏移或端到端训练随机性。',
      '2018—2021为已使用的历史开发诊断，不是独立盲测。','',
      '## 2017固定五折 OOF','']+table(src)+['','## 完整2017冻结后逐年诊断','']+table(fut)
    with (OUT/'final_report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(report)+'\n')
    jwrite('final_decision.json',dict(status=status,flags=flags,criteria=criteria,
      source_BA_delta_H4_H0=delta(h4,h0,'BA'),source_Net_delta_H4_H0=delta(h4,h0,'Net'),
      future_is_independent_blind_test=False,asset_integrity='PASS'))
    with (OUT/'reproduce_commands.md').open('x',encoding='utf-8') as f:
      f.write('# HDBC-V1复现\n\n`cd /home/boyuanjie/projects/tasa-droid`\n\n'
       '`HDBC_OUTPUT='+str(OUT)+' CUDA_VISIBLE_DEVICES=1 bash scripts/care-hdbc-v1/launch_tmux.sh`\n\n'
       '仅允许在原始输出目录具有完整性清单且尚未完成时恢复；不要覆盖完成结果。\n')
    print('HDBC_COLLECT_DONE',flush=True)

def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True)
    for name in ('audit-only','smoke-test','extract-features','train-source','evaluate-source','train-final2017','evaluate-future','collect'):
      g.add_argument('--'+name,action='store_true')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--device',choices=('cpu','cuda'),default='cuda');ap.add_argument('--workers',type=int,default=1)
    a=ap.parse_args();os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if a.workers!=1:raise RuntimeError('single-worker reproducibility contract')
    if a.device=='cuda':
      import torch
      torch.zeros(1,device='cuda')
    if not selftest():raise RuntimeError('directed graph selftest failed')
    if a.evaluate_future and not (OUT/'final2017_frozen.json').exists():raise RuntimeError('future prohibited until full2017 freeze')
    data,source=preflight(a.resume,allow_future=a.evaluate_future)
    if a.audit_only:return
    if a.smoke_test:
      make_features(data,2017);source_train(data,source,a.device,a.resume,smoke=True)
    elif a.extract_features:make_features(data,2017)
    elif a.train_source:source_train(data,source,a.device,a.resume)
    elif a.evaluate_source:evaluate_source(data,source)
    elif a.train_final2017:final_train(data,a.device,a.resume)
    elif a.evaluate_future:future_eval(data,a.device)
    elif a.collect:collect()
if __name__=='__main__':main()
