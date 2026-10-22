#!/usr/bin/env python3
"""DCE-V1 2017-only runner: reads frozen cache/checkpoints, never APKs; writes only DCE_OUTPUT."""
from __future__ import annotations
import argparse,csv,json,os,pickle,sys,time
from pathlib import Path
import numpy as np

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
HERE=ROOT/'scripts/care-dce-v1'
OUT=Path(os.environ.get('DCE_OUTPUT',ROOT/'outputs/care_dce_v1')).resolve()
if ROOT/'outputs' not in OUT.parents:raise RuntimeError('output must be in project outputs')
sys.path[:0]=[str(HERE),str(ROOT/'scripts/care-background-functional-attribution-v1'),
               str(ROOT/'scripts/care-scbe-v1'),str(ROOT/'scripts/care-multiyear-development-v1'),
               str(ROOT/'scripts/care-p31-classifier-family-stage2-v1')]
from bfa_core import SEEDS,digest,formal_split,load_data_contract,load_m0
from train import split_contract,global24
from train_stage2 import raw_bags,transform_saved
from metrics import cascade,paired_intervals
from dce_core import ARMS,extract,fit_heads,predict_head,sigmoid

SPLITS=('S1','S2','S3','S4','S5')
YEARS={2017:799,2018:793,2019:399,2020:397,2021:400}

def serial(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)

def save_json(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,default=serial,allow_nan=False)

def save_csv(name,rows):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);rows=list(rows)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def save_pickle(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)

def get_pickle(path):
    with path.open('rb') as f:return pickle.load(f)

def checkpoint(split,seed,arm):
    if arm==ARMS[0]:return load_m0('T1' if split=='FINAL2017' else split,seed)
    return get_pickle(OUT/'models'/split/str(seed)/(arm+'.pkl'))

def originals():
    old=ROOT/'outputs/care_multiyear_development_v1'
    paths=[HERE/'main.py',HERE/'dce_core.py',ROOT/'scripts/care-background-functional-attribution-v1/bfa_core.py',
           ROOT/'scripts/care-scbe-v1/train.py',ROOT/'scripts/care-multiyear-development-v1/train_stage2.py',
           ROOT/'scripts/care-multiyear-development-v1/myd_common.py',
           ROOT/'scripts/care-p31-classifier-family-stage2-v1/stage1_core.py',
           ROOT/'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py',
           ROOT/'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_models.py',
           ROOT/'scripts/care-svm-c43-remove-bounded-correction-v1/model_variants.py',
           ROOT/'outputs/care-direct-c43-route-conditioned-training-v2/final_decision.json',
           old/'data.npz',old/'feature_rows.pkl',old/'sample_manifest.csv',
           ROOT/'outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv',
           ROOT/'outputs/care-svm-direct-c43-future-v1/predictions_label_free.csv']
    for split in SPLITS+('T1',):
        paths.extend(ROOT/'outputs/care_scbe_v1/models'/split/str(seed)/'M0_REPRODUCED.pkl' for seed in SEEDS)
    return list(dict.fromkeys(paths))

def integrity():
    saved=json.loads((OUT/'asset_manifest_before.json').read_text())
    changed=[]
    for item in saved:
        try:
            if digest(item['path'])!=item:changed.append(item['path'])
        except OSError:changed.append(item['path'])
    return dict(status='PASS' if not changed else 'FAIL',checked=len(saved),changed=changed,
                scope='named reused source code, cache, routes, 2017-only C43 checkpoints; no APK opened; not entire server')

def preflight(resume,allow_future=False):
    if OUT.exists():
        if (OUT/'asset_manifest_before.json').exists():
            if not resume:raise RuntimeError('existing run; use --resume')
        else:
            if resume or any(x.name not in ('run.log','launch_info.txt','run_state.json','phase_status.json') for x in OUT.iterdir()):
                raise RuntimeError('occupied output without integrity manifest')
            save_json('asset_manifest_before.json',[digest(p) for p in originals()])
    else:
        OUT.mkdir(parents=True,exist_ok=False)
        save_json('asset_manifest_before.json',[digest(p) for p in originals()])
    check=integrity()
    if check['status']!='PASS':raise RuntimeError('historical asset changed')
    data,source=load_data_contract()
    if {int(y):int(np.sum(data['years']==y)) for y in YEARS}!=YEARS:raise RuntimeError('year sizes changed')
    src=np.flatnonzero(data['years']==2017)
    if len(src)!=799 or (int(np.sum(data['y'][src]==0)),int(np.sum(data['y'][src]==1)))!=(400,399):
        raise RuntimeError('source class counts changed')
    folds=[]
    for split in SPLITS:
        tr,va,base,route,_=split_contract(data,split,source)
        if not (np.all(data['years'][tr]==2017) and len(va) in (159,160) and int(route.sum())==40):
            raise RuntimeError('fold/route violation')
        for i in va:folds.append(dict(sha256=str(data['ids'][i]),fold=split,year=2017))
    if len(folds)!=799 or len({r['sha256'] for r in folds})!=799:raise RuntimeError('fivefold OOF identity')
    if not (OUT/'dataset_protocol.json').exists():
        save_json('dataset_protocol.json',dict(year_counts=YEARS,training_years=[2017],
            source_benign=400,source_malware=399,folds=[160,160,160,160,159],seeds=SEEDS,
            host_width=142,exact_frozen_head_input_width=113,route='frozen calibrated SVM and MeanOpposition Top25%',
            threshold=.5,future_is_historical_development_diagnostic=True,APK_files_opened=0))
        save_csv('source2017_fold_manifest.csv',folds)
    if not allow_future:
        data['y']=data['y'].copy();data['y'][data['years']>2017]=-1
    print('DCE_PREFLIGHT_PASS output='+str(OUT),flush=True)
    return data,source

def source_oof(source):
    rows={str(s):r for s,r in source.items()}
    if len(rows)!=799:raise RuntimeError('source global Stage-I OOF missing')
    return rows

def nested_stage1(data,source,resume=False,smoke=False):
    from sklearn.model_selection import StratifiedKFold
    from stage1_core import fit_calibrated_member
    records=[];audits=[];groups=[]
    jobs=SPLITS[:1] if smoke else SPLITS
    for split in jobs:
        fold=int(split[1:]);tr,va,_,_,_=split_contract(data,split,source)
        ids=np.asarray(data['ids'][tr]);y=np.asarray(data['y'][tr],int);x=np.asarray(data['p31'][tr],float)
        if np.any(data['years'][tr]!=2017) or set(ids)&set(data['ids'][va]):raise RuntimeError('outer leakage')
        if not np.isfinite(x).all() or x.shape[1]!=31:raise RuntimeError('P31 contract')
        cv=StratifiedKFold(4,shuffle=True,random_state=20261016+fold)
        p=np.full(len(tr),np.nan);inner_id=np.zeros(len(tr),int)
        for inner,(fit,hold) in enumerate(cv.split(x,y),1):
            if set(ids[fit])&set(ids[hold]) or set(ids[fit])&set(data['ids'][va]):raise RuntimeError('nested SHA leakage')
            if smoke and inner>1:break
            prob=fit_calibrated_member('SVM',x[fit],y[fit],x[hold],
                20261016+1000*fold+inner,inner_folds=3,jobs=1)
            p[hold]=prob;inner_id[hold]=inner
            audits.append(dict(outer_fold=fold,inner_fold=inner,fit_n=len(fit),hold_n=len(hold),
                fit_hold_sha_overlap=0,outer_test_in_inner_fit=0,
                calibration_folds=3,calibration_uses_inner_fit_only=True))
            print('DCE_NESTED_SVM_DONE',split,inner,flush=True)
        if smoke:
            valid=np.isfinite(p)
            if not np.any(valid):raise RuntimeError('no smoke direction')
        else:
            valid=np.ones(len(p),bool)
            if not np.isfinite(p).all() or np.any(inner_id==0):raise RuntimeError('incomplete nested OOF')
        for k in np.flatnonzero(valid):
            records.append(dict(split=split,sha256=str(ids[k]),label=int(y[k]),
                inner_fold=int(inner_id[k]),stage1_oof_probability=float(p[k]),
                stage1_oof_direction=int(p[k]>=.5),outer_test_excluded=1))
        direction=p[valid]>=.5
        for d in (0,1):
            z=np.flatnonzero(valid)[direction==d]
            groups.append(dict(split=split,direction=d,n=len(z),benign=int(np.sum(y[z]==0)),
                               malware=int(np.sum(y[z]==1))))
    if not smoke:
        if len(records)!=sum(799-(160 if s!='S5' else 159) for s in SPLITS):
            raise RuntimeError('nested group count')
        save_csv('nested_stage1_oof_directions.csv',records)
        save_json('nested_oof_leakage_audit.json',dict(status='PASS',outer_folds=5,inner_folds=4,
            nested_calibration_folds=3,all_training_rows_held_out_once_per_outer_fold=True,
            global_2017_OOF_not_used_for_outer_training=True,checks=audits))
        save_csv('direction_group_support.csv',groups)
    return records

def nested_lookup():
    rows=read_csv(OUT/'nested_stage1_oof_directions.csv')
    lookup={(r['split'],r['sha256']):r for r in rows}
    if len(rows)!=len(lookup):raise RuntimeError('nested duplicate')
    return lookup

def embeddings_for_pack(pack,device):
    ztrain,ltrain=extract(pack['v4'],pack['model'],pack['global_train'],pack['graph_train'],pack['bags_train'],pack['dev'])
    ztest,ltest=extract(pack['v4'],pack['model'],pack['global_val'],pack['graph_val'],pack['bags_val'],pack['dev'])
    reference=np.asarray(pack['old']['probability'],float)
    delta=float(np.max(np.abs(sigmoid(ltest)-reference)))
    if delta>1e-6:raise RuntimeError('frozen C43 pre-head replay mismatch '+str(delta))
    return ztrain,ltrain,ztest,ltest,delta

def source_fit(data,source,device,resume,smoke=False,smoke_records=None):
    nested=nested_lookup() if not smoke else None
    splits=SPLITS[:1] if smoke else SPLITS
    seeds=SEEDS[:1] if smoke else SEEDS
    audits=[];replays=[];runtime=[]
    for split in splits:
        tr,va,base,route,_=split_contract(data,split,source)
        if smoke:
            # Take only inner-held-out source APKs; their directions are true nested OOF.
            held={r['sha256']:r for r in smoke_records or [] if r['split']==split}
            eligible=np.asarray([j for j,i in enumerate(tr) if str(data['ids'][i]) in held],int)
            ix=np.concatenate([eligible[data['y'][tr[eligible]]==c][:30] for c in (0,1)])
            if len(ix)<40:raise RuntimeError('insufficient nested OOF smoke rows')
            train_direction=np.asarray([int(held[str(data['ids'][tr[j]])]['stage1_oof_direction']) for j in ix],int)
            if len(set(train_direction))!=2:raise RuntimeError('smoke nested directions degenerate')
        else:
            train_direction=np.asarray([int(nested[(split,str(data['ids'][i]))]['stage1_oof_direction']) for i in tr],int)
        for seed in seeds:
            paths=[OUT/'models'/split/str(seed)/(arm+'.pkl') for arm in ARMS[1:]]
            if not smoke and resume and all(p.exists() for p in paths):
                for arm,p in zip(ARMS[1:],paths):
                    obj=get_pickle(p)
                    if obj['arm']!=arm or obj['split']!=split or obj['seed']!=seed:raise RuntimeError('head checkpoint identity')
                    audits.append(obj['audit']);runtime.append(obj['runtime'])
                print('DCE_SOURCE_RESUME',split,seed,flush=True);continue
            t=time.perf_counter();pack=formal_split(data,split,seed,device)
            ztr,ltr,zva,lva,diff=embeddings_for_pack(pack,device)
            if not (np.array_equal(pack['train'],tr) and np.array_equal(pack['val'],va)):
                raise RuntimeError('embedding/index mismatch')
            if smoke:
                # Real head backward pass on a balanced 60-APK subset; not an evaluation.
                ztr=ztr[ix];ltr=ltr[ix];dy=train_direction;y=np.asarray(data['y'][tr[ix]],int)
                zva=zva[:8];lva=lva[:8];test_direction=np.asarray(base[:8]>=.5,int)
            else:
                y=np.asarray(data['y'][tr],int);dy=train_direction;test_direction=np.asarray(base>=.5,int)
            heads,audit=fit_heads(ztr,ltr,y,dy,seed,max_epochs=30,patience=5,
                                  lr=float(pack['args'].learning_rate),weight_decay=float(pack['args'].weight_decay))
            if len(heads)!=4:raise RuntimeError('DCE arms incomplete')
            for arm in ARMS[1:]:
                pred=predict_head(arm,heads[arm],zva,lva,test_direction)
                if not np.isfinite(pred).all():raise RuntimeError('invalid head probability')
                if smoke:continue
                obj=dict(arm=arm,split=split,seed=seed,ids=list(map(str,data['ids'][va])),
                    probability=pred,state_dict=heads[arm]['state_dict'],mean=heads[arm]['mean'],std=heads[arm]['std'],
                    best_epoch=heads[arm]['best_epoch'],train_loss=heads[arm]['train_loss'],
                    validation_loss=heads[arm]['validation_loss'],parameter_n=heads[arm]['parameter_n'],
                    shuffle_seed=heads[arm]['shuffle_seed'],selected_epochs=pack['old']['selected_epochs'],
                    audit=dict(**audit,split=split,arm=arm),
                    runtime=dict(split=split,seed=seed,arm=arm,seconds=time.perf_counter()-t))
                save_pickle(OUT/'models'/split/str(seed)/(arm+'.pkl'),obj)
                audits.append(obj['audit']);runtime.append(obj['runtime'])
                print('DCE_SOURCE_HEAD_DONE',split,seed,arm,flush=True)
            replays.append(dict(split=split,seed=seed,train_n=len(tr),test_n=len(va),
                                head_input_width=113,host_width=142,max_probability_difference=diff,
                                original_background_unmodified=True))
            if smoke:
                save_json('smoke_test.json',dict(status='PASS',real_forward_backward=True,
                    train_head_n=len(ix),validation_head_n=8,heads=list(heads),
                    parameters={a:heads[a]['parameter_n'] for a in heads},
                    frozen_baseline_replay_difference=diff,source_fold=split,seed=seed,
                    nested_oof_direction_used_for_head=True))
                print('DCE_SMOKE_PASS',flush=True);return
    if not (OUT/'encoder_interface_validation.json').exists():
        save_json('encoder_interface_validation.json',dict(status='PASS',records=replays,
            true_tensor='DirectStage2.forward_with_trace correction_input; 113D',
            frozen_model_probability_replayed=True,background_and_142D_host_unchanged=True,
            train_embedding_is_in_sample_for_frozen_C43=True))
    if not (OUT/'baseline_replay.json').exists():save_json('baseline_replay.json',dict(status='PASS',records=replays))
    if not (OUT/'head_training_audit.csv').exists():save_csv('head_training_audit.csv',audits)
    if not (OUT/'runtime_cost.csv').exists():save_csv('runtime_cost.csv',runtime)

def source_eval(data,source):
    per=[];metrics=[];direction_metrics=[];distributions=[];paired=[];capacity=[];shuffle=[];fixed=[]
    for split in SPLITS:
        tr,va,base,route,_=split_contract(data,split,source)
        ids=list(map(str,data['ids'][va]));y=np.asarray(data['y'][va],int)
        for arm in ARMS:
            probs=[]
            for seed in SEEDS:
                obj=checkpoint(split,seed,arm)
                if obj['ids']!=ids:raise RuntimeError('source checkpoint SHA mismatch')
                probs.append(np.asarray(obj['probability'],float))
            p=np.mean(probs,axis=0)
            for i,sid in enumerate(ids):
                per.append(dict(split=split,year=2017,sha256=sid,label=int(y[i]),arm=arm,
                    stage1_probability=float(base[i]),stage1_direction=int(base[i]>=.5),routed=int(route[i]),
                    expert_probability=float(p[i]),final_probability=float(p[i] if route[i] else base[i]),
                    seed_900000=float(probs[0][i]),seed_900001=float(probs[1][i]),seed_900002=float(probs[2][i])))
    groups={arm:sorted([r for r in per if r['arm']==arm],key=lambda r:r['sha256']) for arm in ARMS}
    ref=groups[ARMS[0]];ids=[r['sha256'] for r in ref]
    y=np.asarray([r['label'] for r in ref],int);base=np.asarray([r['stage1_probability'] for r in ref],float)
    route=np.asarray([r['routed'] for r in ref],bool);d=base>=.5
    pp={}
    for arm in ARMS:
        rr=groups[arm]
        if [r['sha256'] for r in rr]!=ids:raise RuntimeError('OOF alignment')
        p=np.asarray([r['expert_probability'] for r in rr],float);pp[arm]=p
        m=cascade(y,base,p,route)
        metrics.append(dict(scope='SOURCE_2017_OOF',year=2017,arm=arm,**m))
        fixed.append(dict(scope='SOURCE_2017_OOF',arm=arm,Fixed=m['Fixed'],Broken=m['Broken'],Net=m['Net'],
                          Fixed_FN=m['Fixed_FN'],Fixed_FP=m['Fixed_FP'],Broken_TN=m['Broken_TN'],Broken_TP=m['Broken_TP']))
        for value,name in ((0,'STAGE1_BENIGN'),(1,'STAGE1_MALICIOUS')):
            sub=route&(d==value)
            from sklearn.metrics import roc_auc_score
            auc=float(roc_auc_score(y[sub],p[sub])) if len(np.unique(y[sub]))==2 else None
            direction_metrics.append(dict(scope='SOURCE_2017_OOF',arm=arm,direction=name,n=int(sub.sum()),
                benign=int(np.sum(sub&(y==0))),malware=int(np.sum(sub&(y==1))),conditional_auc=auc,
                repaired=int(np.sum(sub&((base>=.5)!=y)&((p>=.5)==y))),
                damaged=int(np.sum(sub&((base>=.5)==y)&((p>=.5)!=y))),
                mean_probability=float(p[sub].mean()),std_probability=float(p[sub].std())))
            distributions.append(dict(scope='SOURCE_2017_OOF',arm=arm,direction=name,n=int(sub.sum()),
                mean=float(p[sub].mean()),std=float(p[sub].std()),q05=float(np.quantile(p[sub],.05)),
                q50=float(np.median(p[sub])),q95=float(np.quantile(p[sub],.95))))
    for other in (ARMS[0],ARMS[1],ARMS[3],ARMS[4]):
        for row in paired_intervals(y,base,route,pp[ARMS[2]],pp[other],repeats=500):
            paired.append(dict(scope='SOURCE_2017_OOF',candidate=ARMS[2],reference=other,**row))
        good2=(np.where(route,pp[ARMS[2]],base)>=.5)==y
        good0=(np.where(route,pp[other],base)>=.5)==y
        b=int(np.sum(good2&~good0));c=int(np.sum(~good2&good0))
        from scipy.stats import binomtest
        pval=float(binomtest(min(b,c),b+c,.5).pvalue) if b+c else 1.
        paired.append(dict(scope='SOURCE_2017_OOF',candidate=ARMS[2],reference=other,
                           metric='McNemar_exact',difference=b-c,ci95_low='',ci95_high='',
                           improved=b,worsened=c,p_value=pval))
    for name,rows in [('source2017_oof_predictions.csv',per),('source2017_oof_metrics.csv',metrics),
                      ('source2017_direction_metrics.csv',direction_metrics),
                      ('direction_head_probability_distributions.csv',distributions),
                      ('paired_statistical_analysis.csv',paired),('fixed_broken_analysis.csv',fixed)]:
        if not (OUT/name).exists():save_csv(name,rows)
    if not (OUT/'capacity_control_results.csv').exists():
        save_csv('capacity_control_results.csv',[r for r in metrics if r['arm'] in (ARMS[1],ARMS[2],ARMS[4])])
    if not (OUT/'direction_shuffle_control_results.csv').exists():
        save_csv('direction_shuffle_control_results.csv',[r for r in metrics if r['arm'] in (ARMS[2],ARMS[3])])
    print('DCE_SOURCE_EVAL_DONE',flush=True)

def final_train(data,source,device,resume):
    from myd_common import runtime
    v4,_,args=runtime();args.device=device;args.keep_weight=args.fix_weight=1.;args.keep_consistency=0.
    dev=v4.resolve_device(device)
    tr=np.flatnonzero(data['years']==2017)
    ids=list(map(str,data['ids'][tr]));src=source_oof(source)
    direction=np.asarray([float(src[s]['stage1_prob'])>=.5 for s in ids],int)
    if any(int(src[s]['fold']) not in range(1,6) for s in ids):raise RuntimeError('final OOF directions not fivefold')
    y=np.asarray(data['y'][tr],int)
    if len(tr)!=799:raise RuntimeError('full2017 count')
    results=[];replay=[]
    for seed in SEEDS:
        files=[OUT/'models/FINAL2017'/str(seed)/(arm+'.pkl') for arm in ARMS[1:]]
        if resume and all(p.exists() for p in files):
            for p in files:results.append(get_pickle(p)['training_result'])
            print('DCE_FINAL_RESUME',seed,flush=True);continue
        old=load_m0('T1',seed)
        if old['train_years']!=[2017]:raise RuntimeError('T1 baseline not full2017')
        gx,_=global24(v4,data,tr,tr[:8])
        raw=raw_bags(data,tr)
        bags=transform_saved(v4,raw,old['background'],args)
        graph=v4.transform_role_graphs(data['s7'][tr],old['graph_scaler'])
        model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args).to(dev)
        model.load_state_dict(old['state_dict']);model.eval()
        start=time.perf_counter();z,base=extract(v4,model,gx,graph,bags,dev)
        heads,audit=fit_heads(z,base,y,direction,seed,max_epochs=30,patience=5,
                              lr=float(args.learning_rate),weight_decay=float(args.weight_decay))
        for arm in ARMS[1:]:
            path=OUT/'models/FINAL2017'/str(seed)/(arm+'.pkl')
            if path.exists():
                if not resume:raise RuntimeError('final checkpoint exists')
                obj=get_pickle(path)
                if obj['arm']!=arm or obj['seed']!=seed:raise RuntimeError('final checkpoint identity')
            else:
                h=heads[arm]
                training_result=dict(arm=arm,seed=seed,train_n=799,train_years='2017',
                    best_epoch=h['best_epoch'],train_loss=h['train_loss'],validation_loss=h['validation_loss'],
                    parameter_n=h['parameter_n'],seconds=time.perf_counter()-start)
                obj=dict(arm=arm,split='FINAL2017',seed=seed,ids=ids,
                    state_dict=h['state_dict'],mean=h['mean'],std=h['std'],best_epoch=h['best_epoch'],
                    train_years=[2017],train_n=799,training_result=training_result,
                    training_direction_audit=audit,shuffle_seed=h['shuffle_seed'])
                save_pickle(path,obj)
            results.append(obj['training_result'])
            print('DCE_FINAL_HEAD_DONE',seed,arm,flush=True)
        replay.append(dict(seed=seed,source_train_n=799,head_input_width=z.shape[1],
                           stage1_training_direction_is_fivefold_OOF=True))
    if not (OUT/'final2017_training_results.csv').exists():save_csv('final2017_training_results.csv',results)
    if not (OUT/'final2017_frozen.json').exists():
        manifests=[digest(OUT/'models/FINAL2017'/str(seed)/(arm+'.pkl')) for seed in SEEDS for arm in ARMS[1:]]
        save_json('final2017_frozen.json',dict(status='FROZEN',training_years=[2017],models=manifests,
            stage1_training_direction='historical fivefold source OOF calibrated SVM',
            original_C43_checkpoint='SCBE M0 T1 full2017',replay=replay))
    print('DCE_FINAL2017_FROZEN',flush=True)

def future_eval(data,device):
    frozen=json.loads((OUT/'final2017_frozen.json').read_text())
    for item in frozen['models']:
        if digest(item['path'])!=item:raise RuntimeError('final head changed after freeze')
    from myd_common import runtime
    v4,_,args=runtime();args.device=device;args.keep_weight=args.fix_weight=1.;args.keep_consistency=0.
    dev=v4.resolve_device(device)
    stage={r['sha256']:r for r in read_csv(ROOT/'outputs/care-svm-direct-c43-future-v1/predictions_label_free.csv')}
    if len(stage)!=1989:raise RuntimeError('future Stage-I table incomplete')
    src=np.flatnonzero(data['years']==2017)
    per=[];metrics=[];condition=[];paired=[];fixed=[]
    for year in (2018,2019,2020,2021):
        va=np.flatnonzero(data['years']==year);ids=list(map(str,data['ids'][va]))
        if len(va)!=YEARS[year] or any(s not in stage or int(stage[s]['year'])!=year for s in ids):
            raise RuntimeError('future SHA/year mismatch')
        base=np.asarray([float(stage[s]['stage1_prob']) for s in ids])
        direction=np.asarray(base>=.5,int);route=np.asarray([int(stage[s]['routed'])==1 for s in ids],bool)
        if int(route.sum())!=int(round(.25*len(ids))):raise RuntimeError('frozen Top25% route mismatch')
        score=[float(stage[s]['router_score']) for s in ids]
        gz=global24(v4,data,src,va)[1];raw=raw_bags(data,va)
        predictions={arm:[] for arm in ARMS}
        for seed in SEEDS:
            old=load_m0('T1',seed)
            bags=transform_saved(v4,raw,old['background'],args)
            graph=v4.transform_role_graphs(data['s7'][va],old['graph_scaler'])
            model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args).to(dev)
            model.load_state_dict(old['state_dict']);model.eval()
            z,logit=extract(v4,model,gz,graph,bags,dev)
            p=sigmoid(logit)
            if year==2018 and (old['ids']!=ids or np.max(np.abs(p-old['probability']))>1e-6):
                raise RuntimeError('2018 frozen M0 replay differs')
            predictions[ARMS[0]].append(p)
            for arm in ARMS[1:]:
                obj=checkpoint('FINAL2017',seed,arm)
                predictions[arm].append(predict_head(arm,obj,z,logit,direction))
            del model
            if dev.type=='cuda':v4.torch.cuda.empty_cache()
        y=np.asarray(data['y'][va],int) # only after frozen model integrity and inference
        for arm in ARMS:
            members=np.asarray(predictions[arm]);p=members.mean(0)
            m=cascade(y,base,p,route)
            metrics.append(dict(year=year,arm=arm,**m))
            fixed.append(dict(year=year,arm=arm,Fixed=m['Fixed'],Broken=m['Broken'],Net=m['Net'],
                Fixed_FN=m['Fixed_FN'],Fixed_FP=m['Fixed_FP'],Broken_TN=m['Broken_TN'],Broken_TP=m['Broken_TP']))
            for d,label in ((0,'STAGE1_BENIGN'),(1,'STAGE1_MALICIOUS')):
                sub=route&(direction==d)
                condition.append(dict(year=year,arm=arm,direction=label,n=int(sub.sum()),
                    benign=int(np.sum(sub&(y==0))),malware=int(np.sum(sub&(y==1))),
                    conditional_auc=m['FN_vs_TN_AUC'] if d==0 else m['TP_vs_FP_AUC'],
                    mean_probability=float(p[sub].mean()),std_probability=float(p[sub].std())))
            for i,sid in enumerate(ids):
                per.append(dict(year=year,sha256=sid,label=int(y[i]),arm=arm,
                    stage1_probability=float(base[i]),stage1_direction=int(direction[i]),
                    router_score=score[i],routed=int(route[i]),expert_probability=float(p[i]),
                    final_probability=float(p[i] if route[i] else base[i]),
                    seed_900000=float(members[0,i]),seed_900001=float(members[1,i]),seed_900002=float(members[2,i])))
        for other in (ARMS[0],ARMS[1],ARMS[3],ARMS[4]):
            for row in paired_intervals(y,base,route,np.mean(predictions[ARMS[2]],axis=0),
                                        np.mean(predictions[other],axis=0),repeats=500):
                paired.append(dict(year=year,candidate=ARMS[2],reference=other,**row))
        print('DCE_FUTURE_YEAR_DONE',year,flush=True)
    for name,rows in [('future2018_2021_predictions.csv',per),('future2018_2021_metrics.csv',metrics),
        ('conditional_auc_analysis.csv',condition),('future_fixed_broken.csv',fixed),
        ('future_paired_intervals.csv',paired)]:
        if not (OUT/name).exists():save_csv(name,rows)

def collect():
    check=integrity()
    if check['status']!='PASS':raise RuntimeError('historical asset integrity failed at end')
    save_json('asset_integrity_report.json',check)
    src=read_csv(OUT/'source2017_oof_metrics.csv');fut=read_csv(OUT/'future2018_2021_metrics.csv')
    by={r['arm']:r for r in src}
    a=by[ARMS[2]];b=by[ARMS[0]]
    delta=lambda q,r,key:float(q[key])-float(r[key])
    criteria=dict(BA_gain_at_least_0p5pp=delta(a,b,'BA')>=.005,Net_gain_at_least_6=delta(a,b,'Net')>=6,
        better_than_D1=delta(a,by[ARMS[1]],'BA')>0,better_than_D3=delta(a,by[ARMS[3]],'BA')>0,
        not_worse_than_D4=delta(a,by[ARMS[4]],'BA')>=-.0025,
        FN_TN_AUC_not_down=delta(a,b,'FN_vs_TN_AUC')>=-.01,
        TP_FP_AUC_not_down=delta(a,b,'TP_vs_FP_AUC')>=-.01,
        broken_not_large=int(a['Broken'])<=int(b['Broken'])+3)
    status='DCE_DECISION_CONDITIONING_SUPPORTED' if all(criteria.values()) else 'DCE_NO_RELIABLE_INCREMENT'
    flags=[]
    if delta(a,b,'FN')<0 and delta(a,b,'FP')>0:flags.append('DCE_IMPROVES_FN_HURTS_FP')
    if delta(a,b,'FP')<0 and delta(a,b,'FN')>0:flags.append('DCE_IMPROVES_FP_HURTS_FN')
    if not criteria['FN_TN_AUC_not_down'] or not criteria['TP_FP_AUC_not_down']:flags.append('DCE_HURTS_CONDITIONAL_AUC')
    if status!='DCE_DECISION_CONDITIONING_SUPPORTED' and delta(a,b,'BA')>0 and delta(a,by[ARMS[1]],'BA')<=0:
        flags.append('DCE_GAIN_EXPLAINED_BY_CAPACITY')
    if delta(a,by[ARMS[4]],'BA')<=.001:flags.append('DCE_GAIN_EXPLAINED_BY_SIMPLE_DIRECTION_FEATURE_CANDIDATE')
    if any(float(r['BA'])<float(next(x for x in fut if x['year']==r['year'] and x['arm']==ARMS[0])['BA'])-.005
           for r in fut if r['arm']==ARMS[2]):flags.append('DCE_CROSSYEAR_INSTABILITY_OBSERVED')
    groups=read_csv(OUT/'source2017_direction_metrics.csv')
    for r in groups:
        if r['arm']==ARMS[2] and float(r['std_probability'])<.01:flags.append('DCE_DIRECTION_HEAD_COLLAPSE');break
    report=['# CARE-Droid DCE-V1 2017-only 实验报告','',
      'D0复用冻结Direct-C43，D1-D4冻结其三背景、Host/RoleGraph、113维融合输入及共享判别头，仅训练轻量适配器。',
      '这与既有route-conditioned BCE不同：不改权重、不只训练路由样本；按严格嵌套Stage-I OOF预测方向分头。',
      '外层训练方向来自每折内部四折OOF、每次SVM再用内部三折校准；完整2017最终头使用历史五折Stage-I OOF方向。',
      '原C43训练样本的融合向量属于其拟合内表示；所有对照共享该条件，不能声称为新表示的独立交叉拟合。',
      '2018—2021仅为历史开发诊断，不能用来选择头或超参数。','',
      '## 2017五折 OOF','',
      '| 模型 | BA | FN | FP | Fixed | Broken | Net | FN/TN AUC | TP/FP AUC |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        r=by[arm]
        report.append(f"| {arm} | {float(r['BA']):.4f} | {r['FN']} | {r['FP']} | {r['Fixed']} | {r['Broken']} | {r['Net']} | {float(r['FN_vs_TN_AUC']):.4f} | {float(r['TP_vs_FP_AUC']):.4f} |")
    report+=['','## 完整2017训练后跨年份冻结诊断','',
      '| 年份 | 模型 | BA | FN | FP | Fixed | Broken | Net | FN/TN AUC | TP/FP AUC |',
      '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in fut:
        report.append(f"| {r['year']} | {r['arm']} | {float(r['BA']):.4f} | {r['FN']} | {r['FP']} | {r['Fixed']} | {r['Broken']} | {r['Net']} | {float(r['FN_vs_TN_AUC']):.4f} | {float(r['TP_vs_FP_AUC']):.4f} |")
    report+=['','## 决策', '',status+'；附加观察：'+(', '.join(flags) if flags else '无')+'。',
      'D2相对D0/D1/D3/D4的配对区间和McNemar、各方向样本数及概率分布均保存在同目录CSV。',
      '任何源域正结果仅能支持继续独立验证；不能称未来年份为新盲测。']
    with (OUT/'final_report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(report)+'\n')
    save_json('final_decision.json',dict(status=status,flags=flags,source_criteria=criteria,
        source_delta_BA_D2_D0=delta(a,b,'BA'),source_delta_Net_D2_D0=delta(a,b,'Net'),
        source_delta_FN_D2_D0=delta(a,b,'FN'),source_delta_FP_D2_D0=delta(a,b,'FP'),
        training_years=[2017],future_is_independent_blind_test=False,asset_integrity='PASS'))
    print('DCE_COLLECT_DONE',flush=True)

def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True)
    for opt in ('audit-only','smoke-test','prepare-directions','train-source','source-eval',
                'train-final2017','evaluate-future','collect'):
        g.add_argument('--'+opt,action='store_true')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    ap.add_argument('--workers',type=int,default=1)
    a=ap.parse_args();os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if a.workers!=1:raise RuntimeError('single-worker safe protocol')
    if a.device=='cuda':
        import torch
        probe=torch.zeros(1,device='cuda');del probe
    if a.evaluate_future and not (OUT/'final2017_frozen.json').is_file():
        raise RuntimeError('future labels forbidden before final freeze')
    data,source=preflight(a.resume,allow_future=a.evaluate_future)
    if a.audit_only:return
    if a.smoke_test:
        records=nested_stage1(data,source,smoke=True)
        source_fit(data,source,a.device,a.resume,smoke=True,smoke_records=records)
    elif a.prepare_directions:nested_stage1(data,source,a.resume)
    elif a.train_source:source_fit(data,source,a.device,a.resume)
    elif a.source_eval:source_eval(data,source)
    elif a.train_final2017:final_train(data,source,a.device,a.resume)
    elif a.evaluate_future:future_eval(data,a.device)
    elif a.collect:collect()

if __name__=='__main__':main()
