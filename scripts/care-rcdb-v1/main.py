#!/usr/bin/env python3
"""2017-only RCDB orchestration. Reads frozen feature/model assets, never APKs; writes only RCDB_OUTPUT."""
from __future__ import annotations
import argparse, csv, json, os, pickle, sys, time
from pathlib import Path
import numpy as np

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
HERE = ROOT/'scripts/care-rcdb-v1'
OUT = Path(os.environ.get('RCDB_OUTPUT', ROOT/'outputs/care_rcdb_v1')).resolve()
if ROOT/'outputs' not in OUT.parents:
    raise RuntimeError('RCDB_OUTPUT must be a child of project outputs')
sys.path[:0] = [str(HERE), str(ROOT/'scripts/care-background-functional-attribution-v1'),
                str(ROOT/'scripts/care-scbe-v1'), str(ROOT/'scripts/care-multiyear-development-v1')]
from bfa_core import (SEEDS, anchors, digest, formal_split, load_data_contract,
                      load_m0)
from train import split_contract, global24, train_one
from train_stage2 import raw_bags, transform_saved
from metrics import cascade, paired_intervals
import rcdb

ARMS = ('M0_ORIGINAL','M1_EXTERNAL_NEUTRAL','M2_RCDB','M3_BALANCED_PROTOTYPE')
SOURCE_SPLITS = ('S1','S2','S3','S4','S5')
EXPECTED = {2017:799,2018:793,2019:399,2020:397,2021:400}

def write_json(name, obj):
    p=OUT/name
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,default=serial)

def write_csv(name, rows):
    p=OUT/name
    rows=list(rows)
    p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def serial(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)

def load_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def put_pickle(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)

def load_pickle(p):
    with p.open('rb') as f:return pickle.load(f)

def ckpt(split,seed,arm):
    if arm==ARMS[0]:return load_m0(split,seed)
    if arm==ARMS[1]:
        p=ROOT/'outputs/care_background_functional_attribution_v1/models'/split/str(seed)/'B1_EXTERNAL_NEUTRAL.pkl'
        obj=load_pickle(p)
        if (obj['arm'],obj['split'],obj['seed'])!=('B1_EXTERNAL_NEUTRAL',split,seed):raise RuntimeError('M1 identity')
        return obj
    p=OUT/'models'/split/str(seed)/(arm+'.pkl')
    return load_pickle(p)

def asset_paths():
    old=ROOT/'outputs/care_multiyear_development_v1'
    paths=[HERE/'rcdb.py',HERE/'main.py',ROOT/'scripts/care-scbe-v1/train.py',
           ROOT/'scripts/care-background-functional-attribution-v1/bfa_core.py',
           ROOT/'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py',
           ROOT/'scripts/care-di25-host-gatedgraph-source-v1/run_care_joint_dualprototype_ablation.py',
           ROOT/'scripts/care-multiyear-development-v1/train_stage2.py',
           old/'data.npz',old/'feature_rows.pkl',old/'sample_manifest.csv',
           ROOT/'outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv',
           ROOT/'outputs/care-svm-direct-c43-future-v1/predictions_label_free.csv']
    for split in SOURCE_SPLITS+('T1',):
        for seed in SEEDS:
            paths += [ROOT/'outputs/care_scbe_v1/models'/split/str(seed)/'M0_REPRODUCED.pkl',
                      ROOT/'outputs/care_background_functional_attribution_v1/models'/split/str(seed)/'B1_EXTERNAL_NEUTRAL.pkl']
    return list(dict.fromkeys(paths))

def verify_assets():
    before=json.loads((OUT/'asset_manifest_before.json').read_text())
    changed=[]
    for item in before:
        try:
            now=digest(item['path'])
            if now!=item:changed.append(item['path'])
        except OSError:changed.append(item['path'])
    return dict(status='PASS' if not changed else 'FAIL',checked=len(before),changed=changed,
                scope='named reused code/caches/routes/2017-only M0/M1 checkpoints; no APK files opened; not whole server')

def preflight(resume, allow_future_labels=False):
    if OUT.exists():
        if (OUT/'asset_manifest_before.json').is_file():
            if not resume:raise RuntimeError('existing output cannot be overwritten; use same-run --resume')
        else:
            if resume or any(x.name not in ('run.log','run_state.json','phase_status.json') for x in OUT.iterdir()):
                raise RuntimeError('existing output lacks integrity manifest')
            write_json('asset_manifest_before.json',[digest(p) for p in asset_paths()])
    else:
        OUT.mkdir(parents=True,exist_ok=False)
        write_json('asset_manifest_before.json',[digest(p) for p in asset_paths()])
    check=verify_assets()
    if check['status']!='PASS':raise RuntimeError('historical asset integrity changed')
    data,source=load_data_contract()
    counts={int(y):int(np.sum(data['years']==y)) for y in EXPECTED}
    if counts!=EXPECTED:raise RuntimeError('year inventory changed')
    src=np.flatnonzero(data['years']==2017)
    if len(src)!=799 or (int(np.sum(data['y'][src]==0)),int(np.sum(data['y'][src]==1)))!=(400,399):
        raise RuntimeError('2017 class protocol changed')
    folds=[]
    for s in SOURCE_SPLITS:
        tr,va,base,route,_=split_contract(data,s,source)
        if not (np.all(data['years'][tr]==2017) and len(va) in (159,160) and route.sum()==40):
            raise RuntimeError('source OOF fold/route mismatch')
        for i in va:folds.append(dict(sha256=str(data['ids'][i]),fold=s,year=2017))
    if len({x['sha256'] for x in folds})!=799:raise RuntimeError('OOF fold overlap')
    if not (OUT/'dataset_protocol.json').exists():
        write_json('dataset_protocol.json',dict(status='PASS',year_counts=counts,training_years=[2017],
            training_benign=400,training_malware=399,outer_fold_sizes=[160,160,160,160,159],
            seeds=SEEDS,prototype_slots=24,route='frozen SVM MeanOpposition Top25%',
            threshold=.5,future_role='frozen-model historical diagnostic only',
            future_is_independent_blind_test=False,APK_files_opened=0))
        write_csv('source2017_fold_manifest.csv',folds)
    if not allow_future_labels:
        # The historical bundle physically contains labels for all years.
        # Mask them before any feature/prototype/training routine sees data.
        data['y']=data['y'].copy()
        data['y'][data['years']>2017]=-1
    print('RCDB_PREFLIGHT_PASS output='+str(OUT),flush=True)
    return data,source

def bank_input(pack,y,seed):
    banks,manifest,roles,(train_z,train_role)=rcdb.fit_banks(
        pack['raw_train'],y,pack['old']['background'],seed)
    adapter=rcdb.calibrate(pack['bags_train'],train_z,train_role,banks['M3_BALANCED_PROTOTYPE'])
    prepared={}
    for arm in ARMS[2:]:
        tr=rcdb.transform(pack['bags_train'],pack['raw_train'],banks[arm],pack['old']['background'],adapter)
        va=rcdb.transform(pack['bags_val'],pack['raw_val'],banks[arm],pack['old']['background'],adapter)
        prepared[arm]=(tr,va)
        for before,after in zip(pack['bags_train']+pack['bags_val'],tr+va):
            if not np.array_equal(before[:,108:110],after[:,108:110]) or not np.array_equal(before[:,111:],after[:,111:]):
                raise RuntimeError('Internal/Commonness/Relation changed')
    return banks,manifest,roles,adapter,prepared

def smoke(data,source,device):
    pack=formal_split(data,'S1',SEEDS[0],device)
    tr=pack['train']
    # A genuinely small fit: 40 APK per class, no held-out labels in fitting.
    subset=np.concatenate([np.flatnonzero(data['y'][tr]==c)[:40] for c in (0,1)])
    from myd_common import runtime
    v4,_,args=runtime();args.device=device;args.keep_weight=args.fix_weight=1.;args.keep_consistency=0.
    full_tr=tr[subset]; raw=raw_bags(data,full_tr)
    from train_stage2 import capture_background
    a,b,_,prep=capture_background(v4,raw,raw[:8],data['y'][full_tr],args,SEEDS[0]+1707)
    sample=dict(raw_train=raw,raw_val=raw[:8],bags_train=a,bags_val=b,old={'background':prep})
    banks,manifest,roles,adapter,ready=bank_input(sample,data['y'][full_tr],SEEDS[0])
    if len(manifest)!=48 or any(len(banks[arm][c])!=12 for arm in banks for c in (0,1)):
        raise RuntimeError('RCDB smoke 24-slot invariant')
    # Real forward/backward using the frozen architecture and one optimizer step.
    torch=v4.torch;dev=v4.resolve_device(device)
    v4.seed_everything(SEEDS[0]);model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args).to(dev)
    gtrain,gtest=global24(v4,data,full_tr,full_tr[:8])
    scaler=v4.fit_role_graph_scaler(data['s7'][full_tr])
    graph=v4.transform_role_graphs(data['s7'][full_tr],scaler)
    graph_dummy=v4.transform_role_graphs(data['s7'][full_tr[:8]],scaler)
    pred,state,cost=train_one(v4,args,dev,SEEDS[0],1,gtrain,gtest,graph,graph_dummy,
                            ready['M2_RCDB'][0],ready['M2_RCDB'][1],data['y'][full_tr])
    if len(pred)!=8 or not np.isfinite(pred).all() or cost['parameters']<=0:
        raise RuntimeError('smoke train invalid')
    if not (OUT/'smoke_test.json').exists():
        write_json('smoke_test.json',dict(status='PASS',train_n=80,test_n=8,model_parameters=cost['parameters'],
            prototype_slots=24,host_width=142,shared_internal_unchanged=True,
            residual_recomputed=True,finite_probabilities=True,real_backward=True,
            m0_replay_diff=pack['replay_diff'],adapter=adapter))
    print('RCDB_SMOKE_PASS',flush=True)

def train_oof(data,source,device,resume):
    manifest=[];coverage=[];runtime=[]
    for split in SOURCE_SPLITS:
        for seed in SEEDS:
            p2=OUT/'models'/split/str(seed)/'M2_RCDB.pkl'
            p3=OUT/'models'/split/str(seed)/'M3_BALANCED_PROTOTYPE.pkl'
            if resume and p2.exists() and p3.exists():
                for arm,p in ((ARMS[2],p2),(ARMS[3],p3)):
                    obj=load_pickle(p)
                    if obj['split']!=split or obj['seed']!=seed or obj['arm']!=arm:raise RuntimeError('resume identity')
                    manifest.extend(obj['prototype_manifest']);coverage.extend(obj['role_coverage']);runtime.append(obj['runtime'])
                print('RCDB_OOF_RESUME',split,seed,flush=True);continue
            pack=formal_split(data,split,seed,device)
            y=np.asarray(data['y'][pack['train']],int)
            start=time.perf_counter()
            banks,pm,rc,adapter,ready=bank_input(pack,y,seed)
            build_sec=time.perf_counter()-start
            for arm in ARMS[2:]:
                path=OUT/'models'/split/str(seed)/(arm+'.pkl')
                if path.exists():
                    if not resume:raise RuntimeError('checkpoint exists')
                    obj=load_pickle(path)
                    if obj['arm']!=arm or obj['split']!=split or obj['seed']!=seed:raise RuntimeError('checkpoint identity')
                else:
                    tr,va=ready[arm]
                    pred,state,cost=train_one(pack['v4'],pack['args'],pack['dev'],seed,
                        int(pack['old']['selected_epochs']),pack['global_train'],pack['global_val'],
                        pack['graph_train'],pack['graph_val'],tr,va,y)
                    if cost['parameters']!=sum(x.numel() for x in pack['model'].parameters()):
                        raise RuntimeError('model capacity mismatch')
                    obj=dict(arm=arm,split=split,seed=seed,ids=list(map(str,data['ids'][pack['val']])),
                             probability=pred,state_dict=state,bank=banks[arm],adapter=adapter,
                             background=pack['old']['background'],graph_scaler=pack['old']['graph_scaler'],
                             selected_epochs=int(pack['old']['selected_epochs']),runtime=dict(split=split,seed=seed,
                             arm=arm,bank_seconds=build_sec,**cost),
                             prototype_manifest=[dict(split=split,seed=seed,**m) for m in pm if m['arm']==arm],
                             role_coverage=[dict(split=split,seed=seed,arm=arm,**r) for r in rc])
                    put_pickle(path,obj)
                manifest.extend(obj['prototype_manifest']);coverage.extend(obj['role_coverage']);runtime.append(obj['runtime'])
                print('RCDB_OOF_ARM_DONE',split,seed,arm,flush=True)
    for name,rows in [('rcdb_prototype_manifest.csv',manifest),('rcdb_role_coverage.csv',coverage),
                      ('runtime_cost.csv',runtime)]:
        if not (OUT/name).exists():write_csv(name,rows)
    if not (OUT/'prototype_class_coverage.csv').exists():
        write_csv('prototype_class_coverage.csv',manifest)
    if not (OUT/'shared_rcdb_overlap.csv').exists():
        write_csv('shared_rcdb_overlap.csv',[
            dict(split=r['split'],seed=r['seed'],arm=r['arm'],role=r['role'],
                 class_reference=r['class_reference'],shared_redundancy=r['redundancy'],
                 unique_own_class_coverage=r['unique_rate'],own_class_coverage=r['own_rate'])
            for r in manifest])
    print('RCDB_OOF_TRAIN_DONE',flush=True)

def source_eval(data,source):
    rows=[];metrics=[];compare=[];fixed=[];cond=[];boot=[]
    for split in SOURCE_SPLITS:
        tr,va,base,route,score=split_contract(data,split,source)
        ids=list(map(str,data['ids'][va]));y=np.asarray(data['y'][va],int)
        for arm in ARMS:
            ps=[]
            for seed in SEEDS:
                obj=ckpt(split,seed,arm)
                if obj['ids']!=ids or int(obj['selected_epochs'])!=int(load_m0(split,seed)['selected_epochs']):
                    raise RuntimeError('source checkpoint/OOF mismatch')
                ps.append(np.asarray(obj['probability'],float))
            p=np.mean(ps,axis=0)
            for i,sid in enumerate(ids):
                rows.append(dict(split=split,year=2017,sha256=sid,label=int(y[i]),arm=arm,
                                 stage1_probability=float(base[i]),router_score=None,
                                 routed=int(route[i]),expert_probability=float(p[i]),
                                 final_probability=float(p[i] if route[i] else base[i])))
    # Aggregate all 799 OOF APKs, not fold means.
    by_arm={arm:sorted([r for r in rows if r['arm']==arm],key=lambda r:r['sha256']) for arm in ARMS}
    ref=by_arm[ARMS[0]]
    yy=np.asarray([r['label'] for r in ref],int)
    bb=np.asarray([r['stage1_probability'] for r in ref],float)
    rr=np.asarray([r['routed'] for r in ref],bool)
    for arm in ARMS:
        group=by_arm[arm]
        if [r['sha256'] for r in group]!=[r['sha256'] for r in ref]:raise RuntimeError('OOF SHA misalignment')
        p=np.asarray([r['expert_probability'] for r in group],float)
        m=cascade(yy,bb,p,rr)
        metrics.append(dict(scope='SOURCE_2017_OOF',year=2017,arm=arm,**m))
        fixed.append(dict(scope='SOURCE_2017_OOF',arm=arm,Fixed=m['Fixed'],Broken=m['Broken'],Net=m['Net']))
        cond.append(dict(scope='SOURCE_2017_OOF',arm=arm,FN_vs_TN_AUC=m['FN_vs_TN_AUC'],
                         TP_vs_FP_AUC=m['TP_vs_FP_AUC'],Routed_AUC=m['Routed_AUC']))
        if arm!=ARMS[0]:
            rp=np.asarray([r['expert_probability'] for r in ref],float)
            for stat in paired_intervals(yy,bb,rr,p,rp,repeats=500):
                boot.append(dict(scope='SOURCE_2017_OOF',candidate=arm,reference=ARMS[0],**stat))
    for arm in ARMS[1:]:
        q=next(m for m in metrics if m['arm']==arm);b=metrics[0]
        compare.append(dict(scope='SOURCE_2017_OOF',candidate=arm,reference=ARMS[0],
                            delta_BA=q['BA']-b['BA'],delta_Net=q['Net']-b['Net'],
                            delta_FN=q['FN']-b['FN'],delta_FP=q['FP']-b['FP']))
    for name,vals in [('source2017_oof_predictions.csv',rows),('source2017_oof_metrics.csv',metrics),
                      ('fixed_broken_analysis.csv',fixed),('conditional_auc_analysis.csv',cond),
                      ('baseline_comparison.csv',compare),('paired_intervals.csv',boot)]:
        if not (OUT/name).exists():write_csv(name,vals)
    print('RCDB_SOURCE_EVAL_DONE',flush=True)

def final_train(data,source,device,resume):
    from myd_common import runtime
    v4,_,args=runtime();args.device=device;args.keep_weight=args.fix_weight=1.;args.keep_consistency=0.
    tr=np.flatnonzero(data['years']==2017);dummy=tr[:8]
    if len(tr)!=799:raise RuntimeError('full 2017 train contract')
    y=np.asarray(data['y'][tr],int)
    results=[]
    for seed in SEEDS:
        if resume and all((OUT/'models/FINAL2017'/str(seed)/(arm+'.pkl')).exists() for arm in ARMS[2:]):
            for arm in ARMS[2:]:
                obj=ckpt('FINAL2017',seed,arm);results.append(obj['training_result'])
            print('RCDB_FINAL_RESUME',seed,flush=True);continue
        m0=load_m0('T1',seed)
        m1=ckpt('T1',seed,ARMS[1])
        if m0['train_years']!=[2017] or m1['train_n']!=799:raise RuntimeError('final baseline not 2017-only')
        gx,gz=global24(v4,data,tr,dummy)
        raw=raw_bags(data,tr)
        formal=transform_saved(v4,raw,m0['background'],args)
        gsc=m0['graph_scaler']
        graph=v4.transform_role_graphs(data['s7'][tr],gsc)
        graph_dummy=v4.transform_role_graphs(data['s7'][dummy],gsc)
        # Full-2017 bank and train-only response calibration. Dummy prediction is
        # on training APKs and is never reported as a validation result.
        pack=dict(raw_train=raw,raw_val=raw[:8],bags_train=formal,bags_val=formal[:8],
                  old=m0)
        start=time.perf_counter();banks,pm,rc,adapter,ready=bank_input(pack,y,seed)
        build_sec=time.perf_counter()-start
        for arm in ARMS[2:]:
            path=OUT/'models/FINAL2017'/str(seed)/(arm+'.pkl')
            if path.exists():
                if not resume:raise RuntimeError('final checkpoint already exists')
                obj=load_pickle(path)
                if obj['arm']!=arm or obj['seed']!=seed:raise RuntimeError('final checkpoint mismatch')
            else:
                pred,state,cost=train_one(v4,args,v4.resolve_device(device),seed,
                    int(m0['selected_epochs']),gx,gz,graph,graph_dummy,ready[arm][0],ready[arm][1],y)
                train_result=dict(arm=arm,seed=seed,training_years='2017',training_n=799,
                    selected_epochs=m0['selected_epochs'],prototype_slots=24,bank_seconds=build_sec,**cost)
                obj=dict(arm=arm,split='FINAL2017',seed=seed,train_years=[2017],train_n=799,
                         state_dict=state,bank=banks[arm],adapter=adapter,
                         background=m0['background'],graph_scaler=gsc,
                         selected_epochs=m0['selected_epochs'],training_result=train_result,
                         prototype_manifest=[dict(split='FINAL2017',seed=seed,**m) for m in pm if m['arm']==arm],
                         role_coverage=[dict(split='FINAL2017',seed=seed,arm=arm,**r) for r in rc])
                put_pickle(path,obj)
            results.append(obj['training_result'])
            print('RCDB_FINAL_ARM_DONE',seed,arm,flush=True)
    if not (OUT/'final2017_training_results.csv').exists():write_csv('final2017_training_results.csv',results)
    # This immutable manifest marks the point before any future model inference/labels.
    if not (OUT/'final2017_frozen.json').exists():
        items=[digest(OUT/'models/FINAL2017'/str(seed)/(arm+'.pkl')) for seed in SEEDS for arm in ARMS[2:]]
        write_json('final2017_frozen.json',dict(status='FROZEN',training_years=[2017],models=items,
                                               baseline_model='historical M0/M1 T1 full-2017 checkpoints'))
    print('RCDB_FINAL2017_FROZEN',flush=True)

def future_eval(data,device):
    if not (OUT/'final2017_frozen.json').is_file():raise RuntimeError('final models not frozen')
    frozen=json.loads((OUT/'final2017_frozen.json').read_text())
    for item in frozen['models']:
        if digest(item['path'])!=item:raise RuntimeError('frozen final RCDB model changed')
    from myd_common import runtime
    v4,_,args=runtime();args.device=device;args.keep_weight=args.fix_weight=1.;args.keep_consistency=0.
    dev=v4.resolve_device(device)
    stage={r['sha256']:r for r in load_csv(ROOT/'outputs/care-svm-direct-c43-future-v1/predictions_label_free.csv')}
    if len(stage)!=1989:raise RuntimeError('2017-frozen future stage1 table mismatch')
    tr=np.flatnonzero(data['years']==2017)
    predictions=[];metrics=[];fixed=[];cond=[];comparisons=[];intervals=[]
    for year in (2018,2019,2020,2021):
        va=np.flatnonzero(data['years']==year)
        ids=list(map(str,data['ids'][va]))
        if len(va)!=EXPECTED[year] or any(s not in stage or int(stage[s]['year'])!=year for s in ids):
            raise RuntimeError('future SHA/year alignment failed')
        base=np.asarray([float(stage[s]['stage1_prob']) for s in ids])
        route=np.asarray([int(stage[s]['routed'])==1 for s in ids],bool)
        score=np.asarray([float(stage[s]['router_score']) for s in ids])
        if route.sum()!=int(round(.25*len(va))):raise RuntimeError('future route budget differs')
        gx,gz=global24(v4,data,tr,va)
        raw=raw_bags(data,va)
        per_arm={arm:[] for arm in ARMS}
        for seed in SEEDS:
            m0=load_m0('T1',seed)
            m1=ckpt('T1',seed,ARMS[1])
            final={arm:ckpt('FINAL2017',seed,arm) for arm in ARMS[2:]}
            formal=transform_saved(v4,raw,m0['background'],args)
            graph=v4.transform_role_graphs(data['s7'][va],m0['graph_scaler'])
            neutral=float(m1['feature_protocol']['neutral_value'])
            m1bags=[]
            for b in formal:
                a=b.copy();a[:,107]=neutral
                a[:,110]=(1-np.clip(a[:,109],0,1))*np.sqrt(np.maximum(a[:,107],0)*np.maximum(a[:,108],0))
                m1bags.append(a)
            input_bags={ARMS[0]:formal,ARMS[1]:m1bags}
            for arm in ARMS[2:]:
                obj=final[arm]
                input_bags[arm]=rcdb.transform(formal,raw,obj['bank'],obj['background'],obj['adapter'])
            for arm in ARMS:
                obj=m0 if arm==ARMS[0] else m1 if arm==ARMS[1] else final[arm]
                v4.seed_everything(seed)
                model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args).to(dev)
                model.load_state_dict(obj['state_dict']);model.eval()
                p,_=v4.predict_model(model,gz,np.full(len(va),.5),graph,input_bags[arm],dev)
                p=np.asarray(p,float)
                if not np.isfinite(p).all():raise RuntimeError('nonfinite future probability')
                if year==2018:
                    previous=np.asarray(obj['probability'],float) if arm in ARMS[:2] else None
                    if previous is not None and (obj['ids']!=ids or np.max(np.abs(p-previous))>1e-5):
                        raise RuntimeError('2018 M0/M1 baseline replay mismatch')
                per_arm[arm].append(p)
                del model
                if dev.type=='cuda':v4.torch.cuda.empty_cache()
        y=np.asarray(data['y'][va],int)  # first future-label access: after final freeze
        for arm in ARMS:
            p=np.mean(per_arm[arm],axis=0)
            m=cascade(y,base,p,route)
            metrics.append(dict(year=year,arm=arm,**m))
            fixed.append(dict(scope=str(year),arm=arm,Fixed=m['Fixed'],Broken=m['Broken'],Net=m['Net']))
            cond.append(dict(scope=str(year),arm=arm,FN_vs_TN_AUC=m['FN_vs_TN_AUC'],
                             TP_vs_FP_AUC=m['TP_vs_FP_AUC'],Routed_AUC=m['Routed_AUC']))
            for i,sid in enumerate(ids):
                predictions.append(dict(year=year,sha256=sid,label=int(y[i]),arm=arm,
                    stage1_probability=float(base[i]),router_score=float(score[i]),routed=int(route[i]),
                    expert_probability=float(p[i]),final_probability=float(p[i] if route[i] else base[i])))
        for arm in ARMS[1:]:
            q=next(m for m in metrics if m['year']==year and m['arm']==arm)
            ref=next(m for m in metrics if m['year']==year and m['arm']==ARMS[0])
            comparisons.append(dict(scope=str(year),candidate=arm,reference=ARMS[0],
                delta_BA=q['BA']-ref['BA'],delta_Net=q['Net']-ref['Net'],
                delta_FN=q['FN']-ref['FN'],delta_FP=q['FP']-ref['FP']))
            for stat in paired_intervals(y,base,route,np.mean(per_arm[arm],axis=0),
                                         np.mean(per_arm[ARMS[0]],axis=0),repeats=500):
                intervals.append(dict(scope=str(year),candidate=arm,reference=ARMS[0],**stat))
        print('RCDB_FUTURE_YEAR_DONE',year,flush=True)
    for name,rows in [('future2018_2021_predictions.csv',predictions),('future2018_2021_metrics.csv',metrics),
                      ('future_fixed_broken.csv',fixed),('future_conditional_auc.csv',cond),
                      ('future_baseline_comparison.csv',comparisons),('future_paired_intervals.csv',intervals)]:
        if not (OUT/name).exists():write_csv(name,rows)

def collect():
    if verify_assets()['status']!='PASS':raise RuntimeError('original assets changed after experiment')
    write_json('asset_integrity_report.json',verify_assets())
    source=load_csv(OUT/'source2017_oof_metrics.csv')
    future=load_csv(OUT/'future2018_2021_metrics.csv')
    by={r['arm']:r for r in source}
    m0,m1,m2,m3=[by[a] for a in ARMS]
    def delta(a,b,k):return float(a[k])-float(b[k])
    flags=[]
    if delta(m2,m0,'BA')>=.005 and delta(m2,m1,'BA')>0 and delta(m2,m3,'BA')>0 and delta(m2,m0,'Net')>=6:
        flags.append('RCDB_DISCRIMINATIVE_SELECTION_SUPPORTED_ON_DEVELOPMENT')
    elif delta(m2,m0,'BA')>0 and delta(m2,m3,'BA')<=0:
        flags.append('RCDB_GAIN_EXPLAINED_BY_BALANCED_PROTOTYPES')
    elif delta(m2,m0,'BA')>0:flags.append('RCDB_GAIN_OVER_ORIGINAL_ONLY')
    else:flags.append('RCDB_NO_RELIABLE_INCREMENT')
    if delta(m2,m0,'FN')>0:flags.append('RCDB_INCREASES_FALSE_NEGATIVES')
    ma=[r for r in future if r['arm']=='M2_RCDB'];mb=[r for r in future if r['arm']=='M0_ORIGINAL']
    if sum(float(a['FN_vs_TN_AUC'])<float(b['FN_vs_TN_AUC']) for a,b in zip(ma,mb))>=3:
        flags.append('RCDB_HURTS_CONDITIONAL_DISCRIMINATION')
    report=['# RCDB-V1 实验报告','',
      '本实验只用原始2017年799个APK进行五折/全量训练；2018—2021仅对全量2017冻结模型进行历史开发诊断。',
      'M0/M1复用已核验的2017-only checkpoint。M2/M3相同KMeans候选、12+12 Role配额、同一非负有符号对比适配器及训练预算；仅选择目标不同。',
      'M2以APK级本类支持、异类惩罚和Shared非冗余选择；M3仅依本类支持选择。APK标签是弱监督来源，不是Host恶意语义真值。',
      'RCDB槽位是有符号距离差经训练期校准后的非负响应；Residual按原代数式重算，但语义不再是原External距离残差。',
      '', '## 2017五折 OOF', '',
      '| 模型 | BA | FN | FP | Fixed | Broken | Net | FN/TN AUC | TP/FP AUC |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        r=by[arm]
        report.append(f"| {arm} | {float(r['BA']):.4f} | {r['FN']} | {r['FP']} | {r['Fixed']} | {r['Broken']} | {r['Net']} | {float(r['FN_vs_TN_AUC']):.4f} | {float(r['TP_vs_FP_AUC']):.4f} |")
    report+=['','## 2017全量训练后跨年份冻结诊断','',
      '| 年份 | 模型 | BA | FN | FP | Fixed | Broken | Net | FN/TN AUC | TP/FP AUC |',
      '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in future:
        report.append(f"| {r['year']} | {r['arm']} | {float(r['BA']):.4f} | {r['FN']} | {r['FP']} | {r['Fixed']} | {r['Broken']} | {r['Net']} | {float(r['FN_vs_TN_AUC']):.4f} | {float(r['TP_vs_FP_AUC']):.4f} |")
    report+=['','## 决策边界','',
      '候选状态：'+', '.join(flags)+'。2018—2021已用于历史研发，不能视为独立盲测，也不能用来重选超参数。',
      '若M2未超过M3，不能把增益归因于判别性原型选择；若未超过M1，也不能建议替换原External。',
      '详细机制/成对不确定性见 prototype_class_coverage.csv、shared_rcdb_overlap.csv、paired_intervals.csv。']
    write_json('final_decision.json',dict(status=flags[0],flags=flags,source_delta_BA_M2_M0=delta(m2,m0,'BA'),
        source_delta_BA_M2_M1=delta(m2,m1,'BA'),source_delta_BA_M2_M3=delta(m2,m3,'BA'),
        source_delta_Net_M2_M0=delta(m2,m0,'Net'),future_years=[2018,2019,2020,2021],
        future_is_independent_blind_test=False,training_years=[2017],baseline_integrity='PASS'))
    with (OUT/'final_report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(report)+'\n')
    print('RCDB_COLLECT_DONE',flush=True)

def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    for opt in ('audit-only','smoke-test','train-oof','source-eval','train-final2017','evaluate-future','collect'):
        g.add_argument('--'+opt,action='store_true')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    ap.add_argument('--workers',type=int,default=1)
    a=ap.parse_args()
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if a.workers!=1:raise RuntimeError('single-worker sequential protocol')
    if a.device=='cuda':
        import torch
        probe=torch.zeros(1,device='cuda');del probe
    allow_future_labels=bool(a.evaluate_future)
    if allow_future_labels and not (OUT/'final2017_frozen.json').is_file():
        raise RuntimeError('future labels prohibited before full-2017 freeze')
    data,source=preflight(a.resume,allow_future_labels)
    if a.audit_only:return
    if a.smoke_test:smoke(data,source,a.device)
    elif a.train_oof:train_oof(data,source,a.device,a.resume)
    elif a.source_eval:source_eval(data,source)
    elif a.train_final2017:final_train(data,source,a.device,a.resume)
    elif a.evaluate_future:future_eval(data,a.device)
    elif a.collect:collect()

if __name__=='__main__':main()
