#!/usr/bin/env python3
"""Train and evaluate W0 with BCE, focal, or BCE+pairwise-ranking objectives.

Inputs: frozen W0 tensors/checkpoints, fixed 2017 folds, Manifest135 Stage-I/router
predictions, and cached 2018-2021 tensors. Outputs: strict OOF predictions, independent
new checkpoints, fixed-threshold metrics, frozen future diagnostics, reports, and hashes.
Only the objective changes; architecture, inputs, batching, optimizer, and threshold do not.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

from common import *

def formal_assets():
    paths=[
      MYD/'data.npz',MYD/'feature_rows.pkl',MYD/'dataset_audit.json',MYD/'sample_manifest.csv',
      AUTO/'per_apk_predictions.csv.gz',AUTO/'protocol.json',AUTO/'asset_integrity_check.json',
      ENDPOINT/'source_2017/source_oof_predictions.csv',ENDPOINT/'asset_integrity.json',
      ROOT/'scripts/care-background-functional-attribution-v1/bfa_core.py',
      ROOT/'scripts/care-scbe-v1/train.py',ROOT/'scripts/care-multiyear-development-v1/train_stage2.py',
      ROOT/'scripts/care-multiyear-development-v1/myd_common.py',
      ROOT/'scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py',
    ]
    paths += [SCBE/'models'/split/str(seed)/'M0_REPRODUCED.pkl' for split in SPLITS+('T1',) for seed in MODEL_SEEDS]
    return paths

def preflight(resume=False):
    if OUT.exists() and not resume:raise RuntimeError('output exists; never overwrite '+str(OUT))
    missing=[str(p) for p in formal_assets() if not Path(p).is_file()]
    if missing:raise RuntimeError('missing formal assets '+repr(missing))
    if json.loads((AUTO/'asset_integrity_check.json').read_text()).get('status')!='PASS':raise RuntimeError('router integrity is not PASS')
    if json.loads((ENDPOINT/'asset_integrity.json').read_text()).get('status')!='PASS':raise RuntimeError('recent endpoint integrity is not PASS')
    if not OUT.exists():
        for d in ('config','integrity','models','source_2017','future/2018','future/2019','future/2020','future/2021','reports','logs','smoke'):(OUT/d).mkdir(parents=True,exist_ok=True)
    data=load_data();years=np.asarray(data['years'],int)
    counts={y:int(np.sum(years==y)) for y in YEARS}
    if counts!=YEAR_COUNTS or len(set(map(str,data['ids'])))!=2788:raise RuntimeError('dataset inventory mismatch')
    before=[digest(p) for p in formal_assets()]
    config={'status':'PRE_REGISTERED','stage1':'Manifest135 RF frozen','router':'AUTO_ERROR_LR P(error)>=0.5 frozen; no fixed budget','stage2':'W0 Direct-C43 unchanged','arms':ARMS,'classification_threshold':0.5,'router_threshold':0.5,'T0':'original equal BCEWithLogitsLoss','T1':'symmetric focal gamma=2 alpha_pos=alpha_neg=0.5','T2':'original BCE + 0.1 batch pairwise logistic ranking','optimizer':'Adam lr=8e-4 weight_decay=5e-4','batch_size':16,'max_epochs':30,'early_stop_fraction':0.18,'early_stop_patience':5,'early_stop_min_delta':1e-4,'grad_clip':5.0,'source':'2017 fixed five-fold OOF; three model seeds','future':[2018,2019,2020,2021],'future_used_for_selection':False,'multiple_comparisons':'Holm over T1-T0 and T2-T0','support_rule':'positive fixed-0.5 end-to-end BA delta, APK-cluster class-stratified CI lower bound >0, Holm p<0.05, positive across >=4 folds and >=2 model seeds, Net Fixed >0'}
    atomic_json(OUT/'config/experiment_config.json',config,replace=resume)
    atomic_json(OUT/'integrity/asset_sha256_before.json',{'formal_assets':before},replace=resume)
    atomic_json(OUT/'config/baseline_manifest.json',{'expected_W0_BA':0.9194653299916458,'loss':'equal BCEWithLogitsLoss; keep/fix weights both 1; consistency 0','folds':SPLITS,'model_seeds':MODEL_SEEDS,'manifest_seeds':MANIFEST_SEEDS,'year_counts':YEAR_COUNTS,'W0_checkpoints':[str(SCBE/'models'/s/str(seed)/'M0_REPRODUCED.pkl') for s in SPLITS+('T1',) for seed in MODEL_SEEDS]},replace=resume)
    status('preflight_complete');print('LOSS_PREFLIGHT_PASS',counts,flush=True)

def objective(v4,logits,labels,arm):
    torch=v4.torch;F=torch.nn.functional
    bce=F.binary_cross_entropy_with_logits(logits,labels,reduction='none')
    if arm=='T0_W0_BCE':return torch.mean(bce),False
    if arm=='T1_FOCAL_GAMMA2':
        p=torch.sigmoid(logits);pt=torch.where(labels>.5,p,1-p)
        return torch.mean(.5*torch.square(1-pt)*bce),False
    if arm=='T2_BCE_RANK01':
        pos=logits[labels>.5];neg=logits[labels<.5]
        if pos.numel()==0 or neg.numel()==0:return torch.mean(bce),True
        rank=F.softplus(-(pos[:,None]-neg[None,:])).mean()
        return torch.mean(bce)+.1*rank,False
    raise ValueError(arm)

def train_epochs_custom(v4,model,optimizer,g,graph,bags,y,epochs,args,seed,arm):
    torch=v4.torch;device=g.device;base=torch.zeros(len(y),dtype=torch.float32,device=device);yt=torch.as_tensor(y,dtype=torch.float32,device=device)
    rng=np.random.RandomState(int(seed)+811);single=total_batches=0;final=float('nan')
    for _ in range(int(epochs)):
        model.train();order=rng.permutation(len(y));weighted=0.
        for start in range(0,len(order),int(args.batch_size)):
            ix=order[start:start+int(args.batch_size)];optimizer.zero_grad(set_to_none=True)
            logits=v4.model_forward_batch(model,g,base,graph,bags,ix);it=torch.as_tensor(ix,dtype=torch.long,device=device)
            loss,one=objective(v4,logits,yt[it],arm);single+=int(one);total_batches+=1
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(args.grad_clip));optimizer.step();weighted+=float(loss.detach().cpu())*len(ix)
        final=weighted/len(y)
    return final,{'single_class_batches':single,'total_batches':total_batches,'single_class_fraction':single/max(total_batches,1)}

def eval_custom(v4,model,g,graph_values,bags,y,args,dev,arm):
    torch=v4.torch;gt=torch.as_tensor(g,dtype=torch.float32,device=dev);graph=v4.graph_tensors(graph_values,dev);bt=[torch.as_tensor(x,dtype=torch.float32,device=dev) for x in bags];yt=torch.as_tensor(y,dtype=torch.float32,device=dev);base=torch.zeros(len(y),dtype=torch.float32,device=dev)
    model.eval()
    with torch.no_grad():logits=v4.model_forward_batch(model,gt,base,graph,bt,np.arange(len(y)));loss,_=objective(v4,logits,yt,arm)
    return float(loss.cpu())

def select_epochs(pack,data,arm,seed):
    v4,args,dev=pack['v4'],pack['args'],pack['dev'];torch=v4.torch
    y=np.asarray(data['y'][pack['train']],int);global_values=np.asarray(pack['global_train']);raw=pack['raw_train'];s7=np.asarray(data['s7'][pack['train']])
    splitter=v4.StratifiedShuffleSplit(n_splits=1,test_size=float(args.early_stop_fraction),random_state=int(seed)+int(args.split_seed));fit_ix,val_ix=next(splitter.split(np.zeros(len(y)),y))
    fit_raw=[raw[i] for i in fit_ix];val_raw=[raw[i] for i in val_ix];fit_bags,val_bags,_=v4.v3.fit_scaled_dualproto_bags(fit_raw,val_raw,y[fit_ix],args,int(seed)+707)
    scaler=v4.fit_role_graph_scaler(s7[fit_ix]);fit_graph=v4.transform_role_graphs(s7[fit_ix],scaler);val_graph=v4.transform_role_graphs(s7[val_ix],scaler)
    g=torch.as_tensor(global_values[fit_ix],dtype=torch.float32,device=dev);graph=v4.graph_tensors(fit_graph,dev);bags=[torch.as_tensor(x,dtype=torch.float32,device=dev) for x in fit_bags]
    v4.seed_everything(seed);model=v4.RoleGraphConditionedEvidence(global_values.shape[1],'RGCondAttn',args).to(dev);opt=torch.optim.Adam(model.parameters(),lr=float(args.learning_rate),weight_decay=float(args.weight_decay))
    best=float('inf');best_epoch=1;patience=0
    for epoch in range(1,int(args.max_epochs)+1):
        train_epochs_custom(v4,model,opt,g,graph,bags,y[fit_ix],1,args,int(seed)+epoch*17,arm)
        value=eval_custom(v4,model,global_values[val_ix],val_graph,val_bags,y[val_ix],args,dev,arm)
        if value<best-float(args.early_stop_min_delta):best=value;best_epoch=epoch;patience=0
        else:patience+=1
        if patience>=int(args.early_stop_patience):break
    del model,opt,g,graph,bags
    if dev.type=='cuda':torch.cuda.empty_cache()
    return int(best_epoch),float(best)

def fit_arm(pack,data,arm,seed,selected_epochs=None):
    v4,args,dev=pack['v4'],pack['args'],pack['dev'];torch=v4.torch;y=np.asarray(data['y'][pack['train']],int)
    epochs,sel=(select_epochs(pack,data,arm,seed) if selected_epochs is None else (int(selected_epochs),None))
    v4.seed_everything(seed);model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args).to(dev);opt=torch.optim.Adam(model.parameters(),lr=float(args.learning_rate),weight_decay=float(args.weight_decay))
    g=torch.as_tensor(pack['global_train'],dtype=torch.float32,device=dev);graph=v4.graph_tensors(pack['graph_train'],dev);bags=[torch.as_tensor(x,dtype=torch.float32,device=dev) for x in pack['bags_train']]
    started=time.perf_counter();loss,batch_audit=train_epochs_custom(v4,model,opt,g,graph,bags,y,epochs,args,seed+3301,arm);seconds=time.perf_counter()-started
    pred,_=v4.predict_model(model,pack['global_val'],np.full(len(pack['val']),.5),pack['graph_val'],pack['bags_val'],dev)
    state={k:x.detach().cpu() for k,x in model.state_dict().items()};audit={'selected_epochs':epochs,'selection_loss':sel,'final_loss':loss,'training_seconds':seconds,'batch_audit':batch_audit,'parameters':sum(x.numel() for x in model.parameters())}
    del model,opt,g,graph,bags
    if dev.type=='cuda':torch.cuda.empty_cache()
    return np.asarray(pred),state,audit

def baseline_replay(device='cuda'):
    status('baseline_replay');from bfa_core import formal_split
    data=load_data();years=np.asarray(data['years'],int);source=np.flatnonzero(years==2017);ids=np.asarray(data['ids']).astype(str);pos={ids[i]:j for j,i in enumerate(source)};prob=np.full((3,len(source)),np.nan);rows=[];maxdiff=0.
    for split in SPLITS:
      for si,seed in enumerate(MODEL_SEEDS):
        pack=formal_split(data,split,seed,device=device,train_data=False);maxdiff=max(maxdiff,float(pack['replay_diff']))
        for sid,p in zip(ids[pack['val']],pack['old']['probability']):prob[si,pos[sid]]=p
        rows.append({'split':split,'seed':seed,'n':len(pack['val']),'max_probability_difference':pack['replay_diff'],'selected_epochs':pack['old']['selected_epochs']})
        del pack
    if not np.isfinite(prob).all():raise RuntimeError('incomplete W0 replay')
    ref=pd.read_csv(ENDPOINT/'source_2017/source_oof_predictions.csv');ref=ref[ref.arm=='B0_W0'].drop_duplicates('sha256').set_index('sha256').loc[ids[source]]
    endpoint_diff=float(np.max(np.abs(prob.mean(0)-ref.expert_probability.to_numpy(float))))
    if endpoint_diff>1e-6:raise RuntimeError('BLOCKED_BASELINE_REPLAY endpoint diff '+str(endpoint_diff))
    atomic_json(OUT/'integrity/replay_check.json',{'status':'PASS','checkpoint_replay_max_difference':maxdiff,'endpoint_ensemble_max_difference':endpoint_diff,'fold_seed':rows},replace=True)
    print('BASELINE_REPLAY_PASS',maxdiff,endpoint_diff,flush=True)

def smoke(device='cuda'):
    status('smoke_test');from bfa_core import formal_split
    data=load_data();pack=formal_split(data,'S1',900000,device=device,train_data=True);keep=np.concatenate([np.flatnonzero(data['y'][pack['train']]==v)[:24] for v in (0,1)]);val=np.arange(min(10,len(pack['val'])))
    small=dict(pack);small['train']=pack['train'][keep];small['global_train']=pack['global_train'][keep];small['raw_train']=[pack['raw_train'][i] for i in keep];small['bags_train']=[pack['bags_train'][i] for i in keep];small['graph_train']={k:np.asarray(v)[keep] for k,v in pack['graph_train'].items()};small['val']=pack['val'][val];small['global_val']=pack['global_val'][val];small['bags_val']=[pack['bags_val'][i] for i in val];small['graph_val']={k:np.asarray(v)[val] for k,v in pack['graph_val'].items()}
    out=[]
    for arm in ARMS:
        pred,_,audit=fit_arm(small,data,arm,900000,selected_epochs=1)
        if pred.shape!=(len(val),) or not np.isfinite(pred).all():raise RuntimeError('smoke prediction '+arm)
        out.append({'arm':arm,'status':'PASS','n':len(pred),'audit':audit})
    atomic_json(OUT/'smoke/loss_smoke.json',{'status':'PASS','arms':out},replace=True);print('LOSS_SMOKE_PASS',flush=True)

def train_source(device='cuda'):
    status('source_2017_training');from bfa_core import formal_split
    data=load_data();ids=np.asarray(data['ids']).astype(str)
    for split in SPLITS:
      for seed in MODEL_SEEDS:
        pack=formal_split(data,split,seed,device=device,train_data=True)
        for arm in ARMS:
            path=OUT/'models'/split/str(seed)/(arm+'.pkl')
            if path.is_file():continue
            pred,state,audit=fit_arm(pack,data,arm,seed)
            if arm=='T0_W0_BCE':
                diff=float(np.max(np.abs(pred-np.asarray(pack['old']['probability']))));audit['formal_W0_max_probability_difference']=diff;audit['formal_selected_epochs']=int(pack['old']['selected_epochs'])
                if audit['selected_epochs']!=int(pack['old']['selected_epochs']) or diff>1e-6:raise RuntimeError(f'BLOCKED_BASELINE_REPLAY retrain {split}/{seed} epochs={audit["selected_epochs"]}/{pack["old"]["selected_epochs"]} diff={diff}')
            atomic_pickle(path,{'arm':arm,'split':split,'seed':seed,'target_ids':list(ids[pack['val']]),'train_ids':list(ids[pack['train']]),'probability':pred,'state_dict':state,'audit':audit,'threshold':.5,'future_labels_used':False})
            print('LOSS_ARM_DONE',split,seed,arm,'epochs='+str(audit['selected_epochs']),flush=True)
        del pack

def source_probabilities(data):
    ids=np.asarray(data['ids']).astype(str);source=np.flatnonzero(np.asarray(data['years'],int)==2017);pos={ids[i]:j for j,i in enumerate(source)};values={a:np.full((3,len(source)),np.nan) for a in ARMS};fold=np.zeros(len(source),int)
    for fi,split in enumerate(SPLITS,1):
      for si,seed in enumerate(MODEL_SEEDS):
        for arm in ARMS:
            obj=read_pickle(OUT/'models'/split/str(seed)/(arm+'.pkl'));loc=np.asarray([pos[s] for s in obj['target_ids']],int);values[arm][si,loc]=obj['probability'];fold[loc]=fi
    if any(not np.isfinite(v).all() for v in values.values()):raise RuntimeError('incomplete source probabilities')
    return source,fold,values

def bootstrap(y,cand,ref,repeats=2000,seed=20261022):
    y=np.asarray(y,int);rng=np.random.RandomState(seed);groups=[np.flatnonzero(y==v) for v in (0,1)];vals=[]
    for _ in range(repeats):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);vals.append(np.mean([metric(y[ix],cand[s,ix])['BA']-metric(y[ix],ref[s,ix])['BA'] for s in range(cand.shape[0])]))
    vals=np.asarray(vals);p=min(1.,2*min(float(np.mean(vals<=0)),float(np.mean(vals>=0))))
    return {'bootstrap_mean':float(vals.mean()),'ci95_low':float(np.quantile(vals,.025)),'ci95_high':float(np.quantile(vals,.975)),'p_two_sided':p,'repeats':repeats,'cluster_unit':'independent APK; class stratified; model/Manifest seeds correlated within APK'}

def evaluate_source():
    status('source_2017_evaluation');data=load_data();ids=np.asarray(data['ids']).astype(str);source,fold,probs=source_probabilities(data);sids=list(ids[source]);y=np.asarray(data['y'][source],int);routes=route_rows(2017);performance=[];per=[];finals={};means=[]
    for arm in ARMS:
        expert=probs[arm].mean(0);f=[]
        for ms in MANIFEST_SEEDS:
            q=routes[routes.manifest_seed==ms].set_index('sha256').loc[sids];m,final=cascade(y,q.stage1_probability.to_numpy(float),expert,q.routed.to_numpy(bool));performance.append({'arm':arm,'manifest_seed':ms,**m});f.append(final)
            for i,sid in enumerate(sids):per.append({'arm':arm,'manifest_seed':ms,'sha256':sid,'fold':int(fold[i]),'label':int(y[i]),'routed':int(q.routed.iloc[i]),'stage1_probability':float(q.stage1_probability.iloc[i]),'stage2_probability':float(expert[i]),'final_probability':float(final[i])})
        finals[arm]=np.asarray(f)
        rr=[r for r in performance if r['arm']==arm];keys=[k for k,v in rr[0].items() if k not in ('arm','manifest_seed') and isinstance(v,(int,float,np.integer,np.floating))];means.append({'arm':arm,**{k:float(np.nanmean([r[k] for r in rr])) for k in keys}})
    b0=next(r for r in means if r['arm']=='T0_W0_BCE')
    if abs(b0['BA']-.9194653299916458)>1e-8:raise RuntimeError('BLOCKED_BASELINE_REPLAY T0 BA '+str(b0['BA']))
    comparisons=[]
    for i,arm in enumerate(ARMS[1:]):
        cm=next(r for r in means if r['arm']==arm);fold_delta=[];seed_delta=[]
        for fi in range(1,6):
            ix=fold==fi;fold_delta.append(metric(y[ix],finals[arm].mean(0)[ix])['BA']-metric(y[ix],finals['T0_W0_BCE'].mean(0)[ix])['BA'])
        for si in range(3):seed_delta.append(metric(y,finals[arm][si])['BA']-metric(y,finals['T0_W0_BCE'][si])['BA'])
        comparisons.append({'candidate':arm,'reference':'T0_W0_BCE','BA_difference':cm['BA']-b0['BA'],'Fixed_difference':cm['Fixed']-b0['Fixed'],'Broken_difference':cm['Broken']-b0['Broken'],'Net_Fixed_difference':cm['Net_Fixed']-b0['Net_Fixed'],'FP_difference':cm['FP']-b0['FP'],'FN_difference':cm['FN']-b0['FN'],'positive_folds':int(np.sum(np.asarray(fold_delta)>0)),'fold_deltas':fold_delta,'positive_model_seeds':int(np.sum(np.asarray(seed_delta)>0)),'model_seed_deltas':seed_delta,**bootstrap(y,finals[arm],finals['T0_W0_BCE'],seed=20261022+i)})
    holm(comparisons)
    supported={r['candidate']:(r['BA_difference']>0 and r['ci95_low']>0 and r['p_holm']<.05 and r['Net_Fixed_difference']>0 and r['positive_folds']>=4 and r['positive_model_seeds']>=2) for r in comparisons}
    decision='BOTH_SUPPORTED' if all(supported.values()) else 'T1_FOCAL_SUPPORTED' if supported['T1_FOCAL_GAMMA2'] else 'T2_RANKING_SUPPORTED' if supported['T2_BCE_RANK01'] else 'NO_SUPPORTED_GAIN_2017_OOF'
    # Fold/model-seed diagnostic uses each Stage-II seed and the first matching Manifest seed only for a fixed comparison grid.
    fs=[]
    for arm in ARMS:
      for si,seed in enumerate(MODEL_SEEDS):
       for fi in range(1,6):
        ix=fold==fi;m=metric(y[ix],probs[arm][si,ix]);fs.append({'arm':arm,'model_seed':seed,'fold':fi,**m})
    atomic_csv(OUT/'source_2017/oof_predictions.csv',per,replace=True);atomic_csv(OUT/'source_2017/fold_seed_results.csv',fs,replace=True);atomic_csv(OUT/'source_2017/paired_comparison.csv',comparisons,replace=True);atomic_csv(OUT/'source_2017/bootstrap_intervals.csv',comparisons,replace=True);atomic_csv(OUT/'source_2017/arm_summary.csv',means,replace=True)
    atomic_json(OUT/'source_2017/source_2017_decision.json',{'status':decision,'supported':supported,'baseline_BA':b0['BA'],'comparisons':comparisons,'classification_threshold':.5,'future_used_for_selection':False},replace=True);print('SOURCE_DECISION',decision,flush=True)

def predict_with_state(pack,state,global_values,graphs,bags):
    v4,dev,args=pack['v4'],pack['dev'],pack['args'];model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args).to(dev);model.load_state_dict(state);p,_=v4.predict_model(model,global_values,np.full(len(bags),.5),graphs,bags,dev);del model
    if dev.type=='cuda':v4.torch.cuda.empty_cache()
    return np.asarray(p)

def train_full(device='cuda'):
    status('full_2017_training_and_freeze');from bfa_core import formal_split;from train import global24;from train_stage2 import raw_bags,transform_saved
    data=load_data();ids=np.asarray(data['ids']).astype(str);years=np.asarray(data['years'],int);source=np.flatnonzero(years==2017)
    for seed in MODEL_SEEDS:
        pack=formal_split(data,'T1',seed,device=device,train_data=True)
        for arm in ARMS:
            path=OUT/'models'/'T1'/str(seed)/(arm+'.pkl')
            if path.is_file():continue
            pred18,state,audit=fit_arm(pack,data,arm,seed)
            if arm=='T0_W0_BCE':
                diff=float(np.max(np.abs(pred18-np.asarray(pack['old']['probability']))));audit['formal_W0_2018_max_probability_difference']=diff;audit['formal_selected_epochs']=int(pack['old']['selected_epochs'])
                if audit['selected_epochs']!=int(pack['old']['selected_epochs']) or diff>1e-6:raise RuntimeError('BLOCKED_BASELINE_REPLAY full T0 '+str(diff))
            future={2018:pred18}
            for year in (2019,2020,2021):
                target=np.flatnonzero(years==year);_,g=global24(pack['v4'],data,source,target);raw=raw_bags(data,target);bags=transform_saved(pack['v4'],raw,pack['old']['background'],pack['args']);graphs=pack['v4'].transform_role_graphs(data['s7'][target],pack['old']['graph_scaler']);future[year]=predict_with_state(pack,state,g,graphs,bags)
            atomic_pickle(path,{'arm':arm,'seed':seed,'split':'T1','state_dict':state,'audit':audit,'future_probability':future,'train_ids':list(ids[source]),'threshold':.5,'future_labels_used':False});print('FULL_ARM_DONE',seed,arm,flush=True)
        del pack
    models=sorted((OUT/'models/T1').glob('*/*.pkl'));decision=json.loads((OUT/'source_2017/source_2017_decision.json').read_text())
    atomic_json(OUT/'config/source_freeze_manifest.json',{'status':'FROZEN_BEFORE_FUTURE_LABEL_EVALUATION','source_decision':decision,'model_checkpoints':[digest(p) for p in models],'losses':{'T0':'BCE','T1':'focal gamma2 alpha .5/.5','T2':'BCE + .1 pairwise logistic'},'classification_threshold':.5,'router_threshold':.5,'training_years':[2017],'future_years':[2018,2019,2020,2021],'future_labels_used_for_training_or_selection':False},replace=True)

def evaluate_future():
    status('future_2018_2021_evaluation');freeze=json.loads((OUT/'config/source_freeze_manifest.json').read_text())
    if freeze.get('status')!='FROZEN_BEFORE_FUTURE_LABEL_EVALUATION':raise RuntimeError('source freeze absent')
    data=load_data();ids=np.asarray(data['ids']).astype(str);years=np.asarray(data['years'],int);labels=np.asarray(data['y'],int);allrows=[];summary=[]
    for year in YEARS[1:]:
        ix=np.flatnonzero(years==year);sids=list(ids[ix]);y=labels[ix];routes=route_rows(year);arm_probs={a:np.mean([read_pickle(OUT/'models/T1'/str(seed)/(a+'.pkl'))['future_probability'][year] for seed in MODEL_SEEDS],axis=0) for a in ARMS}
        for arm in ARMS:
            rr=[]
            for ms in MANIFEST_SEEDS:
                q=routes[routes.manifest_seed==ms].set_index('sha256').loc[sids];m,final=cascade(y,q.stage1_probability.to_numpy(float),arm_probs[arm],q.routed.to_numpy(bool));rr.append(m)
                for i,sid in enumerate(sids):allrows.append({'year':year,'arm':arm,'manifest_seed':ms,'sha256':sid,'label':int(y[i]),'routed':int(q.routed.iloc[i]),'stage1_probability':float(q.stage1_probability.iloc[i]),'stage2_probability':float(arm_probs[arm][i]),'final_probability':float(final[i])})
            keys=[k for k,v in rr[0].items() if isinstance(v,(int,float,np.integer,np.floating))];summary.append({'year':year,'arm':arm,**{k:float(np.nanmean([r[k] for r in rr])) for k in keys}})
        atomic_csv(OUT/f'future/{year}/predictions.csv',[r for r in allrows if r['year']==year],replace=True)
    avg={a:float(np.mean([r['BA'] for r in summary if r['arm']==a])) for a in ARMS};atomic_csv(OUT/'future/all_years_summary.csv',summary,replace=True)
    source=json.loads((OUT/'source_2017/source_2017_decision.json').read_text());final={'status':source['status'],'supported':source['supported'],'source_2017_primary':True,'future_development_only':True,'future_average_BA':avg,'future_used_for_selection':False,'classification_threshold':.5}
    atomic_json(OUT/'reports/final_decision.json',final,replace=True)
    src=pd.read_csv(OUT/'source_2017/arm_summary.csv');lines=['# CARE-Droid 第二层训练目标优化与跨年份验证','',f"2017 OOF 正式裁决：**{source['status']}**。",'', '所有分类阈值和路由阈值均固定为 0.5；2018—2021 仅作冻结诊断。输入、W0 架构、背景、RoleGraph、Stage-I 和 Router 均未修改。','', '| 年份 | Arm | BA | AUC | F1 | FPR | FNR | Fixed | Broken | Net |','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _,r in src.iterrows():lines.append(f"| 2017 | {r.arm} | {r.BA:.4f} | {r.AUC:.4f} | {r.F1:.4f} | {r.FPR:.4f} | {r.FNR:.4f} | {r.Fixed:.1f} | {r.Broken:.1f} | {r.Net_Fixed:.1f} |")
    for r in summary:lines.append(f"| {r['year']} | {r['arm']} | {r['BA']:.4f} | {r['AUC']:.4f} | {r['F1']:.4f} | {r['FPR']:.4f} | {r['FNR']:.4f} | {r['Fixed']:.1f} | {r['Broken']:.1f} | {r['Net_Fixed']:.1f} |")
    lines+=['','## 2018—2021 四年等权平均 BA','']+[f'- {a}: {v:.6f}' for a,v in avg.items()]+['','未来结果未参与损失选择、阈值选择、epoch 选择或模型重训。']
    text='\n'.join(lines)+'\n';(OUT/'reports/final_report.md').write_text(text,encoding='utf-8');(OUT/'final_report.md').write_text(text,encoding='utf-8');atomic_json(OUT/'final_decision.json',final,replace=True)

def integrity():
    status('final_integrity');before=json.loads((OUT/'integrity/asset_sha256_before.json').read_text())['formal_assets'];check=verify_manifest(before);check.pop('after',None)
    required=['config/experiment_config.json','config/baseline_manifest.json','config/source_freeze_manifest.json','integrity/replay_check.json','source_2017/oof_predictions.csv','source_2017/fold_seed_results.csv','source_2017/paired_comparison.csv','source_2017/bootstrap_intervals.csv','source_2017/source_2017_decision.json','future/all_years_summary.csv','reports/final_report.md','reports/final_decision.json']
    check['missing_outputs']=[x for x in required if not (OUT/x).is_file()];check['status']='PASS' if check['status']=='PASS' and not check['missing_outputs'] else 'FAIL';atomic_json(OUT/'integrity/asset_sha256_after.json',check,replace=True)
    if check['status']!='PASS':raise RuntimeError('final integrity failed')
    final=json.loads((OUT/'reports/final_decision.json').read_text());status('complete','COMPLETE',decision=final['status'],source=799,future=1989)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('preflight','baseline','smoke','train-source','evaluate-source','train-full','evaluate-future','integrity'));p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cuda');a=p.parse_args();os.environ['PYTHONDONTWRITEBYTECODE']='1'
    {'preflight':lambda:preflight(a.resume),'baseline':lambda:baseline_replay(a.device),'smoke':lambda:smoke(a.device),'train-source':lambda:train_source(a.device),'evaluate-source':evaluate_source,'train-full':lambda:train_full(a.device),'evaluate-future':evaluate_future,'integrity':integrity}[a.action]()
if __name__=='__main__':main()
