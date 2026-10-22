#!/usr/bin/env python3
"""Evaluate source OOF and frozen future semantic Adapters.

The 2017 decision and a source freeze manifest are written before future labels
are accessed. Future results are a one-shot temporal generalization diagnosis.
"""
from __future__ import annotations
import argparse,json,time
import numpy as np
from semantic_common import *
from semantic_model import make_model

def metric(y,p):
 y=np.asarray(y,int); p=np.asarray(p,float); z=p>=.5; tn=int(np.sum((y==0)&~z)); fp=int(np.sum((y==0)&z)); fn=int(np.sum((y==1)&~z)); tp=int(np.sum((y==1)&z)); tnr=tn/max(tn+fp,1); tpr=tp/max(tp+fn,1)
 from sklearn.metrics import roc_auc_score
 try: auc=float(roc_auc_score(y,p))
 except ValueError: auc=float('nan')
 pr=tp/max(tp+fp,1); return {'BA':.5*(tnr+tpr),'Accuracy':float(np.mean(z==y)),'AUC':auc,'F1':2*pr*tpr/max(pr+tpr,1e-12),'TN':tn,'FP':fp,'FN':fn,'TP':tp,'FPR':1-tnr,'FNR':1-tpr}

def cascade(y,s1,expert,route):
 final=np.where(route,expert,s1); bc=(s1>=.5)==y; fc=(final>=.5)==y; r=metric(y,final); fixed=int(np.sum(~bc&fc)); broken=int(np.sum(bc&~fc)); r.update({'Fixed':fixed,'Broken':broken,'Net_Fixed':fixed-broken,'routed':int(route.sum()),'route_rate':float(route.mean()),'stage1_error_coverage':float(np.sum(route&~bc)/max(np.sum(~bc),1))}); return r,final

def source_probabilities(data):
 source=np.flatnonzero(np.asarray(data['years'],int)==2017); pos={int(v):i for i,v in enumerate(source)}; values={a:np.full((len(SEEDS),len(source)),np.nan) for a in ARMS}
 for split in SOURCE_SPLITS:
  for si,seed in enumerate(SEEDS):
   base=read_pickle(SCBE/'models'/split/str(seed)/'M0_REPRODUCED.pkl'); ids=list(map(str,base['ids'])); idx={str(data['ids'][i]):int(i) for i in source}; loc=np.asarray([pos[idx[x]] for x in ids]); values['W0'][si,loc]=np.asarray(base['probability'])
   for arm in ADAPTER_ARMS:
    obj=read_pickle(OUT/'models'/split/str(seed)/(arm+'.pkl'))
    if obj['ids']!=ids: raise RuntimeError('source id mismatch')
    values[arm][si,loc]=np.asarray(obj['probability'])
 if any(not np.isfinite(x).all() for x in values.values()): raise RuntimeError('incomplete OOF')
 return source,values

def boot(y,cand,ref,repeats=1000,seed=20261021):
 rng=np.random.RandomState(seed); groups=[np.flatnonzero(y==q) for q in (0,1)]; vals=[]
 for _ in range(repeats):
  ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]); vals.append(np.mean([metric(y[ix],cand[s,ix])['BA']-metric(y[ix],ref[s,ix])['BA'] for s in range(cand.shape[0])]))
 vals=np.asarray(vals); return {'mean_BA_difference':float(vals.mean()),'ci95_low':float(np.quantile(vals,.025)),'ci95_high':float(np.quantile(vals,.975)),'one_sided_p':float((1+np.sum(vals<=0))/(1+len(vals))),'bootstrap_unit':'independent APK; seeds clustered'}

def source_eval():
 from bfa_core import load_data_contract
 data,_=load_data_contract(); source,probs=source_probabilities(data); y=np.asarray(data['y'][source],int); ids=np.asarray(data['ids'][source]).astype(str); routes=load_auto_routes(); rows=[]; metrics=[]; finals={a:[] for a in ARMS}
 for ms in MANIFEST_SEEDS:
  fr=routes[(routes.year==2017)&(routes.manifest_seed==ms)].set_index('sha256').loc[ids]; s1=fr.stage1_probability.to_numpy(float); route=fr.routed.to_numpy(bool)
  for arm in ARMS:
   expert=probs[arm].mean(0); metrics.append({'year':2017,'manifest_seed':ms,'arm':arm,'scope':'stage2',**metric(y,expert)}); ce,final=cascade(y,s1,expert,route); metrics.append({'year':2017,'manifest_seed':ms,'arm':arm,'scope':'end_to_end',**ce}); finals[arm].append(final)
   for i,sid in enumerate(ids): rows.append({'sha256':sid,'year':2017,'manifest_seed':ms,'label':int(y[i]),'stage1_probability':s1[i],'routed':int(route[i]),'arm':arm,'w0_probability':probs['W0'][:,i].mean(),'arm_probability':expert[i],'final_probability':final[i]})
 finals={k:np.asarray(v) for k,v in finals.items()}; tests=[('C1_S1','W0'),('C1_S1','C0_S1'),('C1_S1','C3_S1'),('C2_S3','W0'),('C2_S3','C0_S3'),('C2_S3','C4_S3')]; comps=[]
 for n,(a,b) in enumerate(tests): r=boot(y,finals[a],finals[b],seed=20261021+n); r.update({'candidate':a,'reference':b}); comps.append(r)
 order=sorted(range(len(comps)),key=lambda i:comps[i]['one_sided_p']); stopped=False
 for rank,i in enumerate(order):
  alpha=.05/(len(comps)-rank); ok=(not stopped) and comps[i]['one_sided_p']<=alpha; comps[i]['holm_alpha']=alpha; comps[i]['holm_reject_null']=ok; stopped=stopped or not ok
 def supported(a): return all(r['mean_BA_difference']>0 and r['ci95_low']>0 and r['holm_reject_null'] for r in comps if r['candidate']==a)
 s1,s3=supported('C1_S1'),supported('C2_S3'); status='S1_S3_SUPPORTED_SOURCE' if s1 and s3 else 'S1_SUPPORTED_SOURCE' if s1 else 'S3_SUPPORTED_SOURCE' if s3 else 'SEMANTIC_ADAPTER_NOT_SUPPORTED_2017_OOF'
 atomic_csv(OUT/'source_2017/fold_seed_results.csv',metrics); atomic_csv(OUT/'source_2017/oof_predictions.csv',rows); atomic_csv(OUT/'source_2017/paired_comparison.csv',comps); atomic_csv(OUT/'source_2017/bootstrap_intervals.csv',comps); atomic_json(OUT/'source_2017/source_2017_decision.json',{'status':status,'S1_supported':s1,'S3_supported':s3,'primary_metric':'2017 end-to-end OOF BA','comparisons':comps,'future_labels_read':False})
 paths=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+list((OUT/'models').glob('**/*.pkl')); atomic_json(OUT/'config/source_freeze_manifest.json',{'status':'FROZEN_BEFORE_FUTURE_LABEL_ACCESS','source_decision':status,'files':[digest(p) for p in sorted(paths)],'future_selection_forbidden':True}); print('SOURCE_DECISION',status,flush=True)

def future_eval(device):
 from bfa_core import load_data_contract
 from myd_common import runtime
 import importlib.util
 def loadmod(name,path):
  spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
 mytrain=loadmod('care_myd_train',ROOT/'scripts/care-scbe-v1/train.py'); stage2=loadmod('care_myd_stage2',ROOT/'scripts/care-multiyear-development-v1/train_stage2.py'); global24=mytrain.global24; raw_bags=stage2.raw_bags; transform_saved=stage2.transform_saved
 if not (OUT/'config/source_freeze_manifest.json').is_file(): raise RuntimeError('source freeze missing')
 data,_=load_data_contract(); v4,_,args=runtime(); args.device=device; args.keep_weight=args.fix_weight=1.; args.keep_consistency=0.; dev=v4.resolve_device(device); years=np.asarray(data['years'],int); train=np.flatnonzero(years==2017)
 fixed=np.load(ROOT/'outputs/care_multiyear_development_v1/predictions/C0R_2017/expert.npz',allow_pickle=False); fpos={str(x):i for i,x in enumerate(fixed['ids'])}; routes=load_auto_routes(); allmetrics=[]; runtime_rows=[]
 for year in YEARS[1:]:
  target=np.flatnonzero(years==year); ids=np.asarray(data['ids'][target]).astype(str); loc=np.asarray([fpos[x] for x in ids]); y=np.asarray(data['y'][target],int); probs={'W0':np.asarray(fixed['seed_probability'][:,loc],float)}; _,gt=global24(v4,data,train,target); raw=raw_bags(data,target)
  for si,seed in enumerate(SEEDS):
   base=read_pickle(SCBE/'models/T1'/str(seed)/'M0_REPRODUCED.pkl'); bags=transform_saved(v4,raw,base['background'],args); graph=v4.transform_role_graphs(data['s7'][target],base['graph_scaler'])
   for arm in ADAPTER_ARMS:
    obj=read_pickle(OUT/'models/T1'/str(seed)/(arm+'.pkl')); ab,audit=build_semantic_bags(bags,raw,ids,arm,seed); model=make_model(v4,args,base['state_dict']).to(dev); model.load_state_dict(obj['state_dict']); model.eval(); st=time.perf_counter(); p=v4.predict_model(model,gt,np.full(len(target),.5),graph,ab,dev)[0]; runtime_rows.append({'year':year,'arm':arm,'seed':seed,'adapter_inference_seconds':time.perf_counter()-st,'apks':len(target),'hosts':audit['hosts']}); probs.setdefault(arm,[]).append(np.asarray(p)); del model
  for arm in ADAPTER_ARMS: probs[arm]=np.asarray(probs[arm])
  pred=[]; mets=[]
  for ms in MANIFEST_SEEDS:
   rr=routes[(routes.year==year)&(routes.manifest_seed==ms)].set_index('sha256').loc[ids]; s1=rr.stage1_probability.to_numpy(float); mask=rr.routed.to_numpy(bool)
   for arm in ARMS:
    expert=probs[arm].mean(0); mets.append({'year':year,'manifest_seed':ms,'arm':arm,'scope':'stage2',**metric(y,expert)}); ce,final=cascade(y,s1,expert,mask); mets.append({'year':year,'manifest_seed':ms,'arm':arm,'scope':'end_to_end',**ce})
    for i,sid in enumerate(ids): pred.append({'sha256':sid,'year':year,'manifest_seed':ms,'label':int(y[i]),'stage1_probability':s1[i],'routed':int(mask[i]),'arm':arm,'w0_probability':probs['W0'][:,i].mean(),'arm_probability':expert[i],'final_probability':final[i]})
  atomic_csv(OUT/f'future/{year}/predictions.csv',pred); atomic_json(OUT/f'future/{year}/metrics.json',mets); allmetrics.extend(mets)
 atomic_csv(OUT/'summary/all_years_comparison.csv',allmetrics); atomic_csv(OUT/'summary/runtime_profile.csv',runtime_rows); decision=json.loads((OUT/'source_2017/source_2017_decision.json').read_text()); atomic_json(OUT/'summary/final_decision.json',{'status':decision['status'],'future_role':'frozen temporal diagnosis only','future_years':[2018,2019,2020,2021],'future_used_for_selection':False}); (OUT/'summary/final_report.md').write_text('# CARE-Droid S1/S3 Semantic Adapter\n\nSource decision: **'+decision['status']+'**.\n\nFuture years were evaluated once after source freeze.\n',encoding='utf-8'); print('FUTURE_EVALUATION_DONE',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--source',action='store_true'); ap.add_argument('--future',action='store_true'); ap.add_argument('--device',default='cuda'); a=ap.parse_args()
 if a.source==a.future: raise SystemExit('choose one mode')
 source_eval() if a.source else future_eval(a.device)



