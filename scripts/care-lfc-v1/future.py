#!/usr/bin/env python3
"""LFC-V1 source-2017-only RF freeze and historical 2018–2021 diagnosis. Inputs: LiteStage shards, formal P31 and frozen HDBC C43; outputs/checkpoints only LFC_OUT. No APK reads or C43 training."""
from __future__ import annotations
import pickle,sys,time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from core import *
sys.path.insert(0,str(ROOT/'scripts/care-p31-stage1-rf-selection-source-v1'))
from stage1_core import make_member,raw_score

SEED=20261018+999
def fit_frozen(x,y):
    splits=list(StratifiedKFold(3,shuffle=True,random_state=SEED).split(x,y))
    oof=np.full(len(y),np.nan)
    for k,(it,iv) in enumerate(splits,1):
        model=make_member('RF',SEED+100+k,2);model.fit(x[it],y[it]);oof[iv]=raw_score(model,x[iv])
    if not np.isfinite(oof).all():raise RuntimeError('inner calibration incomplete')
    cal=LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000,random_state=SEED+999)
    cal.fit(oof.reshape(-1,1),y)
    final=make_member('RF',SEED,2);final.fit(x,y)
    return final,cal
def infer(bundle,x):
    model,cal=bundle
    return np.clip(cal.predict_proba(raw_score(model,x).reshape(-1,1))[:,1],1e-8,1-1e-8)
def main():
    ids,y,_,_,_=source_index();source=shard('source');future=shard('future');fsids=np.array(sorted(future))
    if set(ids)!=set(source):raise RuntimeError('source shard mismatch')
    e=lc.frozen_expert_future()
    if set(fsids)!=set(e):raise RuntimeError('future C43 coverage')
    years=np.array([int(e[s]['year']) for s in fsids]);labels=np.array([int(e[s]['label']) for s in fsids])
    ep=np.array([float(e[s]['expert_probability']) for s in fsids])
    if {a:int((years==a).sum()) for a in (2018,2019,2020,2021)}!={2018:793,2019:399,2020:397,2021:400}:raise RuntimeError('future year counts')
    previous={}
    for r in rows(LITE/'future_frozen_predictions.csv'):
        if r['method']=='LITE_SELECTED' and float(r['budget'])==.25:previous[r['sha256']]=float(r['stage1_probability'])
    if set(previous)!=set(fsids):raise RuntimeError('BestLiteStage future coverage')
    formal={r['sha256']:r for r in rows(ROOT/'artifacts/care-droid-formal-diagnostics-v1/references/future_label_free.csv')}
    if set(formal)!=set(fsids):raise RuntimeError('formal P31 future coverage')
    prob={'BestLiteStage':np.array([previous[s] for s in fsids]),'OriginalP31RF':np.array([float(formal[s]['p_P31__RF']) for s in fsids])}
    train_audit=[]
    for arm in ('ManifestOnly',)+NEW:
        select=role_select([source[s] for s in ids],y) if arm=='RefTop16' else None
        x=feature_matrix(source,ids,arm,select);z=feature_matrix(future,fsids,arm,select)
        path=OUT/'checkpoints'/f'{arm}.pkl';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            with path.open('rb') as f:obj=pickle.load(f)
            if obj['arm']!=arm or obj['train_years']!=[2017] or obj['train_sha']!=list(ids) or obj['seed']!=SEED:raise RuntimeError('existing checkpoint contract differs '+str(path))
            bundle=(obj['model'],obj['calibrator']);fit_s=float(obj['fit_s'])
        else:
            start=time.perf_counter();bundle=fit_frozen(x,y);fit_s=time.perf_counter()-start
            with path.open('xb') as f:pickle.dump(dict(arm=arm,model=bundle[0],calibrator=bundle[1],train_years=[2017],train_sha=list(ids),roles=select,seed=SEED,fit_s=fit_s),f,pickle.HIGHEST_PROTOCOL)
        infer_start=time.perf_counter();prob[arm]=infer(bundle,z);infer_s=time.perf_counter()-infer_start
        train_audit.append(dict(candidate=arm,train_n=len(ids),width=x.shape[1],seed=SEED,fit_s=fit_s,infer_total_s=infer_s,infer_per_APK_s=infer_s/len(fsids),checkpoint=str(path),checkpoint_sha256=sha(path)['sha256'],roles=','.join(map(str,select)) if select is not None else 'all'))
        print('LFC_FINAL2017_FROZEN',arm,flush=True)
    put_csv('final2017_training_audit.csv',train_audit)
    output=[];per=[]
    for yr in (2018,2019,2020,2021):
        ix=np.flatnonzero(years==yr);yrids=fsids[ix]
        for arm in list(MAIN)+list(EXTRA):
            q=prob[arm][ix];route=lc.route_by_score(yrids,1-2*np.abs(q-.5),.25)
            m,fin=lc.cascade(labels[ix],q,ep[ix],route)
            st=lc.metrics(labels[ix],q)
            output.append(dict(year=yr,candidate=arm,scope='Historical Development Diagnostics — Not Independent Blind Testing',stage1_BA=st['BA'],stage1_AUC=st['AUC'],stage1_FN=st['FN'],stage1_FP=st['FP'],stage1_errors=st['errors'],final_BA=m['BA'],final_AUC=m['AUC'],final_FN=m['FN'],final_FP=m['FP'],final_errors=m['errors'],routed=m['routed'],router_error_recall=m['router_error_recall'],fixed=m['fixed'],broken=m['broken'],net=m['net'],feature_success=len(ix),feature_failed=0,average_total_analysis_cost='UNVERIFIED'))
            for j,i in enumerate(ix):per.append(dict(year=yr,sha256=fsids[i],label=int(labels[i]),candidate=arm,stage1_probability=float(q[j]),C43_probability=float(ep[i]),routed=int(route[j]),final_probability=float(fin[j])))
    put_csv('future_frozen_metrics.csv',output);put_csv('future_frozen_predictions.csv',per)
    print('LFC_FUTURE_DONE 2018-2021 1989 APK',flush=True)
if __name__=='__main__':main()
