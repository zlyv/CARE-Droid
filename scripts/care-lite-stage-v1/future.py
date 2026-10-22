#!/usr/bin/env python3
"""LiteStage-V1 2017-full frozen RF committee and 2018–2021 historical diagnostics. Inputs: source/future new shards and frozen C43/P31 predictions. Writes only LiteStage outputs/checkpoint; trains on 2017 only; reads no APK."""
from __future__ import annotations
import json,pickle,sys,time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from common import *

sys.path.insert(0,str(ROOT/'scripts/care-p31-stage1-rf-selection-source-v1'))
from stage1_core import make_member,raw_score,MEMBERS,mean_opposition

SEED=20261017+999
def train_bundle(x,y):
    split=list(StratifiedKFold(3,shuffle=True,random_state=SEED).split(x,y));bundle=[]
    for j,name in enumerate(MEMBERS):
        oof=np.full(len(y),np.nan)
        for k,(fit,cal) in enumerate(split,1):
            model=make_member(name,SEED+10000*j+100*k,2);model.fit(x[fit],y[fit]);oof[cal]=raw_score(model,x[cal])
        if not np.isfinite(oof).all():raise RuntimeError('train calibration incomplete')
        calibrator=LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000,random_state=SEED+10000*j+999)
        calibrator.fit(oof.reshape(-1,1),y)
        model=make_member(name,SEED+10000*j,2);model.fit(x,y);bundle.append((name,model,calibrator))
    return bundle
def infer(bundle,x):
    out=np.zeros((len(x),4),float)
    for j,(_,m,c) in enumerate(bundle):out[:,j]=np.clip(c.predict_proba(raw_score(m,x).reshape(-1,1))[:,1],1e-8,1-1e-8)
    return out

def main():
    selection=json.loads((OUT/'source_selection.json').read_text());arm=selection['global_final_arm']
    lineage,original=source_contract();source,miss=load_shards('source')
    if miss:raise RuntimeError('source missing for frozen model '+str(len(miss)))
    future,missing=load_shards('future')
    rows=readcsv(FUTURE_LINEAGE);year_by={r['sha256'].lower():int(r['year']) for r in rows}
    cover=[dict(year=y,total=sum(v==y for v in year_by.values()),available=sum(year_by[s]==y for s in future),failed=sum(year_by[s]==y for s in missing)) for y in (2018,2019,2020,2021)]
    writecsv('future_feature_coverage.csv',cover)
    if missing:
        writejson('future_blocker.json',dict(status='INCOMPLETE_FUTURE_EXTRATION',missing=len(missing),sample=missing[:20],policy='do not silently exclude hard APKs from historical cohort'))
        print('LITESTAGE_FUTURE_BLOCKED_MISSING',len(missing),flush=True);return
    sids=sorted(source);fsids=sorted(future);y=np.array([int(original[s]['label']) for s in sids]);role=choose_roles([source[s] for s in sids],y,arm)
    x=matrix([source[s] for s in sids],arm,role);z=matrix([future[s] for s in fsids],arm,role)
    t=time.perf_counter();bundle=train_bundle(x,y);prob=infer(bundle,z);training_seconds=time.perf_counter()-t
    ck=OUT/'frozen_2017_lite_committee.pkl'
    with ck.open('xb') as f:pickle.dump(dict(arm=arm,seed=SEED,roles=role,train_sha=sids,bundle=bundle,fit_seconds=training_seconds),f,pickle.HIGHEST_PROTOCOL)
    old=readcsv(ROOT/'artifacts/care-droid-formal-diagnostics-v1/references/future_label_free.csv');legacy={r['sha256']:r for r in old}
    expert=frozen_expert_future()
    if set(fsids)!=set(legacy) or set(fsids)!=set(expert):raise RuntimeError('future SHA alignment')
    years=np.array([year_by[s] for s in fsids]);labels=np.array([int(expert[s]['label']) for s in fsids]);lite=prob[:,1];score=mean_opposition(prob,lite)
    rf=np.array([float(legacy[s]['p_P31__RF']) for s in fsids]);members=np.array([[float(legacy[s][f'p_P31__{m}']) for m in MEMBERS] for s in fsids]);rfscore=mean_opposition(members,rf);ep=np.array([float(expert[s]['expert_probability']) for s in fsids]);output=[];per=[]
    for year in (2018,2019,2020,2021):
        ix=np.flatnonzero(years==year);sy=np.array(fsids)[ix]
        for budget in (.1,.25,.4):
            lr=route_by_score(sy,score[ix],budget);rr=route_by_score(sy,rfscore[ix],budget)
            for name,p,route in [('LITE_SELECTED',lite[ix],lr),('L5_P31_RF',rf[ix],rr)]:
                m,final=cascade(labels[ix],p,ep[ix],route)
                output.append(dict(year=year,method=name,budget=budget,scope='Historical Development Diagnostics — Not Independent Blind Testing',**m))
                if budget==.25:
                    for j,i in enumerate(ix):per.append(dict(year=year,sha256=fsids[i],label=int(labels[i]),method=name,budget=budget,stage1_probability=float(p[j]),routed=int(route[j]),expert_probability=float(ep[i]),final_probability=float(final[j]),feature_extracted_level=future[fsids[i]].get('extracted_level','L4_PRE_SMOKE')))
    writecsv('future_frozen_metrics.csv',output);writecsv('future_frozen_predictions.csv',per)
    writejson('future_training_audit.json',dict(train_year=2017,train_n=len(sids),future_train_labels_used=False,final_arm=arm,train_roles=role.tolist() if role is not None else 'all',committee_members=list(MEMBERS),calibration='inner 3fold source only',training_seconds=training_seconds,checkpoint=str(ck)))
    print('LITESTAGE_FUTURE_DONE',arm,flush=True)
if __name__=='__main__':main()
