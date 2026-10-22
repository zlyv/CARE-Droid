"""Original P31 calibrated classifiers with explicit calibration folds and reloadable model checkpoints."""
from __future__ import annotations
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from myd_common import *
from protocol import split_window

SEED=20260904

def probability(model,x):
    core=family('stage1_core');raw=core.raw_score(model['raw'],np.asarray(x,float)).reshape(-1,1)
    return np.clip(model['calibrator'].predict_proba(raw)[:,1],1e-8,1-1e-8)

def fit_member(name,x,y,ids,seed=SEED,jobs=4):
    core=family('stage1_core');x=np.asarray(x,float);y=np.asarray(y,int);mi=MEMBERS.index(name)
    oof=np.full(len(y),np.nan);splits=[]
    for k,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(x,y),1):
        if set(ids[tr])&set(ids[va]):raise RuntimeError('calibration SHA leakage')
        m=core.make_member(name,seed+10000*mi+100*k,jobs);m.fit(x[tr],y[tr]);oof[va]=core.raw_score(m,x[va])
        splits.append(dict(fit_sha=ids[tr].tolist(),hold_sha=ids[va].tolist()))
    if not np.isfinite(oof).all():raise RuntimeError('calibration OOF incomplete')
    c=LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000,random_state=seed+10000*mi+999);c.fit(oof.reshape(-1,1),y)
    m=core.make_member(name,seed+10000*mi,jobs);m.fit(x,y)
    return dict(raw=m,calibrator=c,calibration_splits=splits,train_sha=ids.tolist(),seed=seed,member=name)

def fitted(data,tr,tag,name,jobs=4):
    meta=dict(dataset=data['signature'],code=implementation_hash(),tag=tag,member=name,train_sha=data['ids'][tr].tolist(),seed=SEED,definition='original classifier family 3-fold OOF sigmoid calibration')
    path=OUTPUT/'models/stage1'/tag/(name+'.pkl');old=load_checkpoint(path,meta)
    if old:return old['model']
    start=time.perf_counter();m=fit_member(name,data['p31'][tr],data['y'][tr],data['ids'][tr],jobs=jobs)
    save_pickle(path,dict(signature=signature(meta),metadata=meta,model=m,fit_seconds=time.perf_counter()-start));dump(path.with_suffix('.json'),meta)
    print('MYD_STAGE1_FIT',tag,name,len(tr),flush=True);return m

def run(data,window,jobs=4,committee=False):
    records=read_csv(OUTPUT/'sample_manifest.csv');records=[dict(r,year=int(r['year']),label=int(r['label'])) for r in records]
    tr,va=split_window(records,window);src=np.flatnonzero(data['years']==2017)
    out={};timings=[]
    for tag,index in [('S0R_2017',src),('S1_'+window,tr)]:
        if window=='T1' and tag.startswith('S1'):models=out['S0R_2017']['models']
        else:models={m:fitted(data,index,tag,m,jobs) for m in (MEMBERS if committee else ['SVM'])}
        start=time.perf_counter();member=np.column_stack([probability(models[m],data['p31'][va]) for m in (MEMBERS if committee else ['SVM'])]);elapsed=time.perf_counter()-start
        out[tag]=dict(models=models,member=member);timings.append(dict(window=window,tag=tag,module='committee' if committee else 'SVM',inference_seconds=elapsed,validation_n=len(va),static_extraction_seconds=0))
    data_out=dict(ids=data['ids'][va],val_index=va,train_index=tr,s0=out['S0R_2017']['member'][:,-1],s1=out['S1_'+window]['member'][:,-1])
    if committee:data_out.update(old_member=out['S0R_2017']['member'],new_member=out['S1_'+window]['member'])
    save_npz(OUTPUT/'predictions'/window/('committee.npz' if committee else 'stage1.npz'),**data_out)
    dump(OUTPUT/'predictions'/window/('committee_timing.json' if committee else 'stage1_timing.json'),timings)
    dump(OUTPUT/'predictions'/window/'split.json',dict(train_sha=data['ids'][tr].tolist(),validation_sha=data['ids'][va].tolist(),train_years=sorted(set(data['years'][tr].tolist())),validation_year=int(data['years'][va][0])))
    return data_out
