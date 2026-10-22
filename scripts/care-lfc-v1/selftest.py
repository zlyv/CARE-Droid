#!/usr/bin/env python3
"""LFC-V1 smoke tests. Reads frozen LiteStage shards and source/C43 contracts; writes only LFC_OUT/smoke_test.json. Trains tiny and small-batch real-feature RFs, no APK reads or C43 training."""
from __future__ import annotations
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from core import *
sys.path.insert(0,str(ROOT/'scripts/care-p31-stage1-rf-selection-source-v1'))
from stage1_core import fit_calibrated_member

def main():
    ids,y,fold,pred,expert=source_index();data=shard('source');v=data[ids[0]]
    if len(ids)!=799 or list(np.bincount(fold)[1:])!=[160,160,160,160,159]:raise RuntimeError('source folds')
    if v['dex_meta'][4]<=0 or int(sum(v['role_ref_count']))<0:raise RuntimeError('smoke shard')
    widths={a:len(feature(v,a)) for a in ('ManifestOnly','RefPresence31','RefCount31','InvokeCount31')}
    if widths!={'ManifestOnly':135,'RefPresence31':166,'RefCount31':198,'InvokeCount31':167}:raise RuntimeError('feature widths '+repr(widths))
    pick=np.r_[np.flatnonzero(y==0)[:20],np.flatnonzero(y==1)[:20]]
    select=role_select([data[ids[i]] for i in pick],y[pick])
    if len(select)!=16 or len(feature(v,'RefTop16',select))!=151:raise RuntimeError('TopK')
    m=RandomForestClassifier(n_estimators=5,random_state=1).fit(np.array([[0.],[1.],[2.],[3.]]),[0,0,1,1])
    if m.predict_proba([[1.5]]).shape!=(1,2):raise RuntimeError('RF forward')
    tinytrain=np.r_[np.flatnonzero(y==0)[:18],np.flatnonzero(y==1)[:18]]
    tinytest=np.r_[np.flatnonzero(y==0)[18:22],np.flatnonzero(y==1)[18:22]]
    x=feature_matrix(data,ids[tinytrain],'RefPresence31');z=feature_matrix(data,ids[tinytest],'RefPresence31')
    sp=fit_calibrated_member('RF',x,y[tinytrain],z,20261018,inner_folds=3,jobs=2)
    if sp.shape!=(8,) or not np.isfinite(sp).all():raise RuntimeError('real-feature RF forward')
    se=np.array([float(expert[s]['expert_probability']) for s in ids[tinytest]])
    rr=route(ids[tinytest],sp,.25);sm,_=lc.cascade(y[tinytest],sp,se,rr)
    if sm['routed']!=2:raise RuntimeError('real-feature C43 pairing')
    rt=route(ids[:40],np.linspace(0,1,40),.25)
    if rt.sum()!=10:raise RuntimeError('route budget')
    e=np.array([float(expert[s]['expert_probability']) for s in ids[:40]])
    b=np.array([float(pred[s]['p_P31__RF']) for s in ids[:40]])
    mm,_=lc.cascade(y[:40],b,e,rt)
    if mm['routed']!=10 or mm['stage1_errors']-mm['fixed']+mm['broken']!=mm['errors']:raise RuntimeError('cascade identity')
    put_json('smoke_test.json',dict(status='PASS',source=799,source_fold_sizes=[160,160,160,160,159],feature_widths=widths,route_25pct=10,C43_same_fold=True,APK_read=False,training='tiny RF only',unverified_end_to_end_cost=True))
    print('LFC_SMOKE_PASS',widths,flush=True)
if __name__=='__main__':main()
