#!/usr/bin/env python3
"""Construct APK-cross-fitted SCBE features; CARE-Droid adapter writes only new experiment outputs, never opens APKs."""
from __future__ import annotations

import hashlib
import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

LAM=0.25
CLIP=2.0
RIDGE_ALPHA=10.0
HOSTS_PER_APK=16
INNER_FOLDS=3

def _hash_seed(seed,value):
    b=hashlib.sha256((str(seed)+':'+str(value)).encode()).digest()
    return int.from_bytes(b[:4],'little')

def responses(raw_bags,prep,v4):
    import haig_background as hb
    core,role,_=hb.split(raw_bags)
    state=prep['state'];sp=prep['sp'];bp=prep['bp'];v2=v4.v3.v2
    out=[]
    for x,w in zip(core,role):
        z=((x-prep['mean'])/prep['std']).astype(np.float32)
        sd=v2.pairwise_rms(z,state['shared_centers'])
        bd=v2.pairwise_rms(z,state['benign_centers'])
        if sd.shape!=(len(z),32) or bd.shape!=(len(z),24):raise RuntimeError('Bank shape changed')
        # 32/24 similarities are diagnostic derivations, not old network outputs.
        shared=np.exp(-sd/np.maximum(state['shared_scale'][None,:],1e-6))
        external=np.exp(-bd/max(float(state['external_scale']),1e-6))
        assigned=hb.hard(bd,w,bp)
        out.append({'x':np.concatenate([shared,w],axis=1).astype(np.float32),
                    'e':external.astype(np.float32),'assigned':assigned})
    return out

def _fit_predictor(bags,indices,seed,placebo=False):
    xs=[];ys=[];weights=[]
    for i in indices:
        x=bags[i]['x'];e=bags[i]['e'];count=min(HOSTS_PER_APK,len(x))
        if count==0:raise RuntimeError('empty bag violates formal Host contract')
        rng=np.random.RandomState(_hash_seed(seed,i));picked=rng.choice(len(x),count,replace=False)
        target=e[picked]
        if placebo and count>1:
            target=target[rng.permutation(count)]
        xs.append(x[picked]);ys.append(target)
        weights.extend([1.0/count]*count)
    xx=np.concatenate(xs);yy=np.concatenate(ys)
    model=Ridge(alpha=RIDGE_ALPHA,fit_intercept=True)
    model.fit(xx,yy,sample_weight=np.asarray(weights))
    return model

def _max_assigned(rows,signal):
    a=np.zeros((len(rows),24),np.float32)
    for i,(row,s) in enumerate(zip(rows,signal)):
        for j in np.unique(row['assigned']):
            a[i,int(j)]=float(np.maximum(s[row['assigned']==j,j],0).max())
    return a

def _class_weight(activation,labels,calibration_indices):
    idx=np.asarray(calibration_indices,int);y=np.asarray(labels,int)[idx]
    if len(np.unique(y))!=2:raise RuntimeError('both classes needed for support calibration')
    thr=np.quantile(activation[idx],.75,axis=0)
    active=(activation[idx]>thr[None,:])&(activation[idx]>0)
    b=y==0;m=y==1
    # APK unit, Beta(2,2) smoothing and fixed low-support shrinkage.
    cb=active[b].sum(0);cm=active[m].sum(0)
    qb=(cb+2.)/(b.sum()+4.);qm=(cm+2.)/(m.sum()+4.)
    raw=np.log(qb/qm)
    w=np.clip(raw*(cb+cm)/(cb+cm+10.),-1.5,1.5)
    return w.astype(np.float32),thr.astype(np.float32),cb,cm

def _role_shuffle(weight,profile,seed):
    out=weight.copy();role=np.argmax(profile,axis=1);changed=0;singleton=0
    for r in np.unique(role):
        idx=np.flatnonzero(role==r)
        if len(idx)<2:singleton+=len(idx);continue
        rng=np.random.RandomState(_hash_seed(seed,'role'+str(r)))
        perm=rng.permutation(len(idx))
        if np.array_equal(perm,np.arange(len(idx))):perm=np.roll(perm,1)
        out[idx]=weight[idx[perm]];changed+=len(idx)
    return out,changed,singleton

def _transform(original,diag,residual,raw_score,weight,scale,kind):
    out=[];strength=[];direction=[]
    for bag,row,res,raw,w in zip(original,diag,residual,raw_score,weight):
        b=np.array(bag,copy=True);assignment=row['assigned'];ix=np.arange(len(assignment))
        if kind=='M1_CLASS_SUPPORT':
            # Original External similarity, centered/scaled using train-only moments.
            signal=raw[ix,assignment]
        else:
            signal=np.clip(res[ix,assignment]/scale[assignment],-CLIP,CLIP)
        if kind=='M2_CONDITION_ONLY':influence=signal
        else:influence=signal*np.tanh(w[assignment])
        ext=np.maximum(0,b[:,107]-LAM*influence)
        if not np.isfinite(ext).all():raise RuntimeError('nonfinite transformed External')
        b[:,107]=ext
        # Internal (108) and Commonness (109) unchanged. Residual is formulaic.
        b[:,110]=(1-np.clip(b[:,109],0,1))*np.sqrt(np.maximum(ext,0)*np.maximum(b[:,108],0))
        if not np.array_equal(b[:,108:110],bag[:,108:110]):raise RuntimeError('Internal/Commonness changed')
        out.append(b.astype(np.float32));strength.append(float(np.mean(np.abs(signal))))
        direction.append(float(np.mean(influence)))
    return out,np.asarray(strength),np.asarray(direction)

def make_arms(v4,raw_train,raw_test,formal_train,formal_test,labels,prep,seed):
    start=time.perf_counter();train=responses(raw_train,prep,v4);test=responses(raw_test,prep,v4)
    n=len(train);y=np.asarray(labels,int)
    if n!=len(y):raise RuntimeError('training labels and bags differ')
    folds=list(KFold(n_splits=INNER_FOLDS,shuffle=True,random_state=seed%100000).split(np.arange(n)))
    oof=[None]*n;oof_null=[None]*n;fold_for=np.empty(n,int)
    for k,(fit,hold) in enumerate(folds):
        model=_fit_predictor(train,fit,seed+k)
        placebo=_fit_predictor(train,fit,seed+k,placebo=True)
        for i in hold:
            oof[i]=train[i]['e']-model.predict(train[i]['x'])
            oof_null[i]=train[i]['e']-placebo.predict(train[i]['x'])
            fold_for[i]=k
    full=_fit_predictor(train,np.arange(n),seed+13000)
    full_null=_fit_predictor(train,np.arange(n),seed+13000,placebo=True)
    test_res=[r['e']-full.predict(r['x']) for r in test]
    test_null=[r['e']-full_null.predict(r['x']) for r in test]
    # Scaling is label-free and only train OOF based; one center cannot dominate.
    scale=np.maximum(np.std(np.concatenate(oof),axis=0),.1).astype(np.float32)
    raw_mean=np.mean(np.concatenate([r['e'] for r in train]),axis=0)
    raw_scale=np.maximum(np.std(np.concatenate([r['e'] for r in train]),axis=0),.1)
    raw_train=[(r['e']-raw_mean)/raw_scale for r in train]
    raw_test=[(r['e']-raw_mean)/raw_scale for r in test]
    activation=_max_assigned(train,oof)
    raw_activation=_max_assigned(train,raw_train)
    w_test,thr,cb,cm=_class_weight(activation,y,np.arange(n))
    w1_test,_,_,_=_class_weight(raw_activation,y,np.arange(n))
    w_train=np.empty((n,24),np.float32);w1_train=np.empty((n,24),np.float32)
    for k,(fit,hold) in enumerate(folds):
        wk,_,_,_=_class_weight(activation,y,fit)
        w1k,_,_,_=_class_weight(raw_activation,y,fit)
        w_train[hold]=wk;w1_train[hold]=w1k
        if set(hold)&set(fit):raise RuntimeError('own APK enters own weight')
    w4_test,shuffled,singleton=_role_shuffle(w_test,prep['bp'],seed)
    w4_train=np.empty_like(w_train)
    for k,(_,hold) in enumerate(folds):
        shuffled_w,_,_=_role_shuffle(w_train[hold[0]],prep['bp'],seed+k)
        w4_train[hold]=shuffled_w
    modes=['M1_CLASS_SUPPORT','M2_CONDITION_ONLY','M3_SCBE','M4_ROLE_SHUFFLE','M5_CAPACITY_PLACEBO']
    arms={}
    for mode in modes:
        if mode=='M1_CLASS_SUPPORT':rt,rv=raw_train,raw_test;wt,wv=w1_train,w1_test
        elif mode=='M4_ROLE_SHUFFLE':rt,rv=oof,test_res;wt,wv=w4_train,w4_test
        elif mode=='M5_CAPACITY_PLACEBO':rt,rv=oof_null,test_null;wt,wv=w_train,w_test
        else:rt,rv=oof,test_res;wt,wv=w_train,w_test
        if mode=='M2_CONDITION_ONLY':wt=np.ones((n,24),np.float32);wv=np.ones(24,np.float32)
        tr,ts,td=_transform(formal_train,train,rt,raw_train,wt,scale,mode)
        te,vs,vd=_transform(formal_test,test,rv,raw_test,[wv]*len(test),scale,mode)
        arms[mode]={'train':tr,'test':te,'train_strength':ts,'test_strength':vs,
                    'train_direction':td,'test_direction':vd}
    all_true=np.concatenate([r['e'] for r in test]);all_pred=np.concatenate([full.predict(r['x']) for r in test])
    denom=float(np.sum((all_true-all_true.mean(axis=0))**2))
    r2=1-float(np.sum((all_true-all_pred)**2))/max(denom,1e-12)
    apk_res=np.asarray([np.mean(np.abs(r)) for r in test_res])
    apk_old=np.asarray([np.mean(b[:,110]) for b in formal_test])
    old_corr=float(np.corrcoef(apk_res,apk_old)[0,1]) if len(test)>2 and np.std(apk_res)>0 and np.std(apk_old)>0 else None
    metrics={'test_conditional_r2':r2,'train_apks':n,'train_host_count':sum(len(r['e']) for r in train),
        'test_host_count':sum(len(r['e']) for r in test),'fit_seconds':time.perf_counter()-start,
        'ridge_alpha':RIDGE_ALPHA,'lambda':LAM,'clip':CLIP,'inner_folds':INNER_FOLDS,
        'positive_support_benign_total':cb.tolist(),'positive_support_malware_total':cm.tolist(),
        'full_weight':w_test.tolist(),'weight_std':float(np.std(w_test)),
        'per_APK_abs_conditional_residual_vs_old_residual_Pearson':old_corr,
        'role_shuffle_changed_prototypes':shuffled,'role_shuffle_singletons':singleton,
        'fixed_bank_self_source_bias_remains':True,'own_label_enters_own_OOF_weight':False}
    return arms,metrics
