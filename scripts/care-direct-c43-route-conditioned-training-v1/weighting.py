from __future__ import annotations
import hashlib
import numpy as np

def opposition(member):
    member=np.asarray(member,float)
    if member.ndim!=2 or member.shape[1]!=4 or not np.isfinite(member).all() or np.any((member<0)|(member>1)):
        raise ValueError('four finite calibrated member probabilities required')
    return np.where((member[:,3]>=.5)[:,None],1-member,member).mean(axis=1)

def top25(score,ids):
    score=np.asarray(score,float); ids=np.asarray(ids).astype(str)
    if len(score)!=len(ids) or len(set(ids))!=len(ids) or not np.isfinite(score).all(): raise ValueError('invalid score/SHA')
    mask=np.zeros(len(ids),bool); mask[np.lexsort((ids,-score))[:int(round(.25*len(ids)))]]=True
    return mask

def host_strata(counts):
    counts=np.asarray(counts,float)
    if not np.isfinite(counts).all() or np.any(counts<0): raise ValueError('invalid Host count')
    return np.searchsorted(np.quantile(np.log1p(counts),[1/3,2/3]),np.log1p(counts),side='right')

def make_weights(y,ids,member,counts,seed=20261017,groups=None):
    y=np.asarray(y,int); ids=np.asarray(ids).astype(str); member=np.asarray(member,float)
    if set(y.tolist())!={0,1} or len(y)!=len(ids) or member.shape!=(len(y),4) or len(counts)!=len(y): raise ValueError('weight shape or label contract')
    score=opposition(member)
    if groups is None: r=top25(score,ids)
    else:
        groups=np.asarray(groups)
        if groups.shape!=y.shape: raise ValueError('inner holdout group shape')
        r=np.zeros(len(ids),bool)
        for g in np.unique(groups):
            ix=np.flatnonzero(groups==g); r[ix]=top25(score[ix],ids[ix])
    raw=np.where(r,4.,1.); real=raw/raw.mean()
    prior=np.empty(len(y),float)
    for label in (0,1): prior[y==label]=real[y==label].mean()
    strata=host_strata(counts); side=member[:,3]>=.5; shuffled=real.copy(); report=[]
    for label in (0,1):
        for s in (False,True):
            for h in range(3):
                group=np.flatnonzero((y==label)&(side==s)&(strata==h))
                # Canonical SHA order makes the draw independent of manifest row order.
                group=group[np.argsort(ids[group],kind='stable')]
                salt=f'{seed}:{label}:{int(s)}:{h}'.encode(); local_seed=int.from_bytes(hashlib.sha256(salt).digest()[:4],'little')
                shuffled[group]=real[group][np.random.RandomState(local_seed).permutation(len(group))]
                if not np.array_equal(np.sort(shuffled[group]),np.sort(real[group])): raise RuntimeError('conditional permutation invariant')
                report.append(dict(label=label,predicted_side=int(s),host_stratum=h,n=len(group),routed=int(r[group].sum()),changed=int(np.sum(shuffled[group]!=real[group]))))
    result=dict(B0_ORIGINAL=np.ones(len(y)),B1_ROUTE_WEIGHTED=real,B2_CLASS_PRIOR=prior,B3_CONDITIONAL_SHUFFLE=shuffled)
    for label in (0,1):
        for arm in ('B2_CLASS_PRIOR','B3_CONDITIONAL_SHUFFLE'):
            if not np.isclose(result[arm][y==label].sum(),real[y==label].sum(),atol=1e-10): raise RuntimeError('class prior not matched')
    return result,r,dict(changed_fraction=float(np.mean(shuffled!=real)),groups=report,
        class_weight_mass={a:[float(w[y==c].sum()) for c in (0,1)] for a,w in result.items()},
        effective_sample_size={a:float(w.sum()**2/np.square(w).sum()) for a,w in result.items()})
