#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from common import auc

def endpoints(y,p,base,route):
    pred=p>=.5
    ba=.5*(float(np.mean(~pred[y==0]))+float(np.mean(pred[y==1])))
    left=route&(base<.5);right=route&(base>=.5)
    return {'BA':ba,'FN_vs_TN_AUC':auc(y[left],p[left]),'TP_vs_FP_AUC':auc(y[right],p[right])}

def strata_indices(y,folds,base,route):
    keys=np.column_stack([folds,y,route.astype(int),(base>=.5).astype(int)])
    return [np.flatnonzero(np.all(keys==key,axis=1)) for key in sorted(set(map(tuple,keys.tolist())))]

def paired_intervals(y,folds,base,route,scores,repeats,seed):
    comparisons=[(a,e) for a in ('M1_HIDDEN_READOUT','M3_CAPACITY_PLACEBO__r00','M4_CONDITIONAL_SHUFFLE__r00')
                 for e in ('BA','FN_vs_TN_AUC','TP_vs_FP_AUC')]+[('M0_DIRECT_C43','BA')]
    point={k:endpoints(y,v,base,route) for k,v in scores.items() if k in {a for a,e in comparisons}|{'M2_HIDDEN_CONTROL'}}
    samples={key:[] for key in comparisons};rng=np.random.default_rng(seed);strata=strata_indices(y,folds,base,route)
    for repeat in range(repeats):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in strata])
        bootstrap={k:endpoints(y[ix],scores[k][ix],base[ix],route[ix]) for k in point}
        for a,e in comparisons:
            if bootstrap['M2_HIDDEN_CONTROL'][e] is not None and bootstrap[a][e] is not None:samples[(a,e)].append(bootstrap['M2_HIDDEN_CONTROL'][e]-bootstrap[a][e])
        if repeats>=1000 and (repeat+1)%250==0:print(f'LCIP_PAIRED_BOOTSTRAP {repeat+1}/{repeats}',flush=True)
    result=[];alpha=.05/len(comparisons)
    for a,e in comparisons:
        values=np.asarray(samples[(a,e)]);n=len(values)
        real=point['M2_HIDDEN_CONTROL'][e];control=point[a][e]
        if n:
            lo,hi=np.quantile(values,[.025,.975]);adjlo,adjhi=np.quantile(values,[alpha/2,1-alpha/2])
            result.append(dict(comparator=a,endpoint=e,difference=real-control,paired_95_low=float(lo),paired_95_high=float(hi),
                multiplicity_adjusted_low=float(adjlo),multiplicity_adjusted_high=float(adjhi),family_size=len(comparisons),
                adjusted_interval_level=1-alpha,valid_bootstraps=n,bootstrap_unit='APK',inference_scope='conditional on fixed predictions/cohort; no upstream refitting or family clusters'))
        else:result.append(dict(comparator=a,endpoint=e,difference=None,paired_95_low=None,paired_95_high=None,multiplicity_adjusted_low=None,multiplicity_adjusted_high=None,family_size=len(comparisons),valid_bootstraps=0))
    return result
