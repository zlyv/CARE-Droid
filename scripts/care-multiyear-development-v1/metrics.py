"""APK-level cascade, routing, conditional AUC and paired uncertainty calculations."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
def auc(y,p):return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan')
def classification(y,p):
    y=np.asarray(y,int);pr=np.asarray(p)>=.5;tn=int(((y==0)&~pr).sum());fp=int(((y==0)&pr).sum());fn=int(((y==1)&~pr).sum());tp=int(((y==1)&pr).sum())
    return dict(n=len(y),BA=.5*(tn/max(tn+fp,1)+tp/max(tp+fn,1)),AUC=auc(y,p),Accuracy=float(np.mean(pr==y)),F1=2*tp/max(2*tp+fp+fn,1),Precision=tp/max(tp+fp,1),Recall=tp/max(tp+fn,1),Specificity=tn/max(tn+fp,1),TP=tp,TN=tn,FP=fp,FN=fn)
def routing(y,base,route):
    error=(base>=.5)!=y;n=int(route.sum());caught=int((error&route).sum());rate=float(error.mean());precision=caught/max(n,1)
    return dict(routed=n,routed_errors=caught,routed_correct=n-caught,stage1_errors=int(error.sum()),unrouted_errors=int((error&~route).sum()),ErrorRecall=caught/max(int(error.sum()),1),RoutePrecision=precision,RoutingLift=precision/rate if rate>0 else float('nan'))
def cascade(y,base,expert,route):
    y=np.asarray(y);base=np.asarray(base);expert=np.asarray(expert);route=np.asarray(route,bool);final=np.where(route,expert,base)
    before=(base>=.5)==y;after=(final>=.5)==y;fixed=~before&after;broken=before&~after;neg=route&(base<.5);pos=route&(base>=.5)
    return dict(**classification(y,final),**routing(y,base,route),Fixed=int(fixed.sum()),Broken=int(broken.sum()),Net=int(fixed.sum()-broken.sum()),
        Fixed_FN=int((fixed&(y==1)).sum()),Fixed_FP=int((fixed&(y==0)).sum()),Broken_TN=int((broken&(y==0)).sum()),Broken_TP=int((broken&(y==1)).sum()),
        Routed_BA=classification(y[route],expert[route])['BA'] if len(np.unique(y[route]))==2 else float('nan'),Routed_AUC=auc(y[route],expert[route]),
        FN_vs_TN_AUC=auc(y[neg],expert[neg]),TP_vs_FP_AUC=auc(y[pos],expert[pos]),FN_TN_n=int(neg.sum()),FN_n=int((neg&(y==1)).sum()),TN_n=int((neg&(y==0)).sum()),
        TP_FP_n=int(pos.sum()),TP_n=int((pos&(y==1)).sum()),FP_n=int((pos&(y==0)).sum()),ClassFlips=int(((base>=.5)!=(final>=.5)).sum()),DamageRate=int(broken.sum())/max(int((before&route).sum()),1))
def paired_intervals(y,base,route,a,b,repeats=2000,seed=20261014):
    groups={}
    for i,k in enumerate(zip(y.tolist(),(base>=.5).tolist(),route.tolist())):groups.setdefault(k,[]).append(i)
    rng=np.random.RandomState(seed);samples=[];keys=('BA','Net','Fixed','Broken','FN_vs_TN_AUC','TP_vs_FP_AUC')
    for _ in range(repeats):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups.values()]);ma=cascade(y[ix],base[ix],a[ix],route[ix]);mb=cascade(y[ix],base[ix],b[ix],route[ix]);samples.append([ma[k]-mb[k] for k in keys])
    point_a=cascade(y,base,a,route);point_b=cascade(y,base,b,route);s=np.asarray(samples);out=[]
    for j,k in enumerate(keys):
        vals=s[:,j];vals=vals[np.isfinite(vals)];lo,hi=np.quantile(vals,[.025,.975]) if len(vals) else (np.nan,np.nan)
        out.append(dict(metric=k,difference=point_a[k]-point_b[k],ci95_low=float(lo),ci95_high=float(hi),valid_bootstrap=len(vals),interval='exploratory paired APK bootstrap, no refitting/family clustering'))
    return out
