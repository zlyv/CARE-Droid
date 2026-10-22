#!/usr/bin/env python3
import math
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def metric(y, p, base, route, method, scope, year=""):
    y=np.asarray(y,int); p=np.asarray(p,float); base=np.asarray(base,float); route=np.asarray(route,bool)
    pred=p>=.5; baseline=base>=.5; correct=pred==y; base_correct=baseline==y
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    fixed=int(np.sum(route & ~base_correct & correct)); broken=int(np.sum(route & base_correct & ~correct))
    return {"scope":scope,"year":year,"method":method,"n":len(y),"routed":int(route.sum()),
            "BA":float(balanced_accuracy_score(y,pred)),"AUC":float(roc_auc_score(y,p)),
            "F1":float(f1_score(y,pred,zero_division=0)),"TN":int(tn),"FP":int(fp),
            "FN":int(fn),"TP":int(tp),"fixed":fixed,"broken":broken,"net":fixed-broken}


def exact(a,b):
    n=a+b
    if not n: return 1.0
    return min(1.0,2*sum(math.comb(n,k) for k in range(min(a,b)+1))/(2**n))
