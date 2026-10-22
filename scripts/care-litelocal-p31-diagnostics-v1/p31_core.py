#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

MEMBERS=("LR","RF","XGB","SVM")

def make(name,seed,jobs=4):
    if name=="LR":return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=1,solver="liblinear",max_iter=5000,random_state=seed))])
    if name=="RF":return RandomForestClassifier(n_estimators=300,max_features="sqrt",min_samples_leaf=2,class_weight="balanced",n_jobs=jobs,random_state=seed)
    if name=="XGB":return XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,reg_lambda=1,objective="binary:logistic",eval_metric="logloss",tree_method="hist",n_jobs=jobs,random_state=seed,verbosity=0)
    if name=="SVM":return Pipeline([("s",StandardScaler()),("m",SVC(C=1,kernel="rbf",gamma="scale",probability=False,random_state=seed))])
    raise KeyError(name)

def raw(model,x):
    if hasattr(model,"decision_function"):return np.asarray(model.decision_function(x),float).reshape(-1)
    p=np.clip(model.predict_proba(x)[:,list(model.classes_).index(1)],1e-8,1-1e-8)
    return np.log(p/(1-p))

def fit_calibrated_committee(x_train,y_train,x_target,seed,inner_folds=4,jobs=4):
    x_train=np.asarray(x_train,float);y_train=np.asarray(y_train,int);x_target=np.asarray(x_target,float)
    splits=list(StratifiedKFold(inner_folds,shuffle=True,random_state=seed).split(x_train,y_train))
    output=np.zeros((len(x_target),4),float);audit=[]
    for j,name in enumerate(MEMBERS):
        oof=np.full(len(x_train),np.nan)
        for k,(fit,cal) in enumerate(splits,1):
            model=make(name,seed+10000*j+100*k,jobs);model.fit(x_train[fit],y_train[fit]);oof[cal]=raw(model,x_train[cal])
        if not np.isfinite(oof).all():raise RuntimeError("incomplete inner OOF calibration")
        platt=LogisticRegression(C=1e6,solver="lbfgs",max_iter=2000,random_state=seed+10000*j+999)
        platt.fit(oof.reshape(-1,1),y_train)
        model=make(name,seed+10000*j,jobs);model.fit(x_train,y_train)
        output[:,j]=np.clip(platt.predict_proba(raw(model,x_target).reshape(-1,1))[:,1],1e-8,1-1e-8)
        audit.append({"member":name,"train_n":len(x_train),"target_n":len(x_target),"seed":seed,"inner_folds":inner_folds,"target_labels_received":False})
    return output,audit

def mean_probability(member_p):return np.asarray(member_p,float).mean(axis=1)
def mean_opposition(member_p,base_p):
    member_p=np.asarray(member_p,float);decision=np.asarray(base_p)>=.5
    return np.where(decision[:,None],1-member_p,member_p).mean(axis=1)
def group_top(score,groups,ids,fraction=.25):
    score=np.asarray(score,float);groups=np.asarray(groups);ids=np.asarray(ids).astype(str);route=np.zeros(len(score),bool)
    for group in sorted(set(groups.tolist())):
        idx=np.flatnonzero(groups==group);n=int(round(len(idx)*fraction));order=np.lexsort((ids[idx],-score[idx]));route[idx[order[:n]]]=True
    return route

def label(value):
    value=str(value or "").strip().casefold()
    if value in {"0","0.0","benign","goodware"}:return 0
    if value in {"1","1.0","malware","malicious"}:return 1
    raise ValueError("unknown label "+repr(value))
