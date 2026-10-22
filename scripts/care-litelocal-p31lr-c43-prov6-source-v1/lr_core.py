#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def model(seed):
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=1, solver="liblinear", max_iter=5000,
                                     random_state=int(seed))),
    ])


def raw_score(fitted, values):
    return np.asarray(fitted.decision_function(values), float).reshape(-1)


def fit_calibrated_lr(x_train, y_train, x_target, seed, inner_folds=4):
    x_train=np.asarray(x_train,float); y_train=np.asarray(y_train,int); x_target=np.asarray(x_target,float)
    oof=np.full(len(y_train),np.nan)
    splits=StratifiedKFold(int(inner_folds),shuffle=True,random_state=int(seed))
    for j,(fit,cal) in enumerate(splits.split(x_train,y_train),1):
        m=model(int(seed)+100*j); m.fit(x_train[fit],y_train[fit]); oof[cal]=raw_score(m,x_train[cal])
    if not np.isfinite(oof).all(): raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete LR calibration OOF")
    calibrator=LogisticRegression(C=1e6,solver="lbfgs",max_iter=2000,random_state=int(seed)+999)
    calibrator.fit(oof.reshape(-1,1),y_train)
    fitted=model(seed); fitted.fit(x_train,y_train)
    return np.clip(calibrator.predict_proba(raw_score(fitted,x_target).reshape(-1,1))[:,1],1e-8,1-1e-8)


def crossfit_lr_probability(x, y, seed):
    x=np.asarray(x,float); y=np.asarray(y,int); out=np.full(len(y),np.nan)
    for j,(fit,hold) in enumerate(StratifiedKFold(4,shuffle=True,random_state=int(seed)).split(x,y),1):
        out[hold]=fit_calibrated_lr(x[fit],y[fit],x[hold],int(seed)+100000*j,3)
    if not np.isfinite(out).all(): raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete nested P31-LR OOF")
    return out


def mean_opposition(member_probability, anchor_probability):
    members=np.asarray(member_probability,float); decision=np.asarray(anchor_probability,float)>=.5
    return np.where(decision[:,None],1-members,members).mean(axis=1)


def foldwise_top25(score, fold, sha256):
    score=np.asarray(score,float); fold=np.asarray(fold,int); sha256=np.asarray(sha256).astype(str)
    route=np.zeros(len(score),bool)
    for value in sorted(set(fold.tolist())):
        idx=np.flatnonzero(fold==value); n=int(round(len(idx)*.25)); order=np.lexsort((sha256[idx],-score[idx])); route[idx[order[:n]]]=True
    return route
