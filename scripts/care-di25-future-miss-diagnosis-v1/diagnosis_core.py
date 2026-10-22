#!/usr/bin/env python3
from __future__ import annotations

import math
import numpy as np

MEMBERS = ("p_lr", "p_rf", "p_xgb", "p_svm")


def entropy(p):
    x = np.clip(np.asarray(p, dtype=float), 1e-12, 1-1e-12)
    return -(x*np.log2(x)+(1-x)*np.log2(1-x))


def uncertainty_scores(stage1, member):
    base=np.asarray(stage1,float); p=np.asarray(member,float)
    mean=p.mean(1); pe=entropy(mean); ee=entropy(p).mean(1)
    pred=base>=.5
    opp=np.where(pred[:,None],1-p,p)
    votes=(p>=.5).mean(1)
    return {
        "A25":1-2*np.abs(base-.5),
        "PredictiveEntropy25":pe,
        "MI_JSD25":np.maximum(pe-ee,0),
        "ProbabilityStd25":p.std(1),
        "ProbabilityRange25":p.max(1)-p.min(1),
        "VoteEntropy25":entropy(votes),
        "MeanOpposition_DI25":opp.mean(1),
        "MaxOpposition25":opp.max(1),
        "LR_Entropy25":entropy(p[:,0]),
        "RF_Entropy25":entropy(p[:,1]),
        "XGB_Entropy25":entropy(p[:,2]),
        "SVM_Entropy25":entropy(p[:,3]),
    }


def exact_year_top(score, years, sha, budget=.25):
    score=np.asarray(score,float); years=np.asarray(years,int); sha=np.asarray(sha,str)
    route=np.zeros(len(score),bool); cutoffs={}
    for year in sorted(set(years.tolist())):
        idx=np.flatnonzero(years==year); n=int(round(len(idx)*budget))
        order=np.lexsort((sha[idx],-score[idx])); chosen=idx[order[:n]]
        route[chosen]=True; cutoffs[year]=float(score[chosen].min())
    return route,cutoffs


def reason(true_support, true_votes, cutoff):
    if true_votes == 0:
        return "A_UNANIMOUS_COMMITTEE_BLIND_SPOT"
    if true_support >= .5:
        return "D_BUDGET_CROWDING_DESPITE_MEAN_TRUE_SUPPORT"
    if true_votes >= 3:
        return "C_WEAK_TRUE_MAJORITY_CONFIDENCE_DILUTION"
    return "B_MINORITY_TRUE_SIGNAL_DILUTED"

