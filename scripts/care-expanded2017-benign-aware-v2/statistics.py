#!/usr/bin/env python3
"""Metric helpers for frozen expanded-2017 comparisons; no filesystem writes.

Single-class cohorts have undefined AUC/BA/recall rather than fabricated zeroes.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def metrics(y, p, threshold=.5):
    y = np.asarray(y, int); p = np.asarray(p, float)
    if len(y) != len(p) or not np.isfinite(p).all():
        raise RuntimeError('invalid metrics inputs')
    pred = p >= threshold
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred))
    neg, pos = tn+fp, tp+fn
    specificity = tn/neg if neg else None
    recall = tp/pos if pos else None
    precision = tp/(tp+fp) if tp+fp else None
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    has_both = bool(neg and pos)
    auc = float(roc_auc_score(y, p)) if has_both else None
    ece = 0.0
    if len(y):
        for lo in np.linspace(0, 1, 11)[:-1]:
            hi = lo+.1; ix = (p >= lo) & (p < hi if hi < 1 else p <= hi)
            if ix.any(): ece += float(ix.mean()) * abs(float(y[ix].mean())-float(p[ix].mean()))
    return {'n': len(y), 'benign_n': neg, 'malware_n': pos,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
            'BA': .5*(specificity+recall) if has_both else None,
            'AUC': auc, 'accuracy': float((tn+tp)/len(y)) if len(y) else None,
            'precision': precision, 'recall': recall, 'specificity': specificity,
            'FPR': fp/neg if neg else None, 'F1': f1,
            'Brier': float(np.mean((p-y)**2)) if len(y) else None,
            'ECE10': ece if len(y) else None,
            'malicious_probability_mean': float(p.mean()) if len(p) else None,
            'malicious_probability_p50': float(np.median(p)) if len(p) else None,
            'malicious_probability_p90': float(np.quantile(p,.9)) if len(p) else None,
            'high_risk_benign_p_ge_0_9': int(np.sum((y == 0) & (p >= .9))),
            'status': 'OK' if has_both else 'UNDEFINED_SINGLE_CLASS'}
