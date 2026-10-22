#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, json, math
from pathlib import Path
import numpy as np

ROOT = Path("artifacts/care-host-complement-size-controlled-audit-v1")
PRIOR = Path("artifacts/care-di25-host-complement-candidate-audit-v1")
SEED = 20260920
BOOTSTRAPS = 10000
THRESHOLD = 0.5
YEARS = (2018, 2019, 2020, 2021)
FOLDS = (1, 2, 3, 4, 5)

def norm(x): return str(x or "").strip().lower()

def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))

def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def load_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def write_text(path,text):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text,encoding="utf-8")
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def ranks(v):
    v=np.asarray(v,float); order=np.argsort(v,kind="mergesort"); out=np.empty(len(v),float); i=0
    while i<len(v):
        j=i+1
        while j<len(v) and v[order[j]]==v[order[i]]: j+=1
        out[order[i:j]]=(i+1+j)/2; i=j
    return out
def auc(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float); n1=int(sum(y==1)); n0=int(sum(y==0))
    if not n1 or not n0:return float("nan")
    r=ranks(p); return float((sum(r[y==1])-n1*(n1+1)/2)/(n1*n0))
def ap(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float); n=int(sum(y==1))
    if not n:return float("nan")
    yy=y[np.argsort(-p,kind="mergesort")]; return float(sum((np.cumsum(yy)/np.arange(1,len(y)+1))*yy)/n)
def metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float); pred=p>=THRESHOLD
    tp=int(sum((y==1)&pred)); tn=int(sum((y==0)&~pred)); fp=int(sum((y==0)&pred)); fn=int(sum((y==1)&~pred))
    rec=tp/(tp+fn) if tp+fn else float("nan"); spec=tn/(tn+fp) if tn+fp else float("nan"); prec=tp/(tp+fp) if tp+fp else 0
    return {"N":len(y),"positive_n":int(sum(y)),"prevalence":float(np.mean(y)),"ROC_AUC":auc(y,p),"PR_AUC":ap(y,p),
            "BA":(rec+spec)/2,"F1":2*prec*rec/(prec+rec) if prec+rec else 0,"Recall_wrong":rec,"Specificity_correct":spec,
            "TN":tn,"FP":fp,"FN":fn,"TP":tp}
def spearman(a,b):
    ra,rb=ranks(a),ranks(b)
    return 0.0 if np.std(ra)==0 or np.std(rb)==0 else float(np.corrcoef(ra,rb)[0,1])
def stratified_indices(y,years,rng):
    chunks=[]
    for year in sorted(set(years)):
        for target in (0,1):
            idx=np.flatnonzero((years==year)&(y==target))
            if len(idx): chunks.append(rng.choice(idx,len(idx),replace=True))
    return np.concatenate(chunks)
def ci(v): return float(np.percentile(v,2.5)),float(np.percentile(v,97.5))

