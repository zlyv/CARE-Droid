#!/usr/bin/env python3
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path("artifacts/care-role-reliability-gate-v1")
BASE = Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv")
CACHE = Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1/cache/A0_ROLE_ONLY.npz")
METHOD = "P31RF__RFRoute__A0_ROLE_ONLY__Bounded"

def norm(x): return str(x).strip().casefold()
def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as h: return list(csv.DictReader(h))
def write_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text("",encoding="utf-8");return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def logit(p):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8);return np.log(p/(1-p))
def entropy(p,axis=-1):
    p=np.asarray(p,float);d=p.sum(axis=axis,keepdims=True);q=np.divide(p,d,out=np.zeros_like(p),where=d>0)
    safe=np.where(q>0,q,1.0)
    return -np.sum(np.where(q>0,q*np.log(safe),0),axis=axis)

