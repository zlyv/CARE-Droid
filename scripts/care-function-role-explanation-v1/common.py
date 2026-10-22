#!/usr/bin/env python3
from __future__ import annotations
import csv,gzip,hashlib,json,os
from pathlib import Path
from typing import Any
import numpy as np

PROJECT=Path(os.environ.get("CFRE_PROJECT","/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT=Path(os.environ.get("CFRE_OUTPUT",str(PROJECT/"outputs/care-function-role-explanation-v1"))).resolve()
FORMAL=PROJECT/"artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv"
P31=PROJECT/"artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"
APK_PATH_CANDIDATES=(PROJECT/"outputs/care-ccbc-source-v3/apk_paths.json",PROJECT/"outputs/care-icde-source-v1/apk_paths.json")

def norm(x): return str(x or "").strip().casefold()
def read_csv(p):
    with Path(p).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write_csv(p,rows):
    rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:p.write_text("",encoding="utf-8");return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with p.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def dump_json(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8");t.replace(p)
def read_json(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha256_file(p):
    d=hashlib.sha256()
    with Path(p).open("rb") as h:
        for b in iter(lambda:h.read(1<<20),b""):d.update(b)
    return d.hexdigest()
def asset(p,purpose):
    p=Path(p).resolve();s=p.stat();return {"path":str(p),"purpose":purpose,"size":s.st_size,"mtime_ns":s.st_mtime_ns,"sha256":sha256_file(p)}
def binary_auc(y,s):
    y=np.asarray(y,int);s=np.asarray(s,float);a=int(y.sum());b=len(y)-a
    if not a or not b:return float("nan")
    order=np.argsort(s,kind="mergesort");z=s[order];r=np.empty(len(s));i=0
    while i<len(s):
        j=i+1
        while j<len(s) and z[j]==z[i]:j+=1
        r[order[i:j]]=(i+1+j)/2;i=j
    return float((r[y==1].sum()-a*(a+1)/2)/(a*b))
def spearman(x,y):
    from scipy.stats import spearmanr
    v=float(spearmanr(np.asarray(x,float),np.asarray(y,float)).statistic);return None if not np.isfinite(v) else v

