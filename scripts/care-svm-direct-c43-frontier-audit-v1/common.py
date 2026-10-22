#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, os
from pathlib import Path
from typing import Any
import numpy as np

PROJECT=Path(os.environ.get("DCFA_PROJECT","/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT=Path(os.environ.get("DCFA_OUTPUT",str(PROJECT/"outputs/care-svm-direct-c43-frontier-audit-v1"))).resolve()
BOUNDED=PROJECT/"outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv"
DIRECT=PROJECT/"outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/source_predictions.csv"
METHOD="P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2"
FOLDS=(1,2,3,4,5)

def norm(v): return str(v or "").strip().casefold()
def ensure(path):
    path=Path(path).resolve()
    try:path.relative_to(OUTPUT)
    except ValueError as exc:raise RuntimeError("BLOCKED_BY_CONTRACT: output outside experiment") from exc
    path.parent.mkdir(parents=True,exist_ok=True);return path
def read_csv(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write_csv(path,rows):
    rows=list(rows);path=ensure(path);fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
def dump_json(path,value):
    path=ensure(path);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8");tmp.replace(path)
def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha256(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as h:
        for b in iter(lambda:h.read(1<<20),b""):d.update(b)
    return d.hexdigest()
def asset(path,purpose):
    path=Path(path).resolve();s=path.stat();return {"absolute_path":str(path),"purpose":purpose,"size_bytes":s.st_size,"mtime_ns":s.st_mtime_ns,"sha256":sha256(path)}

def metrics(y,base,final,route):
    y=np.asarray(y,int);base=np.asarray(base,float);final=np.asarray(final,float);route=np.asarray(route,bool)
    bp=base>=.5;fp=final>=.5;before=bp==y;after=fp==y;fixed=(~before)&after;broken=before&(~after)
    tn=int(np.sum((y==0)&(~fp)));falsep=int(np.sum((y==0)&fp));falsen=int(np.sum((y==1)&(~fp)));tp=int(np.sum((y==1)&fp))
    return {"n":len(y),"BA":float((tn/max(tn+falsep,1)+tp/max(tp+falsen,1))/2),"F1":float(2*tp/max(2*tp+falsep+falsen,1)),
        "TN":tn,"FP":falsep,"FN":falsen,"TP":tp,"errors":falsep+falsen,"fixed":int(fixed.sum()),"broken":int(broken.sum()),"net":int(fixed.sum()-broken.sum()),
        "fixed_FN":int(np.sum(fixed&(y==1))),"fixed_FP":int(np.sum(fixed&(y==0))),"broken_TN":int(np.sum(broken&(y==0))),"broken_TP":int(np.sum(broken&(y==1))),
        "class_flips":int(np.sum(bp!=fp)),"routed":int(route.sum())}

def load():
    raw=[r for r in read_csv(BOUNDED) if norm(r.get("arm"))=="svm"]
    if len(raw)!=799:raise RuntimeError("BLOCKED_BY_CONTRACT: bounded SVM rows changed")
    raw=sorted(raw,key=lambda r:norm(r["sha256"]));ids=[norm(r["sha256"]) for r in raw]
    y=np.asarray([int(float(r["label"])) for r in raw]);fold=np.asarray([int(float(r["fold"])) for r in raw])
    base=np.asarray([float(r["stage1_prob"]) for r in raw]);bounded=np.asarray([float(r["stage2_prob"]) for r in raw]);route=np.asarray([int(float(r["routed"])) for r in raw],bool)
    direct_rows=read_csv(DIRECT);by={norm(r["sha256"]):r for r in direct_rows}
    if set(by)!=set(ids):raise RuntimeError("BLOCKED_BY_CONTRACT: direct-C43 SHA mismatch")
    field=METHOD+"__expert_prob" if METHOD+"__expert_prob" in direct_rows[0] else METHOD+"__prob"
    direct=np.asarray([float(by[sid][field]) for sid in ids])
    if not np.isfinite(direct).all():raise RuntimeError("BLOCKED_BY_CONTRACT: nonfinite direct probability")
    return ids,y,fold,base,bounded,route,direct,field
