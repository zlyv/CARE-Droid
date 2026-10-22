#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, json, os
from pathlib import Path
from typing import Any
import numpy as np

PROJECT = Path(os.environ.get("CRC_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("CRC_OUTPUT", str(PROJECT / "outputs/care-svm-c43-representation-collision-audit-v1"))).resolve()
SCRIPTS = PROJECT / "scripts"
FAMILY_SCRIPT = SCRIPTS / "care-p31-classifier-family-stage2-v1"
DIRECT_SCRIPT = SCRIPTS / "care-svm-c43-remove-bounded-correction-v1"
FAMILY_OUTPUT = PROJECT / "outputs/care-p31-classifier-family-stage2-v1"
DIRECT_OUTPUT = PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1"
FORMAL = FAMILY_OUTPUT / "source_stage2_predictions.csv"
DIRECT = DIRECT_OUTPUT / "source/U2_DIRECT_STAGE2/source_predictions.csv"
P31 = PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"
METHOD_FROZEN = "P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2"
METHOD_REPLAY = METHOD_FROZEN + "__RepresentationReplay"
FOLDS = (1, 2, 3, 4, 5)

def norm(v: Any) -> str: return str(v or "").strip().casefold()

def ensure(path: Path) -> Path:
    path = Path(path).resolve()
    try: path.relative_to(OUTPUT)
    except ValueError as exc: raise RuntimeError("BLOCKED_BY_CONTRACT: unsafe output " + str(path)) from exc
    path.parent.mkdir(parents=True, exist_ok=True); return path

def read_csv(path: Path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as h: return list(csv.DictReader(h))

def write_csv(path: Path, rows):
    rows=list(rows); path=ensure(path); fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def dump_json(path: Path, value):
    path=ensure(path); temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8"); temp.replace(path)

def sha256(path: Path) -> str:
    d=hashlib.sha256()
    with Path(path).open("rb") as h:
        for b in iter(lambda:h.read(1<<20),b""): d.update(b)
    return d.hexdigest()

def asset(path: Path, purpose: str):
    path=Path(path).resolve(); s=path.stat()
    return {"absolute_path":str(path),"purpose":purpose,"size_bytes":s.st_size,"mtime_ns":s.st_mtime_ns,"sha256":sha256(path)}

def auc(y, score):
    y=np.asarray(y,int); score=np.asarray(score,float); p=int(y.sum()); n=len(y)-p
    if p==0 or n==0:return float("nan")
    order=np.argsort(score,kind="mergesort"); ordered=score[order]; ranks=np.empty(len(y),float); a=0
    while a<len(y):
        b=a+1
        while b<len(y) and ordered[b]==ordered[a]:b+=1
        ranks[order[a:b]]=(a+1+b)/2; a=b
    return float((ranks[y==1].sum()-p*(p+1)/2)/(p*n))

def metrics(y, base, final, route):
    y=np.asarray(y,int); base=np.asarray(base,float); final=np.asarray(final,float); route=np.asarray(route,bool)
    bp=base>=.5; fp=final>=.5; before=bp==y; after=fp==y; fixed=(~before)&after; broken=before&(~after)
    tn=int(np.sum((y==0)&(~fp))); falsep=int(np.sum((y==0)&fp)); falsen=int(np.sum((y==1)&(~fp))); tp=int(np.sum((y==1)&fp))
    return {"n":len(y),"BA":float((tn/max(tn+falsep,1)+tp/max(tp+falsen,1))/2),"AUC":auc(y,final),
        "F1":float(2*tp/max(2*tp+falsep+falsen,1)),"TN":tn,"FP":falsep,"FN":falsen,"TP":tp,"errors":falsep+falsen,
        "fixed":int(fixed.sum()),"broken":int(broken.sum()),"net":int(fixed.sum()-broken.sum()),
        "fixed_FN":int(np.sum(fixed&(y==1))),"fixed_FP":int(np.sum(fixed&(y==0))),
        "broken_TN":int(np.sum(broken&(y==0))),"broken_TP":int(np.sum(broken&(y==1))),"routed":int(route.sum())}

def load_reference(sort=False):
    formal=[r for r in read_csv(FORMAL) if norm(r.get("arm"))=="svm"]
    direct=read_csv(DIRECT); by={norm(r["sha256"]):r for r in direct}
    if len(formal)!=799 or len(by)!=799: raise RuntimeError("BLOCKED_BY_CONTRACT: source rows changed")
    if sort: formal=sorted(formal,key=lambda r:norm(r["sha256"]))
    ids=np.asarray([norm(r["sha256"]) for r in formal]); y=np.asarray([int(float(r["label"])) for r in formal])
    fold=np.asarray([int(float(r["fold"])) for r in formal]); base=np.asarray([float(r["stage1_prob"]) for r in formal])
    route=np.asarray([int(float(r["routed"])) for r in formal],bool)
    field=METHOD_FROZEN+"__expert_prob" if METHOD_FROZEN+"__expert_prob" in direct[0] else METHOD_FROZEN+"__prob"
    expert=np.asarray([float(by[sid][field]) for sid in ids]); final=base.copy(); final[route]=expert[route]
    return ids,y,fold,base,route,expert,final,field
