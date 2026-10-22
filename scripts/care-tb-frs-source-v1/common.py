#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, json, math, os
from pathlib import Path
from typing import Any
import numpy as np

PROJECT = Path(os.environ.get("TBFRS_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("TBFRS_OUTPUT", str(PROJECT / "outputs/care-tb-frs-source-v1"))).resolve()
SCRIPTS = PROJECT / "scripts"
FAMILY = SCRIPTS / "care-p31-classifier-family-stage2-v1"
SOURCE_ROWS = PROJECT / "outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv"
DIRECT_ROWS = PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/source_predictions.csv"
HOST_CACHE = PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/cache/A0_ROLE_ONLY.npz"
HAIG_BG = SCRIPTS / "care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py"
V4_SCRIPT = SCRIPTS / "care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py"
SEED = 20261013
FOLDS = (1, 2, 3, 4, 5)
ARMS = (
    "H0_HARD4", "B1_SOFT4", "B2_SIGNED_FIRST", "B2U_UNSIGNED_FIRST",
    "B3_SIGNED_FIRST_SECOND", "B4_UNSIGNED_FIRST_SECOND",
    "B5_COMPONENT_SHUFFLE", "B6_CAPACITY_CONTROL",
)

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

def read_json(path: Path): return json.loads(Path(path).read_text(encoding="utf-8"))

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

def finite(v):
    try:return math.isfinite(float(v))
    except Exception:return False

def bootstrap(y, score, seed, repeats=2000):
    y=np.asarray(y,int); score=np.asarray(score,float); rng=np.random.RandomState(int(seed)); values=[]
    for _ in range(int(repeats)):
        index=rng.randint(0,len(y),len(y)); value=auc(y[index],score[index])
        if finite(value):values.append(value)
    return {"auc":auc(y,score),"ci95_low":float(np.quantile(values,.025)),"ci95_high":float(np.quantile(values,.975)),"repeats":len(values)}

def load_source():
    rows=[r for r in read_csv(SOURCE_ROWS) if norm(r.get("arm"))=="svm"]
    if len(rows)!=799: raise RuntimeError("BLOCKED_BY_CONTRACT: expected 799 SVM source rows")
    ids=np.asarray([norm(r["sha256"]) for r in rows]); y=np.asarray([int(float(r["label"])) for r in rows],int)
    fold=np.asarray([int(float(r["fold"])) for r in rows],int); base=np.asarray([float(r["stage1_prob"]) for r in rows])
    route=np.asarray([int(float(r["routed"])) for r in rows],bool)
    if set(fold)!=set(FOLDS) or int(route.sum())!=200: raise RuntimeError("BLOCKED_BY_CONTRACT: folds or Top25 route changed")
    with np.load(HOST_CACHE,allow_pickle=False) as z:
        bid=[norm(x) for x in z["sha256"]]; values=np.asarray(z["host_values"],np.float32); offsets=np.asarray(z["host_offsets"],int)
    if values.ndim!=2 or values.shape[1]!=169: raise RuntimeError("BLOCKED_BY_CONTRACT: formal A0 Host width changed")
    pos={sid:i for i,sid in enumerate(bid)}; bags=[]
    for sid in ids:
        if sid not in pos: raise RuntimeError("BLOCKED_BY_CONTRACT: missing Host bag "+sid)
        i=pos[sid]; bags.append(values[offsets[i]:offsets[i+1]])
    direct=read_csv(DIRECT_ROWS); by={norm(r["sha256"]):r for r in direct}
    if set(by)!=set(ids): raise RuntimeError("BLOCKED_BY_CONTRACT: Direct-C43 SHA mismatch")
    candidates=[k for k in direct[0] if k.endswith("__expert_prob") and "DirectStage2" in k]
    if not candidates: candidates=[k for k in direct[0] if k.endswith("__prob") and "DirectStage2" in k]
    if len(candidates)!=1: raise RuntimeError("BLOCKED_BY_CONTRACT: Direct-C43 field ambiguous "+repr(candidates))
    direct_score=np.asarray([float(by[sid][candidates[0]]) for sid in ids])
    return rows,ids,y,fold,base,route,bags,direct_score,candidates[0]
