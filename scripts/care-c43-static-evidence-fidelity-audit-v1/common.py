#!/usr/bin/env python3
from __future__ import annotations

import base64, csv, gzip, hashlib, json, os
from pathlib import Path
from typing import Any, Iterable
import numpy as np

PROJECT=Path(os.environ.get("CSEF_PROJECT","/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT=Path(os.environ.get("CSEF_OUTPUT",str(PROJECT/"outputs/care-c43-static-evidence-fidelity-audit-v1"))).resolve()
FORMAL=PROJECT/"outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv"
DIRECT=PROJECT/"outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/source_predictions.csv"
STORE=PROJECT/"outputs/care-svm-c43-representation-collision-audit-v1/representation_store.npz"
REPLAY=PROJECT/"outputs/care-svm-c43-representation-collision-audit-v1/replay_validation.json"
OLD_FP=PROJECT/"scripts/care-p31-lineage-host-audit-v1"
METHOD="P31SVM__SVMRoute__A0_ROLE_C43_RQ__DirectStage2"
PATH_FIELDS=("absolute_path","apk_path","path","file_path","apk","source_path","sample_path","input_path")
SHA_FIELDS=("sha256","sha","sample_id","id")
BIT_BYTES=262144//8
POPCOUNT=np.asarray([int(v).bit_count() for v in range(256)],dtype=np.uint8)

def norm(v:Any)->str:return str(v or "").strip().casefold()
def safe(path:Path)->Path:
    path=Path(path).resolve(); root=(PROJECT/"outputs").resolve()
    if OUTPUT==root or root not in OUTPUT.parents or not OUTPUT.name.startswith("care-c43-static-evidence-fidelity-audit-v"):
        raise RuntimeError("BLOCKED_BY_CONTRACT: unsafe output root "+str(OUTPUT))
    if path!=OUTPUT and OUTPUT not in path.parents:raise RuntimeError("BLOCKED_BY_CONTRACT: write outside output "+str(path))
    path.parent.mkdir(parents=True,exist_ok=True);return path
def read_csv(path:Path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write_csv(path:Path,rows):
    rows=list(rows);path=safe(path);fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
    temp.replace(path)
def dump_json(path:Path,value):
    path=safe(path);temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8");temp.replace(path)
def sha256_file(path:Path)->str:
    d=hashlib.sha256()
    with Path(path).open("rb") as h:
        for block in iter(lambda:h.read(1<<20),b""):d.update(block)
    return d.hexdigest()
def asset(path:Path,purpose:str,content_hash=True):
    path=Path(path).resolve();s=path.stat()
    return {"absolute_path":str(path),"purpose":purpose,"size_bytes":s.st_size,"mtime_ns":s.st_mtime_ns,
            "sha256":sha256_file(path) if content_hash else None}
def is_sha(v):
    v=norm(v);return len(v)==64 and all(ch in "0123456789abcdef" for ch in v)
def row_pair(row):
    sid=next((norm(row.get(k)) for k in SHA_FIELDS if is_sha(row.get(k))),"")
    raw=next((str(row.get(k,"") or "").strip() for k in PATH_FIELDS if str(row.get(k,"") or "").strip().casefold().endswith(".apk")),"")
    if not sid and raw and is_sha(Path(raw).stem):sid=norm(Path(raw).stem)
    return sid,raw
def candidate_pairs(path:Path):
    suffix="".join(path.suffixes).casefold()
    if suffix.endswith(".csv"):
        with path.open("r",encoding="utf-8-sig",newline="") as h:
            for row in csv.DictReader(h):yield row_pair(row)
    elif suffix.endswith(".json"):
        value=json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value,dict):
            for k,v in value.items():
                if is_sha(k) and isinstance(v,str):yield norm(k),v
            for key in ("records","rows","samples","data"):
                for row in value.get(key,[]) if isinstance(value.get(key),list) else []:
                    if isinstance(row,dict):yield row_pair(row)
    elif suffix.endswith((".jsonl",".jsonl.gz")):
        opener=gzip.open if suffix.endswith(".gz") else open
        with opener(path,"rt",encoding="utf-8") as h:
            for line in h:
                if line.strip():
                    row=json.loads(line)
                    if isinstance(row,dict):yield row_pair(row)
def resolve_apk(raw,index):
    value=Path(str(raw or ""));choices=[value] if value.is_absolute() else [PROJECT/value,index.parent/value]
    return next((p.resolve() for p in choices if p.is_file()),None)
def load_apk_paths(wanted:set[str]):
    candidates=[]
    if os.environ.get("CSEF_APK_INDEX"):candidates.append(Path(os.environ["CSEF_APK_INDEX"]))
    candidates += [PROJECT/"outputs/care-ccbc-source-v3/apk_paths.json",PROJECT/"outputs/care-icde-source-v1/apk_asset_audit.json"]
    result={};audit=[]
    for path in candidates:
        if not path.is_file():continue
        found=0
        try:
            for sid,raw in candidate_pairs(path):
                if sid in wanted and sid not in result and raw:
                    apk=resolve_apk(raw,path)
                    if apk is not None:result[sid]=apk;found+=1
            audit.append({"path":str(path.resolve()),"matched":found})
        except Exception as exc:audit.append({"path":str(path.resolve()),"error":repr(exc)})
        if len(result)==len(wanted):break
    for raw in os.environ.get("CSEF_APK_ROOTS","").split(os.pathsep):
        root=Path(raw) if raw else None
        if root and root.is_dir():
            found=0
            for apk in root.rglob("*.apk"):
                sid=norm(apk.stem)
                if sid in wanted and sid not in result:result[sid]=apk.resolve();found+=1
            audit.append({"path":str(root.resolve()),"root_matched":found})
    return result,audit
def sketch_jaccard(left,right):
    a=np.frombuffer(base64.b64decode(left["bits_b64"]),dtype=np.uint8);b=np.frombuffer(base64.b64decode(right["bits_b64"]),dtype=np.uint8)
    if len(a)!=BIT_BYTES or len(b)!=BIT_BYTES:raise RuntimeError("invalid verified fingerprint sketch width")
    inter=int(POPCOUNT[np.bitwise_and(a,b)].sum());union=int(POPCOUNT[np.bitwise_or(a,b)].sum())
    return float(inter/union) if union else 1.0
def load_reference():
    formal=[r for r in read_csv(FORMAL) if norm(r.get("arm"))=="svm"]
    direct=read_csv(DIRECT);by={norm(r["sha256"]):r for r in direct}
    if len(formal)!=799 or len(by)!=799:raise RuntimeError("BLOCKED_BY_CONTRACT: source reference changed")
    ids=np.asarray([norm(r["sha256"]) for r in formal]);y=np.asarray([int(float(r["label"])) for r in formal])
    folds=np.asarray([int(float(r["fold"])) for r in formal]);base=np.asarray([float(r["stage1_prob"]) for r in formal])
    route=np.asarray([int(float(r["routed"])) for r in formal],bool)
    field=METHOD+"__expert_prob" if METHOD+"__expert_prob" in direct[0] else METHOD+"__prob"
    expert=np.asarray([float(by[s][field]) for s in ids]);final=base.copy();final[route]=expert[route]
    return ids,y,folds,base,route,expert,final
