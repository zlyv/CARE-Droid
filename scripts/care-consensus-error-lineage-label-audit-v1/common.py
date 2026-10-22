#!/usr/bin/env python3
from __future__ import annotations

import base64, csv, gzip, hashlib, json, os
from pathlib import Path
from typing import Any, Iterable
import numpy as np

PROJECT=Path(os.environ.get("CELA_PROJECT","/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT=Path(os.environ.get("CELA_OUTPUT",str(PROJECT/"outputs/care-consensus-error-lineage-label-audit-v1"))).resolve()
PATH_FIELDS=("absolute_path","apk_path","path","file_path","apk","source_path","sample_path","input_path")
SHA_FIELDS=("sha256","sha","sample_id","id")
BIT_BYTES=262144//8
POPCOUNT=np.asarray([int(value).bit_count() for value in range(256)],dtype=np.uint8)

def norm(value:Any)->str:return str(value or "").strip().casefold()
def ensure_output_boundary():
    root=(PROJECT/"outputs").resolve()
    if OUTPUT==root or root not in OUTPUT.parents or not OUTPUT.name.startswith("care-consensus-error-lineage-label-audit-v"):
        raise RuntimeError("BLOCKED_BY_CONTRACT: unsafe output "+str(OUTPUT))
def safe(path:Path)->Path:
    ensure_output_boundary();path=Path(path).resolve()
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
def first_existing(paths:Iterable[Path],label:str)->Path:
    checked=[]
    for path in paths:
        checked.append(str(path))
        if path.is_file() and path.stat().st_size:return path.resolve()
    raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {label}; checked={checked!r}")
def discover_assets():
    return {
      "ceiling":first_existing([PROJECT/"outputs/care-static-identifiability-ceiling-audit-v1/sample_error_ceiling.csv"],"identifiability sample audit"),
      "temporal":first_existing([PROJECT/"outputs/care-direct-c43-temporal-direction-audit-v1/sample_direction_audit.csv"],"temporal audit"),
      "old_fingerprint":first_existing([PROJECT/"scripts/care-p31-lineage-host-audit-v1/fingerprints.py"],"verified source lineage fingerprint implementation"),
      "old_common":first_existing([PROJECT/"scripts/care-p31-lineage-host-audit-v1/common.py"],"verified source lineage common implementation"),
    }
def is_sha(value):
    value=norm(value);return len(value)==64 and all(ch in "0123456789abcdef" for ch in value)
def row_pair(row):
    sid=next((norm(row.get(key)) for key in SHA_FIELDS if is_sha(row.get(key))),"")
    raw=next((str(row.get(key,"") or "").strip() for key in PATH_FIELDS if str(row.get(key,"") or "").strip().casefold().endswith(".apk")),"")
    if not sid and raw and is_sha(Path(raw).stem):sid=norm(Path(raw).stem)
    return sid,raw
def candidate_pairs(path:Path):
    suffix="".join(path.suffixes).casefold()
    if suffix.endswith(".csv"):
        with path.open("r",encoding="utf-8-sig",newline="") as h:
            for row in csv.DictReader(h):yield row_pair(row)
    elif suffix.endswith(".jsonl") or suffix.endswith(".jsonl.gz"):
        opener=gzip.open if suffix.endswith(".gz") else open
        with opener(path,"rt",encoding="utf-8") as h:
            for line in h:
                if line.strip():
                    row=json.loads(line)
                    if isinstance(row,dict):yield row_pair(row)
    elif suffix.endswith(".json"):
        value=json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value,dict):
            for key,raw in value.items():
                if is_sha(key) and isinstance(raw,str):yield norm(key),raw
            for key in ("records","rows","samples","cohort","data"):
                if isinstance(value.get(key),list):
                    for row in value[key]:
                        if isinstance(row,dict):yield row_pair(row)
        elif isinstance(value,list):
            for row in value:
                if isinstance(row,dict):yield row_pair(row)
def resolve_apk(raw,index):
    value=Path(str(raw or ""));choices=[value] if value.is_absolute() else [PROJECT/value,index.parent/value]
    return next((path.resolve() for path in choices if path.is_file()),None)
def load_apk_paths(wanted:set[str]):
    candidates=[]
    for name in ("CELA_APK_INDEX","CCBC_FUTURE_APK_INDEX"):
        if os.environ.get(name):candidates.append(Path(os.environ[name]))
    candidates += [PROJECT/"outputs/care-ccbc-source-v3/apk_paths.json",PROJECT/"outputs/care-icde-source-v1/apk_asset_audit.json"]
    patterns=("artifacts/care-p31rf-haig-source-v1/lineage/**/*","artifacts/care-p31rf-haig-source-v1/**/*future*",
              "artifacts/care-future-local-assets-v2/**/*","artifacts/**/*future*lineage*.csv","artifacts/**/*apk*index*.csv")
    seen={str(p.resolve()) for p in candidates if p.exists()}
    for pattern in patterns:
        for path in PROJECT.glob(pattern):
            suffix="".join(path.suffixes).casefold();key=str(path.resolve())
            if path.is_file() and suffix.endswith((".csv",".json",".jsonl",".jsonl.gz")) and key not in seen:
                seen.add(key);candidates.append(path)
    result={};audit=[]
    for path in candidates:
        if not path.is_file():continue
        found=0
        try:
            for sid,raw in candidate_pairs(path):
                if sid in wanted and sid not in result and raw:
                    apk=resolve_apk(raw,path)
                    if apk is not None:result[sid]=apk;found+=1
            if found:audit.append({"path":str(path.resolve()),"matched":found})
        except Exception as exc:audit.append({"path":str(path.resolve()),"error":repr(exc)})
        if len(result)==len(wanted):break
    for name in ("CELA_APK_ROOTS","CCBC_FUTURE_APK_ROOTS"):
        for raw in os.environ.get(name,"").split(os.pathsep):
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
    if len(a)!=BIT_BYTES or len(b)!=BIT_BYTES:raise RuntimeError("invalid verified lineage sketch width")
    inter=int(POPCOUNT[np.bitwise_and(a,b)].sum());union=int(POPCOUNT[np.bitwise_or(a,b)].sum())
    return float(inter/union) if union else 1.0
