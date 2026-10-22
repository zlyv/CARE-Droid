#!/usr/bin/env python3
"""Resumable P31-only DEX invoke scan; never builds an Analysis object or call graph."""
from __future__ import annotations
import csv,gc,json,logging,os,re,time
from pathlib import Path
import numpy as np,yaml

OUT=Path("artifacts/care-litelocal-p31-diagnostics-v1");SHARDS=OUT/"future_p31_shards"
TARGET_RE=re.compile(r"(L[^;]+;)->([^\s(]+)\(")

def suppress():
    os.environ["LOGURU_LEVEL"]="CRITICAL";logging.disable(logging.CRITICAL)
    try:
        from loguru import logger
        logger.remove();logger.disable("androguard")
    except ImportError:pass
def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def canon(v):return str(v or "").strip().casefold().replace(" ","")
def safe(fn,default=None):
    try:return fn()
    except Exception:return default
def dex_class():
    try:
        from androguard.core.dex import DEX
        return DEX
    except ImportError:
        from androguard.core.bytecodes.dvm import DalvikVMFormat
        return DalvikVMFormat
def apk_class():
    try:
        from androguard.core.apk import APK
        return APK
    except ImportError:
        from androguard.core.bytecodes.apk import APK
        return APK
def has_code(method):return callable(getattr(method,"get_code",None)) and safe(method.get_code) is not None
def unwrap(candidate,vm):
    if has_code(candidate):return candidate
    getter=getattr(candidate,"get_method",None);decoded=safe(getter) if callable(getter) else None
    if decoded is not None and has_code(decoded):return decoded
    cls=safe(getattr(candidate,"get_class_name",lambda:""),"");name=safe(getattr(candidate,"get_name",lambda:""),"");desc=safe(getattr(candidate,"get_descriptor",lambda:""),"")
    resolver=getattr(vm,"get_method_descriptor",None);decoded=safe(lambda:resolver(cls,name,desc)) if callable(resolver) else None
    if isinstance(decoded,(list,tuple)):decoded=decoded[0] if decoded else None
    if decoded is not None and has_code(decoded):return decoded
    ig=getattr(candidate,"get_method_idx",None);bg=getattr(vm,"get_method_by_idx",None);decoded=safe(lambda:bg(ig())) if callable(ig) and callable(bg) else None
    return decoded if decoded is not None and has_code(decoded) else None
def instructions(method):
    getter=getattr(method,"get_instructions_idx",None);iterator=getter() if callable(getter) else ((0,x) for x in getattr(method,"get_instructions",lambda:())())
    for _,ins in iterator:yield str(safe(getattr(ins,"get_name",lambda:""),"") or ""),str(safe(getattr(ins,"get_output",lambda:""),"") or "")
def match(rule,cls,name):
    if rule.get("rule_kind","api")!="api":return False
    cls,name=canon(cls),canon(name);prefix=[canon(x) for x in rule.get("class_prefixes",[])];contains=[canon(x) for x in rule.get("class_contains",[])];methods=[canon(x) for x in rule.get("method_names",[])];mp=[canon(x) for x in rule.get("method_prefixes",[])]
    return bool(prefix or contains) and bool(methods or mp) and (any(cls.startswith(x) for x in prefix) or any(x in cls for x in contains)) and (name in methods or any(name.startswith(x) for x in mp))
def extract(row,rules,role_index):
    APK,DEX=apk_class(),dex_class();path=Path(row["absolute_path"]);started=time.perf_counter();role_methods=np.zeros(31,np.float64);sensitive_methods=0;mapped=0;class_n=method_n=0
    apk=APK(str(path))
    for blob in apk.get_all_dex():
        vm=DEX(blob)
        try:class_n+=len(list(vm.get_classes()))
        except Exception:pass
        methods=list(vm.get_methods());method_n+=len(methods);candidates=[]
        try:
            for dex_cls in vm.get_classes():candidates.extend(list(dex_cls.get_methods()))
        except Exception:candidates=[]
        if not candidates:candidates=methods
        for candidate in candidates:
            method=unwrap(candidate,vm)
            if method is None:continue
            found=set()
            for opname,output in instructions(method):
                if not opname.casefold().startswith("invoke-"):continue
                hits=TARGET_RE.findall(output)
                if not hits:continue
                cls,name=hits[-1]
                roles={role_index[r["role"]] for r in rules if match(r,cls,name)}
                if roles:mapped+=1;found.update(roles)
            if found:sensitive_methods+=1;role_methods[list(found)]+=1
        vm=None;gc.collect()
    if sensitive_methods:role_methods/=sensitive_methods
    audit=np.asarray([path.stat().st_size,class_n,method_n,sensitive_methods,mapped,time.perf_counter()-started],float)
    return role_methods.astype(np.float32),audit
def atomic(path,**arrays):
    tmp=path.with_suffix(".tmp.npz");np.savez_compressed(tmp,**arrays);os.replace(tmp,path)
def main():
    suppress();cfg=json.loads((OUT/"preflight/resolved_inputs.json").read_text());rows=read(cfg["future_label_free"]);mapping=yaml.safe_load(Path(cfg["mapping"]).read_text(encoding="utf-8"));roles=list(mapping["roles"]);role_index={r:i for i,r in enumerate(roles)};rules=[r for r in mapping["rules"] if r.get("rule_kind","api")=="api"]
    if len(roles)!=31:raise RuntimeError("BLOCKED_BY_CONTRACT: role mapping changed")
    SHARDS.mkdir(parents=True,exist_ok=True);pending=[r for r in rows if not (SHARDS/(r["sha256"]+".npz")).is_file()]
    print(f"future P31 extraction total=1989 completed={1989-len(pending)} pending={len(pending)}",flush=True)
    for i,row in enumerate(pending,1):
        p,a=extract(row,rules,role_index);atomic(SHARDS/(row["sha256"]+".npz"),p31=p,audit=a)
        if i%25==0 or i==len(pending):print(f"future P31 extracted {i}/{len(pending)}",flush=True)
    ids=[];years=[];p31=[];audit=[]
    for row in rows:
        with np.load(SHARDS/(row["sha256"]+".npz"),allow_pickle=False) as z:p31.append(z["p31"]);audit.append(z["audit"])
        ids.append(row["sha256"]);years.append(int(row["year"]))
    np.savez_compressed(OUT/"future_p31_features_2018_2021.npz",sha256=np.asarray(ids),year=np.asarray(years,np.int16),role_prevalence=np.asarray(p31,np.float32),audit_values=np.asarray(audit,float),audit_names=np.asarray(["apk_bytes","class_count","method_ref_count","sensitive_method_count","mapped_invoke_count","total_seconds"]),role_names=np.asarray(roles))
    print("LITELOCAL_P31_FUTURE_EXTRACTION_PASS");print("future_rows=1989 dimensions=31 callgraph_constructed=false analysis_object_constructed=false labels_loaded=false")
if __name__=="__main__":main()
