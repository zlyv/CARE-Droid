#!/usr/bin/env python3
"""Manifest + DEX-reference extraction without Analysis or call-graph construction."""
from __future__ import annotations
import csv,gc,json,logging,os,time
from pathlib import Path
import numpy as np,yaml
from feature_contract import MANIFEST_NAMES,PERMISSION_GROUPS

ROOT=Path("artifacts/care-litesemantic-noshortcut-source-v1");SHARDS=ROOT/"shards"

def suppress_parser_logs():
    """Silence Androguard's per-map-item DEBUG stream, not our progress/errors."""
    os.environ["LOGURU_LEVEL"]="CRITICAL"
    logging.disable(logging.CRITICAL)
    for name in ("androguard","androguard.core","androguard.core.dex"):
        logger=logging.getLogger(name);logger.handlers.clear();logger.propagate=False;logger.disabled=True
    try:
        from loguru import logger
        logger.remove()
        logger.disable("androguard")
    except ImportError:
        pass

def read(p):
    with Path(p).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def norm(v):return str(v or "").strip().lower()
def canonical(v):return str(v or "").strip().lower().replace(" ","")
def match(rule,cls,name):
    if rule.get("rule_kind","api")!="api":return False
    cp=[canonical(x) for x in rule.get("class_prefixes",[])];cc=[canonical(x) for x in rule.get("class_contains",[])];mn=[canonical(x) for x in rule.get("method_names",[])];mp=[canonical(x) for x in rule.get("method_prefixes",[])]
    return bool(cp or cc) and bool(mn or mp) and (any(cls.startswith(x) for x in cp) or any(x in cls for x in cc)) and (name in mn or any(name.startswith(x) for x in mp))
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
def sval(x):
    try:return str(x)
    except Exception:return ""
def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,ensure_ascii=False,separators=(",",":")),encoding="utf-8");os.replace(tmp,path)
def xml_text(apk):
    try:
        from lxml import etree
        return etree.tostring(apk.get_android_manifest_xml(),encoding="unicode").lower()
    except Exception:return sval(apk.get_android_manifest_xml()).lower()

def extract(row,roles,rules):
    APK=apk_class();DEX=dex_class();path=Path(row.get("absolute_path") or row.get("apk_path"));t0=time.perf_counter();apk=APK(str(path));manifest_seconds=time.perf_counter()-t0
    permissions={str(x).split(".")[-1].upper() for x in (apk.get_permissions() or [])};groups=[]
    for key,wanted in PERMISSION_GROUPS.items():groups.append(float(bool(permissions&set(wanted))))
    component_lists=[list(apk.get_activities() or []),list(apk.get_services() or []),list(apk.get_receivers() or []),list(apk.get_providers() or [])]
    counts=np.asarray([len(x) for x in component_lists],float);presence=(counts>0).astype(float);shares=counts/max(float(counts.sum()),1.0);xml=xml_text(apk)
    contexts=np.asarray([float("boot_completed" in xml),float("accessibilityservice" in xml or "bind_accessibility_service" in xml),float("device_admin" in xml or "bind_device_admin" in xml),float("system_alert_window" in xml or "type_application_overlay" in xml)],float)
    manifest=np.concatenate([np.asarray(groups,float),presence,shares,contexts])
    if len(manifest)!=len(MANIFEST_NAMES):raise RuntimeError("manifest width mismatch")
    role_sets={r:set() for r in roles};class_n=method_n=string_n=dex_n=0;t1=time.perf_counter()
    for blob in apk.get_all_dex():
        dex_n+=1;vm=DEX(blob)
        try:class_n+=len(list(vm.get_classes()))
        except Exception:pass
        try:string_n+=len(list(vm.get_strings()))
        except Exception:pass
        for m in list(vm.get_methods()):
            method_n+=1;cls=canonical(getattr(m,"get_class_name",lambda:"")());name=canonical(getattr(m,"get_name",lambda:"")());sig=cls+"->"+name
            for rule in rules:
                if match(rule,cls,name):role_sets[rule["role"]].add(sig)
        vm=None;gc.collect()
    dex_seconds=time.perf_counter()-t1;rc=np.asarray([len(role_sets[r]) for r in roles],float);presence_r=(rc>0).astype(float);proportion=rc/max(float(rc.sum()),1.0)
    names=["manifest__"+x for x in MANIFEST_NAMES]+["role_presence__"+r for r in roles]+["role_proportion__"+r for r in roles]
    values=np.concatenate([manifest,presence_r,proportion])
    audit={"apk_bytes":path.stat().st_size,"dex_count":dex_n,"class_count":class_n,"method_ref_count":method_n,"string_count":string_n,"permission_count":len(permissions),"component_count":int(counts.sum()),"mapped_api_unique_count":int(rc.sum()),"manifest_seconds":manifest_seconds,"dex_reference_seconds":dex_seconds,"total_seconds":time.perf_counter()-t0}
    return {"sha256":norm(row["sha256"]),"feature_names":names,"feature_values":values.tolist(),"audit":audit}

def main():
    suppress_parser_logs()
    cfg=json.loads((ROOT/"resolved_inputs.json").read_text());rows=read(cfg["lineage"]);mapping=yaml.safe_load(Path(cfg["mapping"]).read_text(encoding="utf-8"));roles=list(mapping["roles"]);rules=[r for r in mapping["rules"] if r.get("rule_kind","api")=="api"]
    if len(roles)!=31:raise RuntimeError("BLOCKED_BY_CONTRACT: role count is not 31")
    SHARDS.mkdir(parents=True,exist_ok=True);pending=[r for r in rows if not (SHARDS/(norm(r["sha256"])+".json")).is_file()]
    print(f"LiteSemantic extraction total=799 completed={799-len(pending)} pending={len(pending)}",flush=True)
    for i,row in enumerate(pending,1):
        result=extract(row,roles,rules);atomic(SHARDS/(result["sha256"]+".json"),result)
        if i%10==0 or i==len(pending):print(f"LiteSemantic extracted {i}/{len(pending)}",flush=True)
    records=[json.loads((SHARDS/(norm(r["sha256"])+".json")).read_text(encoding="utf-8")) for r in rows]
    names=records[0]["feature_names"]
    if len(records)!=799 or any(x["feature_names"]!=names for x in records):raise RuntimeError("BLOCKED_BY_CONTRACT: feature cache incomplete/inconsistent")
    fields=["sha256",*names,*list(records[0]["audit"])]
    with (ROOT/"lite_features_and_audit.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for x in records:w.writerow({"sha256":x["sha256"],**dict(zip(names,x["feature_values"])),**x["audit"]})
    (ROOT/"feature_contract.json").write_text(json.dumps({"feature_names":names,"model_feature_count":len(names),"audit_only_columns":list(records[0]["audit"]),"callgraph_constructed":False,"raw_size_features_in_model":False,"identity_features":False},indent=2)+"\n")
    print("LITESEMANTIC_FEATURE_EXTRACTION_PASS");print(f"rows=799 model_features={len(names)} callgraph_constructed=false")

if __name__=="__main__":main()
