#!/usr/bin/env python3
"""DEX bytecode scan for within-method semantic relations; never builds a call graph."""
from __future__ import annotations
import csv, gc, json, logging, os, time
from pathlib import Path
import numpy as np
import yaml
from local_semantic_core import roles_for_target, summarize_methods, target_from_instruction

ROOT = Path("artifacts/care-litelocal-semantic-source-v1")
SHARDS = ROOT/"shards"

def suppress_logs():
    os.environ["LOGURU_LEVEL"] = "CRITICAL"; logging.disable(logging.CRITICAL)
    try:
        from loguru import logger
        logger.remove(); logger.disable("androguard")
    except ImportError: pass
def read(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle: return list(csv.DictReader(handle))
def norm(value): return str(value or "").strip().casefold()
def safe(callable_obj, default=None):
    try: return callable_obj()
    except Exception: return default
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
def has_code(method):
    return callable(getattr(method, "get_code", None)) and safe(method.get_code) is not None
def unwrap(candidate, vm):
    if has_code(candidate): return candidate
    getter = getattr(candidate, "get_method", None)
    decoded = safe(getter) if callable(getter) else None
    if decoded is not None and has_code(decoded): return decoded
    cls = safe(getattr(candidate,"get_class_name",lambda:""), "")
    name = safe(getattr(candidate,"get_name",lambda:""), "")
    desc = safe(getattr(candidate,"get_descriptor",lambda:""), "")
    resolver = getattr(vm, "get_method_descriptor", None)
    decoded = safe(lambda: resolver(cls,name,desc)) if callable(resolver) else None
    if isinstance(decoded,(list,tuple)): decoded = decoded[0] if decoded else None
    if decoded is not None and has_code(decoded): return decoded
    index_getter=getattr(candidate,"get_method_idx",None);by_index=getattr(vm,"get_method_by_idx",None)
    decoded=safe(lambda:by_index(index_getter())) if callable(index_getter) and callable(by_index) else None
    return decoded if decoded is not None and has_code(decoded) else None
def instructions(method):
    getter = getattr(method,"get_instructions_idx",None)
    if callable(getter): iterator = getter()
    else: iterator = ((0,x) for x in getattr(method,"get_instructions",lambda:())())
    for _, ins in iterator:
        yield str(safe(getattr(ins,"get_name",lambda:""), "") or ""), str(safe(getattr(ins,"get_output",lambda:""), "") or "")
def atomic_npz(path, **arrays):
    tmp = path.with_suffix(".tmp.npz"); np.savez_compressed(tmp, **arrays); os.replace(tmp,path)

def extract(row, rules, role_index):
    APK, DEX = apk_class(), dex_class(); path = Path(row.get("absolute_path") or row.get("apk_path")); started=time.perf_counter()
    apk = APK(str(path)); sequences=[]; dex_n=class_n=method_ref_n=decoded_n=instruction_n=invoke_n=string_n=0
    for blob in apk.get_all_dex():
        dex_n += 1; vm = DEX(blob)
        try: class_n += len(list(vm.get_classes()))
        except Exception: pass
        try: string_n += len(list(vm.get_strings()))
        except Exception: pass
        methods = list(vm.get_methods()); method_ref_n += len(methods)
        candidates=[]
        try:
            for dex_cls in vm.get_classes(): candidates.extend(list(dex_cls.get_methods()))
        except Exception: candidates=[]
        if not candidates: candidates=methods
        for candidate in candidates:
            method = unwrap(candidate,vm)
            if method is None: continue
            decoded_n += 1; seq=[]
            for opname, output in instructions(method):
                instruction_n += 1
                if not opname.casefold().startswith("invoke-"): continue
                invoke_n += 1; cls,name=target_from_instruction(output)
                role_event=roles_for_target(cls,name,rules,role_index) if cls else ()
                if role_event: seq.append(role_event)
            sequences.append(seq)
        vm=None; gc.collect()
    prevalence, cooc, transitions, semantic_audit = summarize_methods(sequences)
    audit = {"apk_bytes":path.stat().st_size,"dex_count":dex_n,"class_count":class_n,
             "method_ref_count":method_ref_n,"decoded_method_count":decoded_n,"string_count":string_n,
             "instruction_count":instruction_n,"invoke_instruction_count":invoke_n,**semantic_audit,
             "total_seconds":time.perf_counter()-started}
    return prevalence,cooc,transitions,audit

def main():
    suppress_logs(); cfg=json.loads((ROOT/"resolved_inputs.json").read_text()); rows=read(cfg["lineage"])
    mapping=yaml.safe_load(Path(cfg["mapping"]).read_text(encoding="utf-8")); roles=list(mapping["roles"])
    if len(roles)!=31: raise RuntimeError("BLOCKED_BY_CONTRACT: role count changed")
    role_index={r:i for i,r in enumerate(roles)}; rules=[r for r in mapping["rules"] if r.get("rule_kind","api")=="api"]
    SHARDS.mkdir(parents=True,exist_ok=True); pending=[]
    for row in rows:
        target=SHARDS/(norm(row["sha256"])+".npz")
        if not target.is_file() or target.stat().st_size<100: pending.append(row)
    print(f"LiteLocal extraction total=799 completed={799-len(pending)} pending={len(pending)}",flush=True)
    for i,row in enumerate(pending,1):
        p,c,t,a=extract(row,rules,role_index); atomic_npz(SHARDS/(norm(row["sha256"])+".npz"), prevalence=p,cooccurrence=c,transitions=t,audit=np.asarray([a[k] for k in AUDIT_NAMES],float))
        if i%10==0 or i==len(pending): print(f"LiteLocal extracted {i}/{len(pending)}",flush=True)
    ps=[];cs=[];ts=[];aud=[];ids=[]
    for row in rows:
        sid=norm(row["sha256"]); path=SHARDS/(sid+".npz")
        with np.load(path,allow_pickle=False) as z:
            ps.append(z["prevalence"]);cs.append(z["cooccurrence"]);ts.append(z["transitions"]);aud.append(z["audit"])
        ids.append(sid)
    arrays={"sha256":np.asarray(ids),"role_prevalence":np.asarray(ps,np.float32),"within_method_cooccurrence":np.asarray(cs,np.float32),"directed_adjacent_transitions":np.asarray(ts,np.float32),"audit_values":np.asarray(aud,float),"audit_names":np.asarray(AUDIT_NAMES),"role_names":np.asarray(roles)}
    if arrays["role_prevalence"].shape!=(799,31) or arrays["within_method_cooccurrence"].shape!=(799,465) or arrays["directed_adjacent_transitions"].shape!=(799,961): raise RuntimeError("BLOCKED_BY_CONTRACT: local semantic cache shape mismatch")
    np.savez_compressed(ROOT/"local_semantic_features_2017.npz",**arrays)
    contract={"source_rows":799,"raw_blocks":{"role_prevalence":31,"within_method_cooccurrence":465,"directed_adjacent_transitions":961},"model_arms":{"P31":31,"COOC16":16,"TRANS16":16,"FUSED63":63},"raw_size_features_in_model":False,"identity_features":False,"callgraph_constructed":False,"analysis_object_constructed":False,"normalization":{"prevalence":"sensitive_method_fraction","cooccurrence":"sensitive_method_fraction","transitions":"unit_sum_over_collapsed_adjacent_transitions"},"audit_only_columns":AUDIT_NAMES}
    (ROOT/"feature_contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    print("LITELOCAL_SEMANTIC_EXTRACTION_PASS");print("rows=799 raw_dimensions=31+465+961 callgraph_constructed=false analysis_object_constructed=false")

AUDIT_NAMES=("apk_bytes","dex_count","class_count","method_ref_count","decoded_method_count","string_count","instruction_count","invoke_instruction_count","sensitive_method_count","mapped_invoke_count","transition_count","total_seconds")
if __name__=="__main__": main()

