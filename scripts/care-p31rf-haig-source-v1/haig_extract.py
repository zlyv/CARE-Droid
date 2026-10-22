#!/usr/bin/env python3
"""Extract Host-centred API interaction graphs; never claims data flow."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,logging,sys,time,traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"vendor/scripts"));sys.path.insert(0,str(HERE/"vendor/src"))
import p70_build_verified_dependency_ag as ag
from tasa_droid.p3.verified_dependency import canonical_reference,method_reference

MAX_SAME_EDGES=1_000_000
MAX_CALL_EDGES=1_000_000

def silence_third_party_debug():
    """Keep HAIG progress/errors while suppressing Androguard's verbose DEBUG stream."""
    for name in ("androguard","androguard.core","androguard.misc","asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr,level="WARNING",backtrace=False,diagnose=False)
    except Exception:
        pass

def stable(value,n=24):return hashlib.sha256(str(value).encode("utf-8",errors="replace")).hexdigest()[:n]
def family(reference):
    cls=canonical_reference(reference).split("->",1)[0].strip("L;").replace("/",".")
    parts=[p for p in cls.split(".") if p]
    take=2 if parts and parts[0] in {"android","java","javax","dalvik","kotlin"} else 3
    return ".".join(parts[:take]) if parts else "unknown"
def read_rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write_gz(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    with gzip.open(tmp,"wt",encoding="utf-8") as h:json.dump(obj,h,separators=(",",":"))
    tmp.replace(path)
def load_gz(path):
    with gzip.open(path,"rt",encoding="utf-8") as h:return json.load(h)

def extract(row,map_roles):
    from androguard.misc import AnalyzeAPK
    apk=dalvik=analysis=None;started=time.perf_counter()
    try:
        apk,dalvik,analysis=AnalyzeAPK(row["absolute_path"]);audit={};methods=ag.build_method_ir(dalvik,analysis,audit)
        method_by={m.signature:m for m in methods};host_calls={};internal_calls={}
        for method in methods:
            sensitive=[];calls=[];invoke_seen=0
            for position,ins in enumerate(method.instructions):
                if not ins.name.casefold().startswith("invoke-"):continue
                invoke_seen+=1;ref=method_reference(ins.output)
                if not ref:continue
                calls.append((int(ins.offset),ref))
                roles=sorted(map_roles(ref))
                if roles:
                    sensitive.append({"reference":ref,"roles":roles,"offset":int(ins.offset),"instruction_position":position,"invoke_position":invoke_seen-1})
            if sensitive:host_calls[method.signature]=sensitive
            internal_calls[method.signature]=calls
        nodes=[];by_host={};node_id=0
        for signature in sorted(host_calls):
            host_id=stable(signature,32);block=[]
            for local_order,item in enumerate(host_calls[signature]):
                node={"node_id":node_id,"canonical_api":item["reference"],"api_hash":stable(item["reference"],32),"roles":item["roles"],"primary_role":item["roles"][0],"api_family":family(item["reference"]),"host_id":host_id,"local_order":local_order,"instruction_position":item["instruction_position"],"instruction_offset":item["offset"],"invoke_position":item["invoke_position"]}
                nodes.append(node);block.append(node);node_id+=1
            by_host[signature]=block
        same=[];same_cap=False
        for signature in sorted(by_host):
            block=by_host[signature]
            for i,left in enumerate(block):
                for right in block[i+1:]:
                    if len(same)>=MAX_SAME_EDGES:same_cap=True;break
                    same.append({"src_node":left["node_id"],"dst_node":right["node_id"],"edge_type":"SAME_HOST_ORDER","instruction_gap":max(0,right["instruction_position"]-left["instruction_position"]-1),"bytecode_offset_gap":right["instruction_offset"]-left["instruction_offset"],"intermediate_invoke_count":max(0,right["invoke_position"]-left["invoke_position"]-1),"same_basic_block":None})
                if same_cap:break
            if same_cap:break
        cross=[];cross_cap=False;host_relations=set()
        for caller in sorted(by_host):
            for _,callee in internal_calls.get(caller,[]):
                if callee not in by_host or callee==caller:continue
                host_relations.add((caller,callee))
        for caller,callee in sorted(host_relations):
            for left in by_host[caller]:
                for right in by_host[callee]:
                    if len(cross)>=MAX_CALL_EDGES:cross_cap=True;break
                    cross.append({"src_node":left["node_id"],"dst_node":right["node_id"],"edge_type":"HOST_CALL","host_call_distance":1})
                if cross_cap:break
            if cross_cap:break
        unique_api=len({n["api_hash"] for n in nodes});unique_pairs=len({(nodes[e["src_node"]]["api_hash"],nodes[e["dst_node"]]["api_hash"],e["edge_type"]) for e in same+cross})
        if any("instruction_position" not in n for n in nodes):raise RuntimeError("HAIG node missing instruction_position")
        if any(e["instruction_gap"]<0 or e["intermediate_invoke_count"]<0 for e in same):raise RuntimeError("HAIG ordered edge has negative gap")
        stats={"num_api_nodes":len(nodes),"num_hosts":len(by_host),"num_same_host_edges":len(same),"num_host_call_edges":len(cross),"unique_api_count":unique_api,"unique_pair_count":unique_pairs,"direct_host_relation_count":len(host_relations),"same_edge_cap_hit":same_cap,"call_edge_cap_hit":cross_cap,"method_count":len(methods),"instruction_count":audit.get("instruction_count",0),"invoke_instruction_count":audit.get("invoke_instruction_count",0),"extraction_seconds":time.perf_counter()-started}
        return {"schema_version":"HAIG-v1","sha256":str(row["sha256"]).casefold(),"year":int(float(row.get("metadata_year") or row.get("year") or 2017)),"status":"OK","nodes":nodes,"same_host_edges":same,"host_call_edges":cross,"stats":stats,"semantics":{"SAME_HOST_ORDER":"linear bytecode order within one Host; not data flow","HOST_CALL":"direct resolved method invocation between two sensitive Hosts; not data flow"}}
    finally:apk=dalvik=analysis=None

def main():
    silence_third_party_debug()
    p=argparse.ArgumentParser();p.add_argument("--lineage",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--worker",type=int,default=0);p.add_argument("--workers",type=int,default=1);p.add_argument("--limit",type=int,default=0);p.add_argument("--progress-every",type=int,default=5);args=p.parse_args()
    mapping=ag.load_yaml(HERE/"vendor/configs/protocol/semantic_mapping_v1.yaml");map_roles=ag.mapping_function(mapping.get("rules",[]));rows=read_rows(args.lineage)
    if args.limit>0:rows=rows[:args.limit]
    selected=[r for i,r in enumerate(rows) if i%args.workers==args.worker];args.output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();failed=0;done=0
    for index,row in enumerate(selected,1):
        sha=str(row["sha256"]).casefold();path=args.output/f"{sha}.json.gz"
        if path.exists():
            try:
                checkpoint=load_gz(path)
                # ERROR checkpoints are diagnostic records, never completed work.
                if checkpoint.get("sha256")==sha and checkpoint.get("status")=="OK":done+=1;continue
            except Exception:pass
        try:write_gz(path,extract(row,map_roles))
        except Exception as exc:
            failed+=1;write_gz(path,{"schema_version":"HAIG-v1","sha256":sha,"year":int(float(row.get("metadata_year") or row.get("year") or 2017)),"status":"ERROR","error_type":type(exc).__name__,"error_message":str(exc)[:1000],"traceback":traceback.format_exc()[-3000:],"nodes":[],"same_host_edges":[],"host_call_edges":[],"stats":{}})
        done+=1
        if done%args.progress_every==0 or index==len(selected):
            elapsed=time.perf_counter()-started;rate=done/max(elapsed,1e-9);eta=(len(selected)-index)/max(rate,1e-9)
            print(f"worker={args.worker} processed={index}/{len(selected)} completed={done} failed={failed} rate={rate:.3f}/s ETA={eta/60:.1f}min",flush=True)
    print(f"HAIG_EXTRACT_WORKER_DONE worker={args.worker} rows={len(selected)} failed={failed}",flush=True)
if __name__=="__main__":main()

