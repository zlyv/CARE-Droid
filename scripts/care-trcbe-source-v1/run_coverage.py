#!/usr/bin/env python3
from __future__ import annotations

import csv,gc,gzip,json,math,os,random,re,sys,traceback
from collections import Counter,defaultdict,deque
from pathlib import Path
from typing import Any

import numpy as np

from common import OUTPUT,PROJECT,ROLE_NAMES,canonical,discover_assets,dump_json,endpoint_hosts,file_record,load_apk_map,read_csv,read_endpoint_rows,sha256_file,stable_id
from taxonomy import CALLBACK_NAMES,RULES,classify,schema

def multi_source_distances(adj:dict[int,list[int]],sources:set[int])->dict[int,int]:
    dist={int(v):0 for v in sources};q=deque(sorted(dist))
    while q:
        u=q.popleft()
        for v in adj.get(u,[]):
            if v not in dist:dist[v]=dist[u]+1;q.append(v)
    return dist

def classify_host(host:int,user:dict[int,int],auto:dict[int,int],neutral:dict[int,int])->tuple[str,int|None,int|None,int|None]:
    u=user.get(host);a=auto.get(host);n=neutral.get(host)
    if u is not None and a is None:return "USER_ONLY",u,None,n
    if a is not None and u is None:return "AUTONOMOUS_ONLY",None,a,n
    if u is not None and a is not None:return "BOTH",u,a,n
    if n is not None:return "NEUTRAL_ONLY",None,None,n
    return "UNREACHABLE",None,None,None

def shuffle_root_types(roots:list[tuple[Any,...]],seed:int)->list[tuple[Any,...]]:
    rng=random.Random(seed);types=[r[2] for r in roots];rng.shuffle(types)
    return [tuple(list(r[:2])+[types[i]]+list(r[3:])) for i,r in enumerate(roots)]

def shuffle_root_host(pairs:list[tuple[int,int]],seed:int)->list[tuple[int,int]]:
    rng=random.Random(seed);hosts=[h for _,h in pairs];rng.shuffle(hosts)
    return [(pairs[i][0],hosts[i]) for i in range(len(pairs))]

def formal_rows(path:Path)->list[dict[str,Any]]:
    out=[]
    for row in read_csv(path):
        sid=str(row.get("sha256","")).strip().casefold();fold=int(float(row["fold"]));routed=int(float(row.get("routed",row.get("A25__routed",0))))
        if len(sid)!=64:raise RuntimeError("formal SHA invalid")
        out.append({"sha256":sid,"fold":fold,"routed":routed})
    if len(out)!=799 or len({r["sha256"] for r in out})!=799:raise RuntimeError("formal lineage mismatch")
    return out

def load_attention(path:Path)->dict[str,dict[str,Any]]:
    out={}
    with np.load(path,allow_pickle=False) as z:
        shas=[str(x).strip().casefold() for x in z["sha256"]];off=np.asarray(z["offsets"],int);att=np.asarray(z["attention"],float);critical=np.asarray(z["critical_index"],int)
        if len(off)!=len(shas)+1:raise RuntimeError("attention offsets mismatch")
        for i,sid in enumerate(shas):out[sid]={"attention":att[off[i]:off[i+1]],"critical":int(critical[i])}
    return out

def _interfaces(value:Any)->list[str]:
    if isinstance(value,(list,tuple)):return [canonical(x) for x in value]
    return [canonical(x) for x in re.findall(r"L[^;]+;",str(value or ""))]

def extract_dex_metadata(apk_path:Path)->dict[str,Any]:
    try:
        from loguru import logger
        logger.remove()
    except Exception:pass
    try:
        from androguard.core.apk import APK
        from androguard.core.dex import DEX
    except Exception as exc:raise RuntimeError("Androguard APK/DEX API unavailable") from exc
    apk=APK(str(apk_path));classes={};methods=[];dex_count=0
    for raw in apk.get_all_dex():
        dex_count+=1;dex=DEX(raw)
        for cls in dex.get_classes():
            name=canonical(cls.get_name());super_name=canonical(cls.get_superclassname());interfaces=_interfaces(cls.get_interfaces())
            record=classes.setdefault(name,{"name":name,"super":super_name,"interfaces":set()});record["interfaces"].update(interfaces)
            for method in cls.get_methods():
                mname=canonical(method.get_name());desc=canonical(method.get_descriptor());access=""
                try:access=str(method.get_access_flags_string())
                except Exception:pass
                signature=canonical(f"{name}->{mname}{desc}")
                methods.append({"class":name,"name":mname,"descriptor":desc,"signature":signature,"node_id":stable_id(signature),"access":access})
        dex=None
    serial={k:{"name":v["name"],"super":v["super"],"interfaces":sorted(v["interfaces"])} for k,v in classes.items()}
    return {"dex_count":dex_count,"classes":serial,"methods":methods}

def type_closure(name:str,classes:dict[str,dict[str,Any]],memo:dict[str,set[str]],stack:set[str]|None=None)->set[str]:
    if name in memo:return memo[name]
    stack=set() if stack is None else set(stack)
    if name in stack:return {name}
    stack.add(name);out={name};record=classes.get(name)
    if record:
        parents=[record.get("super","")]+list(record.get("interfaces",[]))
        for parent in filter(None,parents):
            out.add(parent)
            if parent in classes:out.update(type_closure(parent,classes,memo,stack))
    memo[name]=out;return out

def identify_roots(meta:dict[str,Any],graph_ids:set[int])->tuple[list[dict[str,Any]],dict[str,Any]]:
    classes=meta["classes"];memo={};roots=[];low=[]
    for method in meta["methods"]:
        closure=type_closure(method["class"],classes,memo);rule=classify(closure,method["name"],method["descriptor"])
        if rule:
            roots.append({**method,"family":rule["family"],"category":rule["category"],"confidence":rule["confidence"],"aligned_to_graph":method["node_id"] in graph_ids})
        elif method["name"] in CALLBACK_NAMES:
            low.append({**method,"family":"METHOD_NAME_ONLY","category":"UNKNOWN","confidence":"LOW","aligned_to_graph":method["node_id"] in graph_ids})
    return roots+low,{"class_count":len(classes),"method_count":len(meta["methods"]),"methods_with_access_flags":sum(bool(m.get("access")) for m in meta["methods"]),"classes_with_superclass":sum(bool(v.get("super")) for v in classes.values()),"classes_with_interfaces":sum(bool(v.get("interfaces")) for v in classes.values()),"root_count":len(roots),"low_confidence_name_only_count":len(low),"closure_count":len(memo)}

def payload_graph(payload:dict[str,Any])->tuple[dict[int,list[int]],set[int],dict[int,int]]:
    ids=[int(x) for x in payload.get("node_ids") or []];edges=payload.get("edge_index") or [[],[]];adj=defaultdict(list);id_to_index={v:i for i,v in enumerate(ids)}
    for s,t in zip(edges[0],edges[1]):adj[int(s)].append(int(t))
    return dict(adj),set(ids),id_to_index

def write_csv(path:Path,rows:list[dict[str,Any]],fields:list[str])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)

def discover_raw_graph_candidates()->list[dict[str,Any]]:
    result=[];patterns=("**/*call*graph*","**/*callgraph*","**/*.graphml","**/*.gml")
    seen=set()
    for pattern in patterns:
        for p in (PROJECT/"artifacts").glob(pattern):
            if p.is_file() and str(p.resolve()) not in seen:
                seen.add(str(p.resolve()));result.append({"path":str(p.resolve()),"size_bytes":p.stat().st_size,"suffixes":"".join(p.suffixes)})
                if len(result)>=200:return result
    return result

def hash_manifest(inputs:list[tuple[Path,str]],phase:str)->list[dict[str,Any]]:
    records=[]
    for index,(path,purpose) in enumerate(inputs,1):
        records.append(file_record(path,purpose))
        if index%25==0 or index==len(inputs):print(f"TRCBE_ASSET_HASH phase={phase} {index}/{len(inputs)}",flush=True)
    return records

def main()->None:
    OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/"logs").mkdir(exist_ok=True);(OUTPUT/"cache/dex_metadata").mkdir(parents=True,exist_ok=True)
    dump_json(OUTPUT/"trigger_root_schema.json",schema());schema_hash=sha256_file(OUTPUT/"trigger_root_schema.json")
    assets=discover_assets();formal=formal_rows(assets["formal"]);routed=[r for r in formal if r["routed"]];wanted={r["sha256"] for r in routed};fold_by={r["sha256"]:r["fold"] for r in routed}
    apk_map,apk_map_audit=load_apk_map(wanted)
    if len(apk_map)!=len(wanted):raise RuntimeError(f"BLOCKED_BY_CONTRACT: routed APK paths {len(apk_map)}/{len(wanted)}")
    endpoints=read_endpoint_rows(assets["endpoint"],wanted)
    if len(endpoints)!=len(wanted):raise RuntimeError(f"BLOCKED_BY_CONTRACT: endpoint rows {len(endpoints)}/{len(wanted)}")
    attention=load_attention(assets["attention"])

    formal_scripts=[]
    for root in (PROJECT/"scripts/care-droid-formal-v1",PROJECT/"scripts/care-p31-stage1-rf-selection-source-v1",PROJECT/"scripts/care-di25-compact43-relationquery-source-v1",PROJECT/"scripts/care-di25-graphglobal-path-source-v1",PROJECT/"scripts/care-rolegraph-conditioned-v4-script"):
        if root.is_dir():formal_scripts.extend(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".py",".sh",".json",".yaml",".yml"})
    manifest_inputs=[(assets[k],k) for k in assets]+[(p,"formal script") for p in sorted(set(formal_scripts))]+[(apk_map[sid],"routed source APK") for sid in sorted(wanted)]
    before=hash_manifest(manifest_inputs,"before");dump_json(OUTPUT/"baseline_asset_manifest_before.json",{"git_used":False,"assets":before})

    # Schema audit deliberately confirms what the frozen cache actually retains.
    examples=[];widths=Counter();edge_widths=Counter();numeric_ids=True
    for sid in sorted(wanted)[:25]:
        p=endpoints[sid];ids=p.get("node_ids") or [];base=p.get("node_base_features") or [];ef=p.get("edge_features") or []
        numeric_ids &= all(isinstance(v,(int,float)) for v in ids)
        if base:widths[len(base[0])]+=1
        if ef:edge_widths[len(ef[0])]+=1
        examples.append({"sha256":sid,"keys":sorted(p),"node_count":len(ids),"edge_count":len((p.get("edge_index") or [[],[]])[0]),"node_id_type":type(ids[0]).__name__ if ids else None})
    graph_schema={"cache":str(assets["endpoint"]),"rows":len(endpoints),"payload_examples":examples,"node_feature_widths":dict(widths),"edge_feature_widths":dict(edge_widths),"node_ids_numeric_hashed":numeric_ids,"raw_method_signatures_persisted":False,"bounded_policy_known":{"upstream_hops":2,"downstream_hops":1,"note":"verified from frozen endpoint builder; cache is not a full call graph"},"candidate_raw_graph_assets":discover_raw_graph_candidates(),"future_rows_loaded":0}
    dump_json(OUTPUT/"graph_schema.json",graph_schema)

    root_rows=[];host_rows=[];apk_rows=[];meta_stats=[];failures=[]
    ordered=sorted(wanted)
    for number,sid in enumerate(ordered,1):
        cache_path=OUTPUT/"cache/dex_metadata"/(sid+".json.gz")
        try:
            if cache_path.is_file():
                with gzip.open(cache_path,"rt",encoding="utf-8") as f:meta=json.load(f)
            else:
                meta=extract_dex_metadata(apk_map[sid])
                with gzip.open(cache_path,"wt",encoding="utf-8") as f:json.dump(meta,f,separators=(",",":"))
            payload=endpoints[sid];adj,graph_ids,id_to_index=payload_graph(payload);roots,stat=identify_roots(meta,graph_ids);stat.update({"sha256":sid,"fold":fold_by[sid],"status":"OK"});meta_stats.append(stat)
            hosts=endpoint_hosts(payload);method_ids={int(m["node_id"]) for m in meta["methods"]};host_exact=sum(h["node_id"] in method_ids for h in hosts)
            sources={"USER":set(),"AUTO":set(),"NEUTRAL":set()};root_node_out=0
            for root in roots:
                cat=root["category"]
                if root["aligned_to_graph"]:
                    idx=id_to_index[root["node_id"]];root_node_out+=int(bool(adj.get(idx)))
                    if cat.startswith("USER_"):sources["USER"].add(idx)
                    elif cat.startswith("AUTONOMOUS_"):sources["AUTO"].add(idx)
                    elif cat=="NEUTRAL_OR_AMBIGUOUS":sources["NEUTRAL"].add(idx)
                root_rows.append({"sha256":sid,"fold":fold_by[sid],"family":root["family"],"category":cat,"confidence":root["confidence"],"method_signature":root["signature"],"node_id":root["node_id"],"aligned_to_graph":int(root["aligned_to_graph"]),"has_retained_outgoing_edge":int(root["aligned_to_graph"] and bool(adj.get(id_to_index[root["node_id"]])))})
            ud=multi_source_distances(adj,sources["USER"]);ad=multi_source_distances(adj,sources["AUTO"]);nd=multi_source_distances(adj,sources["NEUTRAL"])
            att_info=attention.get(sid);att_valid=bool(att_info is not None and len(att_info["attention"])==len(hosts));critical=int(att_info["critical"]) if att_valid and hosts else -1
            cats=Counter();weighted=0.0;critical_cat="NO_HOST";any_assigned=0
            for hi,h in enumerate(hosts):
                cat,du,da,dn=classify_host(h["node_index"],ud,ad,nd);cats[cat]+=1;any_assigned+=int(cat!="UNREACHABLE")
                weight=float(att_info["attention"][hi]) if att_valid else math.nan
                if cat in {"USER_ONLY","AUTONOMOUS_ONLY"} and att_valid:weighted+=weight
                if hi==critical:critical_cat=cat
                host_rows.append({"sha256":sid,"fold":fold_by[sid],"host_index":hi,"host_node_id":h["node_id"],"self_role_ids":"|".join(map(str,h["roles"])),"self_roles":"|".join(ROLE_NAMES[r] for r in h["roles"]),"attention":weight,"critical":int(hi==critical),"trigger_type":cat,"user_reachable":int(du is not None),"autonomous_reachable":int(da is not None),"neutral_reachable":int(dn is not None),"min_user_distance":du,"min_autonomous_distance":da,"min_neutral_distance":dn,"user_root_count":len(sources["USER"]),"autonomous_root_count":len(sources["AUTO"]),"neutral_root_count":len(sources["NEUTRAL"]),"exact_dex_host_alignment":int(h["node_id"] in method_ids)})
            total=max(1,len(hosts))
            apk_rows.append({"sha256":sid,"fold":fold_by[sid],"user_root_count":sum(r["category"].startswith("USER_") and r["confidence"]=="HIGH" for r in roots),"autonomous_root_count":sum(r["category"].startswith("AUTONOMOUS_") and r["confidence"]=="HIGH" for r in roots),"neutral_root_count":sum(r["category"]=="NEUTRAL_OR_AMBIGUOUS" for r in roots),"unknown_name_only_count":sum(r["category"]=="UNKNOWN" for r in roots),"aligned_high_root_count":sum(r["aligned_to_graph"] and r["confidence"]=="HIGH" for r in roots),"retained_root_outgoing_count":root_node_out,"host_count":len(hosts),"host_exact_alignment":host_exact/max(1,len(hosts)),"user_only_host_ratio":cats["USER_ONLY"]/total,"autonomous_only_host_ratio":cats["AUTONOMOUS_ONLY"]/total,"both_ratio":cats["BOTH"]/total,"neutral_only_ratio":cats["NEUTRAL_ONLY"]/total,"unreachable_ratio":cats["UNREACHABLE"]/total,"any_root_assignment_ratio":any_assigned/total,"directional_attention_coverage":weighted if att_valid else math.nan,"critical_host_trigger_type":critical_cat,"attention_alignment_valid":int(att_valid)})
        except Exception as exc:
            failures.append({"sha256":sid,"fold":fold_by[sid],"error_type":type(exc).__name__,"error":str(exc)[:1000],"traceback":traceback.format_exc(limit=8)})
        if number%10==0:print(f"TRCBE_DEX_COVERAGE {number}/{len(ordered)} failures={len(failures)}",flush=True)
        gc.collect()

    write_csv(OUTPUT/"root_inventory.csv",root_rows,["sha256","fold","family","category","confidence","method_signature","node_id","aligned_to_graph","has_retained_outgoing_edge"])
    write_csv(OUTPUT/"host_trigger_assignment.csv",host_rows,["sha256","fold","host_index","host_node_id","self_role_ids","self_roles","attention","critical","trigger_type","user_reachable","autonomous_reachable","neutral_reachable","min_user_distance","min_autonomous_distance","min_neutral_distance","user_root_count","autonomous_root_count","neutral_root_count","exact_dex_host_alignment"])
    write_csv(OUTPUT/"apk_evidence.csv",apk_rows,list(apk_rows[0]) if apk_rows else ["sha256"])
    write_csv(OUTPUT/"failure_audit.csv",failures,["sha256","fold","error_type","error","traceback"])

    parsed=len(apk_rows);host_total=len(host_rows);root_high=[r for r in root_rows if r["confidence"]=="HIGH"];directional=[h for h in host_rows if h["trigger_type"] in {"USER_ONLY","AUTONOMOUS_ONLY"}];critical=[h for h in host_rows if h["critical"]];critical_directional=[h for h in critical if h["trigger_type"] in {"USER_ONLY","AUTONOMOUS_ONLY"}]
    attention_valid=[r for r in apk_rows if r["attention_alignment_valid"]];weighted=float(np.mean([r["directional_attention_coverage"] for r in attention_valid])) if attention_valid else 0.0
    exact=float(np.mean([r["host_exact_alignment"] for r in apk_rows])) if apk_rows else 0.0
    user_apks=[r for r in apk_rows if r["user_only_host_ratio"]>0];auto_apks=[r for r in apk_rows if r["autonomous_only_host_ratio"]>0]
    user_folds=len({r["fold"] for r in user_apks});auto_folds=len({r["fold"] for r in auto_apks})
    critical_rate=len(critical_directional)/max(1,len(critical));directional_rate=len(directional)/max(1,host_total)
    root_align=sum(r["aligned_to_graph"] for r in root_high)/max(1,len(root_high));root_out=sum(r["has_retained_outgoing_edge"] for r in root_high)/max(1,sum(r["aligned_to_graph"] for r in root_high))
    coverage={"scope":"2017 formal DI25 routed, label-free","routed_apks_expected":200,"parsed_apks":parsed,"parse_failures":len(failures),"high_confidence_roots":len(root_high),"high_root_graph_alignment":root_align,"aligned_root_outgoing_edge_rate":root_out,"host_count":host_total,"host_exact_signature_alignment":exact,"any_root_host_assignment_rate":sum(h["trigger_type"]!="UNREACHABLE" for h in host_rows)/max(1,host_total),"directional_host_assignment_rate":directional_rate,"critical_host_directional_assignment_rate":critical_rate,"mean_attention_mass_on_directional_hosts":weighted,"routed_apks_with_USER_ONLY":len(user_apks),"routed_apks_with_AUTONOMOUS_ONLY":len(auto_apks),"USER_ONLY_fold_coverage":user_folds,"AUTONOMOUS_ONLY_fold_coverage":auto_folds,"trigger_type_counts":dict(Counter(h["trigger_type"] for h in host_rows)),"root_family_counts":dict(Counter(r["family"] for r in root_high)),"ccbc_host_assignment_reference":0.0254,"future_rows_loaded":0,"labels_read_for_taxonomy_or_coverage":False}
    dump_json(OUTPUT/"coverage_audit.json",coverage)
    total_methods=sum(s["method_count"] for s in meta_stats);total_classes=sum(s["class_count"] for s in meta_stats)
    metadata={"parsed_apks":parsed,"failed_apks":len(failures),"total_classes":total_classes,"total_methods":total_methods,"access_flags_coverage":sum(s["methods_with_access_flags"] for s in meta_stats)/max(1,total_methods),"superclass_coverage":sum(s["classes_with_superclass"] for s in meta_stats)/max(1,total_classes),"interface_declaring_class_fraction":sum(s["classes_with_interfaces"] for s in meta_stats)/max(1,total_classes),"class_superclass_available":all(s["class_count"]>0 for s in meta_stats),"framework_type_strings_recovered_without_android_sdk":bool(root_rows),"custom_app_inheritance_traversed":True,"method_fields":["declaring_class","name","descriptor","return_type_and_parameter_types_via_descriptor","access_flags"],"raw_call_graph_rebuilt":False,"exact_stable_id_contract":"uint64(first8(SHA256(canonical class->name+descriptor)))","kotlin_compose_boundary":"Only generated classes retaining a frozen framework interface/superclass and exact callback descriptor are recognized; opaque lambdas/Compose callbacks are UNKNOWN.","manifest_used":False}
    dump_json(OUTPUT/"method_metadata_coverage.json",metadata)

    criteria={"parsed_apk_rate>=0.90":parsed/max(1,len(ordered))>=.90,"host_exact_alignment>=0.90":exact>=.90,"high_root_graph_alignment>0":root_align>0,"aligned_root_outgoing_edge_rate>0":root_out>0,"critical_directional_assignment>=0.20":critical_rate>=.20,"attention_directional_mass>=0.20":weighted>=.20,"USER_ONLY_apks>=5":len(user_apks)>=5,"AUTONOMOUS_ONLY_apks>=5":len(auto_apks)>=5,"USER_ONLY_folds>=2":user_folds>=2,"AUTONOMOUS_ONLY_folds>=2":auto_folds>=2,"no_fuzzy_matching":True}
    if all(criteria.values()):status="FEASIBLE_FOR_PROBE";engineering="FEASIBLE";reason="Pre-registered label-free coverage gates passed. Direction probe is permitted but not run by this package."
    elif parsed==0 or not root_rows:status="BLOCKED";engineering="BLOCKED";reason="DEX parser or required assets did not yield auditable method metadata."
    elif exact<.50:status="NOT_FEASIBLE";engineering="NOT_FEASIBLE";reason="Host stable IDs cannot be aligned exactly to DEX methods; fuzzy matching is prohibited."
    else:status="NOT_FEASIBLE";engineering="NOT_FEASIBLE";reason="Existing bounded call graph does not provide sufficient high-confidence directional trigger-root coverage."
    decision={"status":status,"engineering_status":engineering,"stage_completed":"B_LABEL_FREE_COVERAGE","continue_to_direction_probe":status=="FEASIBLE_FOR_PROBE","reason":reason,"criteria":criteria,"coverage":coverage,"taxonomy_sha256":schema_hash,"taxonomy_frozen_before_coverage":True,"future_rows_loaded":0}
    dump_json(OUTPUT/"feasibility_decision.json",decision)
    for name in ("sufficiency_probe.json","matched_null_results.json","end_to_end_results.json"):
        dump_json(OUTPUT/name,{"status":"NOT_RUN","gate":"stage_B_feasibility","feasibility_status":status,"future_rows_loaded":0})
    dump_json(OUTPUT/"protocol.json",{"method":"TRCBE","version":"source-v1","stage":"A+B label-free only","formal_method_unchanged":True,"taxonomy_sha256":schema_hash,"coverage_scope":"2017 DI25 routed 200","labels_used":False,"future_rows_loaded":0,"decision_thresholds":criteria})
    (OUTPUT/"protocol_hash.txt").write_text(sha256_file(OUTPUT/"protocol.json")+"\n",encoding="utf-8")

    after=hash_manifest(manifest_inputs,"after");before_map={x["absolute_path"]:x for x in before};after_map={x["absolute_path"]:x for x in after};changed=[{"path":p,"before":before_map.get(p),"after":after_map.get(p)} for p in sorted(set(before_map)|set(after_map)) if before_map.get(p)!=after_map.get(p)]
    dump_json(OUTPUT/"baseline_asset_manifest_after.json",{"git_used":False,"assets":after});dump_json(OUTPUT/"baseline_integrity_check.json",{"status":"PASS" if not changed else "FAIL","changed":changed,"asset_count":len(after)})
    if changed:decision["status"]="BLOCKED";decision["reason"]="Formal asset integrity changed";dump_json(OUTPUT/"feasibility_decision.json",decision)

    report=f"""# CARE-Droid TRCBE source-v1 label-free feasibility report

## Final status

**{decision['status']}** — completed Stage A plus label-free Stage B coverage. Direction models, labels, matched-null probes, end-to-end correction, and future diagnostics were not run.

## Existing graph contract

- Frozen endpoint graph: hashed method IDs, {next(iter(widths),15) if widths else 15}-wide node features, bounded to two upstream and one downstream hop around selected sensitive endpoints.
- Raw method signatures are not persisted. DEX metadata was parsed without rebuilding a call graph, then aligned by the frozen exact stable-ID contract.
- Manifest was not used. No fuzzy class/method matching was used.

## Coverage

- Parsed routed APKs: {parsed}/200; failures={len(failures)}.
- High-confidence roots: {len(root_high)}; graph alignment={root_align:.6f}; aligned roots with retained outgoing calls={root_out:.6f}.
- Host exact DEX alignment={exact:.6f}.
- Any-root Host assignment={coverage['any_root_host_assignment_rate']:.6f}.
- Directional USER_ONLY/AUTONOMOUS_ONLY Host assignment={directional_rate:.6f}.
- Critical Host directional assignment={critical_rate:.6f}.
- Mean C43 attention mass on directional Hosts={weighted:.6f}.
- Routed APKs with USER_ONLY={len(user_apks)} across {user_folds} folds.
- Routed APKs with AUTONOMOUS_ONLY={len(auto_apks)} across {auto_folds} folds.
- Trigger counts: `{json.dumps(coverage['trigger_type_counts'],ensure_ascii=False,sort_keys=True)}`.
- CCBC Host-assignment reference=0.0254.

## Boundaries

- Android framework types are recognized from DEX superclass/interface strings and app-local inheritance closure; no SDK installation or online callback database was used.
- Kotlin/Compose/opaque lambdas, reflection registration, XML onClick, and callbacks outside the bounded frozen graph remain UNKNOWN/unreachable.
- USER/AUTONOMOUS roots are not themselves class labels. BOTH, NEUTRAL_ONLY, and UNREACHABLE are insufficient.
- Labels were not read for taxonomy or coverage; future_rows_loaded=0.

## Decision

{reason}

Formal asset integrity: **{'PASS' if not changed else 'FAIL'}**. Probe is allowed only when `status=FEASIBLE_FOR_PROBE`.
"""
    (OUTPUT/"feasibility_audit.md").write_text(report,encoding="utf-8");(OUTPUT/"coverage_report.md").write_text(report,encoding="utf-8");(OUTPUT/"report.md").write_text(report,encoding="utf-8")
    (OUTPUT/"reproduce_commands.md").write_text("# Reproduce\n\n```bash\ncd /home/boyuanjie/projects/tasa-droid\nbash scripts/care-trcbe-source-v1/launch_tmux.sh\ntail -F outputs/care-trcbe-source-v1.console.log\n```\n",encoding="utf-8")
    print(f"TRCBE_FEASIBILITY_DECISION={decision['status']}");print(f"TRCBE_CRITICAL_DIRECTIONAL_COVERAGE={critical_rate:.6f}");print(f"TRCBE_ATTENTION_DIRECTIONAL_MASS={weighted:.6f}");print("TRCBE_LABEL_FREE_PIPELINE_DONE")

if __name__=="__main__":main()

