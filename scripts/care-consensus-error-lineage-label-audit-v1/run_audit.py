#!/usr/bin/env python3
from __future__ import annotations
import gzip,json,os,subprocess,sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from common import (OUTPUT, asset, discover_assets, dump_json,
                    ensure_output_boundary, load_apk_paths, norm, read_csv,
                    sketch_jaccard, write_csv)

HERE=Path(__file__).resolve().parent
META=OUTPUT/"cache/metadata";DEEP=OUTPUT/"cache/deep_fingerprints"
WORKERS=max(1,int(os.environ.get("CELA_WORKERS","4")));TIMEOUT=int(os.environ.get("CELA_APK_TIMEOUT_SECONDS","1200"))
SKETCHES=("manifest","classes","method_exact","method_shape","internal_edges","external_api","zip_layout")
YEARS=(2017,2018,2019,2020,2021)

def load_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def load_gzip(path):
    with gzip.open(path,"rt",encoding="utf-8") as h:return json.load(h)
def valid_metadata(path,sid):
    try:
        value=load_json(path);return value if value.get("status")=="OK" and value.get("sha256")==sid else None
    except Exception:return None
def run_worker(command):
    try:
        value=subprocess.run(command,capture_output=True,text=True,timeout=TIMEOUT)
        return value.returncode,(value.stderr or value.stdout)[-2000:]
    except subprocess.TimeoutExpired:return 124,"TIMEOUT"
def build_metadata(ids,apk_map):
    META.mkdir(parents=True,exist_ok=True);result={};pending=[]
    for sid in ids:
        path=META/f"{sid}.json";value=valid_metadata(path,sid)
        if value is not None:result[sid]=value
        else:pending.append((sid,path))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs={pool.submit(run_worker,[sys.executable,str(HERE/"metadata_worker.py"),"--apk",str(apk_map[sid]),"--sha256",sid,"--output",str(path)]):(sid,path) for sid,path in pending}
        done=0
        for future in as_completed(jobs):
            sid,path=jobs[future];rc,error=future.result();done+=1;value=valid_metadata(path,sid)
            if rc==0 and value is not None:result[sid]=value
            else:print(f"CELA_METADATA_FAIL sha={sid} rc={rc} error={error!r}",flush=True)
            if done%25==0 or done==len(pending):print(f"CELA_METADATA {len(result)}/{len(ids)} pending_done={done}/{len(pending)}",flush=True)
    return result
def indices(metadata):
    package=defaultdict(set);certificate=defaultdict(set);dex=defaultdict(set)
    for sid,row in metadata.items():
        if row.get("package_sha256"):package[row["package_sha256"]].add(sid)
        for value in row.get("certificate_sha256",[]):certificate[value].add(sid)
        for value in row.get("dex_sha256",[]):dex[value].add(sid)
    return package,certificate,dex
def cross_label(ids,sid,label_by):return sorted(value for value in ids if value!=sid and label_by[value]!=label_by[sid])
def annotate_lineage(rows,metadata):
    label_by={r["sha256"]:int(r["label"]) for r in rows};package,certificate,dex=indices(metadata);out=[]
    for row in rows:
        sid=row["sha256"];m=metadata[sid]
        package_ids=package.get(m.get("package_sha256"),set()) if m.get("package_sha256") else set()
        cert_ids=set().union(*(certificate.get(value,set()) for value in m.get("certificate_sha256",[]))) if m.get("certificate_sha256") else set()
        dex_ids=set().union(*(dex.get(value,set()) for value in m.get("dex_sha256",[]))) if m.get("dex_sha256") else set()
        package_opp=cross_label(package_ids,sid,label_by);cert_opp=cross_label(cert_ids,sid,label_by);dex_opp=cross_label(dex_ids,sid,label_by)
        strong=sorted(set(dex_opp)|set(package_opp).intersection(cert_opp))
        out.append({**row,"same_package_cross_label":int(bool(package_opp)),"shared_certificate_cross_label":int(bool(cert_opp)),
            "exact_dex_cross_label":int(bool(dex_opp)),"strong_cross_label_lineage":int(bool(strong)),
            "package_cross_label_n":len(package_opp),"certificate_cross_label_n":len(cert_opp),"dex_cross_label_n":len(dex_opp),
            "strong_cross_label_ids":";".join(strong[:20])})
    return out
def choose_pairs(rows):
    by_year=defaultdict(list)
    for row in rows:by_year[int(row["year"])].append(row)
    pairs=[];selected=set()
    for error in [r for r in rows if int(r["family_plus_expert_unanimous_wrong"])]:
        sid=error["sha256"];selected.add(sid);year=int(error["year"]);label=int(error["label"])
        pool=by_year[year];matrix=np.asarray([[float(r[f"{arm}_prob"]) for arm in ("LR","RF","XGB","SVM")]+[float(r["expert_prob"])] for r in pool])
        scale=np.maximum(matrix.std(0),1e-6);target=np.asarray([float(error[f"{arm}_prob"]) for arm in ("LR","RF","XGB","SVM")]+[float(error["expert_prob"])])
        def nearest(predicate,count,pair_type):
            candidates=[r for r in pool if r["sha256"]!=sid and predicate(r)]
            candidates.sort(key=lambda r:(float(np.linalg.norm((np.asarray([float(r[f'{arm}_prob']) for arm in ('LR','RF','XGB','SVM')]+[float(r['expert_prob'])])-target)/scale)),r["sha256"]))
            for rank,other in enumerate(candidates[:count],1):
                selected.add(other["sha256"]);pairs.append({"error_sha256":sid,"other_sha256":other["sha256"],"year":year,"pair_type":pair_type,"rank":rank,
                    "error_label":label,"other_label":int(other["label"])})
        nearest(lambda r:int(r["label"])!=label,2,"OPPOSITE_LABEL_SCORE_NEIGHBOR")
        nearest(lambda r:int(r["label"])==label and not int(r["current_wrong"]),1,"SAME_LABEL_CORRECT_CONTROL")
        for other in str(error.get("strong_cross_label_ids","")).split(";"):
            if other:
                selected.add(other);pairs.append({"error_sha256":sid,"other_sha256":other,"year":year,"pair_type":"EXACT_METADATA_CROSS_LABEL","rank":0,"error_label":label,"other_label":1-label})
    unique={}
    for row in pairs:unique[(row["error_sha256"],row["other_sha256"],row["pair_type"])]=row
    return list(unique.values()),selected
def valid_deep(path,sid):
    try:
        value=load_gzip(path);return value if value.get("status")=="OK" and value.get("sha256")==sid else None
    except Exception:return None
def build_deep(ids,apk_map,verified_dir):
    DEEP.mkdir(parents=True,exist_ok=True);result={};pending=[]
    for sid in sorted(ids):
        path=DEEP/f"{sid}.json.gz";value=valid_deep(path,sid)
        if value is not None:result[sid]=value
        else:pending.append((sid,path))
    with ThreadPoolExecutor(max_workers=max(1,min(WORKERS,3))) as pool:
        jobs={pool.submit(run_worker,[sys.executable,str(HERE/"deep_fingerprint_worker.py"),"--apk",str(apk_map[sid]),"--sha256",sid,"--output",str(path),"--verified-dir",str(verified_dir)]):(sid,path) for sid,path in pending}
        done=0
        for future in as_completed(jobs):
            sid,path=jobs[future];rc,error=future.result();done+=1;value=valid_deep(path,sid)
            if rc==0 and value is not None:result[sid]=value
            else:print(f"CELA_DEEP_FAIL sha={sid} rc={rc} error={error!r}",flush=True)
            if done%10==0 or done==len(pending):print(f"CELA_DEEP {len(result)}/{len(ids)} pending_done={done}/{len(pending)}",flush=True)
    return result
def pair_metrics(pairs,deep):
    output=[]
    for row in pairs:
        a=deep.get(row["error_sha256"]);b=deep.get(row["other_sha256"])
        if not a or not b:continue
        ratio=min(a["method_count"],b["method_count"])/max(1,max(a["method_count"],b["method_count"]))
        value={**row,"method_count_ratio":ratio,"same_package":int(bool(a.get("package_sha256")) and a.get("package_sha256")==b.get("package_sha256")),
            "shared_certificate":int(bool(set(a.get("certificate_sha256",[]))&set(b.get("certificate_sha256",[])))),"same_dex":int(a.get("dex_sha256")==b.get("dex_sha256"))}
        for name in SKETCHES:value[name+"_jaccard"]=sketch_jaccard(a["sketches"][name],b["sketches"][name])
        value["strict_near_duplicate"]=int(ratio>=.80 and value["method_shape_jaccard"]>=.90 and value["external_api_jaccard"]>=.90 and value["manifest_jaccard"]>=.70)
        output.append(value)
    return output
def stratum_summary(rows,pair_rows):
    near_by=defaultdict(bool)
    for row in pair_rows:
        if row["pair_type"]!="SAME_LABEL_CORRECT_CONTROL" and int(row["strict_near_duplicate"]):near_by[row["error_sha256"]]=True
    output=[]
    for year in YEARS:
        local=[r for r in rows if int(r["year"])==year]
        for name,predicate in (("CONSENSUS_ERROR",lambda r:int(r["family_plus_expert_unanimous_wrong"])),("OTHER_STAGE1_ERROR",lambda r:int(r["stage1_wrong"]) and not int(r["family_plus_expert_unanimous_wrong"])),("CURRENT_CORRECT",lambda r:not int(r["current_wrong"]))):
            block=[r for r in local if predicate(r)]
            output.append({"year":year,"stratum":name,"n":len(block),"strong_cross_label_lineage_n":sum(int(r["strong_cross_label_lineage"]) for r in block),
                "strong_cross_label_lineage_rate":sum(int(r["strong_cross_label_lineage"]) for r in block)/max(1,len(block)),
                "same_package_cross_label_rate":sum(int(r["same_package_cross_label"]) for r in block)/max(1,len(block)),
                "shared_certificate_cross_label_rate":sum(int(r["shared_certificate_cross_label"]) for r in block)/max(1,len(block)),
                "strict_near_duplicate_error_n":sum(bool(near_by[r["sha256"]]) for r in block) if name=="CONSENSUS_ERROR" else ""})
    return output,near_by
def main():
    ensure_output_boundary();OUTPUT.mkdir(parents=True,exist_ok=True);assets=discover_assets();rows=read_csv(assets["ceiling"])
    # Capture formal-input hashes before any long-running read-only scan.  APK
    # hashes are captured by metadata_worker before parsing and are appended
    # below once the workers complete.
    input_before=[asset(path,key) for key,path in assets.items()]
    if len(rows)!=2788:raise RuntimeError("BLOCKED_BY_CONTRACT: ceiling rows changed")
    for row in rows:
        for key in ("year","label","stage1_wrong","current_wrong","family_plus_expert_unanimous_wrong"):row[key]=int(float(row[key]))
    wanted={norm(r["sha256"]) for r in rows};apk_map,path_audit=load_apk_paths(wanted);dump_json(OUTPUT/"apk_path_audit.json",{"wanted":len(wanted),"resolved":len(apk_map),"sources":path_audit,"missing":sorted(wanted-set(apk_map))})
    if set(apk_map)!=wanted:raise RuntimeError(f"BLOCKED_BY_CONTRACT: APK paths {len(apk_map)}/{len(wanted)}")
    dump_json(OUTPUT/"protocol.json",{"schema_version":"1.0.0","task":"consensus-error software-lineage and label-consistency audit",
        "prior_nonduplicate_boundary":"the earlier source audit tested code lineage inside P31-equivalent groups; this audit tests 2017-2021 cross-model consensus errors and cross-label lineage",
        "lightweight_metadata_scope":"all 2788 APKs","deep_fingerprint_scope":"consensus errors plus deterministic score-neighbors and exact-metadata matches",
        "verified_fingerprint_implementation":str(assets["old_fingerprint"]),"future_diagnostic_only":True,"model_training":False,"model_change_authorized":False,
        "strict_near_duplicate":"method-count ratio>=0.80, method-shape Jaccard>=0.90, external-API Jaccard>=0.90, manifest Jaccard>=0.70",
        "strong_exact_lineage":"shared exact DEX, or same package hash plus shared signing certificate","label_boundary":"mixed labels within lineage are ambiguity evidence, not automatic proof of label noise"})
    metadata=build_metadata(sorted(wanted),apk_map)
    if len(metadata)/len(wanted)<.95:raise RuntimeError(f"BLOCKED_BY_CONTRACT: metadata coverage {len(metadata)}/{len(wanted)}")
    rows=[r for r in rows if r["sha256"] in metadata];annotated=annotate_lineage(rows,metadata);write_csv(OUTPUT/"sample_lineage_audit.csv",annotated)
    pairs,deep_ids=choose_pairs(annotated);deep=build_deep(deep_ids,apk_map,assets["old_fingerprint"].parent)
    if len(deep)/max(1,len(deep_ids))<.90:raise RuntimeError(f"BLOCKED_BY_CONTRACT: deep fingerprint coverage {len(deep)}/{len(deep_ids)}")
    pair_rows=pair_metrics(pairs,deep);write_csv(OUTPUT/"deep_pair_audit.csv",pair_rows)
    summaries,near_by=stratum_summary(annotated,pair_rows);write_csv(OUTPUT/"lineage_stratum_summary.csv",summaries)
    source=[r for r in annotated if r["year"]==2017 and r["family_plus_expert_unanimous_wrong"]];future=[r for r in annotated if r["year"]>2017 and r["family_plus_expert_unanimous_wrong"]]
    source_strong=sum(r["strong_cross_label_lineage"] for r in source);source_near=sum(bool(near_by[r["sha256"]]) for r in source)
    source_rate=(source_strong+source_near-sum(bool(r["strong_cross_label_lineage"] and near_by[r["sha256"]]) for r in source))/max(1,len(source))
    future_rate=(sum(r["strong_cross_label_lineage"] for r in future)+sum(bool(near_by[r["sha256"]]) for r in future)-sum(bool(r["strong_cross_label_lineage"] and near_by[r["sha256"]]) for r in future))/max(1,len(future))
    status="SOURCE_CROSS_LABEL_LINEAGE_AMBIGUITY_SUPPORTED" if source_rate>=.10 else "CONSENSUS_ERRORS_NOT_EXPLAINED_BY_SOFTWARE_LINEAGE"
    decision={"status":status,"source_consensus_errors":len(source),"source_strong_exact_lineage":source_strong,"source_strict_near_duplicate":source_near,"source_explained_union_rate":source_rate,
        "future_consensus_errors":len(future),"future_explained_union_rate_diagnostic":future_rate,"metadata_coverage":len(metadata)/len(wanted),"deep_coverage":len(deep)/max(1,len(deep_ids)),
        "future_used_for_selection":False,"model_change_authorized":False,"label_noise_claimed":False,
        "next_action":"if supported, manually verify the small conflicting lineage set before any dataset correction; otherwise freeze lineage as non-dominant"}
    dump_json(OUTPUT/"final_decision.json",decision)
    # Metadata workers verified every APK content hash before parsing. Rehash all used assets after the audit.
    before=list(input_before)
    before += [{"absolute_path":str(apk_map[sid]),"purpose":"read-only APK "+sid,"size_bytes":metadata[sid]["apk_size"],"mtime_ns":metadata[sid]["mtime_ns"],"sha256":sid} for sid in sorted(metadata)]
    dump_json(OUTPUT/"baseline_asset_manifest_before.json",before)
    after=[]
    for number,row in enumerate(before,1):
        after.append(asset(Path(row["absolute_path"]),row["purpose"]))
        if number%100==0 or number==len(before):print(f"CELA_ASSET_REHASH {number}/{len(before)}",flush=True)
    dump_json(OUTPUT/"baseline_asset_manifest_after.json",after);a={r["absolute_path"]:r for r in before};b={r["absolute_path"]:r for r in after}
    changed=[path for path in a if path not in b or a[path]["sha256"]!=b[path]["sha256"]];integrity={"status":"PASS" if not changed else "FAIL","asset_count":len(before),"changed":changed};dump_json(OUTPUT/"baseline_integrity_check.json",integrity)
    if changed:raise RuntimeError("BLOCKED_BY_CONTRACT: protected asset changed")
    report=f"""# CARE-Droid consensus-error software-lineage and label-consistency audit v1

Final status: **{status}**.

- Source consensus errors: {len(source)}; strong exact-lineage conflicts: {source_strong}; strict cross-label near-duplicates: {source_near}; union rate: {source_rate:.3f}.
- Future consensus errors (diagnostic): {len(future)}; lineage/near-duplicate union rate: {future_rate:.3f}.
- Metadata coverage: {len(metadata)}/{len(wanted)}; deep fingerprint coverage: {len(deep)}/{len(deep_ids)}.

The previous lineage audit tested 2017 P31-equivalent groups. This non-duplicative audit tests cross-model consensus errors across 2017--2021. Mixed labels within a software lineage can reflect label noise, repackaging, rider injection, or genuine behavior change; it is not automatically called a mislabeled sample.

No classifier, route, threshold, correction rule or formal dataset was changed. Future years are diagnostic only. Formal asset integrity: PASS.
"""
    (OUTPUT/"REPORT.md").write_text(report,encoding="utf-8");(OUTPUT/"report.md").write_text(report,encoding="utf-8")
    print("CONSENSUS_ERROR_LINEAGE_LABEL_STATUS="+status,flush=True);print("report="+str(OUTPUT/"REPORT.md"),flush=True)
if __name__=="__main__":main()
