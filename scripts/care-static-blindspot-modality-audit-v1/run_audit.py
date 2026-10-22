#!/usr/bin/env python3
from __future__ import annotations
import json,math,os,subprocess,sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from common import (OUTPUT,asset,discover_assets,dump_json,ensure_output_boundary,
                    load_apk_paths,norm,read_csv,sha256_file,write_csv)

HERE=Path(__file__).resolve().parent;CACHE=OUTPUT/"cache/features";WORKERS=max(1,int(os.environ.get("SBMA_WORKERS","3")));TIMEOUT=int(os.environ.get("SBMA_APK_TIMEOUT_SECONDS","1800"))
YEARS=(2017,2018,2019,2020,2021);ARMS=("LR","RF","XGB","SVM")
FEATURES=("has_native_surface","has_multidex","has_embedded_code_payload","has_dynamic_loader","has_reflection","has_native_loader","has_indirect_dispatch","has_encrypted_zip_entry","has_high_entropy_payload","has_manifest_code_gap","has_two_or_more_blindspot_families")

def load_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def valid(path,sid):
    try:
        value=load_json(path);return value if value.get("status")=="OK" and value.get("sha256")==sid else None
    except Exception:return None
def run_worker(command):
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=TIMEOUT);return result.returncode,(result.stderr or result.stdout)[-2000:]
    except subprocess.TimeoutExpired:return 124,"TIMEOUT"
def build_features(ids,apk_map):
    CACHE.mkdir(parents=True,exist_ok=True);result={};pending=[]
    for sid in ids:
        path=CACHE/f"{sid}.json";value=valid(path,sid)
        if value is not None:result[sid]=value
        else:pending.append((sid,path))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs={pool.submit(run_worker,[sys.executable,str(HERE/"feature_worker.py"),"--apk",str(apk_map[sid]),"--sha256",sid,"--output",str(path)]):(sid,path) for sid,path in pending}
        done=0
        for future in as_completed(jobs):
            sid,path=jobs[future];rc,error=future.result();done+=1;value=valid(path,sid)
            if rc==0 and value is not None:result[sid]=value
            else:print(f"SBMA_FEATURE_FAIL sha={sid} rc={rc} error={error!r}",flush=True)
            if done%20==0 or done==len(pending):print(f"SBMA_FEATURES {len(result)}/{len(ids)} pending_done={done}/{len(pending)}",flush=True)
    return result
def binom_two_sided(success,failure):
    n=success+failure
    if not n:return 1.0
    observed=math.comb(n,success)/(2**n);return min(1.0,sum(math.comb(n,k)/(2**n) for k in range(n+1) if math.comb(n,k)/(2**n)<=observed+1e-15))
def holm(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);out=[1.0]*len(values);running=0.0;m=len(values)
    for rank,index in enumerate(order):
        running=max(running,min(1.0,(m-rank)*values[index]));out[index]=running
    return out
def numeric_row(row,feature):return int(float(row[feature]))
def vector(row):
    values=[float(row[f"{arm}_prob"]) for arm in ARMS]+[float(row["expert_prob"]),math.log1p(float(row["apk_size"])),math.log1p(float(row["method_count"])),math.log1p(float(row["invoke_count"]))]
    return np.asarray(values,float)
def matched_pairs(rows,year,control_kind):
    local=[r for r in rows if int(r["year"])==year];targets=sorted((r for r in local if int(r["family_plus_expert_unanimous_wrong"])),key=lambda r:r["sha256"])
    matrix=np.asarray([vector(r) for r in local]);scale=np.maximum(matrix.std(0),1e-6);used=set();pairs=[]
    for target in targets:
        if control_kind=="CURRENT_CORRECT":eligible=[r for r in local if not int(r["current_wrong"])]
        elif control_kind=="OTHER_STAGE1_ERROR":eligible=[r for r in local if int(r["stage1_wrong"]) and not int(r["family_plus_expert_unanimous_wrong"])]
        else:raise ValueError(control_kind)
        eligible=[r for r in eligible if int(r["label"])==int(target["label"]) and (year!=2017 or int(float(r["fold"]))==int(float(target["fold"]))) and r["sha256"] not in used]
        reused=0
        if not eligible:
            eligible=[r for r in local if ((not int(r["current_wrong"])) if control_kind=="CURRENT_CORRECT" else (int(r["stage1_wrong"]) and not int(r["family_plus_expert_unanimous_wrong"]))) and int(r["label"])==int(target["label"])]
            reused=1
        if not eligible:continue
        chosen=min(eligible,key=lambda r:(float(np.linalg.norm((vector(r)-vector(target))/scale)),r["sha256"]));used.add(chosen["sha256"])
        pairs.append({"year":year,"fold":target.get("fold",""),"label":target["label"],"target_sha256":target["sha256"],"control_sha256":chosen["sha256"],"control_kind":control_kind,"match_distance":float(np.linalg.norm((vector(chosen)-vector(target))/scale)),"control_reused_or_relaxed":reused})
    return pairs
def analyze_pairs(rows_by,pairs,control_kind):
    output=[];raw_p=[]
    for feature in FEATURES:
        target=[numeric_row(rows_by[p["target_sha256"]],feature) for p in pairs];control=[numeric_row(rows_by[p["control_sha256"]],feature) for p in pairs]
        plus=sum(a==1 and b==0 for a,b in zip(target,control));minus=sum(a==0 and b==1 for a,b in zip(target,control));diff=float(np.mean(np.asarray(target)-np.asarray(control))) if target else 0.0;p=binom_two_sided(plus,minus)
        folds=defaultdict(list)
        for pair,a,b in zip(pairs,target,control):folds[int(float(pair.get("fold") or 0))].append(a-b)
        positive=sum(np.mean(v)>0 for key,v in folds.items() if key);evaluable=sum(bool(v) for key,v in folds.items() if key)
        output.append({"control_kind":control_kind,"feature":feature,"pairs":len(target),"target_prevalence":float(np.mean(target)) if target else 0.0,"control_prevalence":float(np.mean(control)) if control else 0.0,"paired_difference":diff,"target_only":plus,"control_only":minus,"discordant_odds_ratio":(plus+.5)/(minus+.5),"mcnemar_exact_p":p,"positive_folds":positive,"evaluable_folds":evaluable});raw_p.append(p)
    adjusted=holm(raw_p)
    for row,value in zip(output,adjusted):row["holm_p"]=value
    return output
def main():
    ensure_output_boundary();OUTPUT.mkdir(parents=True,exist_ok=True);assets=discover_assets();rows=read_csv(assets["ceiling"])
    if len(rows)!=2788:raise RuntimeError("BLOCKED_BY_CONTRACT: sample count changed")
    for row in rows:
        for key in ("year","label","stage1_wrong","current_wrong","family_plus_expert_unanimous_wrong"):row[key]=int(float(row[key]))
    wanted={norm(r["sha256"]) for r in rows};apk_map,path_audit=load_apk_paths(wanted)
    if set(apk_map)!=wanted:raise RuntimeError(f"BLOCKED_BY_CONTRACT: APK paths {len(apk_map)}/{len(wanted)}")
    before=[asset(path,key) for key,path in assets.items()]
    for sid in sorted(wanted):
        status=apk_map[sid].stat();before.append({"absolute_path":str(apk_map[sid]),"purpose":"read-only APK "+sid,"size_bytes":status.st_size,"mtime_ns":status.st_mtime_ns,"sha256":sid})
    dump_json(OUTPUT/"baseline_asset_manifest_before.json",before);dump_json(OUTPUT/"apk_path_audit.json",{"wanted":len(wanted),"resolved":len(apk_map),"sources":path_audit})
    protocol={"schema_version":"1.0.0","task":"static blind-spot modality audit","source_2017_primary":True,"future_2018_2021_diagnostic_only":True,"model_training":False,"model_integration":False,"exact_api_policy":"exact references are used only inside the extractor to map aggregate observability modalities and are discarded","features":FEATURES,"matching":"same year and label; source also same fold; nearest LR/RF/XGB/SVM/expert scores and APK/method/invoke scale","candidate_gate":"source paired difference>=0.15, discordant OR>=2, Holm p<0.05, positive in >=3 folds, and difference versus other Stage-I errors>=0.10"};dump_json(OUTPUT/"protocol.json",protocol)
    features=build_features(sorted(wanted),apk_map);coverage=len(features)/len(wanted)
    consensus_ids={r["sha256"] for r in rows if r["family_plus_expert_unanimous_wrong"]};missing_consensus=sorted(consensus_ids-set(features))
    dump_json(OUTPUT/"feature_extraction.json",{"wanted":len(wanted),"completed":len(features),"coverage":coverage,"failed":sorted(wanted-set(features)),"consensus_wanted":len(consensus_ids),"consensus_completed":len(consensus_ids)-len(missing_consensus),"missing_consensus":missing_consensus})
    if coverage<.95:raise RuntimeError(f"BLOCKED_BY_CONTRACT: feature coverage {len(features)}/{len(wanted)}")
    if missing_consensus:raise RuntimeError(f"BLOCKED_BY_CONTRACT: consensus feature failures {missing_consensus}")
    merged=[]
    for row in rows:
        sid=row["sha256"]
        if sid not in features:continue
        feature=dict(features[sid]);feature.pop("payload_name_hashes",None);feature["has_two_or_more_blindspot_families"]=int(feature["blindspot_family_count"]>=2);merged.append({**row,**feature})
    write_csv(OUTPUT/"sample_modality_features.csv",merged);rows_by={r["sha256"]:r for r in merged}
    source_cc=matched_pairs(merged,2017,"CURRENT_CORRECT");source_other=matched_pairs(merged,2017,"OTHER_STAGE1_ERROR");write_csv(OUTPUT/"source_matched_pairs.csv",source_cc+source_other)
    source_cc_results=analyze_pairs(rows_by,source_cc,"CURRENT_CORRECT");source_other_results=analyze_pairs(rows_by,source_other,"OTHER_STAGE1_ERROR");other_by={r["feature"]:r for r in source_other_results}
    candidates=[]
    for row in source_cc_results:
        row["difference_vs_other_stage1_error"]=other_by[row["feature"]]["paired_difference"]
        row["candidate"]=int(row["paired_difference"]>=.15 and row["discordant_odds_ratio"]>=2 and row["holm_p"]<.05 and row["positive_folds"]>=3 and row["difference_vs_other_stage1_error"]>=.10)
        if row["candidate"]:candidates.append(row["feature"])
    write_csv(OUTPUT/"source_modality_results.csv",source_cc_results+source_other_results)
    future_pairs=[];future_results=[]
    for year in YEARS[1:]:
        pairs=matched_pairs(merged,year,"CURRENT_CORRECT");future_pairs.extend(pairs)
        for result in analyze_pairs(rows_by,pairs,"CURRENT_CORRECT"):
            result["year"]=year;result["source_candidate"]=int(result["feature"] in candidates);future_results.append(result)
    write_csv(OUTPUT/"future_matched_pairs.csv",future_pairs);write_csv(OUTPUT/"future_diagnostic_results.csv",future_results)
    prevalence=[]
    for year in YEARS:
        local=[r for r in merged if r["year"]==year]
        for stratum,predicate in (("CONSENSUS_ERROR",lambda r:r["family_plus_expert_unanimous_wrong"]),("OTHER_STAGE1_ERROR",lambda r:r["stage1_wrong"] and not r["family_plus_expert_unanimous_wrong"]),("CURRENT_CORRECT",lambda r:not r["current_wrong"])):
            block=[r for r in local if predicate(r)]
            for feature in FEATURES:prevalence.append({"year":year,"stratum":stratum,"feature":feature,"n":len(block),"prevalence":float(np.mean([numeric_row(r,feature) for r in block])) if block else 0.0})
    write_csv(OUTPUT/"annual_modality_prevalence.csv",prevalence)
    status="UNMODELED_STATIC_MODALITY_ENRICHMENT_SUPPORTED" if candidates else "UNMODELED_STATIC_MODALITIES_NOT_DOMINANT"
    future_consistency={feature:sum(r["paired_difference"]>0 for r in future_results if r["feature"]==feature) for feature in candidates}
    decision={"status":status,"source_consensus_errors":sum(r["year"]==2017 and r["family_plus_expert_unanimous_wrong"] for r in merged),"candidate_modalities":candidates,"future_directional_years_diagnostic":future_consistency,"feature_coverage":coverage,"future_used_for_selection":False,"model_training_authorized":False,"next_action":"if supported, audit only the named modality with strict nulls before integration; otherwise freeze these common static blind spots as non-dominant"};dump_json(OUTPUT/"final_decision.json",decision)
    after=[]
    for number,row in enumerate(before,1):
        path=Path(row["absolute_path"]);after.append({"absolute_path":str(path),"purpose":row["purpose"],"size_bytes":path.stat().st_size,"mtime_ns":path.stat().st_mtime_ns,"sha256":sha256_file(path)})
        if number%100==0 or number==len(before):print(f"SBMA_ASSET_REHASH {number}/{len(before)}",flush=True)
    dump_json(OUTPUT/"baseline_asset_manifest_after.json",after);a={r["absolute_path"]:r for r in before};b={r["absolute_path"]:r for r in after};changed=[path for path in a if path not in b or a[path]["sha256"]!=b[path]["sha256"]];integrity={"status":"PASS" if not changed else "FAIL","asset_count":len(before),"changed":changed};dump_json(OUTPUT/"baseline_integrity_check.json",integrity)
    if changed:raise RuntimeError("BLOCKED_BY_CONTRACT: protected asset changed")
    report=f"""# CARE-Droid static blind-spot modality audit v1

Final status: **{status}**.

- Feature coverage: {len(features)}/{len(wanted)} ({coverage:.3f}).
- Source consensus errors: {decision['source_consensus_errors']}.
- Source modalities passing the frozen matched gate: {candidates}.
- Future-year directions are diagnostic only: {future_consistency}.

The audit compares consensus errors with same-label, source same-fold controls matched on all five model scores plus APK/method/invoke scale. It does not train a classifier. Exact API references are discarded after mapping reflection, dynamic-loading and native-loading observability flags.

This audit cannot prove theoretical static unidentifiability. It tests whether common unmodelled static modalities are a dominant empirical explanation of the frozen cross-model errors. Formal asset integrity: PASS.
""";(OUTPUT/"REPORT.md").write_text(report,encoding="utf-8");(OUTPUT/"report.md").write_text(report,encoding="utf-8")
    print("STATIC_BLINDSPOT_MODALITY_STATUS="+status,flush=True);print("report="+str(OUTPUT/"REPORT.md"),flush=True)
if __name__=="__main__":main()
