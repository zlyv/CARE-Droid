#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures, gzip, json, os, subprocess, sys
from pathlib import Path
import numpy as np
from common import *

WORKERS=max(1,int(os.environ.get("CSEF_WORKERS","4")))
TIMEOUT=int(os.environ.get("CSEF_APK_TIMEOUT_SECONDS","1200"))
DEEP=OUTPUT/"cache/deep_fingerprints"
SKETCHES=("manifest","classes","method_shape","internal_edges","external_api")

def load_gz(path):
    with gzip.open(path,"rt",encoding="utf-8") as h:return json.load(h)
def fingerprint_one(item):
    sid,apk=item; out=safe(DEEP/(sid+".json.gz"))
    if out.is_file() and out.stat().st_size:return sid,"OK",str(out),0.0
    cmd=[sys.executable,str(Path(__file__).with_name("deep_fingerprint_worker.py")),"--apk",str(apk),"--sha256",sid,"--output",str(out),"--verified-dir",str(OLD_FP)]
    import time;t=time.time()
    try:subprocess.run(cmd,check=True,timeout=TIMEOUT,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True);return sid,"OK",str(out),time.time()-t
    except Exception as exc:return sid,type(exc).__name__,repr(exc),time.time()-t
def cosine_distance(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(1-np.dot(a,b)/den) if den>1e-12 else float(np.linalg.norm(a-b)>1e-12)
def standardized_distance(a,b,reference):
    reference=np.asarray(reference,float);scale=np.nanstd(reference,axis=0);scale=np.where(scale>1e-8,scale,1.0)
    return float(np.sqrt(np.mean(((np.asarray(a)-np.asarray(b))/scale)**2)))
def raw_distance(a,b):
    sims={name:sketch_jaccard(a["sketches"][name],b["sketches"][name]) for name in SKETCHES}
    return float(1-np.mean(list(sims.values()))),sims
def percentile(values,x):
    values=np.asarray(values,float)
    if not len(values):return float("nan")
    return float((np.sum(values<x)+.5*np.sum(values==x))/len(values))
def assemble_layers(folds):
    with np.load(STORE,allow_pickle=False) as z:data={k:np.asarray(z[k]) for k in z.files}
    result={}
    for name in ("s7","fused","hidden"):
        width=int(data[f"f1_test_{name}"].shape[1]);result[name]=np.full((len(folds),width),np.nan,float)
    for f in range(1,6):
        vi=np.asarray(data[f"f{f}_test_index"],int)
        if not np.array_equal(vi,np.flatnonzero(folds==f)):raise RuntimeError("BLOCKED_BY_CONTRACT: representation order mismatch")
        for name in result:result[name][vi]=np.asarray(data[f"f{f}_test_{name}"],float)
    if any(not np.isfinite(v).all() for v in result.values()):raise RuntimeError("BLOCKED_BY_CONTRACT: nonfinite layer representation")
    return result,data
def pair_dist(i,j,layers,data,fold,fps):
    f=int(fold[i]);out={}
    rd,sim=raw_distance(fps[i],fps[j]);out["raw_distance"]=rd
    for key,val in sim.items():out[key+"_jaccard"]=val
    for name in ("s7","fused","hidden"):
        ref=np.asarray(data[f"f{f}_train_{name}"],float)
        out[name+"_distance"]=standardized_distance(layers[name][i],layers[name][j],ref)
        out[name+"_cosine_distance"]=cosine_distance(layers[name][i],layers[name][j])
    return out
def choose_nearest(i,pool,layers,data,fold):
    f=int(fold[i]);ref=np.asarray(data[f"f{f}_train_fused"],float)
    return min(pool,key=lambda j:(standardized_distance(layers["fused"][i],layers["fused"][j],ref),j))
def classify(row):
    raw=row["raw_percentile"]<=.25; s7=row["s7_percentile"]<=.25; fused=row["fused_percentile"]<=.25; hidden=row["hidden_percentile"]<=.25
    if row["raw_percentile"]>=.75 and (s7 or fused):return "RAW_TO_C43_COMPRESSION_CANDIDATE"
    if row["fused_percentile"]>=.75 and hidden:return "C43_TO_HIDDEN_COMPRESSION_CANDIDATE"
    if raw and s7 and fused and hidden:return "MULTILAYER_STATIC_COLLISION"
    if row["hidden_percentile"]>=.75:return "HIDDEN_TO_SCORE_READOUT_CANDIDATE"
    return "INCONCLUSIVE"

def main():
    ids,y,fold,base,route,expert,final=load_reference();layers,data=assemble_layers(fold)
    before=(base>=.5)==y;after=(final>=.5)==y
    failures=np.flatnonzero(route&((~before&~after)|(before&~after)))
    correct=np.flatnonzero(route&after)
    if len(failures)!=52:raise RuntimeError("BLOCKED_BY_CONTRACT: expected 52 Direct-C43 failures")
    apk_paths,path_audit=load_apk_paths(set(ids[route]))
    if len(apk_paths)!=int(route.sum()):
        missing=[s for s in ids[route] if s not in apk_paths]
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: routed APK coverage {len(apk_paths)}/200 missing={missing[:10]}")
    dump_json(OUTPUT/"apk_path_audit.json",{"resolved":len(apk_paths),"sources":path_audit,"future_rows_loaded":0})
    jobs=[(sid,apk_paths[sid]) for sid in ids[route]];results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for k,result in enumerate(pool.map(fingerprint_one,jobs),1):
            results.append(result)
            if k%10==0 or k==len(jobs):print(f"CSEF_DEX_FINGERPRINT {k}/{len(jobs)}",flush=True)
    failed=[{"sha256":sid,"status":status,"detail":detail,"wall_seconds":wall} for sid,status,detail,wall in results if status!="OK"]
    write_csv(OUTPUT/"fingerprint_audit.csv",[{"sha256":sid,"status":status,"detail":detail,"wall_seconds":wall} for sid,status,detail,wall in results])
    if failed:raise RuntimeError("BLOCKED_BY_CONTRACT: verified fingerprint failures "+repr(failed[:5]))
    fps_by_sid={sid:load_gz(DEEP/(sid+".json.gz")) for sid in ids[route]}
    fps={i:fps_by_sid[ids[i]] for i in np.flatnonzero(route)}

    # Cross-label, correctly classified pairs form the fold-local reference distribution at every layer.
    calibration={}
    for f in range(1,6):
        left=[i for i in correct if fold[i]==f and y[i]==0];right=[i for i in correct if fold[i]==f and y[i]==1]
        pairs=[pair_dist(i,j,layers,data,fold,fps) for i in left for j in right]
        if not pairs:raise RuntimeError("BLOCKED_BY_CONTRACT: empty fold calibration")
        calibration[f]={name:[p[name+"_distance"] for p in pairs] for name in ("raw","s7","fused","hidden")}

    traces=[];review=[];keys=[];dossier_dir=safe(OUTPUT/"case_dossiers")
    dossier_dir.mkdir(parents=True,exist_ok=True)
    for serial,i in enumerate(failures,1):
        opposite=[j for j in correct if fold[j]==fold[i] and y[j]!=y[i]]
        same=[j for j in correct if fold[j]==fold[i] and y[j]==y[i] and j!=i]
        if not opposite or not same:raise RuntimeError("BLOCKED_BY_CONTRACT: missing matched controls")
        j=choose_nearest(i,opposite,layers,data,fold);k=choose_nearest(i,same,layers,data,fold)
        # Apply the identical nearest-opposite-label rule to the correctly classified anchor.
        opposite_for_control=[q for q in correct if fold[q]==fold[k] and y[q]!=y[k]]
        j2=choose_nearest(k,opposite_for_control,layers,data,fold)
        failure_dist=pair_dist(i,j,layers,data,fold,fps);control_dist=pair_dist(k,j2,layers,data,fold,fps)
        row={"case_id":f"C{serial:03d}","fold":int(fold[i]),"failure_type":"UNFIXED" if not before[i] else "BROKEN",
             "target_sha256":ids[i],"target_label":int(y[i]),"target_stage1_prob":float(base[i]),"target_direct_prob":float(final[i]),
             "opposite_correct_sha256":ids[j],"same_label_correct_sha256":ids[k],"control_opposite_sha256":ids[j2]}
        for name in ("raw","s7","fused","hidden"):
            row[name+"_distance"]=failure_dist[name+"_distance"]
            row[name+"_percentile"]=percentile(calibration[int(fold[i])][name],failure_dist[name+"_distance"])
            row["control_"+name+"_distance"]=control_dist[name+"_distance"]
            row["control_"+name+"_percentile"]=percentile(calibration[int(fold[i])][name],control_dist[name+"_distance"])
        for name in SKETCHES:row[name+"_jaccard"]=failure_dist[name+"_jaccard"]
        row["automated_localization"]=classify(row);traces.append(row)
        blind={"case_id":row["case_id"],"fold":row["fold"],"apk_A_sha256":ids[i],"apk_B_sha256":ids[j],
               "raw_distance":row["raw_distance"],"s7_distance":row["s7_distance"],"c43_input_distance":row["fused_distance"],
               "hidden_distance":row["hidden_distance"],"automated_localization":row["automated_localization"],
               "raw_reliable_distinguishing_fact":"","fact_description":"","fact_present_in_formal_input":"",
               "fact_retained_in_hidden_or_score":"","review_attribution":"","review_confidence":"","review_notes":""}
        review.append(blind)
        keys.append({"case_id":row["case_id"],"apk_A_sha256":ids[i],"apk_A_label":int(y[i]),"apk_A_failure_type":row["failure_type"],
                     "apk_A_stage1_prob":float(base[i]),"apk_A_direct_prob":float(final[i]),"apk_B_sha256":ids[j],"apk_B_label":int(y[j]),
                     "apk_B_direct_prob":float(final[j]),"matched_correct_anchor_sha256":ids[k],"matched_correct_opposite_sha256":ids[j2]})
        dossier={"case_id":row["case_id"],"blinded_apks":{"A":ids[i],"B":ids[j]},"layer_trace":{q:row[q] for q in row if q.endswith(("_distance","_percentile","_jaccard"))},
                 "automated_localization":row["automated_localization"],"raw_dex_audit":{
                    "A":{"method_count":fps[i].get("method_count"),"class_count":fps[i].get("class_count")},
                    "B":{"method_count":fps[j].get("method_count"),"class_count":fps[j].get("class_count")}},
                 "claim_boundary":"Fingerprint differences are structural observations, not automatic benign/malicious semantic facts."}
        dump_json(dossier_dir/(row["case_id"]+".json"),dossier)
    write_csv(OUTPUT/"case_layer_trace.csv",traces);write_csv(OUTPUT/"blind_review_key.csv",keys)
    form=OUTPUT/"blind_review_form.csv"
    if not form.exists():write_csv(form,review)

    names=("RAW_TO_C43_COMPRESSION_CANDIDATE","C43_TO_HIDDEN_COMPRESSION_CANDIDATE","MULTILAYER_STATIC_COLLISION","HIDDEN_TO_SCORE_READOUT_CANDIDATE","INCONCLUSIVE")
    summary=[];mechanisms=[]
    for name in names:
        fail_count=sum(r["automated_localization"]==name for r in traces)
        # Apply the same rule to each matched correct-pair trace.
        controls=[]
        for r in traces:
            c={q.replace("control_",""):v for q,v in r.items() if q.startswith("control_")}
            controls.append(classify(c))
        control_count=sum(v==name for v in controls);diff=fail_count/len(traces)-control_count/len(traces)
        fold_dirs=sum(1 for f in range(1,6) if sum(r["automated_localization"]==name for r in traces if r["fold"]==f)>sum(controls[n]==name for n,r in enumerate(traces) if r["fold"]==f))
        supported=bool(name!="INCONCLUSIVE" and diff>=.15 and fold_dirs>=3)
        if supported:mechanisms.append(name)
        summary.append({"mechanism":name,"failure_count":fail_count,"failure_rate":fail_count/len(traces),"matched_correct_count":control_count,
                        "matched_correct_rate":control_count/len(traces),"rate_difference":diff,"folds_failure_higher":fold_dirs,"descriptively_enriched":supported})
    write_csv(OUTPUT/"mechanism_summary.csv",summary)
    decision={"status":"AUTOMATED_FIDELITY_LOCALIZATION_COMPLETE_MANUAL_SEMANTIC_REVIEW_REQUIRED","source_rows":799,"routed":200,
              "audited_unfixed":35,"audited_broken":17,"audited_total":52,"candidate_mechanisms":mechanisms,
              "future_rows_loaded":0,"labels_used_for_model_training":False,"model_training_authorized":False,
              "semantic_claim_boundary":"Automated structural differences cannot establish that raw DEX contains a reliable class-direction fact. Blind semantic review is required before selecting extraction, representation, or observability explanations.",
              "next_action":"complete blind_review_form.csv without consulting blind_review_key.csv, then run summarize_review.py"}
    dump_json(OUTPUT/"automated_decision.json",decision);dump_json(OUTPUT/"final_decision.json",decision)
    print("C43_STATIC_EVIDENCE_FIDELITY_AUTOMATED_DONE",json.dumps(decision,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
