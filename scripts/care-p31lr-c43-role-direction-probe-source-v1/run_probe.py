#!/usr/bin/env python3
from __future__ import annotations
import csv,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent;SCRIPTS=PROJECT/"scripts"
for p in (HERE,SCRIPTS,SCRIPTS/"care-stage2only-anomaly-v1/vendor",SCRIPTS/"care-di25-compact43-relationquery-source-v1",SCRIPTS/"care-di25-graphglobal-path-source-v1",SCRIPTS/"care-rolegraph-conditioned-v4-script",SCRIPTS/"care-rgcondattn-dedup-bounded-source-v1",SCRIPTS/"care-malben-host-v2"):
    v=str(p.resolve())
    if v not in sys.path:sys.path.insert(0,v)
import run_care_rolegraph_conditioned_dualstream_v4 as v4
import compact_background as cb
from direction_core import *

ROOT=Path("artifacts/care-p31lr-c43-role-direction-probe-source-v1")
UP=Path("artifacts/care-litelocal-p31lr-c43-prov6-source-v1")
REF=UP/"preflight/lr_reference/source_predictions.csv";PRED=UP/"arms/P0_ZERO6/source_predictions.csv";CACHE=UP/"cache/P0_ZERO6.npz"
METHOD="P31LR__Opp25__C43Prov6Zero__Bounded"
ARGS=SimpleNamespace(shared_prototypes=32,benign_prototypes=24,prototype_regions_per_apk=16,internal_background_coverage=.70)
def read(p):
    with p.open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write(p,rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def logit(p):p=np.clip(np.asarray(p,float),1e-8,1-1e-8);return np.log(p/(1-p))
def load_cache(ids):
    with np.load(CACHE,allow_pickle=False) as z:
        zids=[str(v).strip().casefold() for v in z["sha256"]];values=np.asarray(z["host_values"],np.float32);off=np.asarray(z["host_offsets"],int);real=np.asarray(z["real_host_count"],int)
    idx={s:i for i,s in enumerate(zids)};bags=[];roles=[]
    for sha in ids:
        i=idx[sha];lo=int(off[i]);n=int(real[i]);x=values[lo:lo+max(n,1),:105]
        if x.shape[1]!=105:raise RuntimeError("BLOCKED_BY_CONTRACT: expected Compact43+Role31+Relation31 width 105")
        bags.append(x);roles.append(x[:,43:74])
    return bags,roles,real[[idx[s] for s in ids]]
def background_for_fold(train_bags,val_bags,labels,fold):
    rt,tr,_=cb.split_bags(train_bags);rv,vr,_=cb.split_bags(val_bags);mean,std=v4.v3.care_v1.fit_equal_apk_scaler(rt,np.arange(len(rt),dtype=np.int64));st=cb.weighted_compact(rt,mean,std);sv=cb.weighted_compact(rv,mean,std)
    # external, current-internal, commonness, current-residual,
    # leave-one-host-out internal, leave-one-host-out residual
    accumulated=[np.zeros((len(x),6),float) for x in sv]
    seed_audit=[]
    for seed in (0,1,2):
        fullseed=100000*fold+seed;state=v4.v3.v2.fit_prototype_state(st,labels,ARGS,fullseed);sp,bp=cb.prototype_profiles(st,tr,labels,state,v4.v3.v2)
        for j,(x,w) in enumerate(zip(sv,vr)):
            c=cb.background_columns(x,w,state,sp,bp,ARGS,v4.v3.v2);sd=v4.v3.v2.pairwise_rms(x,state["shared_centers"]);sa=cb.hard_assignment(sd/np.maximum(state["shared_scale"][None,:],1e-6),w,sp);current_check,loo=internal_current_loo(x,sa,ARGS.internal_background_coverage)
            mismatch=float(np.max(np.abs(current_check-c[:,1])))
            if mismatch>1e-5:raise RuntimeError(f"BLOCKED_BY_CONTRACT: current internal reconstruction mismatch max_abs_diff={mismatch:.12g}")
            loo_res=(1-np.clip(c[:,2],0,1))*np.sqrt(np.maximum(c[:,0],0)*np.maximum(loo,0));accumulated[j]+=np.column_stack([c[:,:3],c[:,3],loo,loo_res])
        seed_audit.append({"fold":fold,"seed":fullseed,"shared_n":state["shared_k"],"benign_n":state["benign_k"],"train_only":1})
    return [x/3 for x in accumulated],st,sv,tr,vr,mean,std,seed_audit
def main():
    ref=sorted(read(REF),key=lambda r:r["sha256"]);pred=sorted(read(PRED),key=lambda r:r["sha256"]);ids=[r["sha256"].strip().casefold() for r in ref]
    if ids!=[r["sha256"].strip().casefold() for r in pred]:raise RuntimeError("BLOCKED_BY_CONTRACT: prediction cohort mismatch")
    y=np.asarray([int(r["label"]) for r in ref]);folds=np.asarray([int(r["fold"]) for r in ref]);p=np.asarray([float(r["P31_LR__prob"]) for r in ref]);route=np.asarray([int(r["A25__routed"])==1 for r in ref]);stage=(p>=.5).astype(int);error=route&(stage!=y);truth=np.where(y==1,1,-1)
    expert=np.asarray([float(r[METHOD+"__expert_prob"]) for r in pred]);e0=logit(expert)-logit(p)
    fn_n=int(np.sum(error&(y==1)));fp_n=int(np.sum(error&(y==0)))
    if len(ref)!=799 or int(route.sum())!=200 or int(error.sum())!=99 or fn_n+fp_n!=99:raise RuntimeError("BLOCKED_BY_CONTRACT: current P31-LR/DI25 accounting changed")
    bags,roles,real=load_cache(ids);apk={};host_rows=[];role_audit=[];fold_audit=[]
    for fold in range(1,6):
        train=np.flatnonzero(folds!=fold);val=np.flatnonzero((folds==fold)&route)
        bg,scaled_train,scaled_val,train_roles,val_roles,mean,std,seeds=background_for_fold([bags[i] for i in train],[bags[i] for i in val],y[train],fold)
        models,ra=fit_role_models(scaled_train,train_roles,y[train]);role_audit.extend({"fold":fold,**r} for r in ra);fold_audit.append({"fold":fold,"outer_train_n":len(train),"routed_validation_n":len(val),"validation_labels_used_for_reference":0,"supported_roles":sum(m["supported"] for m in models.values()),"scaler_zero_std_n":int(np.sum(std==1))});fold_audit.extend(seeds)
        for local,i in enumerate(val):
            db,dm,s,support,valid=host_signed_evidence(scaled_val[local],val_roles[local],models);q=bg[local];mean_s=aggregate(s);max_s=aggregate(s,mode="maxabs");e2=aggregate(s,q[:,3]);e3=aggregate(s,q[:,5])
            apk[i]={"sha256":ids[i],"fold":fold,"label":int(y[i]),"stage1_prob":float(p[i]),"stage1_pred":int(stage[i]),"route":1,"stage1_error":int(stage[i]!=y[i]),"error_type":"FN" if y[i]==1 and stage[i]==0 else "FP" if y[i]==0 and stage[i]==1 else "CORRECT","truth_sign":int(truth[i]) if stage[i]!=y[i] else 0,"host_n":len(s),"valid_direction_host_n":int(valid.sum()),"role_conditioned_d_ben":float(np.mean(db)) if len(db) else 0.,"role_conditioned_d_mal":float(np.mean(dm)) if len(dm) else 0.,"external_evidence":float(np.mean(q[:,0])),"internal_evidence":float(np.mean(q[:,1])),"shared_commonness":float(np.mean(q[:,2])),"current_reliability":float(np.mean(q[:,3])),"loo_internal_evidence":float(np.mean(q[:,4])),"loo_reliability":float(np.mean(q[:,5])),"E0_CURRENT_DELTA":float(e0[i]),"E1_ROLE_MEAN":mean_s,"E1_ROLE_MAXABS":max_s,"E2_CURRENT_RELIABILITY":e2,"E3_LOO_RELIABILITY":e3}
            for h in range(len(s)):
                host_rows.append({"sha256":ids[i],"fold":fold,"host_index":h,"label":int(y[i]),"stage1_pred":int(stage[i]),"error_type":apk[i]["error_type"],"truth_sign":apk[i]["truth_sign"],"primary_role":int(primary_roles(val_roles[local])[h]),"role_support_n":int(support[h]),"role_supported":int(valid[h]),"d_ben":float(db[h]),"d_mal":float(dm[h]),"signed_score":float(s[h]),"external":float(q[h,0]),"internal_current":float(q[h,1]),"shared_commonness":float(q[h,2]),"reliability_current":float(q[h,3]),"internal_loo":float(q[h,4]),"reliability_loo":float(q[h,5])})
        print(f"direction probe fold {fold}/5 done",flush=True)
    rows=[apk[i] for i in sorted(apk)];methods=("E0_CURRENT_DELTA","E1_ROLE_MEAN","E1_ROLE_MAXABS","E2_CURRENT_RELIABILITY","E3_LOO_RELIABILITY")
    for r in rows:
        for m in methods:
            aligned=float(r[m])*int(r["truth_sign"]) if r["stage1_error"] else 0.;direction=int(classify_direction([aligned])[0]);r[m+"__aligned_score"]=aligned;r[m+"__direction_correct"]=int(direction>0) if r["stage1_error"] else "";r[m+"__taxonomy"]="CorrectDirection" if direction>0 else "WrongDirection" if direction<0 else "Ambiguous"
    errors=[r for r in rows if r["stage1_error"]];quality=[];taxonomy=[];perfold=[]
    for m in methods:
        aligned=np.asarray([r[m+"__aligned_score"] for r in errors]);fn=np.asarray([r["error_type"]=="FN" for r in errors]);d=classify_direction(aligned);fold_acc=[]
        for f in range(1,6):
            sub=[r for r in errors if r["fold"]==f];fold_acc.append(np.mean([r[m+"__direction_correct"] for r in sub]))
        quality.append({"Method":m,"Overall_direction_acc":float(np.mean(d>0)),"FN_direction_acc":float(np.mean(d[fn]>0)),"FP_direction_acc":float(np.mean(d[~fn]>0)),"Mean_aligned":float(np.mean(aligned)),"Median_aligned":float(np.median(aligned)),"Mean_abs_score":float(np.mean(np.abs(aligned))),"FN_mean_aligned":float(np.mean(aligned[fn])),"FP_mean_aligned":float(np.mean(aligned[~fn])),"Fold_std":float(np.std(fold_acc,ddof=0))});taxonomy.append({"Method":m,"WrongDirection":int(np.sum(d<0)),"Ambiguous":int(np.sum(d==0)),"CorrectDirection":int(np.sum(d>0))})
    for f in range(1,6):
        sub=[r for r in errors if r["fold"]==f];item={"Fold":f,"Routed_errors":len(sub),"FN":sum(r["error_type"]=="FN" for r in sub),"FP":sum(r["error_type"]=="FP" for r in sub)}
        for m in methods:item[m+"_acc"]=float(np.mean([r[m+"__direction_correct"] for r in sub]))
        perfold.append(item)
    q={r["Method"]:r for r in quality};e0q=q["E0_CURRENT_DELTA"];eligible=[]
    for m in ("E1_ROLE_MEAN","E1_ROLE_MAXABS","E2_CURRENT_RELIABILITY","E3_LOO_RELIABILITY"):
        r=q[m];positive_folds=sum(next(x for x in perfold if x["Fold"]==f)[m+"_acc"]>=next(x for x in perfold if x["Fold"]==f)["E0_CURRENT_DELTA_acc"] for f in range(1,6));
        r["Noninferior_folds_vs_E0"]=positive_folds
        if r["FN_direction_acc"]>=e0q["FN_direction_acc"] and r["FP_direction_acc"]>e0q["FP_direction_acc"] and r["Overall_direction_acc"]>e0q["Overall_direction_acc"] and positive_folds>=3:eligible.append(m)
    # Predeclared source-only tie break: overall direction, FP direction, FN direction,
    # then lower fold dispersion.  No future metric participates.
    decision=max(eligible,key=lambda m:(q[m]["Overall_direction_acc"],q[m]["FP_direction_acc"],q[m]["FN_direction_acc"],-q[m]["Fold_std"])) if eligible else "NONE"
    masking=[]
    for cohort,subset in (("ALL_ROUTED",rows),("ROUTED_ERRORS",errors),("FN",[r for r in errors if r["error_type"]=="FN"]),("FP",[r for r in errors if r["error_type"]=="FP"])):
        cur=np.asarray([r["internal_evidence"] for r in subset]);loo=np.asarray([r["loo_internal_evidence"] for r in subset]);masking.append({"cohort":cohort,"n":len(subset),"current_internal_mean":float(cur.mean()),"loo_internal_mean":float(loo.mean()),"loo_minus_current_mean":float((loo-cur).mean()),"loo_gt_current_fraction":float(np.mean(loo>cur+ZERO_TOL))})
    out=ROOT;write(out/"direction_quality.csv",quality);write(out/"per_fold_direction.csv",perfold);write(out/"failure_taxonomy.csv",taxonomy);write(out/"routed_apk_evidence.csv",rows);write(out/"routed_host_evidence.csv",host_rows);write(out/"role_reference_audit.csv",role_audit);write(out/"fold_audit.csv",fold_audit);write(out/"internal_masking_audit.csv",masking)
    protocol={"source_only":True,"source_rows":799,"year":2017,"stage1":"P31 calibrated LR only","committee_role":"routing only","router":"LR-anchored MeanOpposition Top25%","route":200,"routed_errors":99,"FN":fn_n,"FP":fp_n,"reference":"outer-train-only APK-role-equal-weighted class-conditional centroids","distance":"standardized diagonal shrinkage distance","min_class_role_support":MIN_CLASS_ROLE_SUPPORT,"unsupported_role_fallback":"exclude as ambiguous; never use cross-role center","shrinkage":SHRINKAGE,"E2_sign_source":"E1 only; current backgrounds are nonnegative weights","E3_sign_source":"E1 only; LOO internal changes nonnegative weights only","host_attention_artifact_available":False,"attention_gap_resolution":"no frozen per-Host alpha exists; use explicit mean/maxabs/current-reliability/LOO-reliability aggregations without neural replay","historical_malben_loaded":False,"historical_malben_reason":"different Stage-I/route cohort, therefore not an exact paired control","neural_training_invoked":False,"future_rows_loaded":0,"advance_candidate":decision};(out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    lines=["CARE-Droid P31-LR role-conditioned class-direction probe source v1","",f"Strict 2017 OOF; routed=200 errors=99 FN={fn_n} FP={fp_n}. No neural training or future row.","Positive signed evidence always means malware; backgrounds only weight reliability.",""]
    for r in quality:lines.append(f"{r['Method']}: overall={r['Overall_direction_acc']:.6f} FN={r['FN_direction_acc']:.6f} FP={r['FP_direction_acc']:.6f} mean_aligned={r['Mean_aligned']:.6f} median={r['Median_aligned']:.6f} fold_std={r['Fold_std']:.6f}")
    lines += ["",f"DIRECTION_PROBE_ADVANCE_CANDIDATE={decision}","If NONE, do not modify or retrain C43.","Existing per-Host C43 attention was not frozen; no neural replay was used to fabricate it."]
    (out/"REPORT.txt").write_text("\n".join(lines)+"\n");print("P31LR_C43_ROLE_DIRECTION_PROBE_DONE");print("\n".join(lines))
if __name__=="__main__":main()
