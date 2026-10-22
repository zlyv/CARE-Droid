#!/usr/bin/env python3
"""Strict-2017 Global24 group utility and cached-compute cost audit."""
from __future__ import annotations
import argparse,csv,json,math,sys,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent;SCRIPTS=HERE.parent
for p in (HERE,SCRIPTS/"care-multiview-conformal-router-source-v1",SCRIPTS/"care-rolegraph-conditioned-v4-script",SCRIPTS/"care-rgcondattn-dedup-bounded-source-v1",SCRIPTS/"care-committee-uncertainty-router-v1"):sys.path.insert(0,str(p))
import run_multiview_conformal_router_source_v1 as mv
from committee_uncertainty_core import fit_calibrated_committee
from feature_contract import ACQUISITION,GROUPS,ROUTER_VARIANTS,STAGE1_VARIANTS,indices

OUT=Path("artifacts/care-global24-utility-cost-source-v1")


def norm(v):return str(v or "").strip().lower()


def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True);fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def clipped_logit(p):
    x=np.clip(np.asarray(p,float),1e-8,1-1e-8);return np.log(x/(1-x))


def make_lr(seed):
    return Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=1.0,penalty="l2",solver="liblinear",max_iter=5000,random_state=int(seed)))])


def calibrated_stage1(xtr,ytr,xval,keep,seed,inner_folds):
    split=StratifiedKFold(n_splits=inner_folds,shuffle=True,random_state=seed);raw=np.full(len(ytr),np.nan)
    for inner,(fit,cal) in enumerate(split.split(xtr,ytr),1):
        model=make_lr(seed+100*inner);model.fit(xtr[fit][:,keep],ytr[fit]);raw[cal]=model.predict_proba(xtr[cal][:,keep])[:,1]
    if not np.isfinite(raw).all():raise RuntimeError("incomplete inner OOF Stage-I")
    calibrator=LogisticRegression(C=1e6,solver="lbfgs",max_iter=2000,random_state=seed+999);calibrator.fit(clipped_logit(raw).reshape(-1,1),ytr)
    model=make_lr(seed+777);model.fit(xtr[:,keep],ytr);target=model.predict_proba(xval[:,keep])[:,1]
    return np.clip(calibrator.predict_proba(clipped_logit(target).reshape(-1,1))[:,1],1e-8,1-1e-8)


def exact_fold_top(score,folds,ids,budget=.25):
    route=np.zeros(len(score),bool)
    for fold in sorted(set(folds.tolist())):
        q=np.flatnonzero(folds==fold);count=int(round(len(q)*budget));order=np.lexsort((ids[q],-score[q]));route[q[order[:count]]]=True
    return route


def mean_opposition(base,member):
    decision=(np.asarray(base)>=.5).astype(float);return np.where(decision[:,None]==1,1-member,member).mean(1)


def exact(a,b):
    n=int(a+b)
    if not n:return 1.0
    k=min(int(a),int(b));return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))


def stage1_metric(name,y,p,reference):
    pred=p>=.5;ref=reference>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();cand_only=int(np.sum((ref!=y)&(pred==y)));ref_only=int(np.sum((ref==y)&(pred!=y)))
    return {"variant":name,"retained_dimensions":24 if name=="FROZEN_OFFICIAL" else None,"BA":float(balanced_accuracy_score(y,pred)),"AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,pred,zero_division=0)),"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),"errors":int(np.sum(pred!=y)),"candidate_only_correct_vs_official":cand_only,"official_only_correct":ref_only,"paired_net_vs_official":cand_only-ref_only,"exact_p_vs_official":exact(cand_only,ref_only)}


def route_metric(name,y,base,route,reference):
    error=(base>=.5)!=y;high=error&(np.maximum(base,1-base)>=.8);captured=route&error;ref_only=int(np.sum(error&reference&~route));cand_only=int(np.sum(error&route&~reference))
    return {"variant":name,"routed":int(route.sum()),"base_errors":int(error.sum()),"captured_errors":int(captured.sum()),"error_recall":float(captured.sum()/error.sum()),"routing_precision":float(captured.sum()/route.sum()),"highconf_errors":int(high.sum()),"captured_highconf_errors":int((route&high).sum()),"highconf_error_recall":float((route&high).sum()/high.sum()) if high.any() else 0.0,"candidate_only_errors_vs_current_DI":cand_only,"current_DI_only_errors":ref_only,"capture_delta_vs_current_DI":cand_only-ref_only,"exact_p_vs_current_DI":exact(cand_only,ref_only),"route_overlap_with_current_DI":int(np.sum(route&reference))}


def main():
    p=argparse.ArgumentParser();p.add_argument("--features",type=Path,default=Path("artifacts/p1-features/features.csv"));p.add_argument("--schema",type=Path,default=Path("artifacts/p1-features/feature_schema.json"));p.add_argument("--p72-oof",type=Path,default=Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"));p.add_argument("--router",type=Path,default=Path("artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv"));p.add_argument("--inner-folds",type=int,default=4);p.add_argument("--seed",type=int,default=20260925);p.add_argument("--jobs",type=int,default=4);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();manifest,source_rows,_,compact_by_sha,compact_schema,original_names=mv.v4.audit.load_inputs(a);loader_seconds=time.perf_counter()-started
    router_rows=read(a.router);router={norm(r["sha256"]):r for r in router_rows};ids=np.asarray([norm(r["sha256"]) for r in manifest]);y=np.asarray([int(float(router[s]["label"])) for s in ids]);folds=np.asarray([int(float(router[s]["fold"])) for s in ids]);official=np.asarray([float(router[s]["p_stage1"]) for s in ids]);current=np.asarray([int(float(router[s]["route_DI"]))==1 for s in ids])
    if len(manifest)!=799 or len(router)!=799 or set(ids)!=set(router) or {str(r.get("metadata_year","")) for r in source_rows}!={"2017"}:raise RuntimeError("BLOCKED_BY_CONTRACT: source-only loader mismatch")
    stage1={v:np.full(799,np.nan) for v in STAGE1_VARIANTS};members={v:np.full((799,4),np.nan) for v in ROUTER_VARIANTS};timing=[];audits=[];canonical=None
    for fold in sorted(set(folds.tolist())):
        train=np.flatnonzero(folds!=fold);val=np.flatnonzero(folds==fold);t0=time.perf_counter();xtr,xval,names=mv.global_matrices([source_rows[i] for i in train],[source_rows[i] for i in val],compact_by_sha,compact_schema,original_names);matrix_seconds=time.perf_counter()-t0
        if canonical is None:canonical=list(names)
        elif list(names)!=canonical:raise RuntimeError("Global24 names changed across folds")
        for vi,variant in enumerate(STAGE1_VARIANTS):
            keep=np.asarray(indices(canonical,variant),np.int64);t0=time.perf_counter();stage1[variant][val]=calibrated_stage1(xtr,y[train],xval,keep,a.seed+100000*fold+1000*vi,a.inner_folds);elapsed=time.perf_counter()-t0;timing.append({"component":"stage1_L2_calibrated","variant":variant,"fold":fold,"train_n":len(train),"validation_n":len(val),"retained_dimensions":len(keep),"wall_seconds":elapsed})
        for vi,variant in enumerate(ROUTER_VARIANTS):
            keep=np.asarray(indices(canonical,variant),np.int64);mask=np.zeros(24,bool);mask[keep]=True;mtr=xtr.copy();mval=xval.copy();mtr[:,~mask]=0;mval[:,~mask]=0;t0=time.perf_counter();fit=fit_calibrated_committee(mtr,y[train],mval,inner_folds=a.inner_folds,split_seed=a.seed+200000*fold+1000*vi,model_seed=a.seed+300000*fold+1000*vi,jobs=a.jobs);elapsed=time.perf_counter()-t0;members[variant][val]=fit.target_probability;timing.append({"component":"LR_RF_XGB_SVM_calibrated_committee","variant":variant,"fold":fold,"train_n":len(train),"validation_n":len(val),"retained_dimensions":len(keep),"wall_seconds":elapsed});audits.extend({"variant":variant,"outer_fold":fold,**row} for row in fit.audit)
        timing.append({"component":"cached_Global24_matrix_build","variant":"ALL","fold":fold,"train_n":len(train),"validation_n":len(val),"retained_dimensions":24,"wall_seconds":matrix_seconds});print(f"Global24 utility outer fold {fold}/5 done",flush=True)
    if any(not np.isfinite(v).all() for v in stage1.values()) or any(not np.isfinite(v).all() for v in members.values()):raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete OOF output")
    stage_rows=[stage1_metric("FROZEN_OFFICIAL",y,official,official)]
    for variant in STAGE1_VARIANTS:
        row=stage1_metric(variant,y,stage1[variant],official);row["retained_dimensions"]=len(indices(canonical,variant));stage_rows.append(row)
    route_rows=[route_metric("CURRENT_FROZEN_DI25",y,official,current,current)];sample=[]
    routes={}
    for variant in ROUTER_VARIANTS:
        score=mean_opposition(official,members[variant]);route=exact_fold_top(score,folds,ids);routes[variant]=route;route_rows.append(route_metric(variant,y,official,route,current))
    for i,sid in enumerate(ids):
        row={"sha256":sid,"year":2017,"fold":int(folds[i]),"label":int(y[i]),"p_stage1_official":float(official[i]),"stage1_error":int((official[i]>=.5)!=y[i]),"route_current_DI25":int(current[i])}
        for variant in STAGE1_VARIANTS:row["p_stage1__"+variant]=float(stage1[variant][i])
        for variant in ROUTER_VARIANTS:row["route__"+variant]=int(routes[variant][i]);row["U_DI__"+variant]=float(mean_opposition(official,members[variant])[i])
        sample.append(row)
    provenance=[]
    for group,features in GROUPS.items():
        stage,graph_required,note=ACQUISITION[group]
        for feature in features:
            if feature=="method_count":feature_stage="dex_scan";feature_graph=False
            elif group=="code_call":feature_stage="callgraph";feature_graph=True
            else:feature_stage=stage;feature_graph=graph_required
            provenance.append({"feature":feature,"group":group,"acquisition_stage":feature_stage,"requires_callgraph_or_semantic_graph":int(feature_graph),"classification_basis":"feature-name/upstream-contract inference; not runtime measurement","note":note})
    artifact_rows=[]
    for path in (a.features,a.schema,a.p72_oof,a.router):artifact_rows.append({"path":str(path),"bytes":path.stat().st_size if path.is_file() else "unavailable"})
    write(OUT/"stage1_group_ablation.csv",stage_rows);write(OUT/"router_group_ablation.csv",route_rows);write(OUT/"sample_predictions_and_routes.csv",sample);write(OUT/"timing_cached_compute.csv",timing);write(OUT/"committee_fit_audit.csv",audits);write(OUT/"feature_acquisition_provenance.csv",provenance);write(OUT/"artifact_sizes.csv",artifact_rows)
    timing_summary=[]
    for component in sorted({r["component"] for r in timing}):
        values=[float(r["wall_seconds"]) for r in timing if r["component"]==component];timing_summary.append({"component":component,"measurements":len(values),"total_wall_seconds_serial_sum":sum(values),"mean_wall_seconds":float(np.mean(values)),"median_wall_seconds":float(np.median(values)),"p90_wall_seconds":float(np.quantile(values,.9)),"scope":"cached feature/model compute only"})
    timing_summary.append({"component":"source_asset_loader","measurements":1,"total_wall_seconds_serial_sum":loader_seconds,"mean_wall_seconds":loader_seconds,"median_wall_seconds":loader_seconds,"p90_wall_seconds":loader_seconds,"scope":"cached asset loading only"});timing_summary.append({"component":"raw_APK_to_Global24_extraction","measurements":0,"total_wall_seconds_serial_sum":"unavailable","mean_wall_seconds":"unavailable","median_wall_seconds":"unavailable","p90_wall_seconds":"unavailable","scope":"not rerun; no trustworthy phase timing available"});write(OUT/"timing_summary.csv",timing_summary)
    s={r["variant"]:r for r in stage_rows};q={r["variant"]:r for r in route_rows};lines=["CARE-Droid strict-2017 Global24 utility and cost audit v1","","Source-only five-fold OOF. No future row, Stage-II training, route blending, or threshold search.","Official Stage-I remains the frozen Global24 L2 prediction; router ablations alter committee inputs only.","","Stage-I group ablation:"]
    for variant in ["FROZEN_OFFICIAL",*STAGE1_VARIANTS]:
        r=s[variant];lines.append(f"{variant}: dims={r['retained_dimensions']} BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} errors={r['errors']} paired_net_vs_official={r['paired_net_vs_official']}")
    lines += ["","DI25 committee-input group ablation (fixed official Stage-I decisions):"]
    for variant in ["CURRENT_FROZEN_DI25",*ROUTER_VARIANTS]:
        r=q[variant];lines.append(f"{variant}: captured={r['captured_errors']}/{r['base_errors']} recall={r['error_recall']:.6f} precision={r['routing_precision']:.6f} highconf_recall={r['highconf_error_recall']:.6f} delta_vs_current={r['capture_delta_vs_current_DI']}")
    lines += ["","Cost boundary:","Classifier/committee timings are measured from cached Global24 on this run.","Raw APK parsing, code scan, callgraph construction, sensitive mapping, and Global24 extraction were not rerun; their phase times remain unavailable.",f"Contract provenance classifies {sum(int(r['requires_callgraph_or_semantic_graph']) for r in provenance)}/24 features as requiring callgraph or semantic-graph evidence.","Feature usefulness is established at group level only; this audit does not prove every individual dimension is necessary."]
    (OUT/"REPORT.txt").write_text("\n".join(lines)+"\n");(OUT/"protocol.json").write_text(json.dumps({"source_only":True,"source_rows":799,"year":2017,"future_rows_loaded":0,"stage2_training_invoked":False,"outer_folds":5,"inner_calibration_folds":a.inner_folds,"stage1_variants":list(STAGE1_VARIANTS),"router_variants":list(ROUTER_VARIANTS),"router_budget":.25,"router_stage1_decision":"frozen official p_stage1","committee":["LR","RF","XGB","SVM"],"raw_extraction_timing":"unavailable_not_rerun"},indent=2)+"\n")
    print("GLOBAL24_UTILITY_COST_SOURCE_AUDIT_DONE");print("\n".join(lines))


if __name__=="__main__":main()
