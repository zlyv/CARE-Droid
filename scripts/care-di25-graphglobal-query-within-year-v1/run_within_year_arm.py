#!/usr/bin/env python3
"""Run one frozen N0/Q arm under independent same-year five-fold OOF."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent
WITHIN=PROJECT/"scripts/care-route-stage2-within-year-matrix-v1"
MODEL=PROJECT/"scripts/care-di25-graphglobal-query-future-v1"
for p in (HERE,MODEL,WITHIN,WITHIN/"vendor"):sys.path.insert(0,str(p))
import run_within_year_arm as within
from graphglobal_path_models import ARMS,make_graphglobal_path_class

v4=within.v4
VERSION="graphglobal-query-within-year-foldsafe-v1"

def parser():
    p=within.future.parser()
    for action in p._actions:
        if action.dest in {"representation","head","output"}:action.required=False
    p.set_defaults(
        representation="anomaly",head="bounded",output=None,
        resolved=Path("artifacts/care-route-stage2-within-year-matrix-v1/preflight/resolved_inputs.json"),
        future_inter_cache=Path("artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz"),
    )
    p.add_argument("--router-inputs",type=Path,default=Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv"))
    p.add_argument("--arm",required=True,choices=tuple(ARMS))
    return p

def main():
    args=parser().parse_args()
    if args.arm not in {"N0_NO_GRAPHGLOBAL","Q_QUERY_ONLY"}:raise RuntimeError("unexpected arm")
    if args.representation!="anomaly" or args.head!="bounded" or float(args.max_delta)!=2.25:raise RuntimeError("frozen model contract changed")
    seeds=v4.parse_ints(args.seeds)
    if seeds!=[900000,900001,900002]:raise RuntimeError("frozen seed ensemble changed")
    args.output=Path(args.output) if args.output else Path("artifacts/care-di25-graphglobal-query-within-year-v1/arms")/args.arm
    args.output.mkdir(parents=True,exist_ok=True)
    device=v4.resolve_device(args.device)
    data=within.load_data(args)
    v4.AUGMENTED_REGION_DIM=within.future.ORIGINAL_AUGMENTED_DIM
    v4.v3.fit_scaled_dualproto_bags=within.future.ORIGINAL_ANOMALY_FIT
    v4.RoleGraphConditionedEvidence=make_graphglobal_path_class(v4,args.arm)
    v4.predict_model=within.future.ORIGINAL_V4_PREDICT
    output=np.full(1989,np.nan,dtype=np.float64)
    checkpoint=args.output/"within_year_checkpoint.csv"
    audit_checkpoint=args.output/"fit_audit_checkpoint.csv"
    audits=within.read_csv(audit_checkpoint) if audit_checkpoint.is_file() else []
    if checkpoint.is_file():
        saved=within.read_csv(checkpoint);by_id={within.norm(r["sha256"]):r for r in saved}
        if len(saved)!=len(by_id):raise RuntimeError("duplicate checkpoint SHA")
        versions={r.get("version","") for r in saved};arms={r.get("arm","") for r in saved}
        if versions=={VERSION} and arms=={args.arm}:
            for i,sha in enumerate(data.ids):
                if sha in by_id:output[i]=float(by_id[sha]["p_expert"])
            print(f"[resume] {args.arm} restored {int(np.isfinite(output).sum())}/1989",flush=True)
        else:
            print("[invalidate] checkpoint belongs to another version/arm",flush=True);audits=[]
    started=time.perf_counter()
    for year in sorted(set(data.years.tolist())):
        for fold in range(1,6):
            validation=np.flatnonzero((data.years==year)&(data.folds==fold))
            train=np.flatnonzero((data.years==year)&(data.folds!=fold))
            if not len(validation) or len(np.unique(data.labels[train]))!=2:raise RuntimeError(f"invalid same-year split {year}/{fold}")
            if np.isfinite(output[validation]).all():
                print(f"[resume] {args.arm} {year} fold {fold}/5 complete",flush=True);continue
            x_train,x_validation=within.fold_global24(data,train,validation)
            base_train=within.inner_oof_base(data,train,int(args.split_seed)+year*10+fold)
            probability,_,fold_audits=v4.fit_predict_variant(
                x_train,x_validation,data.s7[train],data.s7[validation],
                within.subset_bags(data.bags,train),within.subset_bags(data.bags,validation),
                base_train,data.base[validation],data.labels[train],"RGCondAttn",seeds,args,device)
            probability=np.asarray(probability,dtype=np.float64)
            if probability.shape!=(len(validation),) or not np.isfinite(probability).all():raise RuntimeError(f"incomplete fold output {year}/{fold}")
            params={int(row["parameter_n"]) for row in fold_audits}
            if params!={33036}:raise RuntimeError(f"parameter count changed {year}/{fold}: {params}")
            output[validation]=probability
            audits.extend({"arm":args.arm,"year":year,"outer_fold":fold,**row} for row in fold_audits)
            complete=np.flatnonzero(np.isfinite(output))
            rows=[{"sha256":data.ids[i],"year":int(data.years[i]),"fold":int(data.folds[i]),"arm":args.arm,"p_expert":float(output[i]),"version":VERSION} for i in complete]
            tmp=checkpoint.with_suffix(".tmp");within.write_csv(tmp,rows);tmp.replace(checkpoint)
            atmp=audit_checkpoint.with_suffix(".tmp");within.write_csv(atmp,audits);atmp.replace(audit_checkpoint)
            print(f"{args.arm} same-year {year} fold {fold}/5 done",flush=True)
    elapsed=time.perf_counter()-started
    if not np.isfinite(output).all():raise RuntimeError("incomplete same-year OOF predictions")
    params={int(float(r["parameter_n"])) for r in audits if r.get("parameter_n") not in (None,"")}
    if params!={33036}:raise RuntimeError("final parameter audit failed "+repr(params))
    rows=[{"sha256":data.ids[i],"year":int(data.years[i]),"label":int(data.labels[i]),"fold":int(data.folds[i]),"arm":args.arm,"p_expert":float(output[i]),"real_host_count":int(data.counts[i]),"same_year_oof":1,"version":VERSION} for i in range(1989)]
    within.write_csv(args.output/"within_year_predictions.csv",rows);within.write_csv(args.output/"fit_audit.csv",audits)
    protocol={"arm":args.arm,"protocol":"independent same-year five-fold OOF","years":[2018,2019,2020,2021],
        "labels_used_only_within_each_year_outer_train":True,"outer_validation_labels_used_for_training_or_epoch_selection":False,
        "inner_oof_base_on_outer_train_only":True,"router":"pre-existing same-year OOF DI25","router_not_used_to_train_expert":True,
        "host_only":True,"ego_consumed":False,"parameter_count":33036,"max_delta_logit":2.25,"seeds":seeds,
        "graphglobal_to_attention":args.arm=="Q_QUERY_ONLY","graphglobal_to_delta_direct":False,
        "temporal_generalization_result":False,"training_seconds_this_run":elapsed,"future_inter_audit":data.inter_audit}
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2,default=str)+"\n",encoding="utf-8")
    print(f"GRAPHGLOBAL_QUERY_WITHIN_YEAR_ARM_DONE arm={args.arm} seconds={elapsed:.3f}",flush=True)

if __name__=="__main__":main()
