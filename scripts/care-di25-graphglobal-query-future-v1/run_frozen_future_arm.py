#!/usr/bin/env python3
"""Fit one frozen 2017-only N0/Q arm and write label-free future probabilities."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent
MATRIX=PROJECT/"scripts/care-route-stage2-future-matrix-v1"
for p in (HERE,MATRIX,MATRIX/"vendor"):
    sys.path.insert(0,str(p))
import run_future_arm as matrix
from common import ROOT, SOURCE_ROOT, norm, sha256, write_csv
from graphglobal_path_models import ARMS, make_graphglobal_path_class

v4=matrix.v4

def parser():
    p=matrix.parser()
    for action in p._actions:
        if action.dest in {"representation","head","output"}: action.required=False
    p.set_defaults(representation="anomaly",head="bounded",output=None)
    p.add_argument("--arm",required=True,choices=tuple(ARMS))
    return p

def main():
    args=parser().parse_args()
    if args.representation!="anomaly" or args.head!="bounded" or float(args.max_delta)!=2.25:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen anomaly/bounded/max_delta contract changed")
    seeds=v4.parse_ints(args.seeds)
    if seeds!=[900000,900001,900002]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: deployment seeds changed")
    args.output=Path(args.output) if args.output else ROOT/"arms"/args.arm
    args.output.mkdir(parents=True,exist_ok=True)
    protected=[SOURCE_ROOT/"arms"/args.arm/"source_predictions.csv",SOURCE_ROOT/"results/summary.json"]
    before={str(p):sha256(p) for p in protected}
    data=matrix.load_common(args)
    x_source,x_target,names=v4.audit.compact_matrix_variant(v4.VARIANT,data.source_rows,data.target_rows,data.compact_by_sha,data.schema,data.names)
    global_index,_=v4.evidence.split_view_indices(names)
    global_source=np.asarray(x_source[:,global_index],dtype=np.float32)
    global_target=np.asarray(x_target[:,global_index],dtype=np.float32)
    if global_source.shape!=(799,24) or global_target.shape!=(1989,24):
        raise RuntimeError("Global24 shape contract failed")
    v4.AUGMENTED_REGION_DIM=matrix.ORIGINAL_AUGMENTED_DIM
    v4.v3.fit_scaled_dualproto_bags=matrix.ORIGINAL_ANOMALY_FIT
    v4.RoleGraphConditionedEvidence=make_graphglobal_path_class(v4,args.arm)
    v4.predict_model=matrix.ORIGINAL_V4_PREDICT
    device=v4.resolve_device(args.device)
    started=time.perf_counter()
    probability,diagnostic,audits=v4.fit_predict_variant(
        global_source,global_target,data.source_s7,data.target_s7,
        data.source_bags,data.target_bags,data.base_source,data.base_target,
        data.source_y,"RGCondAttn",seeds,args,device)
    elapsed=time.perf_counter()-started
    probability=np.asarray(probability,dtype=np.float64)
    if probability.shape!=(1989,) or not np.isfinite(probability).all():
        raise RuntimeError("future output is incomplete/non-finite")
    parameters={int(r["parameter_n"]) for r in audits}
    if parameters!={33036}: raise RuntimeError("parameter count changed: "+repr(parameters))
    rows=[{
        "sha256":norm(data.target_rows[i]["sha256"]),"year":int(data.future_input[i]["year"]),
        "arm":args.arm,"p_expert":float(probability[i]),"real_host_count":int(data.target_count[i]),
        "diagnostic_only":1,
    } for i in range(1989)]
    write_csv(args.output/"future_predictions_label_free.csv",rows)
    write_csv(args.output/"fit_audit.csv",audits)
    after={str(p):sha256(p) for p in protected}
    if before!=after: raise RuntimeError("locked source artifacts changed")
    protocol={
        "arm":args.arm,"source_train_n":799,"future_n":1989,"router_applied_during_model_fit":False,
        "source_only_training":True,"future_labels_loaded_by_model":False,
        "future_labels_used_for_training_selection_or_early_stopping":False,
        "local_representation":"Host-only","ego_consumed":False,
        "graphglobal_to_attention":args.arm=="Q_QUERY_ONLY","graphglobal_to_delta_direct":False,
        "correction_input_width":113,"parameter_count":33036,"max_delta_logit":2.25,
        "seeds":seeds,"training_seconds_including_source_only_epoch_selection":elapsed,
        "protected_source_sha256_before":before,"protected_source_sha256_after":after,
        "source_s7_audit":data.source_s7_audit,"future_inter_audit":data.future_inter_audit,
    }
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2,default=str)+"\n",encoding="utf-8")
    print(f"FROZEN_GRAPHGLOBAL_FUTURE_ARM_DONE arm={args.arm} seconds={elapsed:.3f}",flush=True)

if __name__=="__main__": main()
