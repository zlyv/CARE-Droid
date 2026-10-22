#!/usr/bin/env python3
"""Fit on 2017 only and emit label-free 2018-2021 Host-relative probabilities."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent;MATRIX=PROJECT/"scripts/care-route-stage2-future-matrix-v1";MODEL=PROJECT/"scripts/care-di25-graphglobal-path-source-v1";SOURCE=PROJECT/"scripts/care-di25-host-relative-source-v1"
for p in (HERE,SOURCE,MODEL,MATRIX,MATRIX/"vendor"):sys.path.insert(0,str(p))
import run_future_arm as matrix
from graphglobal_path_models import make_graphglobal_path_class
from host_relative_features import ARMS
v4=matrix.v4;ROOT=Path("artifacts/care-di25-host-relative-diagnostics-v1")
def parser():
    p=matrix.parser()
    for a in p._actions:
        if a.dest in {"representation","head","output"}:a.required=False
    p.set_defaults(representation="anomaly",head="bounded",output=None);p.add_argument("--arm",required=True,choices=ARMS);return p
def main():
    a=parser().parse_args();seeds=v4.parse_ints(a.seeds)
    if a.representation!="anomaly" or a.head!="bounded" or float(a.max_delta)!=2.25 or seeds!=[900000,900001,900002]:raise RuntimeError("BLOCKED_BY_CONTRACT: frozen protocol changed")
    a.output=Path(a.output) if a.output else ROOT/"frozen"/a.arm;a.output.mkdir(parents=True,exist_ok=True)
    data=matrix.load_common(a)
    source_cache=Path("artifacts/care-di25-host-relative-source-v1/caches")/(a.arm+"_regions_2017.npz")
    future_cache=ROOT/"caches"/(a.arm+"_regions_2018_2021.npz")
    data.source_bags,data.source_count=matrix.load_bags(source_cache,data.source_rows);data.target_bags,data.target_count=matrix.load_bags(future_cache,data.target_rows)
    x0,x1,names=v4.audit.compact_matrix_variant(v4.VARIANT,data.source_rows,data.target_rows,data.compact_by_sha,data.schema,data.names);gi,_=v4.evidence.split_view_indices(names);g0=np.asarray(x0[:,gi],np.float32);g1=np.asarray(x1[:,gi],np.float32)
    v4.AUGMENTED_REGION_DIM=matrix.ORIGINAL_AUGMENTED_DIM;v4.v3.fit_scaled_dualproto_bags=matrix.ORIGINAL_ANOMALY_FIT;v4.RoleGraphConditionedEvidence=make_graphglobal_path_class(v4,"Q_QUERY_ONLY");v4.predict_model=matrix.ORIGINAL_V4_PREDICT
    device=v4.resolve_device(a.device);started=time.perf_counter();prob,_,audits=v4.fit_predict_variant(g0,g1,data.source_s7,data.target_s7,data.source_bags,data.target_bags,data.base_source,data.base_target,data.source_y,"RGCondAttn",seeds,a,device);elapsed=time.perf_counter()-started
    prob=np.asarray(prob,float)
    if prob.shape!=(1989,) or not np.isfinite(prob).all() or {int(r["parameter_n"]) for r in audits}!={33036}:raise RuntimeError("BLOCKED_BY_CONTRACT: frozen output/parameter audit failed")
    rows=[{"sha256":str(data.target_rows[i]["sha256"]).lower(),"year":int(data.future_input[i]["year"]),"arm":a.arm,"p_expert":float(prob[i]),"real_host_count":int(data.target_count[i]),"diagnostic_only":1} for i in range(1989)]
    matrix.write_csv(a.output/"future_predictions_label_free.csv",rows);matrix.write_csv(a.output/"fit_audit.csv",audits)
    protocol={"arm":a.arm,"protocol":"2017-trained frozen temporal diagnostic","source_train_n":799,"future_n":1989,"source_only_training":True,"future_labels_loaded_by_model":False,"future_labels_used_for_training_selection_or_early_stopping":False,"host_schema":a.arm,"host_width":74,"three_backgrounds_same_host_schema":True,"parameter_count":33036,"max_delta":2.25,"seeds":seeds,"ego_consumed":False,"training_seconds":elapsed}
    (a.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8");print("HOST_RELATIVE_FROZEN_ARM_DONE",a.arm,flush=True)
if __name__=="__main__":main()
