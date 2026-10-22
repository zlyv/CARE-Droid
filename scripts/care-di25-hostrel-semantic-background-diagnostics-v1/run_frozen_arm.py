#!/usr/bin/env python3
"""Fit semantic-background arm on 2017 and emit label-free future probabilities."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent.parent
MATRIX=PROJECT/"scripts/care-route-stage2-future-matrix-v1"; MODEL=PROJECT/"scripts/care-di25-graphglobal-path-source-v1"
for p in (HERE,MODEL,MATRIX,MATRIX/"vendor",PROJECT/"scripts/care-malben-host-v2"): sys.path.insert(0,str(p))
import run_future_arm as matrix
from graphglobal_path_models import make_graphglobal_path_class
from semantic_background import make_fit

v4=matrix.v4; ROOT=Path("artifacts/care-di25-hostrel-semantic-background-diagnostics-v1")
ARMS=("B1_HARD_ROLE","R1_HOSTREL74_HARD_ROLE")
SOURCE_CACHE={"B1_HARD_ROLE":Path("artifacts/care-di25-roleconditioned-background-source-v1/cache/regions_2017_role_sidecar.npz"),"R1_HOSTREL74_HARD_ROLE":Path("artifacts/care-di25-hostrel-semantic-background-source-v1/cache/regions_2017_hostrel_role_sidecar.npz")}


def load_sidecar_bags(path, rows):
    with np.load(path,allow_pickle=False) as saved:
        sha=[matrix.norm(v) for v in saved["sha256"].astype(str)]; values=np.asarray(saved["host_values"],np.float32); offsets=np.asarray(saved["host_offsets"],np.int64); real=np.asarray(saved["real_host_count"],np.int64)
    lookup={sid:i for i,sid in enumerate(sha)}
    if len(lookup)!=len(sha) or values.ndim!=2 or values.shape[1]!=105: raise RuntimeError("BLOCKED_BY_CONTRACT: sidecar cache schema "+str(path))
    bags=[]; counts=[]
    for row in rows:
        sid=matrix.norm(row["sha256"])
        if sid not in lookup: raise RuntimeError("BLOCKED_BY_CONTRACT: sidecar missing "+sid)
        i=lookup[sid]; bag=values[offsets[i]:offsets[i+1]]
        if len(bag)<1: raise RuntimeError("BLOCKED_BY_CONTRACT: empty deterministic bag "+sid)
        bags.append(bag); counts.append(int(real[i]))
    return bags,np.asarray(counts,np.int64)


def parser():
    p=matrix.parser()
    for action in p._actions:
        if action.dest in {"representation","head","output"}: action.required=False
    p.set_defaults(representation="anomaly",head="bounded",output=None)
    p.add_argument("--semantic-arm",required=True,choices=ARMS)
    return p


def main():
    a=parser().parse_args(); seeds=v4.parse_ints(a.seeds)
    if float(a.max_delta)!=2.25 or seeds!=[900000,900001,900002]: raise RuntimeError("BLOCKED_BY_CONTRACT: frozen protocol changed")
    a.output=Path(a.output) if a.output else ROOT/"frozen"/a.semantic_arm; a.output.mkdir(parents=True,exist_ok=True)
    if not Path(a.future_regions).is_file():
        fallback=Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/regions_2018_2021.npz")
        if not fallback.is_file(): raise RuntimeError("BLOCKED_BY_CONTRACT: original future regions missing")
        a.future_regions=fallback
    data=matrix.load_common(a)
    data.source_bags,data.source_count=load_sidecar_bags(SOURCE_CACHE[a.semantic_arm],data.source_rows)
    future_cache=ROOT/"caches"/(a.semantic_arm+"_regions_2018_2021.npz")
    data.target_bags,data.target_count=load_sidecar_bags(future_cache,data.target_rows)
    x0,x1,names=v4.audit.compact_matrix_variant(v4.VARIANT,data.source_rows,data.target_rows,data.compact_by_sha,data.schema,data.names)
    global_indices,_=v4.evidence.split_view_indices(names); g0=np.asarray(x0[:,global_indices],np.float32); g1=np.asarray(x1[:,global_indices],np.float32)
    v4.AUGMENTED_REGION_DIM=matrix.ORIGINAL_AUGMENTED_DIM
    v4.v3.fit_scaled_dualproto_bags=make_fit(v4.v3,"B1_HARD_ROLE")
    v4.RoleGraphConditionedEvidence=make_graphglobal_path_class(v4,"Q_QUERY_ONLY")
    v4.predict_model=matrix.ORIGINAL_V4_PREDICT
    device=v4.resolve_device(a.device); started=time.perf_counter()
    probability,_,audits=v4.fit_predict_variant(g0,g1,data.source_s7,data.target_s7,data.source_bags,data.target_bags,data.base_source,data.base_target,data.source_y,"RGCondAttn",seeds,a,device)
    elapsed=time.perf_counter()-started; probability=np.asarray(probability,float)
    if probability.shape!=(1989,) or not np.isfinite(probability).all() or {int(r["parameter_n"]) for r in audits}!={33036}: raise RuntimeError("BLOCKED_BY_CONTRACT: frozen output/capacity")
    rows=[{"sha256":str(data.target_rows[i]["sha256"]).lower(),"year":int(data.future_input[i]["year"]),"arm":a.semantic_arm,"p_expert":float(probability[i]),"real_host_count":int(data.target_count[i]),"diagnostic_only":1} for i in range(1989)]
    matrix.write_csv(a.output/"future_predictions_label_free.csv",rows); matrix.write_csv(a.output/"fit_audit.csv",audits)
    protocol={"arm":a.semantic_arm,"protocol":"2017-trained frozen temporal diagnostic","source_train_n":799,"future_n":1989,"source_only_training":True,"future_labels_loaded_by_model":False,"future_labels_used_for_training_selection_or_early_stopping":False,"hard_role_background":True,"backgrounds_same_host_schema":True,"host_width":74,"stored_width":105,"parameter_count":33036,"max_delta":2.25,"seeds":seeds,"ego_consumed":False,"training_seconds":elapsed}
    (a.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    print("HOSTREL_SEMANTIC_FROZEN_ARM_DONE",a.semantic_arm,flush=True)


if __name__=="__main__": main()
