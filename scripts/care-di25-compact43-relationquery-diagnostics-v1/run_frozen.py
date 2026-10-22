#!/usr/bin/env python3
"""Fit frozen Compact43 Relation-Query on 2017 and emit label-free future probabilities."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent
MATRIX=PROJECT/"scripts/care-route-stage2-future-matrix-v1";SOURCE=PROJECT/"scripts/care-di25-compact43-relationquery-source-v1"
for p in (HERE,SOURCE,MATRIX,MATRIX/"vendor",PROJECT/"scripts/care-di25-graphglobal-path-source-v1",PROJECT/"scripts/care-malben-host-v2"):sys.path.insert(0,str(p))
import run_future_arm as matrix
from compact_background import make_fit
from compact_models import make_compact_class

v4=matrix.v4;ROOT=Path("artifacts/care-di25-compact43-relationquery-diagnostics-v1")
SOURCE_CACHE=Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz")
FUTURE_CACHE=ROOT/"cache/compact43_relation_future.npz"


def load_bags(path,rows):
    with np.load(path,allow_pickle=False) as z:
        sha=[matrix.norm(v) for v in z["sha256"].astype(str)];values=np.asarray(z["host_values"],np.float32);offsets=np.asarray(z["host_offsets"],np.int64);real=np.asarray(z["real_host_count"],np.int64)
    lookup={sid:i for i,sid in enumerate(sha)}
    if len(lookup)!=len(sha) or values.ndim!=2 or values.shape[1]!=105:raise RuntimeError("BLOCKED_BY_CONTRACT: Compact43 cache schema "+str(path))
    bags=[];counts=[]
    for row in rows:
        sid=matrix.norm(row["sha256"])
        if sid not in lookup:raise RuntimeError("BLOCKED_BY_CONTRACT: cache missing "+sid)
        i=lookup[sid];bag=values[offsets[i]:offsets[i+1]]
        if len(bag)<1:raise RuntimeError("BLOCKED_BY_CONTRACT: empty deterministic bag "+sid)
        bags.append(bag);counts.append(int(real[i]))
    return bags,np.asarray(counts,np.int64)


def parser():
    p=matrix.parser()
    for action in p._actions:
        if action.dest in {"representation","head","output"}:action.required=False
    p.set_defaults(representation="anomaly",head="bounded",output=None)
    return p


def main():
    a=parser().parse_args();seeds=v4.parse_ints(a.seeds)
    if a.representation!="anomaly" or a.head!="bounded" or float(a.max_delta)!=2.25 or seeds!=[900000,900001,900002]:raise RuntimeError("BLOCKED_BY_CONTRACT: frozen protocol changed")
    a.output=Path(a.output) if a.output else ROOT/"frozen/C43_RELATION_QUERY";a.output.mkdir(parents=True,exist_ok=True)
    if not Path(a.future_regions).is_file():
        fallback=Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/regions_2018_2021.npz")
        if not fallback.is_file():raise RuntimeError("BLOCKED_BY_CONTRACT: original future regions missing")
        a.future_regions=fallback
    if not Path(a.future_cache).is_file():
        fallback=Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz")
        if not fallback.is_file():raise RuntimeError("BLOCKED_BY_CONTRACT: original future endpoint cache missing")
        a.future_cache=fallback
    data=matrix.load_common(a);data.source_bags,data.source_count=load_bags(SOURCE_CACHE,data.source_rows);data.target_bags,data.target_count=load_bags(FUTURE_CACHE,data.target_rows)
    x0,x1,names=v4.audit.compact_matrix_variant(v4.VARIANT,data.source_rows,data.target_rows,data.compact_by_sha,data.schema,data.names);indices,_=v4.evidence.split_view_indices(names);g0=np.asarray(x0[:,indices],np.float32);g1=np.asarray(x1[:,indices],np.float32)
    v4.AUGMENTED_REGION_DIM=78;v4.v3.fit_scaled_dualproto_bags=make_fit(v4.v3);v4.RoleGraphConditionedEvidence=make_compact_class(v4,"C43_RELATION_QUERY");v4.predict_model=matrix.ORIGINAL_V4_PREDICT
    device=v4.resolve_device(a.device);started=time.perf_counter();probability,_,audits=v4.fit_predict_variant(g0,g1,data.source_s7,data.target_s7,data.source_bags,data.target_bags,data.base_source,data.base_target,data.source_y,"RGCondAttn",seeds,a,device);elapsed=time.perf_counter()-started
    probability=np.asarray(probability,float)
    if probability.shape!=(1989,) or not np.isfinite(probability).all() or {int(r["parameter_n"]) for r in audits}!={32292}:raise RuntimeError("BLOCKED_BY_CONTRACT: frozen output/capacity")
    rows=[{"sha256":str(data.target_rows[i]["sha256"]).lower(),"year":int(data.future_input[i]["year"]),"arm":"C43_RELATION_QUERY","p_expert":float(probability[i]),"real_host_count":int(data.target_count[i]),"diagnostic_only":1} for i in range(1989)]
    matrix.write_csv(a.output/"future_predictions_label_free.csv",rows);matrix.write_csv(a.output/"fit_audit.csv",audits)
    protocol={"arm":"C43_RELATION_QUERY","protocol":"2017-trained frozen temporal diagnostic","source_train_n":799,"future_n":1989,"source_only_training":True,"future_labels_loaded_by_model":False,"future_labels_used_for_training_selection_or_early_stopping":False,"compact_host_width":43,"duplicate_role_columns":0,"relation_query_width":31,"relation_enters_attention":True,"relation_enters_delta":False,"parameter_count":32292,"max_delta":2.25,"seeds":seeds,"ego_consumed":False,"training_seconds":elapsed}
    (a.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    print("COMPACT43_RELATIONQUERY_FROZEN_DONE",flush=True)


if __name__=="__main__":main()
