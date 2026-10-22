#!/usr/bin/env python3
"""Fit S3 Gate on 2017 and emit label-free 2018-2021 probabilities."""
from __future__ import annotations

import argparse,csv,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent
SOURCE=PROJECT/"scripts/care-di25-noego-needgate-source-v1";FM=PROJECT/"scripts/care-route-stage2-future-matrix-v1"
for p in (HERE,SOURCE,FM,FM/"vendor",PROJECT/"scripts/care-rolegraph-conditioned-v4-script",PROJECT/"scripts/care-rgcondattn-dedup-bounded-source-v1",PROJECT/"scripts/care-stage2only-anomaly-v1/vendor"):
    sys.path.insert(0,str(p))
import run_future_arm as future
import run_care_rolegraph_conditioned_dualstream_v4 as v4
import run_care_rgcondattn_dedup_bounded_source_v1 as anomaly
from noego_representation import fit_seed
from probe_protocol import need_correct,stage1_logit
from safety_core import apply

OUT=Path("artifacts/care-di25-noego-safe-authority-full-v1/future")
CONFIG=Path("artifacts/care-di25-noego-safe-authority-full-v1/source/frozen_deploy_config.json")

def parser():
    p=future.parser()
    for action in p._actions:
        if action.dest in {"representation","head","output"}:
            action.required=False
    p.set_defaults(representation="anomaly",head="bounded",output=OUT/"unused")
    return p

def fit_probe(x,y,xv,seed):
    sc=StandardScaler().fit(x);m=LogisticRegression(C=1,class_weight="balanced",solver="liblinear",max_iter=5000,random_state=seed)
    m.fit(sc.transform(x),y);return m.predict_proba(sc.transform(xv))[:,1]

def load_reconstructed(args):
    """Load label-sealed future rows and rebuild S7 in memory from endpoint caches."""
    future.install_source_label_contract(v4)
    resolved=json.loads(args.resolved.read_text(encoding="utf-8"))
    args.features=Path(resolved["features"]);args.schema=Path(resolved["schema"])
    args.cache=args.source_cache;args.region_cache=args.source_regions;args.rebuild_region_cache=False
    manifest,source_rows,_,compact_by_sha,schema,names=v4.audit.load_inputs(args)
    source_y=np.asarray([r["label"] for r in manifest],dtype=np.int64)
    future_input=future.read_csv(Path(resolved["future_router_inputs_label_free"]))
    target_rows=[]
    for row in future_input:
        target=compact_by_sha[future.norm(row["sha256"])]
        leaked={k:v for k,v in target.items() if "label" in k.casefold() and str(v).strip()}
        if leaked:raise RuntimeError("future label seal failed: "+future.norm(row["sha256"]))
        target_rows.append(target)
    if len(source_rows)!=799 or len(target_rows)!=1989:raise RuntimeError("source/future cohort changed")
    base_source=future.aligned_values(source_rows,Path(resolved["source_scores"]),"p_stage1",source_y)
    base_target=np.asarray([float(r["p_stage1"]) for r in future_input],dtype=np.float64)
    endpoint=future.merge_endpoint_caches(args.source_cache,args.future_cache)
    source_s3,_=v4.audit.future.reconstruct_structure(source_rows,endpoint,31)
    target_s3,_=v4.audit.future.reconstruct_structure(target_rows,endpoint,31)
    _,_,source_inter,_=v4.v3.future_s7.clipped.build_multiscale_features(source_rows,endpoint,source_s3[:,120:],args)
    _,_,target_inter,_=v4.v3.future_s7.clipped.build_multiscale_features(target_rows,endpoint,target_s3[:,120:],args)
    source_s7=np.concatenate([source_s3,np.clip(source_inter,-float(args.z_clip),float(args.z_clip))],axis=1)
    target_s7=np.concatenate([target_s3,np.clip(target_inter,-float(args.z_clip),float(args.z_clip))],axis=1)
    if source_s7.shape!=(799,v4.S7_DIM) or target_s7.shape!=(1989,v4.S7_DIM):raise RuntimeError("in-memory S7 shape contract failed")
    source_bags,source_count=future.load_bags(args.source_regions,source_rows)
    target_bags,target_count=future.load_bags(args.future_regions,target_rows)
    return SimpleNamespace(resolved=resolved,manifest=manifest,source_rows=source_rows,target_rows=target_rows,
        future_input=future_input,source_y=source_y,base_source=base_source,base_target=base_target,
        source_s7=source_s7,target_s7=target_s7,source_bags=source_bags,target_bags=target_bags,
        source_count=source_count,target_count=target_count,compact_by_sha=compact_by_sha,schema=schema,names=names)

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    args=parser().parse_args();config=json.loads(CONFIG.read_text(encoding="utf-8"));threshold=float(config["deploy_threshold"])
    device=v4.resolve_device(args.device);data=load_reconstructed(args)
    xs,xt,names=v4.audit.compact_matrix_variant(v4.VARIANT,data.source_rows,data.target_rows,data.compact_by_sha,data.schema,data.names)
    gi,_=v4.evidence.split_view_indices(names);gs=np.asarray(xs[:,gi],np.float32);gt=np.asarray(xt[:,gi],np.float32)
    source_scores={r["sha256"].strip().lower():r for r in future.read_csv(Path(data.resolved["source_scores"]))}
    source_route=np.asarray([int(float(source_scores[future.norm(r["sha256"])]["route_DI"]))==1 for r in data.source_rows])
    future_route=np.asarray([int(float(r["route_DI"]))==1 for r in data.future_input])
    if int(source_route.sum())!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: source DI25 route changed")
    target=need_correct(data.source_y,data.base_source);scores=[];audits=[]
    v4.AUGMENTED_REGION_DIM=v4.v3.v2.BASE_REGION_DIM+4;v4.RoleGraphConditionedEvidence=anomaly.DedupBoundedRGCondAttn
    for seed in (900000,900001,900002):
        sr,fr,audit=fit_seed(v4,anomaly,gs,data.source_s7,data.target_s7,data.source_bags,data.target_bags,data.base_source,data.source_y,args,device,seed)
        audits.append(audit);sx=np.concatenate([sr,stage1_logit(data.base_source)[:,None]],axis=1);fx=np.concatenate([fr,stage1_logit(data.base_target)[:,None]],axis=1)
        scores.append(fit_probe(sx[source_route],target[source_route],fx,seed))
        print(f"frozen future S3 seed {seed} done",flush=True)
    q=np.mean(scores,axis=0);final,active,limited,delta=apply(data.base_target,q,future_route,threshold)
    rows=[]
    for i,row in enumerate(data.future_input):
        rows.append({"sha256":future.norm(row["sha256"]),"year":int(row["year"]),"p_stage1":float(data.base_target[i]),"route_DI":int(future_route[i]),
            "score_S3":float(q[i]),"frozen_threshold":threshold,"gate_active":int(active[i]),"delta":float(delta[i]),"p_final":float(final[i]),"bound_limited":int(limited[i])})
    write_csv(OUT/"future_predictions_label_free.csv",rows)
    protocol={"source_train_n":799,"future_n":1989,"source_only_training":True,"future_labels_loaded_by_model":False,"ego_consumed":False,
        "deploy_threshold":threshold,"threshold_selected_on":"strict 2017 nested OOF only","future_threshold_selection":False,"future_model_selection":False,"seeds":[900000,900001,900002],"fit_audits":audits}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"protocol.json").write_text(json.dumps(protocol,indent=2,default=str)+"\n",encoding="utf-8")
    print("FROZEN_FUTURE_LABEL_FREE_PREDICTIONS_DONE")

if __name__=="__main__":main()
