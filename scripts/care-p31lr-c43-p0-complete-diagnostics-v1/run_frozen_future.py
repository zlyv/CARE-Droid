#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from common import *
install_paths()
import run_future_arm as future
import run_care_rolegraph_conditioned_dualstream_v4 as v4
from p31_core import fit_calibrated_committee,mean_opposition,group_top

def main():
    parser=future.parser();parser.set_defaults(representation="anomaly",head="bounded",resolved=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json"),output=ROOT/"frozen_future/model");args=parser.parse_args()
    if float(args.max_delta)!=2.25:raise SystemExit("max-delta must remain 2.25")
    seeds=v4.parse_ints(args.seeds)
    if seeds!=[900000,900001,900002]:raise SystemExit("frozen seeds changed")
    device=v4.resolve_device(args.device);cfg=resolved();args.resolved=Path(cfg["future_matrix_resolved"]);data=future.load_common(args)
    with np.load(cfg["source_p31_features"],allow_pickle=False) as z:sids=[norm(v) for v in z["sha256"].astype(str)];sx=np.asarray(z["role_prevalence"],float)
    with np.load(cfg["future_p31_features"],allow_pickle=False) as z:fids=[norm(v) for v in z["sha256"].astype(str)];fx=np.asarray(z["role_prevalence"],float);years=np.asarray(z["year"],int)
    smap={s:i for i,s in enumerate(sids)};fmap={s:i for i,s in enumerate(fids)};sx=np.asarray([sx[smap[norm(r["sha256"])]] for r in data.source_rows]);fx=np.asarray([fx[fmap[norm(r["sha256"])]] for r in data.target_rows]);years=np.asarray([years[fmap[norm(r["sha256"])]] for r in data.target_rows])
    members,audit=fit_calibrated_committee(sx,data.source_y,fx,20260925,inner_folds=5,jobs=4);base_target=members[:,0];score=mean_opposition(members,base_target);route=group_top(score,years,np.asarray([norm(r["sha256"]) for r in data.target_rows]),.25)
    if int(route.sum())!=497:raise RuntimeError("BLOCKED_BY_CONTRACT: frozen P31-LR opposition route changed")
    source_ref=aligned_rows(data.source_rows,Path(cfg["source_lr_reference"]));base_source=np.asarray([float(r["P31_LR__prob"]) for r in source_ref])
    source_bags,_=load_bags(Path(cfg["source_p0_cache"]),data.source_rows,111);target_bags,_=load_bags(Path(cfg["future_c43_cache"]),data.target_rows,105);target_bags=zero6(target_bags)
    xs,xt,names=v4.audit.compact_matrix_variant(v4.VARIANT,data.source_rows,data.target_rows,data.compact_by_sha,data.schema,data.names);gi,_=v4.evidence.split_view_indices(names);gs=np.asarray(xs[:,gi],np.float32);gt=np.asarray(xt[:,gi],np.float32)
    install_p0(v4);nested=NestedAnnualLR([r["sha256"] for r in data.source_rows],sx);v4.inner_oof_global24=nested.inner_oof
    expert,_,audits=v4.fit_predict_variant(gs,gt,data.source_s7,data.target_s7,source_bags,target_bags,base_source,base_target,data.source_y,"RGCondAttn",seeds,args,device);expert=np.asarray(expert,float);final=np.where(route,expert,base_target)
    rows=[{"sha256":norm(r["sha256"]),"year":int(years[i]),"p_stage1":float(base_target[i]),"p_lr":float(members[i,0]),"p_rf":float(members[i,1]),"p_xgb":float(members[i,2]),"p_svm":float(members[i,3]),"U_DI":float(score[i]),"route_DI":int(route[i]),"p_expert":float(expert[i]),"p_final":float(final[i]),"method":METHOD} for i,r in enumerate(data.target_rows)]
    out=ROOT/"frozen_future";write_csv(out/"label_free_predictions.csv",rows);write_csv(out/"stage1_fit_audit.csv",audit);write_csv(out/"stage2_fit_audit.csv",audits)
    (out/"protocol.json").write_text(json.dumps({"protocol":"frozen 2017 to 2018-2021","stage1":"P31 calibrated LR only","committee_role":"routing only","router":"LR-anchored MeanOpposition year-wise Top25%","stage2":"P0_ZERO6 C43-RQ bounded","future_labels_loaded_by_model":False,"route":497,"max_delta":2.25,"seeds":seeds},indent=2)+"\n")
    print("P31LR_C43_P0_FROZEN_FUTURE_LABEL_FREE_DONE");print("future_rows=1989 routed=497 labels_loaded=false")
if __name__=="__main__":main()
