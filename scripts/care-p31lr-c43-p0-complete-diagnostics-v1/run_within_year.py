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

def fold_global(data,tr,va):
    a,b,n=v4.audit.allratio.original_compact_matrix_named([data.target_rows[int(i)] for i in tr],[data.target_rows[int(i)] for i in va],data.compact_by_sha,data.schema);return align_fold_global24(a,b,n,data.names)

def main():
    parser=future.parser();parser.set_defaults(representation="anomaly",head="bounded",resolved=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json"));parser.add_argument("--years",required=True);parser.add_argument("--jobs",type=int,default=4);args=parser.parse_args();requested=[int(v) for v in args.years.split(",")]
    if not set(requested).issubset({2018,2019,2020,2021}) or float(args.max_delta)!=2.25:raise SystemExit("invalid years/max-delta")
    seeds=v4.parse_ints(args.seeds);device=v4.resolve_device(args.device);cfg=resolved();args.resolved=Path(cfg["future_matrix_resolved"]);data=future.load_common(args)
    contract=aligned_rows(data.target_rows,Path(cfg["future_label_fold_contract"]));y=np.asarray([int(float(r["label"])) for r in contract]);years=np.asarray([int(float(r["year"])) for r in contract]);folds=np.asarray([int(float(r["fold"])) for r in contract])
    with np.load(cfg["future_p31_features"],allow_pickle=False) as z:ids=[norm(v) for v in z["sha256"].astype(str)];x=np.asarray(z["role_prevalence"],float)
    lookup={s:i for i,s in enumerate(ids)};x=np.asarray([x[lookup[norm(r["sha256"])]] for r in data.target_rows]);bags,_=load_bags(Path(cfg["future_c43_cache"]),data.target_rows,105);bags=zero6(bags)
    install_p0(v4);nested=NestedAnnualLR([r["sha256"] for r in data.target_rows],x);v4.inner_oof_global24=nested.inner_oof
    expert=np.full(1989,np.nan);members=np.full((1989,4),np.nan);fit_audit=[]
    for year in requested:
        for fold in range(1,6):
            va=np.flatnonzero((years==year)&(folds==fold));tr=np.flatnonzero((years==year)&(folds!=fold));seed=int(args.split_seed)+year*10+fold
            mv,ma=fit_calibrated_committee(x[tr],y[tr],x[va],seed,inner_folds=4,jobs=args.jobs);members[va]=mv
            gt,gv,alignment=fold_global(data,tr,va);base_train=nested.crossfit(x[tr],y[tr],seed);p,_,aud=v4.fit_predict_variant(gt,gv,data.target_s7[tr],data.target_s7[va],subset_bags(bags,tr),subset_bags(bags,va),base_train,mv[:,0],y[tr],"RGCondAttn",seeds,args,device);expert[va]=np.asarray(p,float);fit_audit.extend({"year":year,"outer_fold":fold,**r} for r in aud);print(f"P31LR-P0 same-year {year} fold {fold}/5 done",flush=True)
    selected=np.flatnonzero(np.isin(years,requested))
    if not np.isfinite(expert[selected]).all() or not np.isfinite(members[selected]).all():raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete annual OOF")
    base=members[:,0];score=np.full(1989,np.nan);route=np.zeros(1989,bool)
    for year in requested:
        idx=np.flatnonzero(years==year);score[idx]=mean_opposition(members[idx],base[idx]);route[idx]=group_top(score[idx],years[idx],np.asarray([norm(data.target_rows[i]["sha256"]) for i in idx]),.25)
    rows=[]
    for i in selected:
        final=expert[i] if route[i] else base[i];rows.append({"sha256":norm(data.target_rows[i]["sha256"]),"year":int(years[i]),"fold":int(folds[i]),"label":int(y[i]),"p_stage1":float(base[i]),"p_lr":float(members[i,0]),"p_rf":float(members[i,1]),"p_xgb":float(members[i,2]),"p_svm":float(members[i,3]),"U_DI":float(score[i]),"route_DI":int(route[i]),"p_expert":float(expert[i]),"p_final":float(final),"method":METHOD})
    write_csv(args.output/"oof_predictions.csv",rows);write_csv(args.output/"fit_audit.csv",fit_audit);(args.output/"protocol.json").write_text(json.dumps({"protocol":"independent same-year five-fold OOF","years":requested,"stage1":"P31 calibrated LR only","committee_role":"routing only","router":"LR-anchored MeanOpposition annual Top25%","outer_train_stage1":"nested P31-LR","stage2":"P0_ZERO6 C43-RQ bounded","temporal_generalization":False,"max_delta":2.25},indent=2)+"\n");print("P31LR_C43_P0_WITHIN_YEAR_GROUP_DONE",requested)
if __name__=="__main__":main()
