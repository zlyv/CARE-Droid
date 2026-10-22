#!/usr/bin/env python3
"""Independent same-year OOF for Compact43 Relation-Query."""
from __future__ import annotations
import importlib.util,json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent
BASE_DIR=PROJECT/"scripts/care-di25-graphglobal-query-within-year-v1";SOURCE=PROJECT/"scripts/care-di25-compact43-relationquery-source-v1"
for p in (HERE,SOURCE,BASE_DIR,PROJECT/"scripts/care-di25-graphglobal-path-source-v1",PROJECT/"scripts/care-route-stage2-within-year-matrix-v1",PROJECT/"scripts/care-route-stage2-within-year-matrix-v1/vendor",PROJECT/"scripts/care-malben-host-v2"):sys.path.insert(0,str(p))
spec=importlib.util.spec_from_file_location("graphq_within_base",BASE_DIR/"run_within_year_arm.py");base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from compact_background import make_fit
from compact_models import make_compact_class

v4=base.v4;ROOT=Path("artifacts/care-di25-compact43-relationquery-diagnostics-v1");VERSION="compact43-relationquery-within-v1"


def load_bags(path,rows):
    with np.load(path,allow_pickle=False) as z:
        sha=[base.within.norm(v) for v in z["sha256"].astype(str)];values=np.asarray(z["host_values"],np.float32);offsets=np.asarray(z["host_offsets"],np.int64);real=np.asarray(z["real_host_count"],np.int64)
    lookup={sid:i for i,sid in enumerate(sha)}
    if len(lookup)!=len(sha) or values.ndim!=2 or values.shape[1]!=105:raise RuntimeError("BLOCKED_BY_CONTRACT: future cache schema")
    bags=[];counts=[]
    for row in rows:
        sid=base.within.norm(row["sha256"])
        if sid not in lookup:raise RuntimeError("BLOCKED_BY_CONTRACT: cache missing "+sid)
        i=lookup[sid];bag=values[offsets[i]:offsets[i+1]]
        if len(bag)<1:raise RuntimeError("BLOCKED_BY_CONTRACT: empty deterministic bag "+sid)
        bags.append(bag);counts.append(int(real[i]))
    return bags,np.asarray(counts,np.int64)


def main():
    p=base.parser()
    for action in p._actions:
        if action.dest=="arm":action.required=False
    p.set_defaults(arm="Q_QUERY_ONLY")
    a=p.parse_args();seeds=v4.parse_ints(a.seeds)
    if float(a.max_delta)!=2.25 or seeds!=[900000,900001,900002]:raise RuntimeError("BLOCKED_BY_CONTRACT: within protocol changed")
    a.output=Path(a.output);a.output.mkdir(parents=True,exist_ok=True);device=v4.resolve_device(a.device)
    original=Path("artifacts/care-future-local-assets-v2/regions_2018_2021.npz")
    if not original.is_file():original=Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/regions_2018_2021.npz")
    if not original.is_file():raise RuntimeError("BLOCKED_BY_CONTRACT: original future regions missing")
    a.future_regions=original
    if not Path(a.future_cache).is_file():
        fallback=Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz")
        if not fallback.is_file():raise RuntimeError("BLOCKED_BY_CONTRACT: original future endpoint cache missing")
        a.future_cache=fallback
    data=base.within.load_data(a);data.bags,data.counts=load_bags(ROOT/"cache/compact43_relation_future.npz",data.rows)
    v4.AUGMENTED_REGION_DIM=78;v4.v3.fit_scaled_dualproto_bags=make_fit(v4.v3);v4.RoleGraphConditionedEvidence=make_compact_class(v4,"C43_RELATION_QUERY");v4.predict_model=base.within.future.ORIGINAL_V4_PREDICT
    output=np.full(1989,np.nan,float);checkpoint=a.output/"within_year_checkpoint.csv";audit_path=a.output/"fit_audit_checkpoint.csv";audits=base.within.read_csv(audit_path) if audit_path.is_file() else []
    if checkpoint.is_file():
        saved=base.within.read_csv(checkpoint);by={base.within.norm(r["sha256"]):r for r in saved}
        if len(by)==len(saved) and {r.get("version") for r in saved}=={VERSION}:
            for i,sid in enumerate(data.ids):
                if sid in by:output[i]=float(by[sid]["p_expert"])
            print(f"[resume] C43_RELATION_QUERY restored {int(np.isfinite(output).sum())}/1989",flush=True)
        else:audits=[]
    started=time.perf_counter()
    for year in sorted(set(data.years.tolist())):
        for fold in range(1,6):
            val=np.flatnonzero((data.years==year)&(data.folds==fold));train=np.flatnonzero((data.years==year)&(data.folds!=fold))
            if np.isfinite(output[val]).all():print(f"[resume] C43_RELATION_QUERY {year} fold {fold}/5",flush=True);continue
            xtr,xval=base.within.fold_global24(data,train,val);base_train=base.within.inner_oof_base(data,train,int(a.split_seed)+year*10+fold)
            probability,_,fold_audits=v4.fit_predict_variant(xtr,xval,data.s7[train],data.s7[val],base.within.subset_bags(data.bags,train),base.within.subset_bags(data.bags,val),base_train,data.base[val],data.labels[train],"RGCondAttn",seeds,a,device);probability=np.asarray(probability,float)
            if probability.shape!=(len(val),) or not np.isfinite(probability).all() or {int(r["parameter_n"]) for r in fold_audits}!={32292}:raise RuntimeError(f"BLOCKED_BY_CONTRACT: within output {year}/{fold}")
            output[val]=probability;audits.extend({"arm":"C43_RELATION_QUERY","year":year,"outer_fold":fold,**r} for r in fold_audits)
            complete=np.flatnonzero(np.isfinite(output));rows=[{"sha256":data.ids[i],"year":int(data.years[i]),"fold":int(data.folds[i]),"arm":"C43_RELATION_QUERY","p_expert":float(output[i]),"version":VERSION} for i in complete]
            tmp=checkpoint.with_suffix(".tmp");base.within.write_csv(tmp,rows);tmp.replace(checkpoint);tmp=audit_path.with_suffix(".tmp");base.within.write_csv(tmp,audits);tmp.replace(audit_path)
            print(f"C43_RELATION_QUERY same-year {year} fold {fold}/5 done",flush=True)
    if not np.isfinite(output).all():raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete within predictions")
    rows=[{"sha256":data.ids[i],"year":int(data.years[i]),"label":int(data.labels[i]),"fold":int(data.folds[i]),"arm":"C43_RELATION_QUERY","p_expert":float(output[i]),"real_host_count":int(data.counts[i]),"same_year_oof":1,"version":VERSION} for i in range(1989)]
    base.within.write_csv(a.output/"within_year_predictions.csv",rows);base.within.write_csv(a.output/"fit_audit.csv",audits)
    protocol={"arm":"C43_RELATION_QUERY","protocol":"independent same-year five-fold OOF","years":[2018,2019,2020,2021],"labels_used_only_within_each_year_outer_train":True,"inner_oof_base_on_outer_train_only":True,"compact_host_width":43,"duplicate_role_columns":0,"relation_query_width":31,"relation_enters_attention":True,"relation_enters_delta":False,"parameter_count":32292,"max_delta":2.25,"seeds":seeds,"ego_consumed":False,"temporal_generalization_result":False,"training_seconds_this_run":time.perf_counter()-started}
    (a.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    print("COMPACT43_RELATIONQUERY_WITHIN_DONE",flush=True)


if __name__=="__main__":main()
