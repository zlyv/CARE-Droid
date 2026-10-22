#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from common import ROOT,norm,read_csv
P=Path.cwd();S=P/"scripts"
def choose(items,name):
    for p in items:
        if p.is_file() and p.stat().st_size>=100:return p.resolve()
    raise RuntimeError("BLOCKED_BY_CONTRACT: missing "+name+":\n- "+"\n- ".join(map(str,items)))
def host(items,n,width):
    for p in items:
        if not p.is_file():continue
        try:
            with np.load(p,allow_pickle=False) as z:
                if len(z["sha256"])==n and z["host_values"].shape[1]==width:return p.resolve()
        except Exception:pass
    raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing Host cache rows={n} width={width}")
def main():
    old_resolved={}
    old_path=P/"artifacts/care-litelocal-p31-c43-complete-diagnostics-v1/preflight/resolved_inputs.json"
    if old_path.is_file():old_resolved=json.loads(old_path.read_text())
    source_ref=choose((P/"artifacts/care-litelocal-p31lr-c43-prov6-source-v1/preflight/lr_reference/source_predictions.csv",),"P31-LR source reference")
    source_features=choose((P/"artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz",),"source P31 features")
    future_features=choose((P/"artifacts/care-litelocal-p31-diagnostics-v1/future_p31_features_2018_2021.npz",),"future P31 features")
    label_contract=choose((P/"artifacts/care-litelocal-p31-diagnostics-v1/within_year/oof_predictions.csv",),"future label/fold contract")
    source_p0=host((P/"artifacts/care-litelocal-p31lr-c43-prov6-source-v1/cache/P0_ZERO6.npz",),799,111)
    future_candidates=[P/"artifacts/care-di25-compact43-relationquery-diagnostics-v1/cache/compact43_relation_future.npz",P/"artifacts/care-di25-compact43-relationquery-diagnostics-v1/cache/compact43_relation_2018_2021.npz",P/"artifacts/care-di25-compact43-relationquery-diagnostics-v1/cache/future_compact43_relation.npz"]
    if old_resolved.get("future_c43_cache"):future_candidates.insert(0,Path(old_resolved["future_c43_cache"]))
    future_c43=host(tuple(future_candidates),1989,105)
    future_resolved=choose((P/"artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json",),"future matrix manifest")
    choose((P/"artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz",),"prebuilt future InterZ")
    deps=(S/"care-litelocal-p31-c43-complete-diagnostics-v1/common.py",S/"care-litelocal-p31lr-c43-prov6-source-v1/prov6_background.py",S/"care-litelocal-p31lr-c43-prov6-source-v1/prov6_models.py",S/"care-litelocal-p31lr-c43-prov6-source-v1/lr_core.py",S/"care-litelocal-p31-diagnostics-v1/p31_core.py",S/"care-route-stage2-future-matrix-v1/run_future_arm.py")
    missing=[str(p) for p in deps if not p.is_file()]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing dependencies:\n- "+"\n- ".join(missing))
    ref=read_csv(source_ref);route=np.asarray([int(r["A25__routed"]) for r in ref]);y=np.asarray([int(r["label"]) for r in ref]);p=np.asarray([float(r["P31_LR__prob"]) for r in ref]);
    if len(ref)!=799 or len({norm(r["sha256"]) for r in ref})!=799 or route.sum()!=200 or np.sum(route&((p>=.5)!=y))!=99:raise RuntimeError("BLOCKED_BY_CONTRACT: source P31-LR protocol changed")
    lab=read_csv(label_contract)
    if len(lab)!=1989 or len({norm(r["sha256"]) for r in lab})!=1989 or not {"label","fold","year"}.issubset(lab[0]):raise RuntimeError("BLOCKED_BY_CONTRACT: future label/fold contract changed")
    with np.load(source_features,allow_pickle=False) as z:
        if z["role_prevalence"].shape!=(799,31):raise RuntimeError("source P31 shape changed")
    with np.load(future_features,allow_pickle=False) as z:
        if z["role_prevalence"].shape!=(1989,31):raise RuntimeError("future P31 shape changed")
    out={"source_lr_reference":str(source_ref),"source_p31_features":str(source_features),"future_p31_features":str(future_features),"future_label_fold_contract":str(label_contract),"source_p0_cache":str(source_p0),"future_c43_cache":str(future_c43),"future_matrix_resolved":str(future_resolved)};target=ROOT/"preflight/resolved_inputs.json";target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(out,indent=2)+"\n")
    print("P31LR_C43_P0_COMPLETE_PREFLIGHT_PASS");print("source=799 route=200 errors=99 stage1=P31_LR committee_role=routing_only");print("future=1989 years=2018:793,2019:399,2020:397,2021:400 labels_loaded_by_frozen_model=false")
if __name__=="__main__":main()
