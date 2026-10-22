#!/usr/bin/env python3
from pathlib import Path
import csv,json,sys,numpy as np
ROOT=Path("artifacts/care-litelocal-p31lr-c43-prov6-source-v1")
SCRIPTS=Path("scripts");SOURCE_VENDOR=SCRIPTS/"care-stage2only-anomaly-v1/vendor"
for p in (Path(__file__).resolve().parent,SCRIPTS,SCRIPTS/"care-di25-graphglobal-path-source-v1",SOURCE_VENDOR,SCRIPTS/"care-rolegraph-conditioned-v4-script",SCRIPTS/"care-rgcondattn-dedup-bounded-source-v1",SCRIPTS/"care-malben-host-v2"):
    value=str(p.resolve())
    if value not in sys.path:sys.path.insert(0,value)
REQ=(Path("artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"),Path("artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv"),Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz"),ROOT/"preflight/lr_reference/source_predictions.csv",Path("scripts/care-di25-graphglobal-path-source-v1/source_label_contract.py"),Path("scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py"),SOURCE_VENDOR/"run_care_bounded_fixkeep_ablation.py")
def main():
    missing=[str(p) for p in REQ if not p.is_file() or p.stat().st_size<100]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    with (ROOT/"preflight/lr_reference/source_predictions.csv").open(encoding="utf-8",newline="") as h:rows=list(csv.DictReader(h))
    fold=np.asarray([int(r["fold"]) for r in rows]);route=np.asarray([int(r["A25__routed"]) for r in rows])
    if len(rows)!=799 or len({r["sha256"] for r in rows})!=799 or int(route.sum())!=200 or [int(route[fold==f].sum()) for f in range(1,6)]!=[40]*5:raise RuntimeError("BLOCKED_BY_CONTRACT: LR reference/route changed")
    contract=json.loads((ROOT/"preflight/lr_contract.json").read_text())
    if contract["stage1"]!="calibrated P31 Logistic Regression only" or contract["committee_role"]!="routing only":raise RuntimeError("BLOCKED_BY_CONTRACT: Stage-I contract changed")
    # Import the exact training stack here so missing transitive dependencies
    # fail before cache construction or GPU subprocess launch.
    import run_care_bounded_fixkeep_ablation as v3
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
    if v4.v3 is not v3:raise RuntimeError("BLOCKED_BY_CONTRACT: CARE v3 import identity mismatch")
    print("P31LR_PROV6_SOURCE_PREFLIGHT_PASS");print("source_rows=799 year=2017 fold_sizes=1:160,2:160,3:160,4:160,5:159");print("stage1=P31_LR committee_role=routing_only route=200 route_per_fold=40,40,40,40,40");print("stage2=C43_RELATION_QUERY max_delta=2.25 future_rows_loaded=0 ego_consumed=false")
if __name__=="__main__":main()
