#!/usr/bin/env python3
from pathlib import Path
import csv,json,sys
ROOT=Path("artifacts/care-p31-stage1-rf-selection-source-v1")
REQ=[Path("artifacts/care-litelocal-stage1-attribution-source-v1/source_member_oof_predictions.csv"),Path("artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"),Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),Path("artifacts/care-litelocal-p31lr-c43-prov6-source-v1/cache/P0_ZERO6.npz"),Path("artifacts/care-litelocal-p31lr-c43-prov6-source-v1/arms/P0_ZERO6/source_predictions.csv"),Path("scripts/care-litelocal-p31lr-c43-prov6-source-v1/prov6_models.py"),Path("scripts/care-litelocal-p31lr-c43-prov6-source-v1/prov6_background.py"),Path("scripts/care-di25-graphglobal-path-source-v1/source_label_contract.py"),Path("scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py"),Path("scripts/care-stage2only-anomaly-v1/vendor/run_care_bounded_fixkeep_ablation.py")]
def main():
    missing=[str(p) for p in REQ if not p.is_file() or p.stat().st_size<100]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    protocol=json.loads((ROOT/"preflight/protocol.json").read_text())
    if protocol["future_rows_loaded"]!=0 or protocol["route_budget"]!=.25:raise RuntimeError("BLOCKED_BY_CONTRACT: protocol changed")
    for arm in protocol["protocols"]:
        with (ROOT/"preflight"/arm/"source_predictions.csv").open(encoding="utf-8",newline="") as h:rows=list(csv.DictReader(h))
        if len(rows)!=799 or sum(int(r["A25__routed"]) for r in rows)!=200:raise RuntimeError("BLOCKED_BY_CONTRACT: reference failed "+arm)
    print("P31_RF_SELECTION_SOURCE_PREFLIGHT_PASS")
    print("source_rows=799 year=2017 arms="+",".join(protocol["protocols"]))
    print("P31=fixed route_budget=25% stage2=P0_ZERO6_C43_RQ max_delta=2.25")
    print("future_rows_loaded=0 ego_consumed=false")
if __name__=="__main__":main()
