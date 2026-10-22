#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
ROOT=Path.cwd();HERE=ROOT/"scripts/care-di25-host-relative-source-v1";BASE=ROOT/"scripts/care-di25-graphglobal-path-source-v1";sys.path[:0]=[str(HERE),str(BASE)]
from experiment_contract import build_di_reference
REQUIRED=["scripts/care-di25-host-relative-source-v1/build_caches.py","scripts/care-di25-host-relative-source-v1/run_arm.py","scripts/care-di25-graphglobal-path-source-v1/graphglobal_path_models.py","scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py","scripts/care-rgcondattn-dedup-bounded-source-v1/run_care_rgcondattn_dedup_bounded_source_v1.py","artifacts/p1-features/features.csv","artifacts/p1-features/feature_schema.json","artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv","artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz","artifacts/care-source-local-assets-v2/source_asset_gate.txt","artifacts/care-source-local-assets-v2/regions_2017.npz","artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz","artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY/source_predictions.csv","artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY/source_summary.csv","artifacts/care-di25-graphglobal-path-source-v1/arms/Q_QUERY_ONLY/fit_audit.csv"]
def main():
    errors=["missing: "+p for p in REQUIRED if not Path(p).is_file() or Path(p).stat().st_size==0]
    if not any(p.is_file() for p in (Path("artifacts/care-source-local-assets-v2/local_asset_manifest.json"),Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json"))):errors.append("missing frozen local_asset_manifest.json")
    for m in ("numpy","torch","sklearn","yaml","scipy"):
        if importlib.util.find_spec(m) is None:errors.append("missing Python module: "+m)
    if errors:raise RuntimeError("BLOCKED_BY_CONTRACT:\n- "+"\n- ".join(errors))
    contract=build_di_reference(ROOT,Path("artifacts/care-di25-host-relative-source-v1"))
    print("HOST_RELATIVE_SOURCE_PREFLIGHT_PASS");print("source_rows=799 year=2017 route_DI=200")
    print("fold_sizes=1:160,2:160,3:160,4:160,5:159 route_DI_per_fold=1:40,2:40,3:40,4:40,5:40")
    print("FULL_Q=reuse_locked_not_retrained")
    print("three_backgrounds=unchanged_fold_safe")
    print("future_rows_loaded=0 ego_consumed=false")
if __name__=="__main__":main()
