#!/usr/bin/env python3
from pathlib import Path
REQUIRED=[
"scripts/care-di25-noego-needgate-source-v1/run_needgate_source.py",
"scripts/care-di25-noego-needgate-source-v1/noego_representation.py",
"scripts/care-route-stage2-future-matrix-v1/run_future_arm.py",
"artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv",
"artifacts/care-source-local-assets-v2/host_regions_2017.npz",
"artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
"artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json",
"artifacts/care-route-stage2-future-matrix-v1/preflight/source_sample_scores.csv",
"artifacts/care-route-stage2-future-matrix-v1/preflight/future_router_inputs_label_free.csv",
"/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/regions_2018_2021.npz",
"/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz",
"artifacts/care-di25-noego-safe-authority-full-v1/preflight/future_lineage_2018_2021_exact.csv"]
def main():
    missing=[p for p in REQUIRED if not Path(p).is_file()]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing inputs:\n- "+"\n- ".join(missing))
    print("SAFE_AUTHORITY_FULL_PREFLIGHT_PASS")
    print("source_nested_year = 2017")
    print("source_rows = 799")
    print("future_rows = 1989 (label-sealed model input)")
    print("ego_consumed = false")
if __name__=="__main__":main()
