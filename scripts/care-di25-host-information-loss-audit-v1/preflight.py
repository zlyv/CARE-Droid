#!/usr/bin/env python3
from pathlib import Path
paths=[Path("artifacts/care-source-local-assets-v2/regions_2017.npz"),Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz")]
missing=[str(p) for p in paths if not p.is_file() or p.stat().st_size<100]
if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
print("HOST_INFORMATION_LOSS_PREFLIGHT_PASS")
print("inputs=regions_2017.npz,endpoint_centered_tensor_cache_2017.jsonl.gz")
print("training_invoked=false")
print("apk_reanalysis_invoked=false")
print("labels_loaded=false")
print("future_rows_loaded=0")
