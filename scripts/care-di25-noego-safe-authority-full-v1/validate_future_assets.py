#!/usr/bin/env python3
"""Deep validation of the HDD future assets before any long training starts."""
from __future__ import annotations
import argparse,gzip,json
from pathlib import Path
import numpy as np
def norm(x):return str(x or "").strip().lower()
def main():
    p=argparse.ArgumentParser();p.add_argument("--regions",type=Path,required=True);p.add_argument("--cache",type=Path,required=True);a=p.parse_args()
    for path in (a.regions,a.cache):
        if not path.is_file() or path.stat().st_size<100:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty future asset "+str(path))
    with np.load(a.regions,allow_pickle=False) as z:
        if "sha256" not in z.files:raise RuntimeError("BLOCKED_BY_CONTRACT: future regions has no sha256")
        ids=[norm(x) for x in z["sha256"].astype(str)]
        required={"host_values","host_offsets","real_host_count"}
        if not required.issubset(z.files):raise RuntimeError("BLOCKED_BY_CONTRACT: future regions missing keys "+repr(sorted(required-set(z.files))))
    if len(ids)!=1989 or len(set(ids))!=1989:raise RuntimeError("BLOCKED_BY_CONTRACT: future regions must contain 1989 unique SHA")
    cache_ids=[]
    with gzip.open(a.cache,"rt",encoding="utf-8") as h:
        for line in h:
            if line.strip():cache_ids.append(norm(json.loads(line).get("sha256")))
    if len(cache_ids)!=1989 or len(set(cache_ids))!=1989 or set(cache_ids)!=set(ids):raise RuntimeError("BLOCKED_BY_CONTRACT: future endpoint cache differs from regions cohort")
    print(f"FUTURE_HDD_ASSET_DEEP_GATE_PASS regions_bytes={a.regions.stat().st_size} cache_bytes={a.cache.stat().st_size} sha=1989")
if __name__=="__main__":main()
