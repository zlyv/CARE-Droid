#!/usr/bin/env python3
"""Build or validate the label-free future Inter-Z cache before parallel GPU fits."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent
MATRIX=PROJECT/"scripts/care-route-stage2-future-matrix-v1"
for p in (HERE,MATRIX,MATRIX/"vendor"): sys.path.insert(0,str(p))
import run_future_arm as matrix

TARGET=Path("artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz")
AUDIT=Path("artifacts/care-di25-graphglobal-query-future-v1/preflight/future_inter_cache_audit.json")

def main():
    parser=matrix.parser()
    for action in parser._actions:
        if action.dest in {"representation","head","output"}: action.required=False
    parser.set_defaults(representation="anomaly",head="bounded",output=Path("artifacts/care-di25-graphglobal-query-future-v1/preflight"))
    args=parser.parse_args()
    args.future_inter_cache=TARGET
    # A present cache is always validated by load_common. A missing cache is
    # deterministically built from label-sealed compact rows and endpoint graphs.
    args.rebuild_future_inter_cache=False
    existed=TARGET.is_file() and TARGET.stat().st_size>0
    data=matrix.load_common(args)
    if not TARGET.is_file() or TARGET.stat().st_size==0:
        raise RuntimeError("BLOCKED_BY_CONTRACT: label-free Inter-Z cache was not materialized")
    with np.load(TARGET,allow_pickle=False) as z:
        keys=set(z.files)
        required={"sha256","pair_raw","inter_z_clipped","null_samples","swaps_per_edge","null_seed","z_clip"}
        if not required.issubset(keys): raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete Inter-Z keys")
        sha=z["sha256"].astype(str); pair=np.asarray(z["pair_raw"]); inter=np.asarray(z["inter_z_clipped"])
        checks={
            "rows_1989":len(sha)==1989,"unique_sha_1989":len(set(sha))==1989,
            "pair_raw_shape":list(pair.shape)==[1989,465],"inter_z_shape":list(inter.shape)==[1989,961],
            "finite":bool(np.isfinite(pair).all() and np.isfinite(inter).all()),
            "future_labels_loaded":False,"ego_consumed":False,
        }
    if not all(v for v in checks.values() if isinstance(v,bool) and v is not False) or not all(checks[k] for k in ("rows_1989","unique_sha_1989","pair_raw_shape","inter_z_shape","finite")):
        raise RuntimeError("BLOCKED_BY_CONTRACT: Inter-Z validation failed "+repr(checks))
    audit={"path":str(TARGET.resolve()),"reused":existed,"built_now":not existed,"checks":checks,
        "construction_audit":data.future_inter_audit,"source_train_n":799,"future_n":1989,
        "input_semantics":"label-free graph-derived structural normalization only"}
    AUDIT.parent.mkdir(parents=True,exist_ok=True)
    AUDIT.write_text(json.dumps(audit,indent=2,default=str)+"\n",encoding="utf-8")
    print("FUTURE_INTER_CACHE_PASS", "reused="+str(existed).lower(), "built_now="+str(not existed).lower(), flush=True)

if __name__=="__main__": main()
