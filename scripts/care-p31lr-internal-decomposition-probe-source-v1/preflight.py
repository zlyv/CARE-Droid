#!/usr/bin/env python3
from pathlib import Path
import csv,numpy as np
UP=Path("artifacts/care-litelocal-p31lr-c43-prov6-source-v1")
REQ=[UP/"preflight/lr_reference/source_predictions.csv",UP/"arms/P0_ZERO6/source_predictions.csv",UP/"cache/P0_ZERO6.npz",Path("scripts/care-litelocal-p31lr-c43-prov6-source-v1/prov6_background.py"),Path("scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py")]
def read(p):
    with p.open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def main():
    missing=[str(p) for p in REQ if not p.is_file() or p.stat().st_size<100]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    rows=read(REQ[0]);y=np.asarray([int(r["label"]) for r in rows]);p=np.asarray([float(r["P31_LR__prob"]) for r in rows]);route=np.asarray([int(r["A25__routed"])==1 for r in rows]);fold=np.asarray([int(r["fold"]) for r in rows])
    if len(rows)!=799 or len({r["sha256"] for r in rows})!=799 or route.sum()!=200 or np.sum(route&((p>=.5)!=y))!=99 or [route[fold==f].sum() for f in range(1,6)]!=[40]*5:raise RuntimeError("BLOCKED_BY_CONTRACT: P31-LR route changed")
    with np.load(REQ[2],allow_pickle=False) as z:
        if z["host_values"].shape[1]!=111 or len(z["sha256"])!=799 or not np.isfinite(z["host_values"]).all():raise RuntimeError("BLOCKED_BY_CONTRACT: Host cache changed")
    print("P31LR_INTERNAL_DECOMPOSITION_PREFLIGHT_PASS")
    print("source_rows=799 route=200 routed_errors=99 folds=40,40,40,40,40")
    print("references=outer_train_only future_rows_loaded=0 neural_training_invoked=false")
if __name__=="__main__":main()
