#!/usr/bin/env python3
from __future__ import annotations
import csv,sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent.parent;SCRIPTS=PROJECT/"scripts"
for p in (HERE,SCRIPTS,SCRIPTS/"care-stage2only-anomaly-v1/vendor",SCRIPTS/"care-di25-compact43-relationquery-source-v1",SCRIPTS/"care-di25-graphglobal-path-source-v1",SCRIPTS/"care-rolegraph-conditioned-v4-script",SCRIPTS/"care-rgcondattn-dedup-bounded-source-v1",SCRIPTS/"care-malben-host-v2"):
    q=str(p.resolve())
    if q not in sys.path:sys.path.insert(0,q)

UP=Path("artifacts/care-litelocal-p31lr-c43-prov6-source-v1")
REF=UP/"preflight/lr_reference/source_predictions.csv"
PRED=UP/"arms/P0_ZERO6/source_predictions.csv"
CACHE=UP/"cache/P0_ZERO6.npz"
METHOD="P31LR__Opp25__C43Prov6Zero__Bounded"

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))

def main():
    required=[REF,PRED,CACHE,SCRIPTS/"care-di25-compact43-relationquery-source-v1/compact_background.py",SCRIPTS/"care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py"]
    missing=[str(p) for p in required if not p.is_file() or p.stat().st_size<16]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    # Import the reused training stack now so a missing recovered dependency fails before work starts.
    import run_care_rolegraph_conditioned_dualstream_v4 as v4  # noqa:F401
    import compact_background as cb  # noqa:F401
    ref=read(REF);pred=read(PRED)
    if len(ref)!=799 or len(pred)!=799:raise RuntimeError("BLOCKED_BY_CONTRACT: source row count changed")
    ids=[r["sha256"].strip().casefold() for r in ref]
    if len(set(ids))!=799 or set(ids)!={r["sha256"].strip().casefold() for r in pred}:raise RuntimeError("BLOCKED_BY_CONTRACT: source identity mismatch")
    needed={"sha256","fold","label","P31_LR__prob","A25__routed"}
    if not needed.issubset(ref[0]):raise RuntimeError("BLOCKED_BY_CONTRACT: P31-LR reference schema changed")
    if METHOD+"__expert_prob" not in pred[0]:raise RuntimeError("BLOCKED_BY_CONTRACT: frozen E0 expert probability absent")
    folds=np.asarray([int(r["fold"]) for r in ref]);labels=np.asarray([int(r["label"]) for r in ref]);prob=np.asarray([float(r["P31_LR__prob"]) for r in ref]);route=np.asarray([int(r["A25__routed"])==1 for r in ref]);error=(prob>=.5).astype(int)!=labels
    fold_counts=Counter(folds.tolist());route_counts=Counter(folds[route].tolist())
    if sorted(fold_counts.values())!=[159,160,160,160,160] or any(route_counts[f]!=40 for f in range(1,6)):raise RuntimeError("BLOCKED_BY_CONTRACT: fixed folds/route changed")
    if int(route.sum())!=200 or int(np.sum(route&error))!=99:raise RuntimeError("BLOCKED_BY_CONTRACT: P31-LR route/error accounting changed")
    if not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)):raise RuntimeError("BLOCKED_BY_CONTRACT: invalid Stage-I probability")
    with np.load(CACHE,allow_pickle=False) as z:
        keys=set(z.files);must={"sha256","host_values","host_offsets","real_host_count","host_was_empty"}
        if not must.issubset(keys):raise RuntimeError("BLOCKED_BY_CONTRACT: Host cache schema changed")
        values=np.asarray(z["host_values"]);offsets=np.asarray(z["host_offsets"]);zids=[str(v).strip().casefold() for v in z["sha256"]]
        if values.ndim!=2 or values.shape[1]!=111 or len(offsets)!=800 or len(zids)!=799:raise RuntimeError("BLOCKED_BY_CONTRACT: expected Prov6 control cache width 111")
        if set(zids)!=set(ids) or int(offsets[-1])!=len(values):raise RuntimeError("BLOCKED_BY_CONTRACT: Host cache identity/offset mismatch")
        if not np.isfinite(values).all():raise RuntimeError("BLOCKED_BY_CONTRACT: NaN/Inf in Host cache")
    print("P31LR_C43_ROLE_DIRECTION_PROBE_PREFLIGHT_PASS")
    print("source_rows=799 year=2017 folds="+",".join(f"{k}:{fold_counts[k]}" for k in range(1,6)))
    print("stage1=P31_LR committee_role=routing_only route=200 routed_errors=99")
    print("references=outer_train_only neural_training_invoked=false future_rows_loaded=0 ego_consumed=false")
    print("attention_gap=no_frozen_per_host_alpha; explicit_nonlearned_aggregations_only")

if __name__=="__main__":main()
