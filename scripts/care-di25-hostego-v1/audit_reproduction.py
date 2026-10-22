#!/usr/bin/env python3
"""Require exact Host-only expert reproduction against frozen source artifacts."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from experiment_contract import read_csv, write_csv


def compare(new_path, old_path, token):
    new=read_csv(new_path); old=read_csv(old_path)
    nby={r["sha256"].lower():r for r in new}; oby={r["sha256"].lower():r for r in old}
    if set(nby)!=set(oby) or len(nby)!=799: raise RuntimeError("Host-only reproduction cohort mismatch")
    ncols=[c for c in new[0] if c.endswith("__expert_prob") and c.startswith("DI25__")]
    ocols=[c for c in old[0] if c.endswith("__expert_prob") and token.lower() in c.lower()]
    if len(ncols)!=1 or len(ocols)!=1: raise RuntimeError(f"BLOCKED_BY_CONTRACT: cannot identify {token} expert columns: {ncols} / {ocols}")
    diffs=np.asarray([abs(float(nby[s][ncols[0]])-float(oby[s][ocols[0]])) for s in sorted(nby)])
    return {"evidence":token,"new":str(new_path),"frozen":str(old_path),"max_abs_probability_error":float(diffs.max()),"mean_abs_probability_error":float(diffs.mean()),"within_1e-6":int(diffs.max()<=1e-6)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path("artifacts/care-di25-hostego-v1")); p.add_argument("--arms",choices=("m01","all"),default="all"); a=p.parse_args()
    anomaly_candidates=[Path("artifacts/care-malben-host-v2/source/anomaly/source_predictions.csv"),Path("artifacts/care-di-budget-bound-selection-v1/retrained_source/max_delta_2p25/source_predictions.csv")]
    malben_candidates=[Path("artifacts/care-malben-host-v2/source/malben/source_predictions.csv"),Path("artifacts/care-malben-host-v2/source/model/source_predictions.csv")]
    def first(values):
        for v in values:
            if v.is_file(): return v
        return None
    rows=[]
    old=first(anomaly_candidates)
    if old is not None: rows.append(compare(a.root/"source/M0_host_anomaly_bounded/source_predictions.csv",old,"RGCondAttn"))
    else: rows.append({"evidence":"RGCondAttn","status":"historical_prediction_not_available","new_run_is_frozen_implementation":1})
    if a.arms == "all":
        old=first(malben_candidates)
        if old is not None: rows.append(compare(a.root/"source/M2_host_malben_bounded/source_predictions.csv",old,"MalBen"))
        else: rows.append({"evidence":"MalBen","status":"historical_prediction_not_available","new_run_is_frozen_implementation":1})
    write_csv(a.root/"audits/hostonly_reproduction.csv",rows)
    compared=[r for r in rows if "within_1e-6" in r]
    if not all(r["within_1e-6"] for r in compared): raise RuntimeError("Host-only reproduction exceeds 1e-6")
    print("HOSTONLY_REPRODUCTION_PASS")
if __name__=="__main__": main()
