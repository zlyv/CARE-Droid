#!/usr/bin/env python3
import sys
import numpy as np

from ablation_common import FORMAL, OUTPUT, PRIOR_OUTPUT, PRIOR_SCRIPT, PROJECT, asset, dump_json, formal_rows, metrics


def main():
    if not OUTPUT.is_relative_to((PROJECT / "outputs").resolve()): raise RuntimeError("BLOCKED_BY_CONTRACT: unsafe output")
    required = [FORMAL, PRIOR_OUTPUT / "references/SVM/source_predictions.csv", PRIOR_SCRIPT / "run_source_arm.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing prerequisites " + repr(missing))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = formal_rows(); y=np.asarray([int(float(r["label"])) for r in rows]); fold=np.asarray([int(float(r["fold"])) for r in rows])
    base=np.asarray([float(r["stage1_prob"]) for r in rows]); final=np.asarray([float(r["stage2_prob"]) for r in rows]); route=np.asarray([int(float(r["routed"])) for r in rows],bool)
    stage1=metrics(y,base,base,route); bounded=metrics(y,base,final,route)
    checks={"rows==799":len(rows)==799,"folds_fixed":[int(np.sum(fold==f)) for f in range(1,6)]==[160,160,160,160,159],
            "route==200":int(route.sum())==200,"stage1_errors==97":stage1["errors"]==97,
            "bounded_fixed_broken_net":(bounded["fixed"],bounded["broken"],bounded["net"])==(29,4,25)}
    if not all(checks.values()): raise RuntimeError("BLOCKED_BY_CONTRACT: formal baseline changed "+repr(checks))
    dump_json(OUTPUT/"baseline_asset_manifest_before.json", [asset(path,purpose) for path,purpose in [
        (FORMAL,"frozen SVM family Stage-II source predictions"),
        (PRIOR_OUTPUT/"references/SVM/source_predictions.csv","frozen SVM Stage-I/router source reference")]])
    dump_json(OUTPUT/"preflight.json",{"status":"PASS","checks":checks,"stage1":stage1,"bounded":bounded,"future_rows_loaded":0})
    dump_json(OUTPUT/"protocol.json",{"status":"FROZEN_SOURCE_ABLATION","source_rows":799,"folds":[160,160,160,160,159],
        "stage1":"calibrated P31-SVM","router":"SVM-anchored MeanOpposition Top25%","seeds":[0,1,2],
        "U1":"base logit + raw C43 residual; no tanh and no max-delta clip",
        "U2":"direct C43 logit on routed samples; Stage-I logit input zeroed; FixKeep weighting and keep consistency disabled",
        "future_rows_loaded":0})
    print("SVM_C43_REMOVE_BOUNDED_PREFLIGHT_PASS",flush=True)


if __name__=="__main__":main()
