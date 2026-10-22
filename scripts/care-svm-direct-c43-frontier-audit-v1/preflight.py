#!/usr/bin/env python3
import hashlib
import numpy as np
from common import BOUNDED,DIRECT,FOLDS,OUTPUT,PROJECT,asset,dump_json,load,metrics

def main():
    if not OUTPUT.is_relative_to((PROJECT/"outputs").resolve()):raise RuntimeError("BLOCKED_BY_CONTRACT: unsafe output")
    missing=[str(p) for p in (BOUNDED,DIRECT) if not p.is_file()]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing inputs "+repr(missing))
    ids,y,fold,base,bounded,route,direct,field=load();direct_final=base.copy();direct_final[route]=direct[route]
    s0=metrics(y,base,base,route);b0=metrics(y,base,bounded,route);d0=metrics(y,base,direct_final,route)
    checks={"rows==799":len(ids)==799,"folds_fixed":[int(np.sum(fold==f)) for f in FOLDS]==[160,160,160,160,159],"route==200":int(route.sum())==200,
        "stage1_errors==97":s0["errors"]==97,"bounded_29_4_25":(b0["fixed"],b0["broken"],b0["net"])==(29,4,25),
        "direct_49_17_32":(d0["fixed"],d0["broken"],d0["net"])==(49,17,32)}
    if not all(checks.values()):raise RuntimeError("BLOCKED_BY_CONTRACT: baseline changed "+repr({"checks":checks,"direct":d0,"bounded":b0}))
    OUTPUT.mkdir(parents=True,exist_ok=True);before=[asset(BOUNDED,"frozen SVM bounded-C43 source predictions"),asset(DIRECT,"frozen SVM direct-C43 source predictions")]
    dump_json(OUTPUT/"baseline_asset_manifest_before.json",{"assets":before})
    protocol={"name":"SVM Direct-C43 Unified Decision Frontier Audit v1","domain":"2017 fixed OOF descriptive audit only","stage1":"calibrated P31-SVM",
        "router":"frozen SVM MeanOpposition Top25%","expert":"existing direct C43 probability","thresholds":"one global threshold 0.01..0.99 step 0.01",
        "prohibited":["prediction-side thresholds","class-specific thresholds","gate","future selection","automatic model integration"],
        "practical_region":{"fixed_min":40,"broken_max":10,"net_min":30,"fixed_FN_min":9,"fixed_FP_min":20,"folds_nonnegative_min":4},
        "broad_region":"at least three consecutive 0.01-grid thresholds satisfy every practical-region constraint",
        "strict_pareto":"fixed>29 and broken<=4 with fixed_FN>=9, fixed_FP>=20 and >=4 folds nonnegative",
        "interpretation":"retrospective anatomy may authorize a separate strict nested calibration test; it cannot select a deployable threshold", "future_rows_loaded":0}
    dump_json(OUTPUT/"protocol.json",protocol);(OUTPUT/"protocol_hash.txt").write_text(hashlib.sha256((OUTPUT/"protocol.json").read_bytes()).hexdigest()+"\n",encoding="utf-8")
    dump_json(OUTPUT/"preflight.json",{"status":"PASS","checks":checks,"stage1":s0,"bounded":b0,"direct_t050":d0,"direct_probability_field":field,"future_rows_loaded":0})
    print("DIRECT_C43_FRONTIER_PREFLIGHT_PASS fixed=49 broken=17 net=32",flush=True)
if __name__=="__main__":main()
