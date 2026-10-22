#!/usr/bin/env python3
import hashlib, json
import numpy as np
from audit_common import *

def main():
    if not OUTPUT.is_relative_to((PROJECT/"outputs").resolve()): raise RuntimeError("BLOCKED_BY_CONTRACT: unsafe output")
    required=[FORMAL,DIRECT,P31,FAMILY_SCRIPT/"model_adapter.py",DIRECT_SCRIPT/"model_variants.py"]
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing assets "+repr(missing))
    ids,y,fold,base,route,expert,final,field=load_reference(False); value=metrics(y,base,final,route)
    checks={"rows==799":len(ids)==799,"unique_sha==799":len(set(ids.tolist()))==799,
        "fold_sizes":[int(np.sum(fold==f)) for f in FOLDS]==[160,160,160,160,159],"route==200":int(route.sum())==200,
        "stage1_errors==97":int(np.sum((base>=.5)!=y))==97,"direct_49_17_32":(value["fixed"],value["broken"],value["net"])==(49,17,32)}
    if not all(v for v in checks.values()): raise RuntimeError("BLOCKED_BY_CONTRACT: frozen reference changed "+repr(checks))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    before=[asset(FORMAL,"frozen SVM Stage-I/bounded reference"),asset(DIRECT,"frozen SVM Direct-C43 predictions"),asset(P31,"frozen source P31")]
    dump_json(OUTPUT/"baseline_asset_manifest_before.json",{"assets":before})
    protocol={"name":"SVM Direct-C43 Frozen Representation Collision Audit v1","domain":"2017 fixed outer-fold OOF only",
      "reference":{"threshold":0.5,"fixed":49,"broken":17,"net":32},
      "replay":"same Direct-C43 architecture, folds, seeds and training arguments; replay only exports hidden states",
      "representations":["113-D fused correction input","final-linear-layer penultimate hidden activation"],
      "conditional_domains":["Stage-I predicted benign: FN vs TN","Stage-I predicted malware: TP vs FP"],
      "distances":["outer-train standardized Euclidean","cosine","Ledoit-Wolf Mahalanobis"],"neighbors":[5,10,20],
      "frozen_probes":["logistic","shrinkage LDA","nearest centroid","10-NN"],
      "nulls":["P31","count-like size summary","within-stratum matched representation shuffle"],
      "prohibited":["new C43 training target","threshold selection","calibration","gate","future data","deployment selection"],
      "future_rows_loaded":0}
    dump_json(OUTPUT/"protocol.json",protocol); (OUTPUT/"protocol_hash.txt").write_text(hashlib.sha256((OUTPUT/"protocol.json").read_bytes()).hexdigest()+"\n",encoding="utf-8")
    dump_json(OUTPUT/"preflight.json",{"status":"PASS","checks":checks,"direct":value,"probability_field":field,"future_rows_loaded":0})
    print("C43_REPRESENTATION_COLLISION_PREFLIGHT_PASS",flush=True)
if __name__=="__main__":main()
