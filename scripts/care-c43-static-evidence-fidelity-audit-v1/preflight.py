#!/usr/bin/env python3
from common import *
def main():
    required=[FORMAL,DIRECT,STORE,REPLAY,OLD_FP/"fingerprints.py"]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing assets "+repr(missing))
    replay=json.loads(REPLAY.read_text(encoding="utf-8"))
    ids,y,folds,base,route,expert,final=load_reference();before=(base>=.5)==y;after=(final>=.5)==y
    counts={"source":len(ids),"routed":int(route.sum()),"routed_stage1_errors":int(np.sum(route&~before)),
            "unfixed":int(np.sum(route&~before&~after)),"broken":int(np.sum(route&before&~after))}
    if counts!={"source":799,"routed":200,"routed_stage1_errors":84,"unfixed":35,"broken":17}:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen cohort mismatch "+repr(counts))
    with np.load(STORE,allow_pickle=False) as z:
        keys=set(z.files)
        for fold in range(1,6):
            for name in ("test_index","test_s7","test_fused","test_hidden"):
                if f"f{fold}_{name}" not in keys:raise RuntimeError("BLOCKED_BY_CONTRACT: representation key missing "+f"f{fold}_{name}")
    assets=[asset(p,purpose) for p,purpose in [(FORMAL,"frozen SVM Stage-I predictions"),(DIRECT,"frozen Direct-C43 predictions"),(STORE,"frozen layer representation store"),(REPLAY,"exact replay validation"),(OLD_FP/"fingerprints.py","verified DEX fingerprint implementation")]]
    dump_json(OUTPUT/"baseline_asset_manifest_before.json",{"assets":assets,"formal_scope":"source 2017 only","future_rows_loaded":0})
    dump_json(OUTPUT/"preflight.json",{"status":"PASS","counts":counts,"replay":replay,"future_rows_loaded":0})
    print("C43_STATIC_EVIDENCE_FIDELITY_PREFLIGHT_PASS",counts)
if __name__=="__main__":main()
