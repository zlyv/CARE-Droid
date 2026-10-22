#!/usr/bin/env python3
from pathlib import Path
import ast
from audit_common import *

REQUIRED=(
    PRIOR/"preflight/source_manifest.json",
    PRIOR/"features/candidate_feature_schema.json",
    PRIOR/"features/candidate_features_summary.csv",
    PRIOR/"predictions/candidate_oof_predictions.csv",
    PRIOR/"reports/need_correction_metrics.csv",
    PRIOR/"reports/conditional_rescue.csv",
    PRIOR/"reports/paired_bootstrap.csv",
)

def main():
    missing=[str(p) for p in REQUIRED if not p.is_file()]
    if missing: raise RuntimeError("BLOCKED_BY_CONTRACT: missing prior audit artifacts: "+str(missing))
    manifest=load_json(REQUIRED[0]); cohort=Path(manifest["cohort_file"])
    if not cohort.is_file(): raise RuntimeError("BLOCKED_BY_CONTRACT: canonical cohort missing")
    rows=read_csv(cohort); routed=[r for r in rows if int(float(r["route_DI"]))==1]
    if len(rows)!=1989 or len({norm(r["sha256"]) for r in rows})!=1989 or len(routed)!=int(manifest["routed_n"]):
        raise RuntimeError("BLOCKED_BY_CONTRACT: cohort lineage changed")
    pred=read_csv(REQUIRED[3]); candidates={r["candidate"] for r in pred}
    if not {"A","B","C","ABC"}.issubset(candidates): raise RuntimeError("BLOCKED_BY_CONTRACT: prior OOF candidates missing")
    files=list(REQUIRED)+[cohort]
    before={str(p.resolve()):sha256(p) for p in files}
    (ROOT/"preflight").mkdir(parents=True,exist_ok=True)
    write_json(ROOT/"preflight/source_manifest.json",{"status":"PASS","cohort_file":str(cohort.resolve()),"routed_n":len(routed),"input_sha256":before})
    write_text(ROOT/"preflight/original_sha256_before.txt","".join(f"{h}  {p}\n" for p,h in before.items()))
    forbidden={"torch","tensorflow","xgboost","lightgbm","androguard"}; audit=[]
    for p in Path(__file__).resolve().parent.glob("*.py"):
        tree=ast.parse(p.read_text(encoding="utf-8")); imports=set()
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):imports.update(a.name.split('.')[0] for a in n.names)
            elif isinstance(n,ast.ImportFrom) and n.module:imports.add(n.module.split('.')[0])
        bad=sorted(imports&forbidden);audit.append({"file":str(p),"forbidden":bad})
    if any(r["forbidden"] for r in audit):raise RuntimeError("BLOCKED_BY_CONTRACT: forbidden dependency")
    write_json(ROOT/"preflight/contract.json",{"status":"PASS","same_year_OOF":True,"diagnostic_only":True,"fixed_C":1.0,"fixed_threshold":.5,
               "bootstrap_iterations":10000,"old_prediction_modification":False,"neural_training":False,"APK_reanalysis":False,"static_audit":audit})
    print("HOST_COMPLEMENT_SIZE_CONTROLLED_PREFLIGHT_PASS")
if __name__=="__main__":main()
