#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path

UP=Path("artifacts/care-litesemantic-noshortcut-source-v1")
OUT=Path("artifacts/care-litesemantic-sizeconditioned-source-v1")
REQ=(UP/"lite_features_and_audit.csv",UP/"feature_contract.json",UP/"source_oof_predictions.csv",UP/"resolved_inputs.json",Path("scripts/care-litesemantic-noshortcut-source-v1/run_source_oof.py"))
def read(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def norm(v):return str(v or "").strip().lower()
def main():
    missing=[str(p) for p in REQ if not p.is_file() or p.stat().st_size<10]
    if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: missing/empty inputs:\n- "+"\n- ".join(missing))
    rows=read(REQ[0]);pred=read(REQ[2]);contract=json.loads(REQ[1].read_text());resolved=json.loads(REQ[3].read_text())
    if len(rows)!=799 or len({norm(r["sha256"]) for r in rows})!=799 or len(pred)!=799 or len(contract["feature_names"])!=82 or resolved.get("year")!=2017:raise RuntimeError("BLOCKED_BY_CONTRACT: upstream LiteSemantic contract changed")
    # Semantic roles named package_installation/package_uninstallation are
    # legitimate behavior features.  Do not confuse them with package_name,
    # which is an identity shortcut.  Enforce an allow-listed namespace plus
    # exact forbidden identity/time fields instead of substring matching.
    names=[str(n).strip().lower() for n in contract["feature_names"]]
    allowed_prefixes=("manifest__","role_presence__","role_proportion__")
    forbidden_exact={"package_name","application_name","certificate","certificate_sha256",
                     "timestamp","signing_time","year","metadata_year","min_sdk","target_sdk"}
    unexpected=[n for n in names if not n.startswith(allowed_prefixes)]
    forbidden_found=[n for n in names if n in forbidden_exact or n.split("__")[-1] in forbidden_exact]
    if unexpected or forbidden_found:
        raise RuntimeError("BLOCKED_BY_CONTRACT: identity/time or unexpected model feature: "+repr({"unexpected":unexpected,"forbidden":forbidden_found}))
    OUT.mkdir(parents=True,exist_ok=True);audit={"passed":True,"source_rows":799,"year":2017,"model_features":82,"apk_reanalysis":False,"future_rows_loaded":0,"raw_size_enters_classifier":False,"upstream_callgraph_constructed":False}
    (OUT/"preflight.json").write_text(json.dumps(audit,indent=2)+"\n")
    print("LITESEMANTIC_SIZECONDITIONED_PREFLIGHT_PASS")
    print("source_rows=799 year=2017 model_features=82")
    print("apk_reanalysis=false future_rows_loaded=0 raw_size_enters_classifier=false")

if __name__=="__main__":main()
