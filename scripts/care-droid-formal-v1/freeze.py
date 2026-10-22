#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from common import ROOT,sha256_file,dump
repro=json.loads((ROOT/"formal_reproduction_audit.json").read_text())
if repro["status"]!="PASS":raise RuntimeError("FORMAL_REPRODUCTION_FAILED")
script_root=Path("scripts/care-droid-formal-v1");required=("formal_protocol.json","formal_fold_assignments.csv","formal_feature_schema.json","existing_experiment_audit.json","formal_reproduction_audit.json","formal_predictions.csv","formal_ablation_results.csv","formal_ablation_fold_results.csv","random25_seed_results.csv","routing_analysis.csv","budget_curve.csv","formal_prediction_change_audit.csv","runtime_raw.csv","cost_summary.csv","internal_ablation_results.csv","final_summary.json","REPORT.txt")
missing=[name for name in required if not (ROOT/name).exists()]
if missing:raise RuntimeError("BLOCKED_BY_CONTRACT: freeze outputs missing "+str(missing))
artifact_files=[path for path in ROOT.iterdir() if path.is_file() and path.name not in {"formal_freeze_manifest.json","run.log"}]
files=[]
for path in sorted(artifact_files+list(script_root.glob("*.py"))+list(script_root.glob("*.sh"))+list(script_root.glob("*.md")),key=lambda p:str(p)):
    files.append({"path":str(path),"sha256":sha256_file(path),"bytes":path.stat().st_size})
manifest={"formal_method":"CARE_DROID_FORMAL_V1","freeze_complete":True,"reproduction_pass":True,"files":files,"manifest_self_hash_excluded":True,"future_rows_loaded":0}
dump(ROOT/"formal_freeze_manifest.json",manifest);print("FORMAL_FREEZE_COMPLETE");print("CARE_DROID_FORMAL_V1_FROZEN = true")

