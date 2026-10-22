#!/usr/bin/env python3
from __future__ import annotations
import json,platform,subprocess,sys
from pathlib import Path
from common import ROOT,SOURCE,CACHE,METHOD,BASE_FIELD,ROUTE_FIELD,ROUTER_SCORE_FIELD,RANDOM_SEEDS,BUDGETS,MAX_DELTA,ROLE_NAMES,read_csv,write_csv,dump,sha256_file,sha256_text,norm
if not SOURCE.exists() or not CACHE.exists():raise RuntimeError("BLOCKED_BY_CONTRACT: formal source/cache missing")
rows=read_csv(SOURCE);fold_rows=[{"sha256":norm(r["sha256"]),"fold":int(r["fold"]),"label":int(r["label"])} for r in rows];write_csv(ROOT/"formal_fold_assignments.csv",fold_rows)
lineage="\n".join(f"{r['sha256']},{r['fold']},{r['label']}" for r in fold_rows)
try:git_commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=False).stdout.strip() or None
except Exception:git_commit=None
role_schema_hash=sha256_text(json.dumps(ROLE_NAMES,separators=(",",":")))
source_protocol_candidates=[Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1/protocol.json"),Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1/preflight.json")]
lineage_files=[]
for path in (SOURCE,CACHE,*source_protocol_candidates):
    if path.exists():lineage_files.append({"path":str(path),"sha256":sha256_file(path),"bytes":path.stat().st_size})
protocol={
 "formal_method_name":"CARE-Droid P31-RF RF-DI25 Role-C43-RQ Bounded","formal_version":"CARE_DROID_FORMAL_V1",
 "source_year":2017,"source_rows":799,"benign_rows":400,"malware_rows":399,"fold_assignment_path":str(ROOT/"formal_fold_assignments.csv"),"fold_sizes":[160,160,160,160,159],
 "stage1":{"name":"P31-RF","feature_schema":"P31 sensitive semantic-role prevalence","model":"calibrated Random Forest","probability_column_legacy_name":BASE_FIELD,"threshold":.5},
 "router":{"name":"RF-anchored MeanOpposition DI25","score_column":ROUTER_SCORE_FIELD,"route_column":ROUTE_FIELD,"budget":.25,"rule":"top fold-wise 25% MeanOpposition, deterministic SHA tie-break"},
 "stage2":{"name":"Role C43-RQ A0_ROLE_ONLY","prediction_prefix":METHOD,"host_content":"Compact43","localization":"RoleGraph / Relation31 query","training_scope":"all outer-train APKs","ego_consumed":False},
 "bounded_correction":{"formula":"final_logit = stage1_logit + 2.25*tanh(raw_delta)","max_delta":MAX_DELTA,"final_threshold":.5,"saved_formal_candidate":METHOD+"__expert_prob"},
 "random25_seeds":list(RANDOM_SEEDS),"budget_curve":[int(x*100) for x in BUDGETS],"formal_budget_percent":25,"budget_curve_used_for_selection":False,
 "role_schema":list(ROLE_NAMES),"role_schema_sha256":role_schema_hash,"semantic_mapping_lineage":"materialized in frozen A0 Host cache; exact external mapping file is recorded only if present in source protocol",
 "source_lineage_sha256":sha256_text(lineage),"lineage_files":lineage_files,"git_commit":git_commit,"python_version":sys.version,"platform":platform.platform(),
 "formal_artifact_paths":{"source_predictions":str(SOURCE),"host_cache":str(CACHE),"formal_root":str(ROOT)},"future_rows_loaded":0,"selection_uses_future":False
}
dump(ROOT/"formal_protocol.json",protocol);dump(ROOT/"formal_feature_schema.json",{"source_prediction_columns":list(rows[0]),"role_names":list(ROLE_NAMES),"stage1_feature":"P31","host_cache":str(CACHE)})
print("FORMAL_PROTOCOL_BUILT")

