#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from common import ROOT,read_csv,write_csv,dump
base=Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1");candidates=list(base.rglob("aggregate_metrics.csv")) if base.exists() else []
rows=[]
for path in candidates:
    try:data=read_csv(path)
    except Exception:continue
    for row in data:
        arm=row.get("arm",row.get("method",""))
        if arm in {"A0_ROLE_ONLY","A1_API_NO_ROLE","A2_H3_NO_ROLE","A3_ROLE_PLUS_H3","A4_H3_SHUFFLE_NO_ROLE"}:rows.append({"experiment":"Host-content HAIG ablation","arm":arm,"status":"REUSED_EXISTING_DESIGN_SPACE_EVIDENCE","formal_internal_ablation":0,"reason":"current P31-RF/DI25 source protocol, but changes prohibited Host content representations rather than toggling a canonical internal C43-RQ module",**row})
if not rows:rows=[{"experiment":"C43 internal-module audit","arm":"NONE","status":"NO_COMPATIBLE_FORMAL_INTERNAL_TOGGLE_FOUND","formal_internal_ablation":0,"reason":"older Relation/attention/localization ablations use a different Stage-I/router or alter experimental definitions; no new module deletion is invented"}]
write_csv(ROOT/"internal_ablation_results.csv",rows);dump(ROOT/"internal_ablation_audit.json",{"compatible_formal_internal_ablations":0,"design_space_rows_reused":len(rows) if rows[0]["arm"]!="NONE" else 0,"decision":"No B-series result is mixed into the formal A-series table","future_rows_loaded":0})
print("INTERNAL_ABLATION_AUDIT_DONE")

