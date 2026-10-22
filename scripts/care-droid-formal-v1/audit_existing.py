#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
from common import ROOT,SOURCE,CACHE,METHOD,sha256_file,dump
base=Path.cwd().resolve();hits=[]
terms=("p31-rf","meanopposition","random25","random_25","full100","full routing","runtime","timing","bounded_delta","prebound","drop_","without_","ablation")
for top in (base/"scripts",base/"artifacts"):
    if not top.exists():continue
    for root,dirs,files in os.walk(top):
        dirs[:]=[d for d in dirs if d not in {"__pycache__","vendor","checkpoints","nested","fold_checkpoints"} and d!="care-droid-formal-v1" and not any(token in d.lower() for token in ("future","same-year","same_year","diagnostic","2018","2019","2020","2021","2022","2023","2024"))]
        for name in files:
            path=Path(root)/name
            if any(token in str(path).lower() for token in ("future","same-year","same_year","diagnostic","2018","2019","2020","2021","2022","2023","2024")):continue
            if path.suffix.lower() not in {".py",".json",".txt",".md",".csv",".log"} or path.stat().st_size>4_000_000:continue
            text=path.read_text(encoding="utf-8",errors="ignore")[:200000].lower();found=[t for t in terms if t in text or t in name.lower()]
            if found:hits.append({"path":str(path.relative_to(base)),"terms":found})
items=[]
if SOURCE.exists():items += [
 {"arm":"A2_DI25_FORMAL","status":"REUSED_EXISTING_EXPERIMENT","artifact":str(SOURCE),"sha256":sha256_file(SOURCE),"reason":"current P31-RF / RF-DI25 / A0 Role C43-RQ bounded prediction artifact"},
 {"arm":"A0_STAGE1_ONLY","status":"DERIVED_FROM_FROZEN_PREDICTIONS","artifact":str(SOURCE),"reason":"uses the formal Stage-I probability column only"},
 {"arm":"A1_RANDOM25","status":"MISSING_REQUIRES_LABEL_FREE_ROUTE_DERIVATION","artifact":str(SOURCE),"reason":"reuse the same frozen per-APK C43 candidate with new label-free matched random routes"},
 {"arm":"A3_FULL100","status":"DERIVED_IF_ALL_CANDIDATES_AVAILABLE","artifact":str(SOURCE),"reason":"route all rows to the same frozen per-APK bounded C43 candidate"}]
items.append({"arm":"A4_PREBOUND","status":"A4_NOT_RUN_NO_CANONICAL_PREBOUND_DECISION","reason":"seed-level raw_delta/pre-bound prediction was not saved; inverse reconstruction after probability/delta averaging is invalid"})
timing=[h for h in hits if any(x in h["terms"] for x in ("runtime","timing"))]
dump(ROOT/"existing_experiment_audit.json",{"formal_source_available":SOURCE.exists(),"formal_cache_available":CACHE.exists(),"arms":items,"search_hits":hits[:300],"timing_candidates":timing[:100],"future_files_read":0,"future_rows_loaded":0})
print("FORMAL_EXISTING_EXPERIMENT_AUDIT_DONE")

