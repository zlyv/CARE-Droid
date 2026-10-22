#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
from common import ROOT,dump
base=Path.cwd().resolve();self_name="care-local-role-neighborhood-v1"
terms=("knn","k-nearest","nearest_neighbor","nearest-neighbor","neighborhood","memory_bank","memory bank","local prototype","retrieval","cosine neighbor","role neighbor","instance retrieval","support set","nearest prototype")
hits=[]
for top in (base/"scripts",base/"artifacts"):
    if not top.exists():continue
    for root,dirs,files in os.walk(top):
        dirs[:]=[d for d in dirs if d not in {"__pycache__","vendor","checkpoints","fold_checkpoints","nested"} and self_name not in d]
        for name in files:
            path=Path(root)/name
            if path.suffix.lower() not in {".py",".json",".txt",".md",".csv"} or path.stat().st_size>3_000_000:continue
            text=path.read_text(encoding="utf-8",errors="ignore")[:250000].lower()
            found=[term for term in terms if term in text or term in name.lower()]
            if found:hits.append({"path":str(path.relative_to(base)),"terms":found})
exact=[]
for hit in hits:
    text=(base/hit["path"]).read_text(encoding="utf-8",errors="ignore").lower()
    markers=("p31-rf","rf-anchored meanopposition","same role","outer-train memory","k nearest benign","k nearest malware")
    if all(marker in text for marker in markers):exact.append(hit)
known=[
 {"name":"Class-Conditional Role Direction v1","status":"PARTIAL","reusable":"fold-safe Compact43/SelfRole31 representation and D0 outcome definitions","difference":"global/per-Role centroids, not local neighbors"},
 {"name":"P31-C43 directionality/evidence audits","status":"PARTIAL","reusable":"class-conditional distance conventions","difference":"background/prototype distances rather than outer-train same-Role kNN"}
]
status="FOUND" if exact else "PARTIAL"
dump(ROOT/"existing_experiment_audit.json",{"status":status,"equivalent":bool(exact),"exact":exact,"known_partial":known,"search_hits":hits[:250],"future_rows_loaded":0})
print("LOCAL_ROLE_NEIGHBOR_EXISTING_EXPERIMENT:",status)
for item in known:print("PARTIAL:",item["name"],"-",item["difference"])
if exact:print("LOCAL_ROLE_NEIGHBOR_EXISTING_EQUIVALENT = TRUE")

