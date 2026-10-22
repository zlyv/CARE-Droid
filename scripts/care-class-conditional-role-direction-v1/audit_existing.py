#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
from common import ROOT,dump
PROJECT=Path.cwd().resolve();SELF="care-class-conditional-role-direction-v1"
terms=("dualprototype","dual_prototype","class_conditional","class-conditional","malben","benign prototype","malware prototype","role prototype","shared_role_prototypes","direction probe")
hits=[]
for top in (PROJECT/"scripts",PROJECT/"artifacts"):
 if top.exists():
  for base,dirs,files in os.walk(top):
   dirs[:]=[d for d in dirs if d not in {"__pycache__","vendor","checkpoints"} and SELF not in d]
   for n in files:
    p=Path(base)/n
    if p.suffix.lower() not in {".py",".txt",".md",".json",".csv"} or p.stat().st_size>3_000_000:continue
    try:t=p.read_text(encoding="utf-8",errors="ignore")[:150000].lower()
    except Exception:continue
    found=[q for q in terms if q in t or q in n.lower()]
    if found:hits.append({"path":str(p.relative_to(PROJECT)),"terms":found,"current_tokens":sum(q in t for q in ("p31rf","rfroute","a0_role_only","fp-vs-tp","fn-vs-tn","strict_nested_role_direction"))})
cert=[]
for h in hits:
 t=(PROJECT/h["path"]).read_text(encoding="utf-8",errors="ignore").lower()
 if "strict_nested_role_direction" in t and "p31rf" in t and "a0_role_only" in t:cert.append(h)
known=[
 {"name":"P31-LR role-conditioned class-direction probe","reason":"per-role benign/malware reference and source OOF are reusable concepts, but Stage-I was P31-LR, route contained 99 errors, no current RF-DI25/A0 correction integration"},
 {"name":"P31-C43 class-conditional Host directionality audit","reason":"read-only nearest-reference audit on another Stage-I/representation; no D1-D3 bounded source integration"},
 {"name":"MalBen / CARE dual-prototype","reason":"old Global24/A25 or anomaly-background prototypes; not symmetric per-role class direction on current formal line"}
]
status="FOUND" if cert else ("PARTIAL" if hits or known else "NOT_FOUND")
dump(ROOT/"existing_experiment_audit.json",{"status":status,"equivalent":bool(cert),"certified":cert,"nearby":sorted(hits,key=lambda x:(-x["current_tokens"],x["path"]))[:100],"known_partial":known,"future_rows_loaded":0})
print("CLASS_CONDITIONAL_ROLE_EXISTING_EXPERIMENT:",status)
if cert:print("CLASS_CONDITIONAL_ROLE_REUSE_EXISTING_RESULT")
elif status=="PARTIAL":
 for item in known:print("PARTIAL:",item["name"],"-",item["reason"])

