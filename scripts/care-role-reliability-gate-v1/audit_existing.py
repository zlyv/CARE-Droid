#!/usr/bin/env python3
from __future__ import annotations
import os,re
from pathlib import Path
from common import ROOT,dump

PROJECT=Path.cwd().resolve();SELF="care-role-reliability-gate-v1"
TERMS=("gate","gating","fixkeep","benefit","beneficial","harm","harmful","reliability","fixed","broken","correction")
EXACT=("p31rf","rfroute","c43","fixed","broken","nested","future_rows_loaded")
hits=[]
for top in (PROJECT/"scripts",PROJECT/"artifacts"):
    if not top.exists():continue
    for base,dirs,files in os.walk(top):
        dirs[:]=[d for d in dirs if d not in {"__pycache__","vendor","checkpoints"} and SELF not in d]
        for name in files:
            p=Path(base)/name
            if p.suffix.lower() not in {".py",".txt",".md",".json",".csv"} or p.stat().st_size>3_000_000:continue
            key=(str(p)+" "+name).lower()
            if not any(t in key for t in TERMS):
                try:head=p.read_text(encoding="utf-8",errors="ignore")[:120000].lower()
                except Exception:continue
                if not any(t in head for t in TERMS):continue
            else:
                try:head=p.read_text(encoding="utf-8",errors="ignore")[:120000].lower()
                except Exception:head=""
            score=sum(t in head or t in key for t in EXACT)
            if score>=3:hits.append({"path":str(p.relative_to(PROJECT)),"equivalence_token_score":score,"tokens":[t for t in EXACT if t in head or t in key]})

# Known nearby experiments are deliberately recorded even if a server copy was pruned.
known=[
 {"name":"error-aware-residual-2017-v1","expected_script":"outputs/error-aware-residual-2017-v1-script/run_error_aware_residual_2017.py","stage1":"Compact76","router":"uncertainty Top20%","stage2":"Stats/Pair residual","target":"coarse error / preservation","why_not_equivalent":"not P31-RF, not RF-DI25, not Role C43-RQ, not post-hoc FIXED-vs-BROKEN acceptance"},
 {"name":"care-bounded-fixkeep-v3","expected_artifact":"artifacts/care-bounded-fixkeep-v3-full","stage1":"Global24","router":"A25","stage2":"old dual-prototype CARE","target":"weighted correction loss","why_not_equivalent":"not P31-RF/current C43 and no correction acceptance classifier"},
 {"name":"global24-rolefree-reliability-gate-v1","expected_artifact":"artifacts/global24-rolefree-reliability-gate-v1","stage1":"Global24","router":"budgeted role-free reliability","stage2":"none/current-incompatible","target":"routing coverage","why_not_equivalent":"routing gate, not current correction acceptance gate"},
]
equivalent=[]
for h in hits:
    if h["equivalence_token_score"]==len(EXACT):equivalent.append(h)
# Token co-occurrence is not enough to certify semantic equivalence. Require an explicit current gate marker.
certified=[]
for h in equivalent:
    p=PROJECT/h["path"]
    text=p.read_text(encoding="utf-8",errors="ignore").lower()
    if "correction_acceptance_gate" in text and "strict_nested_c43" in text:certified.append(h)
status="FOUND" if certified else ("PARTIAL" if hits or known else "NOT_FOUND")
out={"status":status,"equivalent":bool(certified),"certified_matches":certified,"nearby_matches":sorted(hits,key=lambda x:(-x["equivalence_token_score"],x["path"]))[:80],"known_partial_experiments":known,"search_roots":["scripts","artifacts"],"future_rows_loaded":0}
dump(ROOT/"existing_experiment_audit.json",out)
print("ROLE_RELIABILITY_EXISTING_EXPERIMENT:",status)
if status=="FOUND":print("ROLE_RELIABILITY_REUSE_EXISTING_RESULT")

