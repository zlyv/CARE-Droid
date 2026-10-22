#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
from common import ROOT,dump
P=Path.cwd().resolve();SELF="care-d0-correction-score-calibration-v1"
terms=("isotonic","monotonic","delta_scale","delta scale","adaptive_delta","adaptive correction","correction strength","soft_gate","needgate","fixkeep","benefit_auc","abs_candidate_delta")
hits=[]
for top in (P/"scripts",P/"artifacts"):
    if not top.exists():continue
    for base,dirs,files in os.walk(top):
        dirs[:]=[d for d in dirs if d not in {"__pycache__","vendor","checkpoints","nested"} and SELF not in d]
        for n in files:
            q=Path(base)/n
            if q.suffix.lower() not in {".py",".json",".txt",".md",".csv"} or q.stat().st_size>3_000_000:continue
            try:t=q.read_text(encoding="utf-8",errors="ignore")[:200000].lower()
            except Exception:continue
            found=[x for x in terms if x in t or x in n.lower()]
            if found:hits.append({"path":str(q.relative_to(P)),"terms":found})
# Exact equivalence requires an explicit current-line continuous monotonic
# magnitude calibration marker; gates and FixKeep losses are not equivalent.
exact=[]
for h in hits:
    t=(P/h["path"]).read_text(encoding="utf-8",errors="ignore").lower()
    if all(x in t for x in ("p31-rf","a0 role c43-rq","monotonic magnitude scaling","direction unchanged")):exact.append(h)
known=[
 {"name":"Role Evidence Reliability Gate v1","status":"PARTIAL","reusable":"A0 outcome table and abs_candidate_delta audit","not_equivalent":"learned accept/reject benefit gate; not continuous monotonic magnitude scaling"},
 {"name":"FixKeep / NeedGate / soft-gate histories","status":"PARTIAL","reusable":"bounded-delta and fixed/broken conventions","not_equivalent":"loss weighting or acceptance gates, generally old Global24/A25 lines; not current P31-RF A0 strength calibration"}
]
status="FOUND" if exact else "PARTIAL"
dump(ROOT/"existing_experiment_audit.json",{"status":status,"equivalent":bool(exact),"exact":exact,"search_hits":hits[:200],"known_partial":known,"future_rows_loaded":0})
print("D0_SCORE_CALIBRATION_EXISTING_EXPERIMENT:",status)
for x in known:print("PARTIAL:",x["name"],"-",x["not_equivalent"])
if exact:print("D0_SCORE_CALIBRATION_REUSE_EXISTING_RESULT")

