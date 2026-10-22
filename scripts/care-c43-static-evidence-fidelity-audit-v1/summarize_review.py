#!/usr/bin/env python3
from __future__ import annotations
from collections import Counter
from common import *

YES={"yes","y","是"};NO={"no","n","否"};UNC={"uncertain","u","不确定"}
def answer(value):
    v=norm(value)
    if v in YES:return "YES"
    if v in NO:return "NO"
    if v in UNC:return "UNCERTAIN"
    return ""
def main():
    form=OUTPUT/"blind_review_form.csv"
    if not form.is_file():raise RuntimeError("BLOCKED_BY_CONTRACT: review form missing")
    rows=read_csv(form);completed=[]
    for row in rows:
        raw=answer(row.get("raw_reliable_distinguishing_fact"));inside=answer(row.get("fact_present_in_formal_input"));kept=answer(row.get("fact_retained_in_hidden_or_score"))
        if raw:
            if raw=="NO":derived="NO_RELIABLE_STATIC_FACT_OBSERVED"
            elif raw=="YES" and inside=="NO":derived="RAW_TO_C43_ENCODING_GAP"
            elif raw=="YES" and inside=="YES" and kept=="NO":derived="C43_REPRESENTATION_OR_DECISION_LOSS"
            elif raw=="YES" and inside=="YES" and kept=="YES":derived="STATIC_FACT_PRESENT_BUT_CLASS_DIRECTION_UNRESOLVED"
            else:derived="INCONCLUSIVE"
            completed.append({**row,"derived_attribution":derived})
    if not completed:
        print("C43_STATIC_EVIDENCE_FIDELITY_REVIEW_PENDING completed=0/52");return
    write_csv(OUTPUT/"semantic_review_completed.csv",completed)
    counts=Counter(r["derived_attribution"] for r in completed);full=len(completed)==52
    decision={"status":"SEMANTIC_REVIEW_COMPLETE" if full else "SEMANTIC_REVIEW_PARTIAL","completed":len(completed),"required":52,
              "attribution_counts":dict(counts),"future_rows_loaded":0,"model_training_authorized":False,
              "claim_boundary":"Review observations localize candidate failure stages; they do not by themselves establish population-level causal mechanisms."}
    dump_json(OUTPUT/"semantic_review_decision.json",decision)
    print("C43_STATIC_EVIDENCE_FIDELITY_REVIEW",decision)
if __name__=="__main__":main()
