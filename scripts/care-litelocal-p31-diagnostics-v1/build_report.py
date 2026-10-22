#!/usr/bin/env python3
from pathlib import Path

ROOT=Path("artifacts/care-litelocal-p31-diagnostics-v1")

def main():
    parts=[
        "CARE-Droid LiteLocal P31 complete diagnostic v1",
        "",
        "Candidate: P31 role prevalence + calibrated LR/RF/XGB/SVM probability mean.",
        "No inter-method call graph, Host, RoleGraph, Ego, co-occurrence, transition, or Stage-II model is used.",
        "Frozen future and independent same-year results answer different questions and must not be conflated.",
    ]
    for title,path in (("SOURCE SIZE-CONDITIONED ATTRIBUTION",ROOT/"source_shortcut/REPORT.txt"),("FROZEN 2017 -> 2018-2021",ROOT/"frozen_future/REPORT.txt"),("INDEPENDENT SAME-YEAR OOF",ROOT/"within_year/REPORT.txt")):
        if not path.is_file():raise RuntimeError("BLOCKED_BY_CONTRACT: missing report "+str(path))
        parts.extend(["",title,"="*len(title),path.read_text(encoding="utf-8").strip()])
    parts.extend(["","Evidence boundary:","2018-2021 have already been inspected during method development. Frozen-future results are temporal diagnostics, not a new untouched test. Same-year OOF results are not temporal generalization."])
    (ROOT/"REPORT.txt").write_text("\n".join(parts)+"\n",encoding="utf-8")
    print("LITELOCAL_P31_COMPLETE_REPORT_PASS")
    print("\n".join(parts))

if __name__=="__main__":main()
