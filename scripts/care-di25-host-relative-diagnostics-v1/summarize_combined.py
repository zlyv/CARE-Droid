#!/usr/bin/env python3
"""Apply the predeclared source + frozen + within-year development rule."""
from __future__ import annotations
import csv,json
from pathlib import Path
ROOT=Path("artifacts/care-di25-host-relative-diagnostics-v1")
SOURCE=Path("artifacts/care-di25-host-relative-source-v1/results/aggregate_metrics.csv")
ARMS=("H0_FULL_Q_LOCKED","HM_R","HC_R","HMC_R")
def read(p):
    with Path(p).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def num(r,k):return float(r[k])
def pick(rows,arm,scope="aggregate"):
    hits=[r for r in rows if r.get("arm",r.get("method"))==arm and r.get("scope",scope)==scope]
    if len(hits)!=1:raise RuntimeError(f"BLOCKED_BY_CONTRACT: cannot resolve {arm}/{scope}: {len(hits)}")
    return hits[0]
def main():
    source=read(SOURCE);frozen=read(ROOT/"frozen_results/metrics.csv");within=read(ROOT/"within_results/metrics.csv")
    source_name={"H0_FULL_Q_LOCKED":"H0_FULL_Q_LOCKED","HM_R":"HM_R","HC_R":"HC_R","HMC_R":"HMC_R"}
    # This rule is fixed before diagnostic outcomes are read:
    # no source BA loss > 1 pp; both pooled diagnostic BA/net must not be below H0;
    # no frozen annual BA loss > 1 pp; maximize mean of the three BA values.
    h0s=pick(source,source_name["H0_FULL_Q_LOCKED"]);h0f=pick(frozen,"H0_FULL_Q_LOCKED","pooled_2018_2021");h0w=pick(within,"H0_FULL_Q_LOCKED","pooled_2018_2021")
    out=[]
    for arm in ARMS:
        s=pick(source,source_name[arm]);f=pick(frozen,arm,"pooled_2018_2021");w=pick(within,arm,"pooled_2018_2021")
        annual=[r for r in frozen if r["method"]==arm and r["scope"]=="annual"]
        min_annual_delta=min(num(r,"BA")-num(next(x for x in frozen if x["method"]=="H0_FULL_Q_LOCKED" and x["scope"]=="annual" and x["year"]==r["year"]),"BA") for r in annual)
        eligible=(num(s,"BA")>=num(h0s,"BA")-.01 and num(f,"BA")>=num(h0f,"BA") and int(float(f["net"]))>=int(float(h0f["net"])) and num(w,"BA")>=num(h0w,"BA") and int(float(w["net"]))>=int(float(h0w["net"])) and min_annual_delta>=-.01)
        out.append({"arm":arm,"source_BA":num(s,"BA"),"source_net":int(float(s["net"])),"frozen_pooled_BA":num(f,"BA"),"frozen_pooled_net":int(float(f["net"])),"within_pooled_BA":num(w,"BA"),"within_pooled_net":int(float(w["net"])),"min_frozen_annual_BA_delta_vs_H0":min_annual_delta,"mean_three_BA":(num(s,"BA")+num(f,"BA")+num(w,"BA"))/3,"eligible":eligible})
    eligible=[r for r in out if r["eligible"]]
    selected=max(eligible,key=lambda r:(r["mean_three_BA"],r["frozen_pooled_BA"],r["frozen_pooled_net"],r["within_pooled_BA"]))["arm"] if eligible else "H0_FULL_Q_LOCKED"
    write(ROOT/"combined_selection.csv",out)
    decision={"selection_rule_predeclared":True,"diagnostic_years_used_for_selection":[2018,2019,2020,2021],"independent_final_test":False,"selected_development_method":selected,"eligible_candidates":[r["arm"] for r in eligible],"final_holdout_recommendation":[2022,2023,2024],"warning":"The selected method is a development freeze only; 2018-2021 cannot be reported as an unseen final evaluation."}
    (ROOT/"COMBINED_DECISION.json").write_text(json.dumps(decision,indent=2)+"\n",encoding="utf-8")
    lines=["CARE-Droid Host-relative combined development decision v1","","2017 source, frozen 2017-to-2018/2021, and independent same-year OOF are jointly considered.","Because 2018-2021 participate in selection, they are diagnostic/development evidence, not an independent final test.",""]
    for r in out:lines.append(f"{r['arm']}: source_BA={r['source_BA']:.6f} frozen_BA={r['frozen_pooled_BA']:.6f} within_BA={r['within_pooled_BA']:.6f} mean_BA={r['mean_three_BA']:.6f} eligible={str(r['eligible']).lower()}")
    lines += ["",f"SELECTED_DEVELOPMENT_METHOD={selected}","Reserve 2022-2024 for an untouched final temporal evaluation."]
    (ROOT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");print("HOST_RELATIVE_COMBINED_SELECTION_PASS");print((ROOT/"REPORT.txt").read_text(encoding="utf-8"))
if __name__=="__main__":main()
