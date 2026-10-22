#!/usr/bin/env python3
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));from metrics_core import metric,exact
ROOT=Path("artifacts/care-di25-host-relative-diagnostics-v1");ROUTER=Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv");H0=Path("artifacts/care-di25-graphglobal-query-within-year-v1/arms/Q_QUERY_ONLY/within_year_predictions.csv");ARMS=("H0_FULL_Q_LOCKED","HM_R","HC_R","HMC_R");YEARS=(2018,2019,2020,2021)
def read(p):
    with Path(p).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def norm(x):return str(x or "").strip().lower()
def idx(rows):
    d={norm(r["sha256"]):r for r in rows}
    if len(rows)!=1989 or len(d)!=1989:raise RuntimeError("BLOCKED_BY_CONTRACT: cohort changed")
    return d
def main():
    rr=read(ROUTER);ids=[norm(r["sha256"]) for r in rr];years=np.asarray([int(r["year"]) for r in rr]);y=np.asarray([int(r["label"]) for r in rr]);base=np.asarray([float(r["p_stage1"]) for r in rr]);route=np.asarray([int(float(r["route_DI"]))==1 for r in rr])
    pred={"H0_FULL_Q_LOCKED":idx(read(H0))}
    for arm in ARMS[1:]:
        protocol=json.loads((ROOT/"within"/arm/"protocol.json").read_text(encoding="utf-8"))
        if not protocol.get("three_backgrounds_same_host_schema") or protocol.get("ego_consumed") or protocol.get("protocol")!="independent same-year five-fold OOF":raise RuntimeError("BLOCKED_BY_CONTRACT: protocol "+arm)
        pred[arm]=idx(read(ROOT/"within"/arm/"within_year_predictions.csv"))
    expert={a:np.asarray([float(pred[a][s]["p_expert"]) for s in ids]) for a in ARMS};final={a:np.where(route,expert[a],base) for a in ARMS};rows=[];paired=[]
    for arm in ARMS:
        for year in YEARS:
            q=years==year;rows.append(metric(y[q],final[arm][q],base[q],route[q],arm,"annual",year))
        rows.append(metric(y,final[arm],base,route,arm,"pooled_2018_2021"))
    h0=(final["H0_FULL_Q_LOCKED"]>=.5)==y
    for arm in ARMS[1:]:
        c=(final[arm]>=.5)==y
        for scope,q in [(str(v),years==v) for v in YEARS]+[("pooled_2018_2021",np.ones(len(y),bool))]:
            n01=int(np.sum((~h0[q])&c[q]));n10=int(np.sum(h0[q]&(~c[q])));paired.append({"scope":scope,"comparison":arm+"_minus_H0","candidate_only_correct":n01,"H0_only_correct":n10,"paired_net":n01-n10,"exact_p":exact(n01,n10)})
    out=ROOT/"within_results";write(out/"metrics.csv",rows);write(out/"paired_vs_H0.csv",paired)
    lines=["CARE-Droid Host-relative independent within-year OOF diagnostic v1","","Each year independently rebuilds Stage-I, DI25, scaler, and all three same-schema backgrounds.","This is representation sufficiency evidence, not temporal generalization.",""]
    for scope in [*(str(v) for v in YEARS),"pooled_2018_2021"]:
        lines.append(scope)
        for r in rows:
            if (str(r["year"])==scope if scope!="pooled_2018_2021" else r["scope"]==scope):lines.append(f"  {r['method']}: BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']}")
    (out/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");print("HOST_RELATIVE_WITHIN_SUMMARY_PASS");print((out/"REPORT.txt").read_text(encoding="utf-8"))
if __name__=="__main__":main()
