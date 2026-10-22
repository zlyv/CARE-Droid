#!/usr/bin/env python3
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));from metrics_core import metric,exact
ROOT=Path("artifacts/care-di25-compact43-relationquery-diagnostics-v1")
FUTURE_ROUTER=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/future_router_inputs_label_free.csv")
WITHIN_ROUTER=Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv")
RESOLVED=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json")
H0_FROZEN=Path("artifacts/care-di25-graphglobal-query-future-v1/arms/Q_QUERY_ONLY/future_predictions_label_free.csv")
H0_WITHIN=Path("artifacts/care-di25-graphglobal-query-within-year-v1/arms/Q_QUERY_ONLY/within_year_predictions.csv")
YEARS=(2018,2019,2020,2021)


def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def norm(x):return str(x or "").strip().lower()


def parse_label(x):
    value=norm(x)
    if value in {"1","1.0","malware","malicious","true"}:return 1
    if value in {"0","0.0","benign","goodware","false"}:return 0
    raise RuntimeError("unknown label "+repr(x))


def index(rows,expected=1989):
    result={norm(r.get("sha256") or r.get("sample_id")):r for r in rows}
    if len(rows)!=expected or len(result)!=expected:raise RuntimeError("BLOCKED_BY_CONTRACT: cohort changed")
    return result


def summarize_protocol(name,router_path,h0_path,candidate_path,label_map=None):
    rr=read(router_path);ids=[norm(r.get("sha256") or r.get("sample_id")) for r in rr];years=np.asarray([int(float(r["year"])) for r in rr]);base=np.asarray([float(r.get("p_stage1") or r.get("Global24__prob")) for r in rr]);route=np.asarray([int(float(r["route_DI"]))==1 for r in rr])
    y=np.asarray([label_map[s] for s in ids]) if label_map is not None else np.asarray([int(float(r["label"])) for r in rr])
    h0=index(read(h0_path));candidate=index(read(candidate_path));expert={"H0_LOCKED":np.asarray([float(h0[s]["p_expert"]) for s in ids]),"C43_RELATION_QUERY":np.asarray([float(candidate[s]["p_expert"]) for s in ids])}
    final={"Global24":base,"H0_LOCKED":np.where(route,expert["H0_LOCKED"],base),"C43_RELATION_QUERY":np.where(route,expert["C43_RELATION_QUERY"],base)}
    rows=[];paired=[];samples=[]
    for arm in final:
        for year in YEARS:
            q=years==year;rows.append(metric(y[q],final[arm][q],base[q],route[q] if arm!="Global24" else np.zeros(q.sum(),bool),arm,"annual",year))
        rows.append(metric(y,final[arm],base,route if arm!="Global24" else np.zeros(len(y),bool),arm,"pooled_2018_2021"))
    hcorrect=(final["H0_LOCKED"]>=.5)==y;ccorrect=(final["C43_RELATION_QUERY"]>=.5)==y
    for scope,q in [(str(year),years==year) for year in YEARS]+[("pooled_2018_2021",np.ones(len(y),bool))]:
        n01=int(np.sum((~hcorrect[q])&ccorrect[q]));n10=int(np.sum(hcorrect[q]&(~ccorrect[q])));paired.append({"scope":scope,"comparison":"C43_RELATION_QUERY_minus_H0","candidate_only_correct":n01,"H0_only_correct":n10,"paired_net":n01-n10,"exact_p":exact(n01,n10)})
    for i,sid in enumerate(ids):samples.append({"sha256":sid,"year":int(years[i]),"label":int(y[i]),"route_DI":int(route[i]),"p_stage1":float(base[i]),"p_H0":float(final["H0_LOCKED"][i]),"p_C43_RELATION_QUERY":float(final["C43_RELATION_QUERY"][i]),"H0_correct":int(hcorrect[i]),"C43_correct":int(ccorrect[i])})
    return rows,paired,samples


def make_report(title,rows,paired,path,boundary):
    lines=[title,"",boundary,"C43 Relation31 affects Host attention only; it never directly enters the correction head.","H0 predictions are locked and were not retrained.",""]
    for scope in [*(str(y) for y in YEARS),"pooled_2018_2021"]:
        lines.append(scope)
        for row in rows:
            match=str(row["year"])==scope if scope!="pooled_2018_2021" else row["scope"]==scope
            if match:lines.append(f"  {row['method']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} FP={row['FP']} FN={row['FN']} fixed={row['fixed']} broken={row['broken']} net={row['net']}")
        pair=next(r for r in paired if r["scope"]==scope);lines.append(f"  Paired C43 vs H0: candidate_only={pair['candidate_only_correct']} H0_only={pair['H0_only_correct']} net={pair['paired_net']} p={pair['exact_p']:.6f}")
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text("\n".join(lines)+"\n")


def main():
    frozen_protocol=json.loads((ROOT/"frozen/C43_RELATION_QUERY/protocol.json").read_text());within_protocol=json.loads((ROOT/"within/C43_RELATION_QUERY/protocol.json").read_text())
    for protocol in (frozen_protocol,within_protocol):
        if protocol.get("relation_enters_delta") is not False or protocol.get("ego_consumed") is not False or int(protocol.get("parameter_count",-1))!=32292:raise RuntimeError("BLOCKED_BY_CONTRACT: candidate protocol changed")
    resolved=json.loads(RESOLVED.read_text());label_rows=index(read(Path(resolved["future_label_file_posthoc_only"])));label_map={sid:parse_label(row.get("label",row.get("metadata_label"))) for sid,row in label_rows.items()}
    frozen,fpair,fsamples=summarize_protocol("frozen",FUTURE_ROUTER,H0_FROZEN,ROOT/"frozen/C43_RELATION_QUERY/future_predictions_label_free.csv",label_map)
    within,wpair,wsamples=summarize_protocol("within",WITHIN_ROUTER,H0_WITHIN,ROOT/"within/C43_RELATION_QUERY/within_year_predictions.csv")
    write(ROOT/"frozen_results/metrics.csv",frozen);write(ROOT/"frozen_results/paired_vs_H0.csv",fpair);write(ROOT/"frozen_results/sample_predictions.csv",fsamples);make_report("CARE-Droid Compact43 Relation-Query frozen 2017-to-future diagnostic v1",frozen,fpair,ROOT/"frozen_results/REPORT.txt","Candidate is fitted on 2017 only. Future labels are opened only by this summarizer.")
    write(ROOT/"within_results/metrics.csv",within);write(ROOT/"within_results/paired_vs_H0.csv",wpair);write(ROOT/"within_results/sample_predictions.csv",wsamples);make_report("CARE-Droid Compact43 Relation-Query independent within-year OOF diagnostic v1",within,wpair,ROOT/"within_results/REPORT.txt","Each year independently rebuilds Stage-I, DI25, preprocessing, backgrounds, and Stage-II in strict five-fold OOF.")
    combined=(ROOT/"frozen_results/REPORT.txt").read_text()+"\n"+(ROOT/"within_results/REPORT.txt").read_text()+"\n2018-2021 are already-inspected development diagnostics, not an independent final test.\n"
    (ROOT/"REPORT.txt").write_text(combined);print("COMPACT43_RELATIONQUERY_DIAGNOSTICS_SUMMARY_PASS");print(combined)


if __name__=="__main__":main()
