#!/usr/bin/env python3
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); from metrics_core import metric, exact
ROOT=Path("artifacts/care-di25-hostrel-semantic-background-diagnostics-v1")
FUTURE_ROUTER=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/future_router_inputs_label_free.csv")
WITHIN_ROUTER=Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv")
RESOLVED=Path("artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json")
H0_FROZEN=Path("artifacts/care-di25-graphglobal-query-future-v1/arms/Q_QUERY_ONLY/future_predictions_label_free.csv")
H0_WITHIN=Path("artifacts/care-di25-graphglobal-query-within-year-v1/arms/Q_QUERY_ONLY/within_year_predictions.csv")
ARMS=("H0_LOCKED","B1_HARD_ROLE","R1_HOSTREL74_HARD_ROLE"); YEARS=(2018,2019,2020,2021)


def read(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))


def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def norm(x): return str(x or "").strip().lower()


def label(x):
    value=norm(x)
    if value in {"1","1.0","malware","malicious","true"}: return 1
    if value in {"0","0.0","benign","goodware","false"}: return 0
    raise RuntimeError("unknown label "+repr(x))


def index(rows):
    result={norm(r.get("sha256") or r.get("sample_id")):r for r in rows}
    if len(rows)!=1989 or len(result)!=1989: raise RuntimeError("BLOCKED_BY_CONTRACT: cohort changed")
    return result


def summarize_protocol(name,router_path,h0_path,prediction_name,label_map=None):
    rr=read(router_path); ids=[norm(r.get("sha256") or r.get("sample_id")) for r in rr]; years=np.asarray([int(float(r["year"])) for r in rr]); base=np.asarray([float(r.get("p_stage1") or r.get("Global24__prob")) for r in rr]); route=np.asarray([int(float(r["route_DI"]))==1 for r in rr])
    y=np.asarray([label_map[s] for s in ids]) if label_map is not None else np.asarray([int(float(r["label"])) for r in rr])
    pred={"H0_LOCKED":index(read(h0_path))}
    for arm in ARMS[1:]:
        protocol=json.loads((ROOT/name/arm/"protocol.json").read_text())
        if not protocol.get("backgrounds_same_host_schema") or protocol.get("ego_consumed") or protocol.get("future_labels_loaded_by_model",False): raise RuntimeError("BLOCKED_BY_CONTRACT: protocol "+name+"/"+arm)
        pred[arm]=index(read(ROOT/name/arm/prediction_name))
    expert={arm:np.asarray([float(pred[arm][sid]["p_expert"]) for sid in ids]) for arm in ARMS}
    final={arm:np.where(route,expert[arm],base) for arm in ARMS}; rows=[]; paired=[]
    for arm in ARMS:
        for year in YEARS:
            q=years==year; rows.append(metric(y[q],final[arm][q],base[q],route[q],arm,"annual",year))
        rows.append(metric(y,final[arm],base,route,arm,"pooled_2018_2021"))
    h0=(final["H0_LOCKED"]>=.5)==y
    for arm in ARMS[1:]:
        correct=(final[arm]>=.5)==y
        for scope,q in [(str(year),years==year) for year in YEARS]+[("pooled_2018_2021",np.ones(len(y),bool))]:
            n01=int(np.sum((~h0[q])&correct[q])); n10=int(np.sum(h0[q]&(~correct[q])))
            paired.append({"scope":scope,"comparison":arm+"_minus_H0","candidate_only_correct":n01,"H0_only_correct":n10,"paired_net":n01-n10,"exact_p":exact(n01,n10)})
    return rows,paired


def report(title,rows,path,boundary):
    lines=[title,"",boundary,"All candidate backgrounds are rebuilt in the same 74-D representation used by their validation Hosts.",""]
    for scope in [*(str(y) for y in YEARS),"pooled_2018_2021"]:
        lines.append(scope)
        for row in rows:
            if (str(row["year"])==scope if scope!="pooled_2018_2021" else row["scope"]==scope): lines.append(f"  {row['method']}: BA={row['BA']:.6f} AUC={row['AUC']:.6f} F1={row['F1']:.6f} fixed={row['fixed']} broken={row['broken']} net={row['net']}")
    path.write_text("\n".join(lines)+"\n")


def main():
    resolved=json.loads(RESOLVED.read_text()); label_rows=index(read(Path(resolved["future_label_file_posthoc_only"]))); label_map={sid:label(row.get("label",row.get("metadata_label"))) for sid,row in label_rows.items()}
    frozen,fpair=summarize_protocol("frozen",FUTURE_ROUTER,H0_FROZEN,"future_predictions_label_free.csv",label_map)
    within,wpair=summarize_protocol("within",WITHIN_ROUTER,H0_WITHIN,"within_year_predictions.csv")
    write(ROOT/"frozen_results/metrics.csv",frozen); write(ROOT/"frozen_results/paired_vs_H0.csv",fpair); report("CARE-Droid HostRel semantic-background frozen temporal diagnostic v1",frozen,ROOT/"frozen_results/REPORT.txt","Models are fitted on 2017 only. Future labels are opened only by this summarizer.")
    write(ROOT/"within_results/metrics.csv",within); write(ROOT/"within_results/paired_vs_H0.csv",wpair); report("CARE-Droid HostRel semantic-background independent within-year OOF diagnostic v1",within,ROOT/"within_results/REPORT.txt","Each year independently rebuilds Stage-I, DI25, preprocessing, backgrounds, and Stage-II in strict five-fold OOF.")
    combined=(ROOT/"frozen_results/REPORT.txt").read_text()+"\n"+(ROOT/"within_results/REPORT.txt").read_text()+"\n2018-2021 are already-inspected development diagnostics and cannot authorize revision from an independent final test.\n"
    (ROOT/"REPORT.txt").write_text(combined)
    print("HOSTREL_SEMANTIC_DIAGNOSTICS_SUMMARY_PASS"); print(combined)


if __name__=="__main__": main()
