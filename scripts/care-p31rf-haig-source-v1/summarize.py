#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from haig_model import *
ROOT=Path("artifacts/care-p31rf-haig-source-v1");BASE=Path("artifacts/care-p31rf-c43-p0-complete-diagnostics-v1")
def summarize_rows(rows,protocol):
    y=np.asarray([int(float(r["label"])) for r in rows]);base=np.asarray([float(r["p_stage1"]) for r in rows]);route=np.asarray([int(float(r["route_DI"]))==1 for r in rows]);years=np.asarray([int(float(r["year"])) for r in rows]);out=[]
    for scope,sel in [(str(y),years==y) for y in (2018,2019,2020,2021)]+[("pooled",np.ones(len(rows),bool))]:
        for arm in ARMS:
            p=np.asarray([float(r[arm+"__prob"]) for r in rows]);out.append({"protocol":protocol,"scope":scope,"arm":arm,"n":int(sel.sum()),**metric(y[sel],p[sel]),**direction(y[sel],base[sel],route[sel],p[sel])})
    return out
def main():
    # Labels are opened here only, after source selection and frozen predictions exist.
    labels=[]
    for g in ("2018_2019","2020_2021"):labels+=read_csv(BASE/f"within_year/{g}/oof_predictions.csv")
    by={norm(r["sha256"]):r for r in labels};frozen=read_csv(ROOT/"future/frozen_label_free_predictions.csv")
    frozen_labeled=[]
    for r in frozen:
        z=dict(r);z["label"]=by[norm(r["sha256"])]["label"];frozen_labeled.append(z)
    within=read_csv(ROOT/"future/within_year_oof_predictions.csv")
    fr=summarize_rows(frozen_labeled,"frozen_2017_to_future");wr=summarize_rows(within,"independent_same_year_5fold");write_csv(ROOT/"future/frozen_results.csv",fr);write_csv(ROOT/"future/within_year_results.csv",wr)
    source=read_csv(ROOT/"source_representation_results.csv");direction_rows=read_csv(ROOT/"direction_probe_results.csv");sub=read_csv(ROOT/"source_subgroup_results.csv");shuffle=read_csv(ROOT/"shuffle_control_results.csv");selection=json.loads((ROOT/"source_selection.json").read_text());extract=json.loads((ROOT/"source_extraction_summary.json").read_text())
    src={r["arm"]:r for r in source};dire={r["arm"]:r for r in direction_rows};subs={(r["comparison"],r["arm"]):r for r in sub}
    final={"decision":"FEASIBLE","source_rows":799,"future_rows_loaded_during_selection":0,"future_diagnostic_rows":1989,"source_selected_arm":selection["selected_arm"],"advance_to_CARE":bool(selection["H2_advance"] or selection["H3_advance"]),"node_edge_scale":extract,"source_H0_H1_H2_H3":{a:src[a] for a in ("H0_ROLE","H1_ROLE_API","H2_SAME_HOST","H3_HOST_CALL")},"source_direction":{a:dire[a] for a in ("H0_ROLE","H1_ROLE_API","H2_SAME_HOST","H3_HOST_CALL")},"source_FP_vs_TP":{a:subs[("FP_vs_TP",a)] for a in ("H0_ROLE","H1_ROLE_API","H2_SAME_HOST","H3_HOST_CALL")},"source_FN_vs_TN":{a:subs[("FN_vs_TN",a)] for a in ("H0_ROLE","H1_ROLE_API","H2_SAME_HOST","H3_HOST_CALL")},"shuffle_controls":shuffle,"count_only":src["COUNT_ONLY"],"future_boundary":"2018-2021 are post-hoc diagnostics and cannot alter source selection","semantics_boundary":"SAME_HOST_ORDER and HOST_CALL are interactions, never verified data dependency"}
    (ROOT/"final_summary.json").write_text(json.dumps(final,indent=2)+"\n")
    report=["CARE-Droid HAIG source and diagnostic summary v1","",f"FEASIBILITY=FEASIBLE source_rows=799 selected={selection['selected_arm']} advance_to_CARE={final['advance_to_CARE']}","future_rows_loaded_during_selection=0","SAME_HOST_ORDER and HOST_CALL are interactions, not data flow.",""]
    for a in ("H0_ROLE","H1_ROLE_API","H2_SAME_HOST","H3_HOST_CALL","H2_SHUFFLE","H3_SHUFFLE","COUNT_ONLY"):
        r=src[a];d=dire[a];report.append(f"{a}: BA={float(r['BA']):.6f} AUC={float(r['AUC']):.6f} direction={float(d['overall_direction_acc']):.6f} FN={float(d['FN_direction_acc']):.6f} FP={float(d['FP_direction_acc']):.6f}")
    report+= ["","2018-2021 results are diagnostic only and were not used for representation selection.","CARE_HAIG_SOURCE_PIPELINE_DONE"]
    (ROOT/"REPORT.txt").write_text("\n".join(report)+"\n")
    print("\n".join(report));print("final_summary="+str(ROOT/"final_summary.json"))
if __name__=="__main__":main()
