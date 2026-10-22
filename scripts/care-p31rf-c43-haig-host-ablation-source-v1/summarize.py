#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score
ROOT=Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1");ARMS=("A0_ROLE_ONLY","A1_API_NO_ROLE","A2_H3_NO_ROLE","A3_ROLE_PLUS_H3","A4_H3_SHUFFLE_NO_ROLE")
def read(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write(p,rows):
    with Path(p).open("w",encoding="utf-8",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ref=sorted(read("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv"),key=lambda x:x["sha256"]);sha=[r["sha256"] for r in ref];y=np.asarray([int(r["label"]) for r in ref]);base=np.asarray([float(r["Global24__prob"]) for r in ref]);route=np.asarray([int(r["A25__routed"])==1 for r in ref]);baseok=(base>=.5)==y;fold=np.asarray([int(r["fold"]) for r in ref]);agg=[];fr=[];ok={}
    for arm in ARMS:
        method="P31RF__RFRoute__"+arm+"__Bounded";rows=sorted(read(ROOT/"arms"/arm/"source_predictions.csv"),key=lambda x:x["sha256"])
        if [r["sha256"] for r in rows]!=sha:raise RuntimeError("BLOCKED_BY_CONTRACT: cohort mismatch")
        p=np.asarray([float(r[method+"__prob"]) for r in rows]);q=p>=.5;fixed=~baseok&(q==y);broken=baseok&~(q==y);row={"arm":arm,"BA":balanced_accuracy_score(y,q),"AUC":roc_auc_score(y,p),"F1":f1_score(y,q),"fixed":int(fixed.sum()),"broken":int(broken.sum()),"net":int(fixed.sum()-broken.sum()),"fixed_FN":int(np.sum(fixed&(y==1))),"fixed_FP":int(np.sum(fixed&(y==0))),"broken_TN_to_FP":int(np.sum(broken&(y==0))),"broken_TP_to_FN":int(np.sum(broken&(y==1)))};agg.append(row);ok[arm]=q==y
        for f in range(1,6):
            s=fold==f;fx=~baseok[s]&ok[arm][s];br=baseok[s]&~ok[arm][s];fr.append({"arm":arm,"fold":f,"BA":balanced_accuracy_score(y[s],q[s]),"fixed":int(fx.sum()),"broken":int(br.sum()),"net":int(fx.sum()-br.sum())})
    counts=[]
    for arm in ARMS:counts.extend(int(float(r["parameter_n"])) for r in read(ROOT/"arms"/arm/"fit_audit.csv"))
    if len(set(counts))!=1:raise RuntimeError("BLOCKED_BY_CONTRACT: arms are not capacity matched")
    a={r["arm"]:r for r in agg};paired=[]
    for cand,ctrl in (("A1_API_NO_ROLE","A0_ROLE_ONLY"),("A2_H3_NO_ROLE","A1_API_NO_ROLE"),("A3_ROLE_PLUS_H3","A2_H3_NO_ROLE"),("A2_H3_NO_ROLE","A4_H3_SHUFFLE_NO_ROLE")):
        paired.append({"comparison":cand+"_minus_"+ctrl,"candidate_only_correct":int(np.sum(ok[cand]&~ok[ctrl])),"control_only_correct":int(np.sum(~ok[cand]&ok[ctrl]))})
    replace=a["A2_H3_NO_ROLE"]["net"]>=a["A0_ROLE_ONLY"]["net"] and a["A2_H3_NO_ROLE"]["broken"]<=a["A0_ROLE_ONLY"]["broken"]+1 and sum(next(r for r in fr if r["arm"]=="A2_H3_NO_ROLE" and r["fold"]==f)["net"]>=next(r for r in fr if r["arm"]=="A0_ROLE_ONLY" and r["fold"]==f)["net"] for f in range(1,6))>=3
    role_extra=a["A3_ROLE_PLUS_H3"]["net"]>a["A2_H3_NO_ROLE"]["net"] and a["A3_ROLE_PLUS_H3"]["broken"]<=a["A2_H3_NO_ROLE"]["broken"]
    authentic=(paired[-1]["candidate_only_correct"]-paired[-1]["control_only_correct"])>0
    decision="H3_REPLACES_ROLE" if replace and authentic and not role_extra else ("ROLE_REMAINS_COMPLEMENTARY" if role_extra and authentic else "NO_ADVANCE")
    write(ROOT/"results_aggregate.csv",agg);write(ROOT/"results_per_fold.csv",fr);write(ROOT/"paired.csv",paired);summary={"decision":decision,"replace_role_supported":replace,"role_adds_independent_value":role_extra,"real_host_call_beats_shuffle":authentic,"parameter_count":counts[0],"future_rows_loaded":0};(ROOT/"final_summary.json").write_text(json.dumps(summary,indent=2)+"\n");lines=["CARE-Droid P31-RF C43 Host-content HAIG ablation source v1","","Strict 2017 OOF; P31-RF/DI25/max_delta unchanged. Role removal applies only to Host content; RoleGraph/Relation31 remain localization inputs.",""]+[f"{r['arm']}: BA={r['BA']:.6f} AUC={r['AUC']:.6f} F1={r['F1']:.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']} fixed_FN={r['fixed_FN']} fixed_FP={r['fixed_FP']}" for r in agg]+["",f"SOURCE_DECISION={decision}","No future diagnostic is authorized automatically."];(ROOT/"REPORT.txt").write_text("\n".join(lines)+"\n");print("P31RF_C43_HAIG_HOST_ABLATION_SUMMARY_PASS");print("\n".join(lines))
if __name__=="__main__":main()

