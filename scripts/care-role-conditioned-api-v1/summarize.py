#!/usr/bin/env python3
from __future__ import annotations
import csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score
from role_api_core import write_csv,norm,ROLE_NAMES
ROOT=Path("artifacts/care-role-conditioned-api-v1");OLD=Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1")
ARMS=("R0_LOCKED_ROLE","R1_UNCONDITIONAL_API","R2_FIXED_ROLE_API","R3_LEARNED_ROLE_API","R4_SHUFFLED_WITHIN_ROLE")
def read(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def logit(p):p=np.clip(np.asarray(p,float),1e-7,1-1e-7);return np.log(p/(1-p))
def safe_auc(y,p):return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float("nan")
def metrics(y,p):return {"BA":float(balanced_accuracy_score(y,p>=.5)),"AUC":safe_auc(y,p),"F1":float(f1_score(y,p>=.5,zero_division=0))}
def load_prob(path,needle,ids):
    rows=read(path);by={norm(r["sha256"]):r for r in rows};cols=[x for x in rows[0] if x.endswith("__prob") and needle in x and "expert" not in x]
    if len(cols)!=1:raise RuntimeError(f"cannot identify probability {needle}: {cols}")
    return np.asarray([float(by[s][cols[0]]) for s in ids])
def main():
    ref=read("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv");ids=[norm(r["sha256"]) for r in ref];y=np.asarray([int(float(r["label"])) for r in ref]);fold=np.asarray([int(float(r["fold"])) for r in ref]);base=np.asarray([float(r["Global24__prob"]) for r in ref]);route=np.asarray([int(float(r["A25__routed"]))==1 for r in ref]);baseok=(base>=.5)==y
    probs={"R0_LOCKED_ROLE":load_prob(OLD/"arms/A0_ROLE_ONLY/source_predictions.csv","A0_ROLE_ONLY",ids)}
    for arm in ARMS[1:]:probs[arm]=load_prob(ROOT/"arms"/arm/"source_predictions.csv",arm,ids)
    agg=[];per=[];direction=[];sub=[];correct={}
    err=route&~baseok;truth=np.where(y==1,1.,-1.)
    for arm,p in probs.items():
        ok=(p>=.5)==y;correct[arm]=ok;fixed=~baseok&ok;broken=baseok&~ok;row={"arm":arm,**metrics(y,p),"fixed":int(fixed.sum()),"broken":int(broken.sum()),"net":int(fixed.sum()-broken.sum()),"fixed_FN":int(np.sum(fixed&(y==1))),"fixed_FP":int(np.sum(fixed&(y==0))),"broken_TN_to_FP":int(np.sum(broken&(y==0))),"broken_TP_to_FN":int(np.sum(broken&(y==1)))};agg.append(row)
        delta=logit(p)-logit(base);aligned=delta*truth;right=aligned>1e-12;amb=np.abs(aligned)<=1e-12;fn=err&(y==1);fp=err&(y==0)
        direction.append({"arm":arm,"routed_errors":int(err.sum()),"FN":int(fn.sum()),"FP":int(fp.sum()),"overall_direction_acc":float(np.mean(right[err])),"FN_direction_acc":float(np.mean(right[fn])),"FP_direction_acc":float(np.mean(right[fp])),"wrong_direction":int(np.sum(err&~right&~amb)),"ambiguous":int(np.sum(err&amb)),"correct_direction":int(np.sum(err&right)),"mean_aligned":float(np.mean(aligned[err]))})
        for comp,sel in (("FP_vs_TP",route&(base>=.5)),("FN_vs_TN",route&(base<.5))):sub.append({"comparison":comp,"arm":arm,"n":int(sel.sum()),**metrics(y[sel],p[sel])})
        for f in range(1,6):
            s=fold==f;fx=~baseok[s]&ok[s];br=baseok[s]&~ok[s];per.append({"arm":arm,"fold":f,**metrics(y[s],p[s]),"fixed":int(fx.sum()),"broken":int(br.sum()),"net":int(fx.sum()-br.sum())})
    paired=[]
    for cand,ctrl in (("R1_UNCONDITIONAL_API","R0_LOCKED_ROLE"),("R2_FIXED_ROLE_API","R0_LOCKED_ROLE"),("R3_LEARNED_ROLE_API","R0_LOCKED_ROLE"),("R3_LEARNED_ROLE_API","R1_UNCONDITIONAL_API"),("R3_LEARNED_ROLE_API","R4_SHUFFLED_WITHIN_ROLE")):
        paired.append({"comparison":cand+"_minus_"+ctrl,"candidate_only_correct":int(np.sum(correct[cand]&~correct[ctrl])),"control_only_correct":int(np.sum(~correct[cand]&correct[ctrl]))})
    write_csv(ROOT/"source_results.csv",agg);write_csv(ROOT/"source_fold_results.csv",per);write_csv(ROOT/"direction_probe_results.csv",direction);write_csv(ROOT/"subgroup_results.csv",sub);write_csv(ROOT/"shuffle_control_results.csv",paired)
    prediction=[]
    for i,s in enumerate(ids):
        row={"sha256":s,"fold":int(fold[i]),"label":int(y[i]),"P31RF_prob":float(base[i]),"routed":int(route[i])}
        for arm,p in probs.items():row[arm+"__prob"]=float(p[i])
        prediction.append(row)
    write_csv(ROOT/"source_predictions.csv",prediction)
    # Capacity and vocabulary audits.
    vocab=[];params={}
    for arm in ARMS[1:]:
        va=read(ROOT/"arms"/arm/"api_projection_audit.csv");vocab.extend(r for r in va if int(float(r["full_outer_fit"]))==1)
        fa=read(ROOT/"arms"/arm/"fit_audit.csv");params[arm]=sorted({int(float(r["parameter_n"])) for r in fa})
    if len({tuple(v) for v in params.values()})!=1:raise RuntimeError("BLOCKED_BY_CONTRACT: R1-R4 are not capacity matched")
    write_csv(ROOT/"api_vocabulary_audit.csv",vocab)
    # Gate audit combines learned alphas with source coverage; do not use it for retuning.
    raw_stats=read(ROOT/"cache/role_api_statistics.csv");write_csv(ROOT/"role_api_statistics.csv",raw_stats);stats={int(r["role"]):r for r in raw_stats};gates=read(ROOT/"arms/R3_LEARNED_ROLE_API/gate_snapshots.csv");gate=[]
    for role in range(31):
        a=np.asarray([float(r["alpha"]) for r in gates if int(r["role"])==role]);s=stats[role];gate.append({"role":role,"role_name":ROLE_NAMES[role],"alpha_mean":float(a.mean()),"alpha_std":float(a.std()),"api_occurrences":s["api_occurrences"],"unique_api_count":s["unique"],"apk_coverage":s["apk"],"benign_apk_coverage":s["benign_apk"],"malware_apk_coverage":s["malware_apk"],"routed_error_coverage":s["routed_error_apk"]})
    write_csv(ROOT/"role_api_gate_audit.csv",sorted(gate,key=lambda r:-r["alpha_mean"]))
    # Shortcut audit: validation feature norm from first full-fit seed per APK.
    apk={int(r["apk_index"]):r for r in read(ROOT/"cache/apk_api_audit.csv")};shortcut=[]
    for arm in ARMS[1:]:
        rows=read(ROOT/"arms"/arm/"apk_feature_audit.csv");seen={}
        for r in rows:seen.setdefault(int(r["apk_index"]),float(r["api_residual_input_norm"]))
        score=np.asarray([seen[i] for i in range(799)]);variables={"HostCount":np.asarray([float(apk[i]["real_host_count"]) for i in range(799)]),"MethodCount":np.asarray([float(apk[i]["method_count"]) for i in range(799)]),"SensitiveAPICount":np.asarray([float(apk[i]["sensitive_api_occurrence_count"]) for i in range(799)]),"UniqueAPICount":np.asarray([float(apk[i]["unique_api_count"]) for i in range(799)]),"RoleCount":np.asarray([float(apk[i]["role_count"]) for i in range(799)]),"GraphNodeCount":np.asarray([float(apk[i]["graph_node_count"]) for i in range(799)]),"GraphEdgeCount":np.asarray([float(apk[i]["graph_edge_count"]) for i in range(799)])}
        for name,x in variables.items():
            shortcut.append({"arm":arm,"variable":name,"pearson":float(pearsonr(score,x).statistic) if np.std(x)>0 else 0.,"spearman":float(spearmanr(score,x).statistic) if np.std(x)>0 else 0.,"linear_R2":float(LinearRegression().fit(x[:,None],score).score(x[:,None],score)) if np.std(x)>0 else 0.})
    write_csv(ROOT/"shortcut_audit.csv",shortcut);count=read(ROOT/"count_only_results.csv")[0]
    by={r["arm"]:r for r in agg};subby={(r["arm"],r["comparison"]):r for r in sub};r0=by["R0_LOCKED_ROLE"];r3=by["R3_LEARNED_ROLE_API"]
    folds=sum(next(r for r in per if r["arm"]=="R3_LEARNED_ROLE_API" and int(r["fold"])==f)["BA"]>=next(r for r in per if r["arm"]=="R0_LOCKED_ROLE" and int(r["fold"])==f)["BA"] for f in range(1,6))
    real_shuffle=next(r for r in paired if r["comparison"].endswith("R4_SHUFFLED_WITHIN_ROLE"));fp_gain=float(subby[("R3_LEARNED_ROLE_API","FP_vs_TP")]["AUC"])-float(subby[("R0_LOCKED_ROLE","FP_vs_TP")]["AUC"]);fn_change=float(subby[("R3_LEARNED_ROLE_API","FN_vs_TN")]["AUC"])-float(subby[("R0_LOCKED_ROLE","FN_vs_TN")]["AUC"])
    advance=(float(r3["BA"])>float(r0["BA"]) and int(r3["net"])>int(r0["net"]) and fp_gain>0 and fn_change>=-.02 and folds>=3 and real_shuffle["candidate_only_correct"]>real_shuffle["control_only_correct"])
    summary={"decision":"ADVANCE_R3" if advance else "NO_ADVANCE","R0_R4":agg,"FP_vs_TP_AUC_gain_R3_vs_R0":fp_gain,"FN_vs_TN_AUC_change_R3_vs_R0":fn_change,"R3_noninferior_BA_folds":int(folds),"real_vs_shuffled":real_shuffle,"count_only":count,"parameter_counts":params,"future_rows_loaded":0,"supports_exact_API_refines_not_replaces_role":bool(advance),"recommend_integrate_R3_into_CARE":bool(advance)}
    (ROOT/"final_summary.json").write_text(json.dumps(summary,indent=2)+"\n");lines=["CARE-Droid Role-conditioned API refinement source v1","","Strict 2017 five-fold OOF; P31-RF/DI25/max_delta fixed; future_rows_loaded=0."]+[f"{r['arm']}: BA={float(r['BA']):.6f} AUC={float(r['AUC']):.6f} F1={float(r['F1']):.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']} fixed_FN={r['fixed_FN']} fixed_FP={r['fixed_FP']}" for r in agg]+["",f"FP-vs-TP AUC gain R3-R0={fp_gain:+.6f}",f"FN-vs-TN AUC change R3-R0={fn_change:+.6f}",f"R3 noninferior folds={folds}/5",f"SOURCE_DECISION={summary['decision']}","No future diagnostic is authorized automatically."];(ROOT/"REPORT.txt").write_text("\n".join(lines)+"\n");print("SHORTCUT_AUDIT_DONE");print("SHUFFLE_CONTROL_DONE");print("CARE_ROLE_CONDITIONED_API_SOURCE_PIPELINE_DONE");print("\n".join(lines))
if __name__=="__main__":main()
