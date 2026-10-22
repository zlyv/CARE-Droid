#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score
from role_api_core import norm,write_csv
ROOT=Path("artifacts/care-role-conditioned-api-v1");OLD=Path("artifacts/care-p31rf-c43-haig-host-ablation-source-v1")
def read(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def load(path,needle,ids):
    rows=read(path);by={norm(r["sha256"]):r for r in rows};cols=[c for c in rows[0] if c.endswith("__prob") and needle in c and "expert" not in c]
    if len(cols)!=1:raise RuntimeError(f"probability column mismatch {needle}: {cols}")
    return np.asarray([float(by[s][cols[0]]) for s in ids])
def logit(p):p=np.clip(np.asarray(p,float),1e-7,1-1e-7);return np.log(p/(1-p))
def metric(y,p):return {"BA":float(balanced_accuracy_score(y,p>=.5)),"AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,p>=.5,zero_division=0))}
def main():
    ref=read("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv");ids=[norm(r["sha256"]) for r in ref];y=np.asarray([int(float(r["label"])) for r in ref]);fold=np.asarray([int(float(r["fold"])) for r in ref]);base=np.asarray([float(r["Global24__prob"]) for r in ref]);route=np.asarray([int(float(r["A25__routed"]))==1 for r in ref]);baseok=(base>=.5)==y
    probs={"R0_LOCKED_ROLE":load(OLD/"arms/A0_ROLE_ONLY/source_predictions.csv","A0_ROLE_ONLY",ids),"R2_FIXED_ROLE_API":load(ROOT/"arms/R2_FIXED_ROLE_API/source_predictions.csv","R2_FIXED_ROLE_API",ids),"R2S_FIXED_SHUFFLED":load(ROOT/"arms/R2S_FIXED_SHUFFLED/source_predictions.csv","R2S_FIXED_SHUFFLED",ids)}
    agg=[];per=[];direction=[];sub=[];correct={};err=route&~baseok;truth=np.where(y==1,1.,-1.)
    for arm,p in probs.items():
        ok=(p>=.5)==y;correct[arm]=ok;fixed=~baseok&ok;broken=baseok&~ok;agg.append({"arm":arm,**metric(y,p),"fixed":int(fixed.sum()),"broken":int(broken.sum()),"net":int(fixed.sum()-broken.sum()),"fixed_FN":int(np.sum(fixed&(y==1))),"fixed_FP":int(np.sum(fixed&(y==0))),"broken_TN_to_FP":int(np.sum(broken&(y==0))),"broken_TP_to_FN":int(np.sum(broken&(y==1)))})
        aligned=(logit(p)-logit(base))*truth;right=aligned>1e-12;fn=err&(y==1);fp=err&(y==0);direction.append({"arm":arm,"overall_direction_acc":float(np.mean(right[err])),"FN_direction_acc":float(np.mean(right[fn])),"FP_direction_acc":float(np.mean(right[fp])),"wrong_direction":int(np.sum(err&~right)),"correct_direction":int(np.sum(err&right)),"mean_aligned":float(np.mean(aligned[err]))})
        for name,s in (("FP_vs_TP",route&(base>=.5)),("FN_vs_TN",route&(base<.5))):sub.append({"comparison":name,"arm":arm,"n":int(s.sum()),**metric(y[s],p[s])})
        for f in range(1,6):
            s=fold==f;fx=~baseok[s]&ok[s];br=baseok[s]&~ok[s];per.append({"arm":arm,"fold":f,**metric(y[s],p[s]),"fixed":int(fx.sum()),"broken":int(br.sum()),"net":int(fx.sum()-br.sum())})
    cand=correct["R2_FIXED_ROLE_API"];ctrl=correct["R2S_FIXED_SHUFFLED"];paired={"comparison":"R2_FIXED_ROLE_API_minus_R2S_FIXED_SHUFFLED","real_only_correct":int(np.sum(cand&~ctrl)),"shuffle_only_correct":int(np.sum(~cand&ctrl)),"paired_net":int(np.sum(cand&~ctrl)-np.sum(~cand&ctrl))}
    # Correlate the actual bounded correction delta, not only representation norm, with audit size variables.
    apk={int(r["apk_index"]):r for r in read(ROOT/"cache/apk_api_audit.csv")};shortcut=[]
    for arm in ("R2_FIXED_ROLE_API","R2S_FIXED_SHUFFLED"):
        delta=logit(probs[arm])-logit(base)
        for name,column in (("HostCount","real_host_count"),("MethodCount","method_count"),("SensitiveAPICount","sensitive_api_occurrence_count"),("UniqueAPICount","unique_api_count"),("RoleCount","role_count"),("GraphNodeCount","graph_node_count"),("GraphEdgeCount","graph_edge_count")):
            x=np.asarray([float(apk[i][column]) for i in range(799)]);shortcut.append({"arm":arm,"variable":name,"pearson_delta":float(pearsonr(delta,x).statistic) if np.std(x)>0 else 0.,"spearman_delta":float(spearmanr(delta,x).statistic) if np.std(x)>0 else 0.,"linear_R2_delta":float(LinearRegression().fit(x[:,None],delta).score(x[:,None],delta)) if np.std(x)>0 else 0.})
    write_csv(ROOT/"r2s_source_results.csv",agg);write_csv(ROOT/"r2s_per_fold.csv",per);write_csv(ROOT/"r2s_direction.csv",direction);write_csv(ROOT/"r2s_subgroups.csv",sub);write_csv(ROOT/"r2s_delta_shortcut.csv",shortcut);write_csv(ROOT/"r2s_paired.csv",[paired])
    by={r["arm"]:r for r in agg};real=by["R2_FIXED_ROLE_API"];shuf=by["R2S_FIXED_SHUFFLED"];noninferior=sum(next(r for r in per if r["arm"]=="R2_FIXED_ROLE_API" and r["fold"]==f)["BA"]>=next(r for r in per if r["arm"]=="R2S_FIXED_SHUFFLED" and r["fold"]==f)["BA"] for f in range(1,6));advance=float(real["BA"])>float(shuf["BA"]) and int(real["net"])>int(shuf["net"]) and paired["paired_net"]>0 and noninferior>=3
    summary={"decision":"FREEZE_R2_FOR_TEMPORAL_DIAGNOSTIC" if advance else "R2_EXACT_API_NOT_ESTABLISHED","real":real,"shuffle":shuf,"paired":paired,"real_noninferior_folds":int(noninferior),"future_rows_loaded":0};(ROOT/"r2s_final_summary.json").write_text(json.dumps(summary,indent=2)+"\n");lines=["CARE-Droid R2 fixed-alpha matched shuffle control","",f"REAL R2: BA={real['BA']:.6f} AUC={real['AUC']:.6f} fixed={real['fixed']} broken={real['broken']} net={real['net']}",f"SHUFFLED R2S: BA={shuf['BA']:.6f} AUC={shuf['AUC']:.6f} fixed={shuf['fixed']} broken={shuf['broken']} net={shuf['net']}",f"paired real-only={paired['real_only_correct']} shuffle-only={paired['shuffle_only_correct']} net={paired['paired_net']}",f"noninferior folds={noninferior}/5",f"DECISION={summary['decision']}","future_rows_loaded=0"];(ROOT/"R2S_REPORT.txt").write_text("\n".join(lines)+"\n");print("R2_FIXED_SHUFFLE_CONTROL_DONE");print("\n".join(lines))
if __name__=="__main__":main()
