#!/usr/bin/env python3
import json,hashlib
from pathlib import Path
import joblib,numpy as np
from scipy.stats import spearmanr
from haig_model import *
ROOT=Path("artifacts/care-p31rf-haig-source-v1");REF=Path("artifacts/care-p31-stage1-rf-selection-source-v1/preflight/C_RF_RFROUTE/source_predictions.csv")
def main():
    rows=read_csv(ROOT/"lineage/source_2017.csv");graphs,fail=load_graphs(rows,ROOT/"graphs/source");ref=read_csv(REF);by={norm(r["sha256"]):r for r in ref};contract=[by[norm(r["sha256"])] for r in rows]
    y=np.array([int(float(r["label"])) for r in contract]);fold=np.array([int(float(r["fold"])) for r in contract]);base=np.array([float(r["Global24__prob"]) for r in contract]);route=np.array([int(float(r["A25__routed"]))==1 for r in contract])
    pred={a:np.full(len(rows),np.nan) for a in ARMS};norms={a:np.full(len(rows),np.nan) for a in ARMS};fold_rows=[];contracts=[];vocabularies={}
    for f in range(1,6):
        tr=np.flatnonzero(fold!=f);va=np.flatnonzero(fold==f);bundle=fit_bundle([graphs[i] for i in tr],y[tr],32000+f);p,n=predict_bundle(bundle,[graphs[i] for i in va])
        for a in ARMS:pred[a][va]=p[a];norms[a][va]=n[a]
        for a in ARMS:fold_rows.append({"fold":f,"arm":a,"n":len(va),**metric(y[va],p[a]),**direction(y[va],base[va],route[va],p[a])})
        digest=hashlib.sha256("\n".join(sorted(norm(rows[i]["sha256"]) for i in tr)).encode()).hexdigest()
        vaudit=bundle["encoder"].vocabulary_audit([graphs[i] for i in va]);vocabularies[str(f)]={ch:v["tokens"] for ch,v in vaudit.items()}
        contracts.append({"fold":f,"outer_train_rows":len(tr),"outer_validation_rows":len(va),"outer_train_sha_hash":digest,"hash_dimension":HASH_DIM,"channel_dimension":CHANNEL_DIM,"learned_api_vocabulary":True,"exact_min_train_APK_df":EXACT_MIN_DF,"api_vocab_size":vaudit["api"]["size"],"same_pair_vocab_size":vaudit["same"]["size"],"call_pair_vocab_size":vaudit["call"]["size"],"api_vocab_sha256":vaudit["api"]["sha256"],"same_vocab_sha256":vaudit["same"]["sha256"],"call_vocab_sha256":vaudit["call"]["sha256"],"validation_api_backoff_rate":vaudit["api"]["backoff_rate"],"validation_same_pair_backoff_rate":vaudit["same"]["backoff_rate"],"validation_call_pair_backoff_rate":vaudit["call"]["backoff_rate"],"unseen_api_policy":"family and role backoff","future_rows_loaded":0})
        print(f"HAIG source outer fold {f}/5 done",flush=True)
    result=[];direction_rows=[]
    for a in ARMS:result.append({"arm":a,"n":len(y),**metric(y,pred[a])});direction_rows.append({"arm":a,**direction(y,base,route,pred[a])})
    subgroup=[]
    for name,sel in (("FP_vs_TP",route&(base>=.5)),("FN_vs_TN",route&(base<.5))):
        for a in ARMS:subgroup.append({"comparison":name,"arm":a,"n":int(sel.sum()),**metric(y[sel],pred[a][sel])})
    stats=[g.get("stats",{}) for g in graphs];audit_vars={"graph_method_node_count":np.array([s.get("method_count",0) for s in stats]),"graph_invoke_edge_proxy_count":np.array([s.get("invoke_instruction_count",0) for s in stats]),"Host_count":np.array([s.get("num_hosts",0) for s in stats]),"sensitive_API_count":np.array([s.get("num_api_nodes",0) for s in stats]),"same_edge_count":np.array([s.get("num_same_host_edges",0) for s in stats]),"host_call_edge_count":np.array([s.get("num_host_call_edges",0) for s in stats])}
    shortcut=[]
    def rho(a,b):
        v=float(spearmanr(a,b).statistic);return v if np.isfinite(v) else 0.0
    for a in ARMS:
        for name,v in audit_vars.items():shortcut.append({"arm":a,"audit_variable":name,"corr_embedding_norm":rho(norms[a],v),"corr_score":rho(pred[a],v)})
    for name,v in audit_vars.items():shortcut.append({"arm":"RAW_COUNT_AUDIT","audit_variable":name,"corr_with_label":rho(v,y)})
    prediction=[]
    for i,r in enumerate(rows):
        x={"sha256":norm(r["sha256"]),"fold":int(fold[i]),"label":int(y[i]),"stage1_prob":float(base[i]),"route":int(route[i])}
        for a in ARMS:x[a+"__prob"]=float(pred[a][i]);x[a+"__embedding_norm"]=float(norms[a][i])
        prediction.append(x)
    shuffle=[{"real_arm":"H2_SAME_HOST","shuffle_arm":"H2_SHUFFLE","real_direction_acc":q1["overall_direction_acc"],"shuffle_direction_acc":q2["overall_direction_acc"],"delta":q1["overall_direction_acc"]-q2["overall_direction_acc"]} for q1,q2 in [(next(r for r in direction_rows if r["arm"]=="H2_SAME_HOST"),next(r for r in direction_rows if r["arm"]=="H2_SHUFFLE"))]]
    q1=next(r for r in direction_rows if r["arm"]=="H3_HOST_CALL");q2=next(r for r in direction_rows if r["arm"]=="H3_SHUFFLE");shuffle.append({"real_arm":"H3_HOST_CALL","shuffle_arm":"H3_SHUFFLE","real_direction_acc":q1["overall_direction_acc"],"shuffle_direction_acc":q2["overall_direction_acc"],"delta":q1["overall_direction_acc"]-q2["overall_direction_acc"]})
    write_csv(ROOT/"source_representation_results.csv",result);write_csv(ROOT/"source_fold_results.csv",fold_rows);write_csv(ROOT/"direction_probe_results.csv",direction_rows);write_csv(ROOT/"source_subgroup_results.csv",subgroup);write_csv(ROOT/"shortcut_audit.csv",shortcut);write_csv(ROOT/"source_oof_predictions.csv",prediction);write_csv(ROOT/"failures.csv",fail);write_csv(ROOT/"shuffle_control_results.csv",shuffle);write_csv(ROOT/"fold_feature_contracts.csv",contracts)
    (ROOT/"fold_vocabularies.json").write_text(json.dumps(vocabularies,sort_keys=True)+"\n")
    q={r["arm"]:r for r in direction_rows};sub={(r["comparison"],r["arm"]):r for r in subgroup};fold_by={(r["fold"],r["arm"]):r for r in fold_rows}
    def qualifies(a,control):
        return q[a]["FP_direction_acc"]>=q[control]["FP_direction_acc"]+.05 and q[a]["FN_direction_acc"]>=q[control]["FN_direction_acc"]-.05 and q[a]["overall_direction_acc"]>q[control]["overall_direction_acc"] and sub[("FP_vs_TP",a)]["AUC"]>sub[("FP_vs_TP",control)]["AUC"] and sum(fold_by[(f,a)]["overall_direction_acc"]>=fold_by[(f,control)]["overall_direction_acc"] for f in range(1,6))>=3
    h1=qualifies("H1_ROLE_API","H0_ROLE")
    h2=qualifies("H2_SAME_HOST","H1_ROLE_API") and q["H2_SAME_HOST"]["overall_direction_acc"]>=q["H2_SHUFFLE"]["overall_direction_acc"]+.02
    h3=qualifies("H3_HOST_CALL","H2_SAME_HOST") and q["H3_HOST_CALL"]["overall_direction_acc"]>=q["H3_SHUFFLE"]["overall_direction_acc"]+.02
    selected="H3_HOST_CALL" if h3 else ("H2_SAME_HOST" if h2 else ("H1_ROLE_API" if h1 else "H0_ROLE"))
    selection={"selected_arm":selected,"H1_advance":h1,"H2_advance":h2,"H3_advance":h3,"selection_source_only":True,"future_rows_loaded":0,"frozen_arms_for_diagnostic":list(ARMS),"rule":"FP direction improves >=5pp and FP-vs-TP improves; FN drop <=5pp; overall improves; >=3/5 folds noninferior; relation arms additionally beat topology-preserving shuffle by >=2pp"}
    (ROOT/"source_selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    bundle=fit_bundle(graphs,y,32999);(ROOT/"models").mkdir(parents=True,exist_ok=True);joblib.dump(bundle,ROOT/"models/frozen_source_bundle.joblib",compress=3)
    print("SOURCE_PROBE_DONE");print("SOURCE_SELECTED_ARM="+selected)
if __name__=="__main__":main()

