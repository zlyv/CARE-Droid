#!/usr/bin/env python3
from __future__ import annotations
import math
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from common import ROOT,read_csv,write_csv,dump

ID={"sha256","fold","label","route","stage1_prob","candidate_prob","stage1_pred","candidate_pred","outcome","prediction_changed","candidate_delta","stage1_margin","stage2_margin","router_score"}
COUNT={"host_count","active_role_count","role_occurrence_mass"}
rows=read_csv(ROOT/"routed_outcome_table.csv");changed=[r for r in rows if r["outcome"] in {"FIXED","BROKEN"}]
features=[k for k in rows[0] if k not in ID]
audit=[];foldrows=[]
for name in features:
  for comparison,label_value in (("OVERALL",None),("FN_SIDE",1),("FP_SIDE",0)):
    cohort=[r for r in changed if label_value is None or int(r["label"])==label_value]
    a=np.asarray([float(r[name]) for r in cohort]);y=np.asarray([r["outcome"]=="FIXED" for r in cohort],int)
    if len(set(y.tolist()))<2:continue
    fx=a[y==1];br=a[y==0]
    raw=float(roc_auc_score(y,a));sep=max(raw,1-raw);direction=1 if raw>=.5 else -1
    u,p=mannwhitneyu(fx,br,alternative="two-sided")
    consistent=0;eligible=0
    for f in range(1,6):
        s=np.asarray([int(r["fold"])==f for r in cohort]);yf=y[s];af=a[s]
        if len(set(yf.tolist()))<2:continue
        fa=float(roc_auc_score(yf,af));eligible+=1;consistent+=int((fa-.5)*direction>0)
        foldrows.append({"comparison":comparison,"feature":name,"fold":f,"fixed_n":int((yf==1).sum()),"broken_n":int((yf==0).sum()),"raw_auc":fa,"oriented_auc":fa if direction>0 else 1-fa})
    audit.append({"comparison":comparison,"feature":name,"feature_group":"count_only" if name in COUNT else "role_reliability","fixed_n":len(fx),"broken_n":len(br),"fixed_mean":fx.mean(),"fixed_std":fx.std(),"fixed_median":np.median(fx),"broken_mean":br.mean(),"broken_std":br.std(),"broken_median":np.median(br),"raw_auc":raw,"separability_auc":sep,"orientation":direction,"mann_whitney_u":float(u),"p_value":float(p),"rank_biserial_effect":2*raw-1,"eligible_folds":eligible,"consistent_folds":consistent})
write_csv(ROOT/"fixed_broken_feature_audit.csv",audit);write_csv(ROOT/"fixed_broken_fold_audit.csv",foldrows)
rel=[r for r in audit if r["comparison"]=="OVERALL" and r["feature_group"]=="role_reliability"]
top_auc=max(rel,key=lambda r:(r["separability_auc"],r["consistent_folds"]))
# The predeclared rule is conjunctive.  Do not test only the largest-AUC
# feature: a slightly smaller-AUC feature may be the one that is consistent
# across folds (as happened for mean_external_distance in the first run).
qualified=[r for r in rel if float(r["separability_auc"])>=.65 and int(r["consistent_folds"])>=3]
best=max(qualified,key=lambda r:(r["separability_auc"],r["consistent_folds"])) if qualified else top_auc
fold_outcomes={f:{"fixed":sum(int(r["fold"])==f and r["outcome"]=="FIXED" for r in changed),"broken":sum(int(r["fold"])==f and r["outcome"]=="BROKEN" for r in changed)} for f in range(1,6)}
eligible=sum(v["fixed"]>0 and v["broken"]>0 for v in fold_outcomes.values())
# Predeclared diagnostic gate: enough outcomes, >=.65 separation, and same orientation in >=3 eligible folds.
advance=len([r for r in changed if r["outcome"]=="FIXED"])>=5 and len([r for r in changed if r["outcome"]=="BROKEN"])>=5 and eligible>=3 and bool(qualified)
decision={"advance_to_strict_nested_gate":bool(advance),"predeclared_rule":"FIXED>=5, BROKEN>=5, >=3 folds with both outcomes, at least one label-free reliability feature with AUC>=0.65 and >=3 orientation-consistent folds","top_auc_feature":top_auc,"qualifying_features":qualified,"selected_qualifying_feature":best if qualified else None,"best_feature":best,"fold_outcomes":fold_outcomes,"future_rows_loaded":0}
dump(ROOT/"diagnostic_decision.json",decision)
print("FIXED_VS_BROKEN_AUC",best["separability_auc"],"feature",best["feature"],"consistent",best["consistent_folds"])
print("ROLE_RELIABILITY_DIAGNOSTIC_ADVANCE",int(advance))

