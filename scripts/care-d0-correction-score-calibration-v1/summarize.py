#!/usr/bin/env python3
from __future__ import annotations
import json
import numpy as np
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score
from common import ROOT,BASE,METHOD,read_csv,write_csv,dump

decision=json.loads((ROOT/"strength_signal_decision.json").read_text());prov=json.loads((ROOT/"d0_auc_provenance_audit.json").read_text());src=read_csv(BASE);an=read_csv(ROOT/"d0_correction_score_anatomy.csv")
y=np.asarray([int(r["label"]) for r in src]);b=np.asarray([float(r["Global24__prob"]) for r in src]);p=np.asarray([float(r[METHOD+"__prob"]) for r in src]);fold=np.asarray([int(r["fold"]) for r in src]);bo=(b>=.5)==y;ok=(p>=.5)==y;fixed=(~bo)&ok;broken=bo&(~ok)
m0={"arm":"M0_CURRENT","BA":float(balanced_accuracy_score(y,p>=.5)),"AUC":float(roc_auc_score(y,p)),"F1":float(f1_score(y,p>=.5)),"fixed":int(fixed.sum()),"broken":int(broken.sum()),"net":int(fixed.sum()-broken.sum()),"prediction_changed":int(np.sum((b>=.5)!=(p>=.5))),"mean_absolute_delta":float(np.mean([float(r["abs_bounded_delta"]) for r in an])),"median_absolute_delta":float(np.median([float(r["abs_bounded_delta"]) for r in an]))}
fold_rows=[]
for f in range(1,6):
    z=fold==f;fo=(p[z]>=.5)==y[z];fb=bo[z];fold_rows.append({"arm":"M0_CURRENT","fold":f,"BA":balanced_accuracy_score(y[z],p[z]>=.5),"AUC":roc_auc_score(y[z],p[z]),"F1":f1_score(y[z],p[z]>=.5),"fixed":int(np.sum(~fb&fo)),"broken":int(np.sum(fb&~fo)),"net":int(np.sum(~fb&fo)-np.sum(fb&~fo)),"status":"EVALUATED"})
write_csv(ROOT/"calibration_fold_results.csv",fold_rows)
write_csv(ROOT/"calibration_predictions.csv",[{"sha256":r["sha256"],"fold":r["fold"],"label":r["label"],"arm":"M0_CURRENT","stage1_prob":r["Global24__prob"],"final_prob":r[METHOD+"__prob"],"bounded_delta":r[METHOD+"__bounded_delta"]} for r in src])
side=[{"arm":"M0_CURRENT","fixed_FN":int(np.sum(fixed&(y==1))),"broken_TP_to_FN":int(np.sum(broken&(y==1))),"FN_side_net":int(np.sum(fixed&(y==1))-np.sum(broken&(y==1))),"fixed_FP":int(np.sum(fixed&(y==0))),"broken_TN_to_FP":int(np.sum(broken&(y==0))),"FP_side_net":int(np.sum(fixed&(y==0))-np.sum(broken&(y==0)))}]
write_csv(ROOT/"fp_fn_calibration_audit.csv",side)
write_csv(ROOT/"shuffle_control_results.csv",[{"status":"NOT_RUN","reason":"primary D0 strength score failed the predeclared fold-stability gate; M1-M3 and their shuffle control are prohibited","future_rows_loaded":0}])
if decision["supported"]:raise RuntimeError("BLOCKED_BY_PROTOCOL: unexpected strength support on locked data; this audit-only package must not silently invent a calibration after seeing source outcomes")
summary={"status":"MONOTONIC_CORRECTION_CALIBRATION_NOT_SUPPORTED","stopping_decision":decision["decision"],"reason":"The deployable abs_bounded_delta score has a high pooled AUC but fails the strict-majority fold-consistency requirement; calibration arms M1-M3 were not run.","D0_0805_provenance":prov,"M0":m0,"M1_M2_M3":"NOT_RUN_SIGNAL_UNSTABLE","shuffle_control":"NOT_RUN_BECAUSE_NO_CALIBRATION_ADVANCED","future_rows_loaded":0,"selection_uses_future":False}
dump(ROOT/"final_summary.json",summary)
lines=["CARE-Droid D0 Correction-Score Anatomy v1","","Strict 2017 source-only; P31-RF / RF-DI25 / A0 Role C43-RQ fixed; future_rows_loaded=0.",f"D0_0805_SCORE_NAME={prov['score_name']}",f"D0_0805_SCORE_FORMULA={prov['score_formula']}",f"D0_0805_LABEL_FREE={str(prov['label_free']).lower()}",f"D0_0805_DEPLOYABLE={str(prov['available_at_inference']).lower()}",f"D0_0805_POOLED_AUC={prov['pooled_raw_auc']:.6f}",f"fixed={prov['fixed_n']} broken={prov['broken_n']} eligible_folds={prov['eligible_folds']} consistent_folds={prov['consistent_folds_higher_is_fixed']}"]
for r in prov["fold_results"]:lines.append(f"fold{r['fold']}: fixed={r['fixed_n']} broken={r['broken_n']} auc={r['raw_auc_fixed_positive'] if r['auc_status']=='DEFINED' else r['auc_status']}")
lines += ["",decision["decision"],f"M0: BA={m0['BA']:.6f} AUC={m0['AUC']:.6f} F1={m0['F1']:.6f} fixed={m0['fixed']} broken={m0['broken']} net={m0['net']}","M1/M2/M3: NOT RUN (stopped by protocol before calibration)","future_rows_loaded=0",summary["status"]]
(ROOT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");print("\n".join(lines))

