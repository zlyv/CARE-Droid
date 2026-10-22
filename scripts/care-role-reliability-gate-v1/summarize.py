#!/usr/bin/env python3
from __future__ import annotations
import json,math
import numpy as np
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score
from common import ROOT,BASE,METHOD,read_csv,write_csv,dump

decision=json.loads((ROOT/"diagnostic_decision.json").read_text())
if not decision["advance_to_strict_nested_gate"]:
    src=read_csv(BASE);y=np.asarray([int(r["label"]) for r in src]);b=np.asarray([float(r["Global24__prob"]) for r in src]);p=np.asarray([float(r[METHOD+"__prob"]) for r in src]);bo=(b>=.5)==y;ok=(p>=.5)==y;fx=(~bo)&ok;br=bo&(~ok)
    agg=[{"arm":"G0_CURRENT","BA":balanced_accuracy_score(y,p>=.5),"AUC":roc_auc_score(y,p),"F1":f1_score(y,p>=.5),"fixed":int(fx.sum()),"broken":int(br.sum()),"net":int(fx.sum()-br.sum())}]
    for name in ("gate_fold_results.csv","gate_predictions.csv","coverage_risk_curve.csv","fp_fn_gate_audit.csv","shuffle_control_results.csv","count_only_results.csv","feature_importance.csv"):write_csv(ROOT/name,[])
    summary={"status":"ROLE_RELIABILITY_GATE_NOT_SUPPORTED","reason":"predeclared FIXED-vs-BROKEN diagnostic gate failed; strict nested gate was not trained","diagnostic":decision,"aggregate":agg,"future_rows_loaded":0}
else:
    agg=read_csv(ROOT/"gate_aggregate.csv");per=read_csv(ROOT/"gate_fold_results.csv");a={r["arm"]:r for r in agg};s={r["arm"]:r for r in read_csv(ROOT/"fp_fn_gate_audit.csv")}
    def num(arm,k):return float(a[arm][k])
    consistency=sum(int(r["net"])>=int(next(x for x in per if x["arm"]=="G0_CURRENT" and x["fold"]==r["fold"])["net"]) for r in per if r["arm"]=="G3_COMBINED")
    auc3=num("G3_COMBINED","benefit_auc");aucs=num("C1_G3_SHUFFLE","benefit_auc");aucc=num("C2_COUNT_ONLY","benefit_auc")
    train_audit=read_csv(ROOT/"gate_training_audit.csv");degenerate=sorted({int(r["outer_fold"]) for r in train_audit if r.get("reason")})
    supported=(not degenerate and np.isfinite(auc3) and auc3>=.65 and int(a["G3_COMBINED"]["net"])>int(a["G0_CURRENT"]["net"]) and int(a["G3_COMBINED"]["broken"])<=int(a["G0_CURRENT"]["broken"]) and num("G3_COMBINED","fixed_retention")>=.60 and num("G3_COMBINED","broken_suppression")>=.25 and consistency>=3 and (not np.isfinite(aucs) or auc3-aucs>=.05) and (not np.isfinite(aucc) or auc3-aucc>=.03))
    summary={"status":"ROLE_RELIABILITY_GATE_SUPPORTED" if supported else "ROLE_RELIABILITY_GATE_NOT_SUPPORTED","predeclared_support_rule":"all five outer gates estimable; G3 benefit AUC>=.65, net>G0, broken<=G0, fixed retention>=.60, broken suppression>=.25, >=3/5 noninferior-net folds, AUC gain >=.05 vs shuffle and >=.03 vs count-only","diagnostic":decision,"aggregate":agg,"side_metrics":list(s.values()),"degenerate_outer_folds":degenerate,"G3_noninferior_net_folds":consistency,"real_minus_shuffle_benefit_auc":auc3-aucs if np.isfinite(auc3) and np.isfinite(aucs) else None,"real_minus_count_benefit_auc":auc3-aucc if np.isfinite(auc3) and np.isfinite(aucc) else None,"future_rows_loaded":0}
dump(ROOT/"final_summary.json",summary)
lines=["CARE-Droid Role Evidence Reliability Gate v1","","Strict 2017 source-only; P31-RF / RF-DI25 / A0 Role C43-RQ fixed; future_rows_loaded=0.",f"Existing experiment audit: {json.loads((ROOT/'existing_experiment_audit.json').read_text())['status']}"]
for r in summary["aggregate"]:lines.append(f"{r['arm']}: BA={float(r['BA']):.6f} AUC={float(r['AUC']):.6f} F1={float(r['F1']):.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']}"+(f" benefit_AUC={float(r['benefit_auc']):.6f} fixed_retention={float(r['fixed_retention']):.6f} broken_suppression={float(r['broken_suppression']):.6f}" if "benefit_auc" in r and str(r['benefit_auc']).lower()!='nan' else ""))
lines += ["",summary["status"],"No future diagnostic was loaded or run."]
(ROOT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");print("\n".join(lines));print(summary["status"])

