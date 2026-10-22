#!/usr/bin/env python3
from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np
from core import failure_type,load,write

OUT=Path("artifacts/care-di25-future-stage2-failure-mechanism-v1")

def div(a,b):return float(a/b) if b else 0.0

def main():
    ids,years,y,base,route,q,n0,hosts,member=load()
    base_ok=(base>=.5)==y;q_ok=(q>=.5)==y;n0_ok=(n0>=.5)==y
    target=np.isin(years,(2020,2021))&~base_ok&~route&~q_ok
    if int(target.sum())!=46:raise RuntimeError("BLOCKED_BY_CONTRACT: expected 46 jointly unrepairable misses, got "+str(int(target.sum())))
    rows=[]
    for i in np.flatnonzero(target):
        kind,lb,lq,delta,desired,aligned,required=failure_type(y[i],base[i],q[i])
        true_prob=np.where(y[i]==1,member[i],1-member[i])
        if n0_ok[i]:graph_effect="N0_CORRECT_Q_WRONG_GRAPHGLOBAL_QUERY_HARM_ASSOCIATION"
        elif abs(q[i]-y[i])<abs(n0[i]-y[i]):graph_effect="BOTH_WRONG_Q_MOVED_TOWARD_TRUE"
        elif abs(q[i]-y[i])>abs(n0[i]-y[i]):graph_effect="BOTH_WRONG_Q_MOVED_AWAY_FROM_TRUE"
        else:graph_effect="BOTH_WRONG_NO_CHANGE"
        rows.append({"sha256":ids[i],"year":int(years[i]),"label":int(y[i]),"class":"malware_FN" if y[i] else "benign_FP",
          "p_stage1":base[i],"p_Q":q[i],"p_N0":n0[i],"stage1_confidence":max(base[i],1-base[i]),
          "stage1_logit":lb,"Q_logit":lq,"effective_delta":delta,"desired_sign":desired,"aligned_delta":aligned,
          "required_delta_to_cross":required,"failure_type":kind,"real_host_count":int(hosts[i]),"host_missing":int(hosts[i]==0),
          "N0_correct":int(n0_ok[i]),"graphglobal_query_effect":graph_effect,
          "committee_true_support_mean":float(true_prob.mean()),"committee_true_votes":int(((member[i]>=.5)==y[i]).sum())})
    summary=[]
    for scope,mask in [("2020",np.asarray([r["year"]==2020 for r in rows])),
                       ("2021",np.asarray([r["year"]==2021 for r in rows])),
                       ("pooled_2020_2021",np.ones(len(rows),bool))]:
        take=[r for r,m in zip(rows,mask) if m];fc=Counter(r["failure_type"] for r in take);gc=Counter(r["graphglobal_query_effect"] for r in take)
        summary.append({"scope":scope,"n":len(take),"malware_FN":sum(r["label"]==1 for r in take),"benign_FP":sum(r["label"]==0 for r in take),
          "highconfidence":sum(r["stage1_confidence"]>=.8 for r in take),"host_missing":sum(r["host_missing"] for r in take),
          "host_count_mean":float(np.mean([r["real_host_count"] for r in take])),"N0_correct_Q_wrong":sum(r["N0_correct"] for r in take),
          **{"failure__"+k:v for k,v in sorted(fc.items())},**{"graph_effect__"+k:v for k,v in sorted(gc.items())}})
    OUT.mkdir(parents=True,exist_ok=True);write(OUT/"joint_failure_samples.csv",rows);write(OUT/"summary.csv",summary)
    lines=["CARE-Droid 2020/2021 DI-missed Stage-II failure mechanism audit v1","",
      "Read-only decomposition of the 46 Stage-I-wrong, DI-unrouted, Q-still-wrong APKs.",
      "No training, inference, threshold search, route/model change, or feature re-extraction.",
      "Q versus N0 is a block-level association for GraphGlobal->Attention, not causal feature attribution.",""]
    for s in summary:
        lines.append(f"{s['scope']}: N={s['n']} malware_FN={s['malware_FN']} benign_FP={s['benign_FP']} highconf={s['highconfidence']} host_missing={s['host_missing']} N0_correct_Q_wrong={s['N0_correct_Q_wrong']}")
        for k,v in s.items():
            if k.startswith("failure__") or k.startswith("graph_effect__"):lines.append("  "+k+"="+str(v))
    pooled=summary[-1];types={k.replace("failure__",""):v for k,v in pooled.items() if k.startswith("failure__")}
    primary=max(types,key=types.get)
    lines += ["","PRIMARY_FAILURE_MECHANISM="+primary,
      "Interpretation boundary:",
      "WRONG_DIRECTION suggests misleading/shifted learned evidence but does not identify a specific raw API or role.",
      "RIGHT_DIRECTION_TOO_WEAK points to correction authority/evidence strength.",
      "BOUND_LIMIT means even the allowed 2.25 logit correction cannot cross the Stage-I decision boundary.",
      "N0-correct/Q-wrong cases associate the GraphGlobal-guided attention path with harm; both-wrong cases remain shared RoleToken/Host/head failures.",
      "Exact role/API attribution requires saved attention/critical-host internals or a separately authorized instrumented replay."]
    (OUT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("STAGE2_FAILURE_MECHANISM_AUDIT_DONE");print("\n".join(lines))

if __name__=="__main__":main()

