#!/usr/bin/env python3
"""Post-hoc explanation of frozen DI25 misses in 2020 and 2021."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np

from diagnosis_core import MEMBERS, exact_year_top, reason, uncertainty_scores

OUT=Path("artifacts/care-di25-future-miss-diagnosis-v1")
EXPECTED={2018:793,2019:399,2020:397,2021:400}
SCORE_CANDIDATES=(
 Path("artifacts/care-committee-decision-instability-router-future-v1/future_sample_scores.csv"),
 Path("artifacts/care-route-stage2-future-matrix-v1/preflight/future_sample_scores.csv"),
 Path("artifacts/care-route-stage2-future-matrix-v1/preflight/future_router_inputs_label_free.csv"),
 Path("artifacts/care-di25-hostego-future-v1/preflight/future_sample_scores.csv"),
)
LABEL_CANDIDATES=(
 Path("artifacts/care-di25-graphglobal-query-future-v1/results/future_sample_predictions.csv"),
 Path("artifacts/care-di25-hostego-future-v1/results/future_sample_predictions.csv"),
 Path("artifacts/care-di25-noego-safe-authority-full-v1/preflight/future_lineage_2018_2021_exact.csv"),
 Path("artifacts/care-future-local-assets-v2/future_lineage_2018_2021.csv"),
 Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-future-local-assets-v2/future_lineage_2018_2021.csv"),
 Path("artifacts/care-route-stage2-future-matrix-v1/preflight/future_cohort_label_free.csv"),
)
RESOLVED_CANDIDATES=(
 Path("artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json"),
 Path("artifacts/care-di25-hostego-future-v1/preflight/resolved_inputs.json"),
)


def read(path):
    with path.open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))


def norm(v):return str(v).strip().lower()


def id_of(r):return norm(r.get("sha256") or r.get("sample_id") or r.get("id") or "")


def label_of(v):
    x=str(v).strip().lower()
    if x in ("1","1.0","malware","malicious"):return 1
    if x in ("0","0.0","benign","goodware"):return 0
    raise ValueError("unknown label "+repr(v))


def resolve_scores():
    required={"p_stage1",*MEMBERS}
    for p in SCORE_CANDIDATES:
        if p.is_file() and p.stat().st_size>100:
            rows=read(p)
            if rows and required.issubset(rows[0]) and "year" in rows[0] and id_of(rows[0]):
                return p,rows
    raise RuntimeError("BLOCKED_BY_CONTRACT: no complete frozen future committee probability file")


def attach_labels(rows):
    def raw_label(r):
        return r.get("label") if str(r.get("label","")).strip() else r.get("metadata_label")
    if any(k in rows[0] for k in ("label","metadata_label")) and all(str(raw_label(r) or "").strip() for r in rows):
        return np.asarray([label_of(raw_label(r)) for r in rows]),"embedded_frozen_diagnostic"
    ids=[id_of(r) for r in rows]
    dynamic=[]
    for resolved in RESOLVED_CANDIDATES:
        if resolved.is_file():
            try:
                value=json.loads(resolved.read_text(encoding="utf-8")).get("future_label_file_posthoc_only")
                if value:dynamic.append(Path(value))
            except (OSError,ValueError,TypeError):
                pass
    seen=set()
    for p in (*dynamic,*LABEL_CANDIDATES):
        key=str(p.resolve()) if p.exists() else str(p)
        if key in seen:continue
        seen.add(key)
        if not p.is_file() or p.stat().st_size<100:continue
        lr=read(p)
        if not lr or not any(k in lr[0] for k in ("label","metadata_label")):continue
        try:m={id_of(r):label_of(raw_label(r)) for r in lr if id_of(r) and str(raw_label(r) or "").strip()}
        except ValueError:continue
        if all(i in m for i in ids):return np.asarray([m[i] for i in ids]),str(p)
    raise RuntimeError("BLOCKED_BY_CONTRACT: diagnostic labels cannot be joined exactly")


def write(path,rows):
    if not rows:
        raise RuntimeError("refusing to write empty audit table: "+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)


def q(a,p=.5):return float(np.quantile(a,p)) if len(a) else float("nan")


def main():
    path,rows=resolve_scores(); sha=np.asarray([id_of(r) for r in rows]); years=np.asarray([int(r["year"]) for r in rows])
    counts=Counter(years.tolist())
    if len(rows)!=1989 or len(set(sha.tolist()))!=1989 or dict(sorted(counts.items()))!=EXPECTED:
        raise RuntimeError("BLOCKED_BY_CONTRACT: future cohort changed "+repr(dict(counts)))
    base=np.asarray([float(r["p_stage1"]) for r in rows]); member=np.column_stack([[float(r[k]) for r in rows] for k in MEMBERS])
    if not np.isfinite(base).all() or not np.isfinite(member).all():raise RuntimeError("invalid frozen probabilities")
    score=uncertainty_scores(base,member)
    stored_di=np.asarray([float(r["U_DI"]) for r in rows],float) if "U_DI" in rows[0] else None
    di_score_max_abs_diff=(float(np.max(np.abs(stored_di-score["MeanOpposition_DI25"])))
                           if stored_di is not None else 0.0)
    if stored_di is not None:
        if di_score_max_abs_diff > 1e-10:
            raise RuntimeError("BLOCKED_BY_CONTRACT: stored U_DI formula changed, max_abs_diff="+str(di_score_max_abs_diff))
        # Preserve the exact serialized values used by the historical frozen route.
        score["MeanOpposition_DI25"]=stored_di
    routes={}; cutoffs={}
    for name,value in score.items():routes[name],cutoffs[name]=exact_year_top(value,years,sha,.25)
    route_reconciliation=[];route_diff=0
    if "route_DI" in rows[0]:
        frozen_di=np.asarray([int(float(r["route_DI"]))==1 for r in rows],bool)
        route_diff=int(np.sum(frozen_di!=routes["MeanOpposition_DI25"]))
        for i in np.flatnonzero(frozen_di!=routes["MeanOpposition_DI25"]):
            route_reconciliation.append({"sha256":sha[i],"year":int(years[i]),
                "U_DI":float(score["MeanOpposition_DI25"][i]),
                "route_DI_frozen":int(frozen_di[i]),
                "route_DI_sha_tiebreak_recomputed":int(routes["MeanOpposition_DI25"][i]),
                "reason":"historical frozen route retained; difference is deterministic cutoff tie handling"})
        routes["MeanOpposition_DI25"]=frozen_di
        cutoffs["MeanOpposition_DI25"]={yr:float(score["MeanOpposition_DI25"][frozen_di&(years==yr)].min()) for yr in EXPECTED}
    y,label_source=attach_labels(rows) # labels opened only after every route is frozen
    pred=base>=.5; error=pred!=y; di=routes["MeanOpposition_DI25"]
    expected_route={2018:198,2019:100,2020:99,2021:100}
    if {yr:int(di[years==yr].sum()) for yr in EXPECTED}!=expected_route:raise RuntimeError("DI route counts changed")

    detail=[]; summaries=[]; member_rows=[]; alternative=[]
    for year in (2020,2021):
        mask=years==year; caught=mask&error&di; missed=mask&error&~di
        cutoff=cutoffs["MeanOpposition_DI25"][year]
        type_count=Counter()
        for i in np.flatnonzero(missed):
            true_prob=np.where(y[i]==1,member[i],1-member[i]); votes=int(((member[i]>=.5)==y[i]).sum())
            why=reason(float(true_prob.mean()),votes,cutoff);type_count[why]+=1
            row={"sha256":sha[i],"year":year,"label":int(y[i]),"error_type":"FN" if y[i]==1 else "FP",
                 "p_stage1":base[i],"stage1_confidence":max(base[i],1-base[i]),"DI_score_true_class_mean":true_prob.mean(),
                 "DI_cutoff":cutoff,"gap_below_cutoff":cutoff-true_prob.mean(),"true_member_votes":votes,
                 "diagnosis":why}
            for j,k in enumerate(MEMBERS):row[k]=member[i,j];row[k+"__true_support"]=true_prob[j];row[k+"__correct"]=int((member[i,j]>=.5)==y[i])
            for name in routes:row["captured_by__"+name]=int(routes[name][i])
            detail.append(row)
        summaries.append({"year":year,"total_n":int(mask.sum()),"stage1_errors":int((mask&error).sum()),
            "DI_routed":int((mask&di).sum()),"DI_caught_errors":int(caught.sum()),"DI_missed_errors":int(missed.sum()),
            "DI_error_recall":float(caught.sum()/(mask&error).sum()),"DI_cutoff":cutoff,
            "missed_FN":int((missed&(y==1)).sum()),"missed_FP":int((missed&(y==0)).sum()),
            "missed_highconfidence":int((missed&(np.maximum(base,1-base)>=.8)).sum()),
            "missed_median_cutoff_gap":q(cutoff-score["MeanOpposition_DI25"][missed]),
            **{"reason__"+k:int(v) for k,v in sorted(type_count.items())}})
        for j,k in enumerate(MEMBERS):
            member_rows.append({"year":year,"member":k,"missed_n":int(missed.sum()),
                "correct_on_DI_missed":int((((member[:,j]>=.5)==y)&missed).sum()),
                "accuracy_on_DI_missed":float((((member[:,j]>=.5)==y)&missed).sum()/missed.sum()),
                "mean_true_support_on_DI_missed":float(np.where(y[missed]==1,member[missed,j],1-member[missed,j]).mean())})
        for name,route in routes.items():
            alternative.append({"year":year,"candidate":name,"DI_missed_n":int(missed.sum()),
                "DI_missed_captured":int((missed&route).sum()),"DI_missed_capture_rate":float((missed&route).sum()/missed.sum()),
                "all_error_capture":int((mask&error&route).sum()),"all_error_recall":float((mask&error&route).sum()/(mask&error).sum())})

    OUT.mkdir(parents=True,exist_ok=True);write(OUT/"missed_errors_2020_2021.csv",detail);write(OUT/"year_summary.csv",summaries)
    write(OUT/"member_behavior_on_misses.csv",member_rows);write(OUT/"alternative_uncertainty_capture.csv",alternative)
    if route_reconciliation:write(OUT/"route_tie_reconciliation.csv",route_reconciliation)
    lines=["CARE-Droid frozen DI25 2020/2021 miss diagnosis v1","",
           "Post-hoc diagnostic only. All routes were frozen before labels were opened.",
           "No training, threshold search, router modification, Stage-II invocation, or future-driven selection.",
           "score_file="+str(path),"label_source="+label_source,
           "frozen_route_used=true",
           "sha_tiebreak_recompute_difference_count="+str(route_diff),
           "stored_vs_formula_U_DI_max_abs_diff="+str(di_score_max_abs_diff),""]
    for s in summaries:
        lines.append(f"{s['year']}: errors={s['stage1_errors']} caught={s['DI_caught_errors']} missed={s['DI_missed_errors']} "
                     f"recall={s['DI_error_recall']:.6f} missed_FN={s['missed_FN']} missed_FP={s['missed_FP']} "
                     f"missed_highconf={s['missed_highconfidence']} cutoff={s['DI_cutoff']:.6f} median_gap={s['missed_median_cutoff_gap']:.6f}")
        for k,v in s.items():
            if k.startswith("reason__"):lines.append("  "+k.replace("reason__","")+"="+str(v))
    lines += ["","Interpretation:",
              "A = all four committee members share the Stage-I error (committee representation blind spot).",
              "B = only a minority member signal points to the true class and averaging dilutes it.",
              "C = hard-vote majority is correct but calibrated confidence averaging remains below 0.5.",
              "D = mean committee support already favors the true class, but the sample loses the fixed-budget ranking competition.",
              "Alternative-score capture is descriptive only and must not be used to tune a future-aware router."]
    (OUT/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("DI25_FUTURE_MISS_DIAGNOSIS_DONE");print("\n".join(lines))

if __name__=="__main__":main()

