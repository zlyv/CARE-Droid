#!/usr/bin/env python3
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from audit_common import *


def quantile(values, q):
    values = np.asarray(values, float); return float(np.quantile(values, q)) if len(values) else float("nan")


def main():
    prediction_path = OUTPUT / "proxy_predictions.npz"
    if not prediction_path.is_file(): raise RuntimeError("BLOCKED_BY_CONTRACT: proxy predictions missing")
    ids, y, folds, base, route, frozen = load_reference(); category = categories(y, base, frozen)
    with np.load(prediction_path, allow_pickle=False) as archive:
        original = np.asarray(archive["original"], float); predictions = {key:np.asarray(archive[key],float) for key in archive.files if "__r" in key}
    edits = read_csv(OUTPUT / "transformation_edits.csv"); edit_by={(norm(r["sha256"]),r["transform"],int(r["repeat"])):r for r in edits}
    replay_max = float(np.max(np.abs(original - frozen)))
    rows=[]; aggregate=[]; per_fold=[]; rng=np.random.default_rng(SEED); null_records=[]
    for name in TRANSFORMS:
        for repeat in range(3):
            transformed=predictions[f"{name}__r{repeat}"]; drift=np.abs(transformed-original); flip=(transformed>=.5)!=(original>=.5)
            routed=np.flatnonzero(route); broken=(category=="BROKEN")&route; comparison=(category!="BROKEN")&route
            score=auc((category[routed]=="BROKEN").astype(int),drift[routed])
            strata=np.asarray([f"{folds[i]}|{int(base[i]>=.5)}" for i in routed]); null=[]
            broken_label=(category[routed]=="BROKEN").astype(int)
            for permutation in range(500):
                shuffled=broken_label.copy()
                for stratum in np.unique(strata):
                    idx=np.flatnonzero(strata==stratum); shuffled[idx]=shuffled[rng.permutation(idx)]
                null.append(auc(shuffled,drift[routed]))
                null_records.append({"transform":name,"repeat":repeat,"permutation":permutation,"AUC":null[-1]})
            fold_direction=0
            for fold in range(1,6):
                b=(folds==fold)&broken; n=(folds==fold)&comparison
                bm=float(np.mean(drift[b])) if b.any() else float("nan"); nm=float(np.mean(drift[n])) if n.any() else float("nan")
                if math.isfinite(bm) and math.isfinite(nm) and bm>nm: fold_direction+=1
                per_fold.append({"transform":name,"repeat":repeat,"fold":fold,"routed":int(np.sum(route&(folds==fold))),
                    "broken_n":int(b.sum()),"nonbroken_n":int(n.sum()),"broken_mean_abs_drift":bm,"nonbroken_mean_abs_drift":nm,
                    "flip_rate":float(np.mean(flip[route&(folds==fold)]))})
            applicability=[]; edited_nodes=[]; edited_edges=[]
            for i in routed:
                edit=edit_by[(ids[i],name,repeat)]; applicable=str(edit.get("applicable","")).casefold() in {"true","1"}
                applicability.append(applicable); edited_nodes.append(float(edit.get("edited_nodes",0))); edited_edges.append(float(edit.get("edited_edges",0)))
                rows.append({"sha256":ids[i],"fold":int(folds[i]),"label":int(y[i]),"category":category[i],"stage1_prob":float(base[i]),
                    "original_direct_prob":float(original[i]),"transform":name,"repeat":repeat,"transformed_prob":float(transformed[i]),
                    "abs_drift":float(drift[i]),"signed_drift":float(transformed[i]-original[i]),"prediction_flip":int(flip[i]),
                    "malware_to_benign":int(original[i]>=.5 and transformed[i]<.5),"benign_to_malware":int(original[i]<.5 and transformed[i]>=.5),
                    "applicable":int(applicable),"edited_nodes":edited_nodes[-1],"edited_edges":edited_edges[-1]})
            aggregate.append({"transform":name,"repeat":repeat,"routed":len(routed),"applicability":float(np.mean(applicability)),
                "mean_abs_drift":float(np.mean(drift[routed])),"median_abs_drift":float(np.median(drift[routed])),"p90_abs_drift":quantile(drift[routed],.9),
                "max_abs_drift":float(np.max(drift[routed])),"flip_count":int(np.sum(flip[routed])),"flip_rate":float(np.mean(flip[routed])),
                "malware_to_benign":int(np.sum((original[routed]>=.5)&(transformed[routed]<.5))),
                "benign_to_malware":int(np.sum((original[routed]<.5)&(transformed[routed]>=.5))),
                "broken_mean_abs_drift":float(np.mean(drift[broken])),"nonbroken_mean_abs_drift":float(np.mean(drift[comparison])),
                "broken_vs_nonbroken_AUC":score,"shuffle_AUC_95pct":quantile(null,.95),"folds_broken_drift_higher":fold_direction,
                "drift_vs_edited_nodes_spearman":spearman(drift[routed],edited_nodes),"drift_vs_edited_edges_spearman":spearman(drift[routed],edited_edges)})
    write_csv(OUTPUT/"apk_proxy_drift.csv",rows); write_csv(OUTPUT/"transform_summary.csv",aggregate); write_csv(OUTPUT/"fold_summary.csv",per_fold)
    write_csv(OUTPUT/"matched_label_shuffle_null.csv",null_records)
    # Average repeat-level evidence before applying the frozen gate.
    by_name=defaultdict(list)
    for row in aggregate: by_name[row["transform"]].append(row)
    averaged={name:{key:float(np.mean([float(r[key]) for r in values])) for key in (
        "applicability","mean_abs_drift","flip_rate","broken_mean_abs_drift","nonbroken_mean_abs_drift","broken_vs_nonbroken_AUC","shuffle_AUC_95pct","folds_broken_drift_higher")}
        for name,values in by_name.items()}
    order_ok=all(row["flip_count"]==0 and row["max_abs_drift"]<=1e-5 for row in aggregate if row["transform"]=="T0_ORDER")
    candidates=[]
    for name in ("T1_WRAPPER_SENSITIVE","T2_DEAD_SUBGRAPH","T3_DEAD_EDGE_NEAR_HOST"):
        value=averaged[name]
        matched_ok = value["mean_abs_drift"] > averaged["N1_WRAPPER_FAR"]["mean_abs_drift"] if name=="T1_WRAPPER_SENSITIVE" else True
        criteria={"applicability>=0.80":value["applicability"]>=.80,"broken_AUC>=0.65":value["broken_vs_nonbroken_AUC"]>=.65,
            "broken_AUC>shuffle95+0.03":value["broken_vs_nonbroken_AUC"]>=value["shuffle_AUC_95pct"]+.03,
            "folds>=4":value["folds_broken_drift_higher"]>=4,"matched_location_control":matched_ok}
        if all(criteria.values()): candidates.append({"transform":name,"criteria":criteria,**value})
    real_names=("T1_WRAPPER_SENSITIVE","T2_DEAD_SUBGRAPH","T3_DEAD_EDGE_NEAR_HOST")
    any_real_transform=max(averaged[name]["applicability"] for name in real_names)>0
    if not order_ok:
        status="IMPLEMENTATION_INVARIANCE_FAILED"
    elif not any_real_transform:
        status="IMPLEMENTATION_TRANSFORM_NOT_APPLIED"
    elif candidates:
        status="GO_REAL_APK_PILOT"
    else:
        status="CACHE_PROXY_NOT_LINKED_TO_BROKEN"
    decision={"status":status,"representation_level_proxy":True,"real_apk_transformation_run":False,"future_rows_loaded":0,
        "frozen_direct_reference":{"fixed":49,"broken":17,"net":32},"replay_max_abs":replay_max,"T0_order_exact":order_ok,
        "passing_candidates":candidates,"averaged":averaged,"any_real_transform_applied":any_real_transform,
        "interpretation":"Only RoleGraph/S7 structure and Host row order were perturbed; Compact43 Host descriptors were held fixed. A pass authorizes a real transformed-APK pilot, not EC-C43 training."}
    dump_json(OUTPUT/"final_decision.json",decision)
    report=f"""# CARE-Droid C43 equivalence-sensitivity proxy audit v1

Final status: **{status}**.

This is a representation-level proxy audit. It does not claim transformed-APK semantic equivalence or verified robustness. P31, SVM routing, Direct-C43 weights, Compact43 Host descriptors and threshold 0.5 remain frozen. Only endpoint-derived RoleGraph/S7 structure and Host row order are perturbed.

## Frozen reference

- Routed APKs: 200
- Direct-C43: fixed=49, broken=17, net=32
- Original replay max probability difference: {replay_max:.8g}
- T0 order invariance: {order_ok}

## Proxy results

| Transform | Applicability | Mean drift | Flip rate | Broken AUC | Shuffle95 | Folds | 
|---|---:|---:|---:|---:|---:|---:|
"""
    for name in TRANSFORMS:
        v=averaged[name]; report += f"| {name} | {v['applicability']:.4f} | {v['mean_abs_drift']:.6f} | {v['flip_rate']:.4f} | {v['broken_vs_nonbroken_AUC']:.4f} | {v['shuffle_AUC_95pct']:.4f} | {v['folds_broken_drift_higher']:.2f} |\n"
    report += f"\n## Decision boundary\n\nPassing candidates: `{json.dumps([r['transform'] for r in candidates])}`.\n\nOnly a passing proxy authorizes a 20--30 APK real bytecode transformation pilot. Training EC-C43 remains unauthorized until cache-to-real agreement is measured. Future rows loaded: 0.\n"
    ensure(OUTPUT/"report.md").write_text(report,encoding="utf-8")
    print("C43_EQUIVALENCE_SENSITIVITY_STATUS="+status,flush=True)


if __name__=="__main__": main()
