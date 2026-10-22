#!/usr/bin/env python3
import numpy as np
from common import FOLDS,OUTPUT,dump_json,load,metrics,write_csv

def contiguous_regions(rows,key):
    good=[r for r in rows if r[key]];regions=[]
    if not good:return regions
    current=[good[0]]
    for row in good[1:]:
        if round(row["threshold"]-current[-1]["threshold"],10)==.01:current.append(row)
        else:regions.append(current);current=[row]
    regions.append(current)
    return [{"start":r[0]["threshold"],"end":r[-1]["threshold"],"grid_points":len(r),"width":round(r[-1]["threshold"]-r[0]["threshold"],10),
             "min_fixed":min(x["fixed"] for x in r),"max_broken":max(x["broken"] for x in r),"min_net":min(x["net"] for x in r)} for r in regions]

def main():
    ids,y,fold,base,bounded,route,direct,field=load();thresholds=np.round(np.arange(.01,1,.01),2);rows=[];fold_rows=[]
    bounded_folds={}
    for f in FOLDS:
        z=fold==f;bounded_folds[f]=metrics(y[z],base[z],bounded[z],route[z])
    for threshold in thresholds:
        final=base.copy();final[route]=(direct[route]>=threshold).astype(float)
        value=metrics(y,base,final,route);fold_values=[]
        for f in FOLDS:
            z=fold==f;fv=metrics(y[z],base[z],final[z],route[z]);fold_values.append(fv)
            fold_rows.append({"threshold":threshold,"fold":f,**fv,"bounded_net":bounded_folds[f]["net"],"net_vs_bounded":fv["net"]-bounded_folds[f]["net"]})
        nonnegative=sum(v["net"]>=0 for v in fold_values);noninferior=sum(v["net"]>=bounded_folds[f]["net"] for f,v in zip(FOLDS,fold_values))
        practical=(value["fixed"]>=40 and value["broken"]<=10 and value["net"]>=30 and value["fixed_FN"]>=9 and value["fixed_FP"]>=20 and nonnegative>=4)
        strict=(value["fixed"]>29 and value["broken"]<=4 and value["fixed_FN"]>=9 and value["fixed_FP"]>=20 and nonnegative>=4)
        rows.append({"threshold":threshold,**value,"folds_net_nonnegative":nonnegative,"folds_net_noninferior_to_bounded":noninferior,"practical_region":int(practical),"strict_pareto":int(strict)})
    practical_regions=contiguous_regions(rows,"practical_region");strict_regions=contiguous_regions(rows,"strict_pareto")
    broad=[r for r in practical_regions if r["grid_points"]>=3]
    pareto=[]
    for row in rows:
        dominated=any((other["fixed"]>=row["fixed"] and other["broken"]<=row["broken"] and (other["fixed"]>row["fixed"] or other["broken"]<row["broken"])) for other in rows)
        if not dominated:pareto.append(row)
    best_net=max(rows,key=lambda r:(r["net"],-r["broken"],abs(r["threshold"]-.5)))
    min_broken_40=min((r for r in rows if r["fixed"]>=40),key=lambda r:(r["broken"],-r["net"]),default=None)
    if strict_regions:status="DIRECT_C43_STRICT_PARETO_REGION_FOUND"
    elif broad:status="DIRECT_C43_BROAD_FRONTIER_WORTH_NESTED_TEST"
    else:status="DIRECT_C43_GLOBAL_THRESHOLD_NOT_STABLY_SUPPORTED"
    decision={"status":status,"future_rows_loaded":0,"deployable_threshold_selected":False,"strict_pareto_regions":strict_regions,"practical_regions":practical_regions,
        "broad_practical_regions":broad,"best_net_descriptive":best_net,"minimum_broken_with_fixed_at_least_40":min_broken_40,
        "bounded_reference":{"fixed":29,"broken":4,"net":25},"direct_t050_reference":{"fixed":49,"broken":17,"net":32},
        "next_action":"run a separately frozen strict nested calibration only if a strict or broad stable region exists; otherwise stop threshold tuning"}
    write_csv(OUTPUT/"threshold_sweep.csv",rows);write_csv(OUTPUT/"fold_threshold_sweep.csv",fold_rows);write_csv(OUTPUT/"pareto_frontier.csv",pareto)
    dump_json(OUTPUT/"frontier_decision.json",decision);dump_json(OUTPUT/"final_decision.json",decision)
    print("DIRECT_C43_FRONTIER_STATUS="+status,flush=True)
    print("best_net threshold={threshold:.2f} fixed={fixed} broken={broken} net={net}".format(**best_net),flush=True)
    if min_broken_40:print("fixed>=40 minimum_broken threshold={threshold:.2f} fixed={fixed} broken={broken} net={net}".format(**min_broken_40),flush=True)
if __name__=="__main__":main()
