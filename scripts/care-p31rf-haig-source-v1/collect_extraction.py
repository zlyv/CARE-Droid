#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from haig_model import read_csv,load_graphs,write_csv

def main():
    p=argparse.ArgumentParser();p.add_argument("--lineage",type=Path,required=True);p.add_argument("--graphs",type=Path,required=True);p.add_argument("--scope",required=True);a=p.parse_args()
    rows=read_csv(a.lineage);graphs,fail=load_graphs(rows,a.graphs);out=[]
    for g in graphs:out.append({"sha256":g.get("sha256"),"year":g.get("year"),"status":g.get("status"),**g.get("stats",{})})
    root=Path("artifacts/care-p31rf-haig-source-v1");write_csv(root/f"{a.scope}_apk_graph_stats.csv",out);write_csv(root/f"{a.scope}_failures.csv",fail)
    if a.scope=="source":write_csv(root/"apk_graph_stats.csv",out)
    ok=[g for g in graphs if g.get("status")=="OK"]
    def vals(k):return np.asarray([float(g.get("stats",{}).get(k,0)) for g in ok])
    summary={"scope":a.scope,"rows":len(rows),"ok":len(ok),"failed":len(rows)-len(ok)}
    for k in ("num_api_nodes","num_hosts","num_same_host_edges","num_host_call_edges","unique_api_count","unique_pair_count","extraction_seconds"):
        v=vals(k);summary[k]={"mean":float(v.mean()) if len(v) else 0,"median":float(np.median(v)) if len(v) else 0,"p90":float(np.quantile(v,.9)) if len(v) else 0,"max":float(v.max()) if len(v) else 0,"total":float(v.sum()) if len(v) else 0}
    summary["same_edge_cap_hits"]=sum(bool(g.get("stats",{}).get("same_edge_cap_hit")) for g in ok);summary["call_edge_cap_hits"]=sum(bool(g.get("stats",{}).get("call_edge_cap_hit")) for g in ok)
    (root/f"{a.scope}_extraction_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    if a.scope=="source":(root/"extraction_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    coverage=[{"scope":a.scope,"rows":len(rows),"successful_rows":len(ok),"API_node_coverage":float(np.mean(vals("num_api_nodes")>0)) if ok else 0,"same_host_edge_coverage":float(np.mean(vals("num_same_host_edges")>0)) if ok else 0,"host_call_edge_coverage":float(np.mean(vals("num_host_call_edges")>0)) if ok else 0,"mean_extraction_seconds":summary["extraction_seconds"]["mean"],"failed":len(rows)-len(ok)}]
    write_csv(root/f"{a.scope}_coverage_cost.csv",coverage)
    print(f"HAIG_{a.scope.upper()}_EXTRACTION_SUMMARY rows={len(rows)} ok={len(ok)} failed={len(rows)-len(ok)}")
    if len(ok)!=len(rows):raise RuntimeError(f"BLOCKED_BY_CONTRACT: {a.scope} extraction has {len(rows)-len(ok)} failed APKs; inspect failures.csv")
    if summary["same_edge_cap_hits"] or summary["call_edge_cap_hits"]:raise RuntimeError("BLOCKED_BY_CONTRACT: graph edge cap hit; representation would be truncated")
if __name__=="__main__":main()
